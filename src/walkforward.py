"""Walk-forward training and evaluation.

The data is split sequentially into expanding train windows and fixed-size test
windows. Each fold trains a fresh model, produces predictions (one per day
after a lookback warmup), and logs regression metrics.
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from .config import load_config
from .models.lstm_model import LSTMReturnModel
from .models.ts_transformer import TimeSeriesTransformer
from .models.etsformer_wrapper import ETSFormerWrapper
from .datasets.timeseries import make_dataloader
from .evaluation.regression_metrics import compute_regression_metrics, aggregate_fold_metrics
from torch.optim import Adam
from torch.nn import MSELoss
from torch.cuda.amp import autocast, GradScaler
from .utils.logger import get_logger

SCALERS = {
    'standard': StandardScaler,
    'robust': RobustScaler,
    'minmax': MinMaxScaler
}


def build_model(input_size: int, cfg: dict):
    mtype = cfg['model'].get('type', 'lstm')
    if mtype == 'transformer':
        t = cfg['model']['transformer']
        return TimeSeriesTransformer(
            input_size=input_size,
            d_model=t['d_model'],
            n_heads=t['n_heads'],
            n_layers=t['n_layers'],
            dim_feedforward=t['dim_feedforward'],
            dropout=t['dropout']
        )
    if mtype == 'etsformer':
        e = cfg['model']['etsformer']
        return ETSFormerWrapper(
            input_size=input_size,
            time_features=e.get('time_features', 4),
            model_dim=e.get('model_dim', 128),
            layers=e.get('layers', 2),
            heads=e.get('heads', 4),
            K=e.get('K', 4),
            dropout=e.get('dropout', 0.1)
        )
    return LSTMReturnModel(input_size=input_size, hidden_size=cfg['model']['hidden_size'], num_layers=cfg['model']['num_layers'], dropout=cfg['model']['dropout'])


def train_fold(train_df, feature_cols, target_col, cfg, device):
    """Train a single fold model on scaled training dataframe."""
    lookback = cfg['model'].get('lookback', 30)
    loader = make_dataloader(train_df, feature_cols, target_col, lookback, cfg['model']['batch_size'], shuffle=True)
    model = build_model(len(feature_cols), cfg).to(device)
    opt = Adam(model.parameters(), lr=cfg['model']['lr'])
    crit = MSELoss()
    model_type = cfg['model'].get('type', 'lstm')
    use_amp = device.startswith('cuda') and model_type != 'etsformer'
    scaler = GradScaler(enabled=use_amp)
    model.train()
    for _ in range(cfg['model']['epochs']):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            with autocast(enabled=use_amp):
                pred = model(xb)
                loss = crit(pred, yb)
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward(); opt.step()
    return model


def predict_fold(model, test_df, feature_cols, target_col, device, cfg):
    """Generate predictions for the test slice (one per row after lookback)."""
    model.eval()
    lookback = cfg['model'].get('lookback', 30)
    if len(test_df) <= lookback:
        return pd.DataFrame(columns=list(test_df.columns) + ['model_pred'])
    X = test_df[feature_cols].values.astype('float32')
    preds = np.full(len(test_df), np.nan, dtype='float32')
    with torch.no_grad():
        for end in range(lookback, len(test_df)):
            start = end - lookback
            window = torch.tensor(X[start:end]).unsqueeze(0).to(device)
            p = model(window).cpu().item()
            preds[end] = p
    out = test_df.copy()
    out['model_pred'] = preds
    return out.dropna(subset=['model_pred'])


def walk_forward(df: pd.DataFrame, cfg: dict, logger):
    """Run sequential expanding-window walk-forward evaluation.

    Returns predictions dataframe, list of per-fold metric dicts, and
    aggregate metric summary.
    """
    feature_cols = [c for c in df.columns if c not in ('date', 'ticker', 'next_return')]
    target_col = 'next_return'
    folds = []
    fold_metrics: list[dict] = []
    min_train = cfg['walkforward']['min_train_size']
    test_win = cfg['walkforward']['test_window']
    max_folds = cfg['walkforward']['max_folds']
    scaler_cls = SCALERS[cfg['walkforward']['scaler']]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    total_len = len(df)
    lookback = cfg['model'].get('lookback', 30)

    if total_len < (min_train + test_win + lookback + 1):
        original = (min_train, test_win)
        min_train = max(lookback * 2, int(total_len * 0.5))
        residual = total_len - min_train
        test_win = max(lookback + 5, residual - lookback - 1) if residual > lookback else lookback + 5
        logger.warning(f"Adjusted walk-forward windows due to limited data: train {original[0]} -> {min_train}, test {original[1]} -> {test_win}, total_len={total_len}")

    start_test = min_train
    fold_idx = 0
    while start_test + lookback < total_len:
        fold_idx += 1
        end_test = min(start_test + test_win, total_len)
        train_df = df.iloc[:start_test].copy()
        test_df = df.iloc[start_test:end_test].copy()
        scaler = scaler_cls()
        scaler.fit(train_df[feature_cols])
        train_df[feature_cols] = scaler.transform(train_df[feature_cols])
        test_df[feature_cols] = scaler.transform(test_df[feature_cols])
        model = train_fold(train_df, feature_cols, target_col, cfg, device)
        test_with_preds = predict_fold(model, test_df, feature_cols, target_col, device, cfg)
        test_with_preds['fold'] = fold_idx
        folds.append(test_with_preds)
        regm = compute_regression_metrics(test_with_preds, pred_col='model_pred', target_col=target_col)
        regm['fold'] = fold_idx
        regm['train_size'] = len(train_df)
        regm['test_size'] = len(test_df)
        fold_metrics.append(regm)
        logger.info(
            f"Fold {fold_idx} | train {len(train_df)} | test {len(test_df)} | RMSE {regm['rmse']:.6f} | MAE {regm['mae']:.6f} | DirAcc {regm['directional_accuracy']:.3f}")
        if max_folds and fold_idx >= max_folds:
            break
        start_test += test_win
    out = pd.concat(folds).reset_index(drop=True)
    agg = aggregate_fold_metrics(fold_metrics)
    return out, fold_metrics, agg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--features_csv', default='data/processed/features.csv')
    args = parser.parse_args()
    cfg = load_config(args.config)
    logger = get_logger('walkforward', log_dir='data/artifacts/logs')
    df = pd.read_csv(args.features_csv, parse_dates=['date'])
    out, fold_metrics, agg_metrics = walk_forward(df, cfg, logger)
    artifacts_dir = Path('data/artifacts/walkforward')
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(artifacts_dir / 'predictions.csv', index=False)
    pd.DataFrame(fold_metrics).to_csv(artifacts_dir / 'fold_regression_metrics.csv', index=False)
    pd.DataFrame([agg_metrics]).to_csv(artifacts_dir / 'aggregate_regression_metrics.csv', index=False)
    logger.info(f"Saved predictions to {artifacts_dir / 'predictions.csv'}")
    logger.info(f"Aggregate regression metrics: {agg_metrics}")


if __name__ == '__main__':
    main()
