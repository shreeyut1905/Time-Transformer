import argparse
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from torch.optim import Adam
from torch.nn import MSELoss
from torch.utils.data import DataLoader
from .datasets.timeseries import make_dataloader
from .models.lstm_model import LSTMReturnModel
from .models.ts_transformer import TimeSeriesTransformer
from .models.etsformer_wrapper import ETSFormerWrapper
from .utils.logger import get_logger

SCALERS = {
    'standard': StandardScaler,
    'robust': RobustScaler,
    'minmax': MinMaxScaler
}


def train_one(model, loader, optimizer, criterion, device):
    model.train()
    total = 0
    for xb, yb in loader:
        print(f"xb shape: {xb.shape}, yb shape: {yb.shape}")
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        total += loss.item() * xb.size(0)
    return total / len(loader.dataset)


def evaluate(model, loader, criterion, device):
    model.eval()
    total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb)
            loss = criterion(preds, yb)
            total += loss.item() * xb.size(0)
    return total / len(loader.dataset)


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


def run_train(train_df: pd.DataFrame, val_df: pd.DataFrame, feature_cols, target_col: str, cfg: dict, device: str, artifacts_dir: Path):
    lookback = cfg['model'].get('lookback', 30)
    train_loader = make_dataloader(train_df, feature_cols, target_col, lookback, cfg['model']['batch_size'], shuffle=True)
    val_loader = make_dataloader(val_df, feature_cols, target_col, lookback, cfg['model']['batch_size'], shuffle=False)
    model = build_model(len(feature_cols), cfg).to(device)
    optimizer = Adam(model.parameters(), lr=cfg['model']['lr'])
    criterion = MSELoss()
    best_val = float('inf')
    best_path = artifacts_dir / 'best_model.pt'
    for epoch in range(1, cfg['model']['epochs']+1):
        tr_loss = train_one(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        if val_loss < best_val:
            best_val = val_loss
            torch.save({'state_dict': model.state_dict(), 'val_loss': best_val}, best_path)
        print(f"Epoch {epoch} | train {tr_loss:.6f} | val {val_loss:.6f} | best {best_val:.6f}")
    return best_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--features_csv', default='data/processed/features.csv')
    parser.add_argument('--scaler', default='standard')
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()

    logger = get_logger('train')
    df = pd.read_csv(args.features_csv, parse_dates=['date'])
    feature_cols = [c for c in df.columns if c not in ('date', 'ticker', 'next_return')]
    target_col = 'next_return'
    split = int(len(df)*0.8)
    train_df = df.iloc[:split].copy()
    val_df = df.iloc[split:].copy()
    scaler_cls = SCALERS[args.scaler]
    scaler = scaler_cls()
    scaler.fit(train_df[feature_cols])
    train_df[feature_cols] = scaler.transform(train_df[feature_cols])
    val_df[feature_cols] = scaler.transform(val_df[feature_cols])

    cfg = {
        'model': {
            'type': 'etsformer',
            'hidden_size': 64,  # for LSTM fallback
            'num_layers': 2,
            'dropout': 0.2,
            'lr': 7e-4,
            'epochs': args.epochs,
            'batch_size': 64,
            'lookback': 90,
            'transformer': {
                'd_model': 128,
                'n_heads': 4,
                'n_layers': 4,
                'dim_feedforward': 256,
                'dropout': 0.1
            },
            'etsformer': {
                'time_features': 8,
                'model_dim': 128,
                'layers': 2,
                'heads': 4,
                'K': 4,
                'dropout': 0.2
            }
        }
    }
    artifacts_dir = Path('data/artifacts/single_train')
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    run_train(train_df, val_df, feature_cols, target_col, cfg, device, artifacts_dir)


if __name__ == '__main__':
    main()
