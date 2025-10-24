
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from ..config import load_config
from .metrics import compute_performance
from ..utils.logger import get_logger


def baseline_signals(df: pd.DataFrame, short=10, long=50):
    df = df.copy()
    df['sma_short'] = df['adj_close'].rolling(short).mean()
    df['sma_long'] = df['adj_close'].rolling(long).mean()
    df['baseline_signal'] = (df['sma_short'] > df['sma_long']).astype(int)
    return df


def model_signals(preds: pd.DataFrame, cfg: dict):
    
    df = preds.copy()
    if 'model_pred' not in df.columns:
        return df
    signal_cfg = cfg['backtest'].get('signal', {})
    method = signal_cfg.get('method', 'raw')
    if method == 'raw':
        thr_bps = signal_cfg.get('raw_threshold_bps', 0)
        thr = thr_bps / 10000.0
        df['model_signal'] = (df['model_pred'] > thr).astype(int)
    elif method == 'vol_adj':
        vol_window = signal_cfg.get('vol_window', 20)
        vol_mult = signal_cfg.get('vol_multiplier', 0.3)
        df['realized_vol'] = df['return'].rolling(vol_window).std()
        df['model_signal'] = (df['model_pred'] > vol_mult * df['realized_vol']).astype(int)
    else:
        df['model_signal'] = (df['model_pred'] > 0).astype(int)
    return df


def apply_execution_delay(signals: pd.Series) -> pd.Series:
    """Assumption: Shift signals forward one bar to simulate entering next period."""
    return signals.shift(1)


def backtest(df: pd.DataFrame, signal_col: str, cost_bps: float, initial_capital: float):
    """Run a simple long-only backtest with per-trade bps costs."""
    df = df.copy()
    df['position'] = apply_execution_delay(df[signal_col]).fillna(0)
    df['trade'] = df['position'].diff().abs().fillna(0)
    df['cost'] = df['trade'] * cost_bps / 10000.0
    df['strategy_ret'] = df['position'] * df['return'] - df['cost']
    df['equity'] = (1 + df['strategy_ret']).cumprod() * initial_capital
    return df


def plot_equity(curves: dict, outpath: Path):
    
    plt.figure(figsize=(10,6))
    for name, series in curves.items():
        plt.plot(series.index, series.values, label=name)
    plt.legend()
    plt.title('Equity Curves')
    plt.ylabel('Equity')
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--run_all', action='store_true')
    parser.add_argument('--predictions_csv', default='data/artifacts/walkforward/predictions.csv')
    parser.add_argument('--features_csv', default='data/processed/features.csv')
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = get_logger('backtest', log_dir='data/artifacts/logs')
    feat = pd.read_csv(args.features_csv, parse_dates=['date'])
    preds = pd.read_csv(args.predictions_csv, parse_dates=['date']) if Path(args.predictions_csv).exists() else None
    df = feat.merge(preds[['date','model_pred']] if preds is not None else pd.DataFrame(), on='date', how='left')
    df = baseline_signals(df)
    if preds is not None:
        df = model_signals(df, cfg)
    reports_dir = Path(cfg['report']['output_dir'])
    reports_dir.mkdir(parents=True, exist_ok=True)

    curves = {}
    metrics = {}
    base_bt = backtest(df, 'baseline_signal', cfg['backtest']['cost_bps'], cfg['backtest']['initial_capital'])
    curves['Baseline'] = base_bt.set_index('date')['equity']
    metrics['Baseline'] = compute_performance(curves['Baseline'], base_bt.set_index('date')['strategy_ret'], cfg['backtest']['risk_free_rate'])
    if 'model_signal' in df.columns:
        model_bt = backtest(df, 'model_signal', cfg['backtest']['cost_bps'], cfg['backtest']['initial_capital'])
        curves['Model'] = model_bt.set_index('date')['equity']
        metrics['Model'] = compute_performance(curves['Model'], model_bt.set_index('date')['strategy_ret'], cfg['backtest']['risk_free_rate'])

    if cfg['report']['save_plots']:
        plot_equity(curves, reports_dir / 'equity_curves.png')
    if cfg['report']['save_csv']:
        for name, series in curves.items():
            series.to_csv(reports_dir / f'equity_{name.lower()}.csv')
    if cfg['report']['save_json']:
        with (reports_dir / 'metrics.json').open('w') as f:
            json.dump(metrics, f, indent=2)
    logger.info(f"Metrics: {metrics}")


if __name__ == '__main__':
    main()
