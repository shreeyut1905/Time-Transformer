import pandas as pd
import numpy as np


def compute_performance(equity_curve: pd.Series, daily_returns: pd.Series, rf_rate: float = 0.0) -> dict:
    equity_curve = equity_curve.dropna()
    daily_returns = daily_returns.loc[equity_curve.index]
    total_return = equity_curve.iloc[-1]/equity_curve.iloc[0] - 1
    years = len(equity_curve)/252
    cagr = (equity_curve.iloc[-1]/equity_curve.iloc[0]) ** (1/years) - 1 if years > 0 else np.nan
    excess = daily_returns - rf_rate/252
    sharpe = np.sqrt(252) * excess.mean() / (excess.std() + 1e-9)
    neg = excess[excess < 0]
    sortino = np.sqrt(252) * excess.mean() / (neg.std() + 1e-9)
    rolling_max = equity_curve.cummax()
    drawdown = equity_curve/rolling_max - 1
    max_dd = drawdown.min()
    trade_rets = daily_returns[daily_returns != 0]
    wins = trade_rets[trade_rets > 0]
    losses = trade_rets[trade_rets < 0]
    hit_rate = len(wins) / max(len(trade_rets), 1)
    avg_win = wins.mean() if len(wins) else 0.0
    avg_loss = losses.mean() if len(losses) else 0.0
    return {
        'total_return': float(total_return),
        'CAGR': float(cagr),
        'Sharpe': float(sharpe),
        'Sortino': float(sortino),
        'MaxDrawdown': float(max_dd),
        'HitRate': float(hit_rate),
        'AvgWin': float(avg_win),
        'AvgLoss': float(avg_loss),
    }
