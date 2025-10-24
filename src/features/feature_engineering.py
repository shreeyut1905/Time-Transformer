"""Feature engineering for price/return based modeling.

All features are computed per ticker, then rows with any NA (including the forward
target) are dropped. Target is next day's simple return (next_return).
"""

import pandas as pd
import numpy as np
from scipy.stats import skew as _skew, kurtosis as _kurt


def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))


def build_features(
    df: pd.DataFrame,
    sma_windows=(5, 10, 20, 50),
    rsi_window: int = 14,
    volatility_window: int = 20
) -> pd.DataFrame:
    
    df = df.sort_values(['ticker', 'date']).copy()

    df['return'] = df.groupby('ticker')['adj_close'].pct_change()
    df['log_return'] = np.log1p(df['return'])

    grp_price = df.groupby('ticker')['adj_close']
    for w in sma_windows:
        sma_col = f'sma_{w}'
        df[sma_col] = grp_price.transform(lambda x, win=w: x.rolling(win).mean())
        df[f'rel_sma_{w}'] = df['adj_close'] / df[sma_col] - 1

    df['rsi'] = grp_price.transform(lambda x: compute_rsi(x, window=rsi_window))
    df['volatility'] = df.groupby('ticker')['return'].transform(lambda x: x.rolling(volatility_window).std())
    df['rolling_max'] = grp_price.transform(lambda x: x.rolling(50).max())
    df['drawdown_pct'] = df['adj_close'] / df['rolling_max'] - 1

    def _macd(x: pd.Series):
        ema12 = x.ewm(span=12, adjust=False).mean()
        ema26 = x.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        return macd, signal, macd - signal

    macd_tuple = grp_price.transform(lambda x: _macd(x)[0])
    df['macd'] = macd_tuple
    df['macd_signal'] = grp_price.transform(lambda x: _macd(x)[1])
    df['macd_hist'] = grp_price.transform(lambda x: _macd(x)[2])

    bb_mid = grp_price.transform(lambda x: x.rolling(20).mean())
    bb_std = grp_price.transform(lambda x: x.rolling(20).std())
    df['bb_upper'] = bb_mid + 2 * bb_std
    df['bb_lower'] = bb_mid - 2 * bb_std
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / (bb_mid + 1e-9)
    df['bb_pos'] = (df['adj_close'] - bb_mid) / (2 * bb_std + 1e-9)

    for _, sub in df.groupby('ticker'):
        tr_high = sub['high']
        tr_low = sub['low']
        prev_close = sub['adj_close'].shift(1)
        tr_range = np.maximum(tr_high - tr_low, np.maximum((tr_high - prev_close).abs(), (tr_low - prev_close).abs()))
        df.loc[sub.index, 'atr14'] = tr_range.rolling(14).mean()

    for lag in (1, 2, 3, 5, 10):
        df[f'return_lag_{lag}'] = df.groupby('ticker')['return'].shift(lag)

    for _, sub in df.groupby('ticker'):
        r = sub['return']
        roll = r.rolling(20)
        df.loc[sub.index, 'ret_skew_20'] = roll.apply(lambda x: _skew(x, bias=False) if len(x.dropna()) == 20 else np.nan, raw=False)
        df.loc[sub.index, 'ret_kurt_20'] = roll.apply(lambda x: _kurt(x, fisher=True, bias=False) if len(x.dropna()) == 20 else np.nan, raw=False)

    df['next_return'] = df.groupby('ticker')['return'].shift(-1)
    return df.dropna().reset_index(drop=True)
