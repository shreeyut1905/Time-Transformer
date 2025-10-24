
import pandas as pd
import numpy as np
from math import sqrt


def _clean(df: pd.DataFrame, pred_col: str, target_col: str):
    return df[[pred_col, target_col]].dropna()


def compute_regression_metrics(df: pd.DataFrame, pred_col: str = 'model_pred', target_col: str = 'next_return') -> dict:
    data = _clean(df, pred_col, target_col)
    if data.empty:
        return {
            'mse': np.nan,
            'rmse': np.nan,
            'mae': np.nan,
            'r2': np.nan,
            'corr_pearson': np.nan,
            'corr_spearman': np.nan,
            'directional_accuracy': np.nan,
            'count': 0
        }
    y = data[target_col].values
    p = data[pred_col].values
    mse = float(np.mean((p - y) ** 2))
    rmse = float(sqrt(mse))
    mae = float(np.mean(np.abs(p - y)))
    ss_tot = np.sum((y - y.mean()) ** 2)
    ss_res = np.sum((y - p) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot != 0 else np.nan
    corr_pearson = float(np.corrcoef(p, y)[0, 1]) if len(data) > 1 else np.nan
    corr_spearman = float(pd.Series(p).corr(pd.Series(y), method='spearman')) if len(data) > 1 else np.nan
    nonzero_mask = (y != 0)
    nz_count = nonzero_mask.sum()
    if nz_count == 0:
        directional_accuracy = np.nan
    else:
        directional_accuracy = float(((np.sign(p[nonzero_mask]) == np.sign(y[nonzero_mask]))).mean())
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'corr_pearson': corr_pearson,
        'corr_spearman': corr_spearman,
        'directional_accuracy': directional_accuracy,
        'count': int(len(data))
    }


def aggregate_fold_metrics(fold_metrics: list[dict]) -> dict:
    if not fold_metrics:
        return {}
    df = pd.DataFrame(fold_metrics)
    summary = {}
    for col in ['mse', 'rmse', 'mae', 'r2', 'corr_pearson', 'corr_spearman', 'directional_accuracy']:
        summary[f'{col}_mean'] = float(df[col].mean())
        summary[f'{col}_std'] = float(df[col].std())
    summary['total_observations'] = int(df['count'].sum())
    summary['folds'] = int(len(df))
    return summary
