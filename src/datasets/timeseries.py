
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd


@dataclass
class WindowConfig:
    lookback: int = 30
    horizon: int = 1  # predict next_return

class ReturnDataset(Dataset):
    def __init__(self, df: pd.DataFrame, feature_cols, target_col: str, lookback: int = 30):
        self.df = df.reset_index(drop=True)
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.lookback = lookback
        self.X = self.df[feature_cols].values.astype('float32')
        self.y = self.df[target_col].values.astype('float32')
       

    def __len__(self):
        return len(self.df) - self.lookback
        

    def __getitem__(self, idx):
        x = self.X[idx: idx + self.lookback]
        y = self.y[idx + self.lookback]
        return torch.from_numpy(x), torch.tensor(y)
     


def make_dataloader(df, feature_cols, target_col, lookback, batch_size, shuffle=False):
    ds = ReturnDataset(df, feature_cols, target_col, lookback)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=True)
