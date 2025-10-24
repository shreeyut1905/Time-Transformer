import argparse
from pathlib import Path
import pandas as pd
from datetime import datetime
from ..features.feature_engineering import build_features


def load_raw(tickers, raw_dir: str = 'data/raw') -> pd.DataFrame:
    frames = []
    for t in tickers:
        fp = Path(raw_dir)/f"{t}.csv"
        df = pd.read_csv(fp, parse_dates=['date'])
        frames.append(df)
    data = pd.concat(frames)
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tickers', nargs='+', required=True)
    parser.add_argument('--raw_dir', default='data/raw')
    parser.add_argument('--outdir', default='data/processed')
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    data = load_raw(args.tickers, args.raw_dir)
    feat = build_features(data)
    feat.to_csv(outdir / 'features.csv', index=False)
    print(f"Saved features to {outdir / 'features.csv'}")


if __name__ == '__main__':
    main()
