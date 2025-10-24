import argparse
from datetime import datetime
from pathlib import Path
import yfinance as yf
import pandas as pd


def fetch_ticker(ticker: str, start: str, end: str | None) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
    if df.empty:
        raise ValueError(f"No data returned for {ticker}")
    # yfinance can return MultiIndex columns even for a single ticker (level0=field, level1=ticker)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [lvl0.lower().replace(' ', '_') for (lvl0, *_rest) in df.columns.values]
    else:
        df.columns = [c.lower().replace(' ', '_') for c in df.columns]
    df.index.name = 'date'
    df['ticker'] = ticker
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tickers', nargs='+', required=True)
    parser.add_argument('--start', required=True)
    parser.add_argument('--end', default=None)
    parser.add_argument('--outdir', default='data/raw')
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    all_frames = []
    for t in args.tickers:
        print(f"Downloading {t}...")
        df = fetch_ticker(t, args.start, args.end)
        df.to_csv(outdir / f"{t}.csv")
        all_frames.append(df)
    if len(all_frames) > 1:
        concat = pd.concat(all_frames)
        concat.to_csv(outdir / "all_tickers.csv")
    print("Done.")


if __name__ == '__main__':
    main()
