# End-to-End Time Series Research Experiment



1. Data collection (equities via Yahoo Finance / yfinance)
2. Feature engineering (technical indicators + returns)
3. Baseline rules-based strategy (SMA crossover)
4. Deep learning model --> etsformer(from lucidrains) , lstm(create from scratch)
5. Walk-forward validation with expanding window
6. Backtest with realistic assumptions (execution delay, transaction costs)
7. Performance reporting: CAGR, Sharpe, Sortino, Max Drawdown, Hit Rate, Avg Win/Loss + regression metrics

## Project Layout

```
config.yaml                #  configuration
requirements.txt           
src/
  config.py                # Config loader
  utils/logger.py          # Logging utility
  data/fetch_data.py       # Download raw price data
  data/prepare.py          # Clean & merge, produce feature-ready dataset
  features/feature_engineering.py  # Technical feature functions
  datasets/timeseries.py   # PyTorch Dataset & DataLoader builders
  models/lstm_model.py     # LSTM model definition
  models/ts_transformer.py # Transformer time-series model
  train.py                 # Train model on a single (train/val) split
  walkforward.py           # Orchestrates walk-forward training & prediction
  evaluation/regression_metrics.py  # Regression quality metrics
  backtest/backtester.py   # Signal generation & portfolio simulation
  backtest/metrics.py      # Performance metric calculations

reports/
  (generated artifacts: equity curves, metrics JSON/CSV, figures)
data/
  raw/                     # Raw downloaded data
  processed/               # Feature-engineered dataset cache
  artifacts/               # Model weights, predictions, logs
```

## Quick Start

### 1. Create & Activate Virtual Environment (Windows PowerShell example)

```
python -m venv .venv
./.venv/Scripts/Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Fetch Data
```
python -m src.data.fetch_data --tickers SPY --start 2015-01-01 --end 2025-09-01
```

### 3. Prepare & Feature Engineer
```
python -m src.data.prepare --tickers SPY
```

### 4. Run Walk-Forward (Model Training + Prediction) (Uses configurable lookback & enhanced features)
```
python -m src.walkforward --config config.yaml
```

After this step you can inspect regression (forecast quality) metrics:
```
type data\artifacts\walkforward\aggregate_regression_metrics.csv
```

### 5. Backtest & Report
```
python -m src.backtest.backtester --config config.yaml --run_all
```

Artifacts (equity curves, metrics) will appear under `reports/` and `data/artifacts/`.

## Performance Reporting

Trading  metrics in backtest:
- CAGR, Sharpe, Sortino, Max Drawdown, Hit Rate, Avg Win/Loss

Model  metrics:
- MSE, RMSE, MAE
- R² 
- Pearson & Spearman correlation between prediction and realized next-day return
- Directional Accuracy 

## Configuration
Edit `config.yaml` to change tickers, walk-forward windows, model hyperparameters, and cost assumptions.

## Transaction Costs & Realism
Assumptions:
- Execution at next day's open given signal generated on prior close (1-bar delay)
- Flat or long-only (model) / long-flat SMA baseline
- Commission + slippage modeled as basis points per turnover event

