# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

megaMM is a Hidden Markov Model (HMM) regime detection system for mega-cap stocks. It uses pomegranate's DenseHMM to identify market regimes from OHLCV-derived features and produces next-day close tail probabilities (P(up>=5%), P(down<=-5%), P(mid)).

## Commands

Install and activate:
```bash
uv venv && source .venv/bin/activate && uv pip install -e .
```

Full pipeline:
```bash
qt-hmm download      # Download OHLCV data via yfinance
qt-hmm features      # Compute feature set
qt-hmm walkforward   # Train models with K=3..6, evaluate, save best
qt-hmm predict       # Generate predictions using best model
```

Add `--force` flag to redownload/recompute cached data.

## Architecture

```
src/megamm/
├── cli.py              # Typer CLI entry point (qt-hmm command)
├── config.py           # Frozen dataclass configs loaded from configs/*.json
├── paths.py            # Artifact path helpers (ARTIFACTS = Path("artifacts"))
├── dataset.py          # Panel dataclass (B,T,D tensor), align_intersection, to_tensor
├── data/
│   └── download.py     # yfinance wrapper
├── features/
│   ├── schema.py       # FEATURE_COLUMNS list (11 features), RETURN_DIM
│   ├── compute.py      # Feature engineering (returns, vol, RSI, etc.)
│   ├── normalize.py    # Z-score normalization
│   └── build.py        # Orchestrates feature pipeline per ticker
├── models/
│   ├── train.py        # train_dense_hmm() returns TrainResult
│   ├── transition.py   # estimate_transition_matrix_from_forward_backward()
│   ├── predict.py      # predict_one() returns Prediction with tail probs
│   ├── params.py       # Extracts emission means/covariances from model
│   ├── math.py         # Gaussian tail probability helpers
│   └── artifacts.py    # save_best/load_best for model persistence
├── backtest/
│   ├── walkforward.py  # Expanding window walk-forward evaluation
│   └── metrics.py      # log_loss_3class, brier_3class, tail_metrics
├── engine/
│   └── predict_latest.py  # Production prediction entry point
└── debug/
    └── inspect_model.py   # Diagnostic for pomegranate internals
```

## Key Data Structures

- **Panel**: `(X: Tensor[B,T,D], tickers: List[str], dates: DatetimeIndex)` - batched feature tensor
- **Prediction**: `(p_up, p_mid, p_down, p_z_t, p_z_t1, mu_r_t1, var_r_t1)` - model output
- **TrainResult**: `(model: DenseHMM, k: int, train_seconds: float, device: str)`

## Configuration

- `configs/config.json`: Data params, feature windows, model hyperparams (K candidates, max_iter, device), eval windows, tail thresholds
- `configs/universe.json`: Ticker list (30 mega-caps)

Model device is set in `configs/config.json` under `model.device` (cpu/mps/cuda).

## Causality Constraint

The system avoids lookahead bias by computing regime posteriors using only data up to time t via `model.predict_proba()` on prefix sequences.

## Artifacts Layout

```
artifacts/
├── data/raw/{TICKER}.parquet      # Raw OHLCV
├── data/features/{TICKER}.parquet # Computed features
├── models/best/                   # Best model (model.pt, meta.json, config.json)
├── reports/walkforward_metrics.csv
└── predictions/predictions.{parquet,csv}
```

## pomegranate Notes

**GPU Support**: Training and inference support CUDA and MPS devices via `cfg.model.device` in `configs/config.json`. Set to `"cuda"` for NVIDIA GPUs, `"mps"` for Apple Silicon, or `"cpu"` for CPU-only. The system:

1. Creates the model on CPU, then moves it to the target device with `.to(device)`
2. Moves training/inference data to the same device
3. Automatically falls back to CPU if the requested device fails (e.g., MPS KMeans issues)
4. Saves models on CPU for portability, loads to configured device for inference

**MPS Caveat**: Apple MPS support can be unstable with pomegranate's internal KMeans during initialization. If MPS training fails, the system automatically falls back to CPU. CUDA is the most reliable GPU option.

**Transition Matrix**: The DenseHMM API can vary by version. If transition matrix extraction fails, run `python -m megamm.debug.inspect_model artifacts/models/best/model.pt` and update `models/transition.py` accordingly.
