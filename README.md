# Mega-Cap HMM Regime → Next-Day Close Tail Probabilities (pomegranate + yfinance)

This repo is a **minimal, ready-to-commit skeleton** that:

- downloads ~30 mega-cap tickers with `yfinance`
- computes a causal OHLCV-derived feature set
- trains **pomegranate** `DenseHMM` for **K = 3..6**
- runs **walk-forward** evaluation (expanding window)
- writes daily `predictions.parquet` with `P(up/mid/down)` per ticker

## What this is (and what it isn't)

- This is a **regime model** (latent states) trained on *today's features*.
- It produces **probabilities** for next-day close return being:
  - `up`: >= +5%
  - `down`: <= -5%
  - `mid`: otherwise
- It is **not** a trading strategy by itself.

## Key correctness note (no lookahead)
For evaluation and daily prediction, we compute the regime posterior at time *t* using **only data up to t** by calling `model.predict_proba()` on the *prefix* sequence. This avoids using future observations.

This is slower than a custom forward-filter implementation, but it's correct and simple.

## Quickstart

### 1) Install
Use a fresh venv (example with uv, but pip is fine too):

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

or:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 2) Download data
```bash
qt-hmm download
```

### 3) Build features
```bash
qt-hmm features
```

### 4) Walk-forward train + evaluate (K=3..6)
```bash
qt-hmm walkforward
```

This writes a CSV report to `artifacts/reports/walkforward_metrics.csv` and saves the best model to `artifacts/models/best/`.

### 5) Produce latest predictions
```bash
qt-hmm predict
```

This writes:
- `artifacts/predictions/predictions.parquet`
- `artifacts/predictions/predictions.csv`

## Configuration
Edit:
- `configs/universe.json` (tickers)
- `configs/config.json` (windows, K candidates, walk-forward params)

Device selection:
- `model.device`: `"auto"` (CUDA → MPS → CPU), `"cuda"`, `"mps"`, or `"cpu"`.
- Check availability with `qt-hmm device`.

## Outputs
- Raw data: `artifacts/data/raw/{TICKER}.parquet`
- Features: `artifacts/data/features/{TICKER}.parquet`
- Models: `artifacts/models/{run_id}/...`
- Best model: `artifacts/models/best/...`
- Reports: `artifacts/reports/...`
- Predictions: `artifacts/predictions/predictions.parquet`

## Notes / gotchas
- **pomegranate API details** can vary slightly by version. This skeleton includes defensive getters for
  emission means/covariances and transition matrix extraction.
- If you see an error extracting transition matrix, run:
  - `python -m qt_hmm.debug.inspect_model artifacts/models/best/model.pt`
  and paste the output; then update `qt_hmm/models/transition.py`.
- **CUDA requires a CUDA-enabled PyTorch build**. If `torch.cuda.is_available()` is false, `"cuda"` will fall back to CPU. On Apple Silicon, use `"mps"` or `"auto"`.

## License
MIT
