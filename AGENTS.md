# Repository Guidelines

## Project Structure & Module Organization
- `src/megamm/` houses library code and the `qt-hmm` CLI (`src/megamm/cli.py`).
- `configs/` stores runtime settings (`configs/config.json`, `configs/universe.json`).
- `artifacts/` is for generated data/models/reports and is gitignored.
- Key domains: `data/`, `features/`, `models/`, `backtest/`, `engine/`, `reporting/`, `debug/`.

## Build, Test, and Development Commands
- `uv venv && source .venv/bin/activate && uv pip install -e .` (or `python -m venv .venv && pip install -e .`): create a dev env.
- `qt-hmm download`: fetch OHLCV data via yfinance.
- `qt-hmm features`: compute features; add `--force` to recompute.
- `qt-hmm walkforward`: train/evaluate K=3..6 and save best model to `artifacts/models/best/`.
- `qt-hmm predict`: generate `artifacts/predictions/*`.
- `python -m megamm.debug.inspect_model artifacts/models/best/model.pt`: debug pomegranate internals if needed.

## Coding Style & Naming Conventions
- Python 3.12, 4-space indentation, snake_case for modules/functions; keep new code consistent with existing imports and Typer CLI patterns.
- Prefer small, focused modules under `src/megamm/<area>/`; keep config schema aligned between `configs/config.json` and `src/megamm/config.py`.
- Data artifacts should stay in `artifacts/` (ignored by git).

## Testing Guidelines
- No automated test suite is present. Validate changes by running the pipeline and inspecting `artifacts/reports/walkforward_metrics.csv` and `artifacts/predictions/predictions.parquet`.
- If you add tests, use a `tests/` directory and `test_*.py` naming.

## Commit & Pull Request Guidelines
- Recent history uses Conventional Commits (e.g., `feat(reporting): ...`, `refactor(backtest): ...`); follow that style when possible.
- PRs should describe the change, list commands run, and call out config/schema changes. Do not commit `artifacts/` or `.venv/`.
