"""Statistics computation functions for megaMM pipeline."""
from __future__ import annotations
from typing import List, Tuple, Dict, Any
import numpy as np
import pandas as pd


def compute_summary_stats(data: pd.DataFrame | pd.Series, name: str = "Data") -> Dict[str, Any]:
    """Compute comprehensive summary statistics.

    Args:
        data: DataFrame or Series to analyze
        name: Name for the data

    Returns:
        Dictionary of statistics
    """
    if isinstance(data, pd.Series):
        data = data.to_frame(name=name)

    stats = {
        "name": name,
        "n_rows": len(data),
        "n_cols": len(data.columns),
        "memory_mb": data.memory_usage(deep=True).sum() / 1024 / 1024,
        "columns": {},
    }

    for col in data.columns:
        s = data[col]
        col_stats = {
            "dtype": str(s.dtype),
            "non_null": s.notna().sum(),
            "null_pct": s.isna().mean() * 100,
        }

        if np.issubdtype(s.dtype, np.number):
            col_stats.update({
                "mean": s.mean(),
                "std": s.std(),
                "min": s.min(),
                "p25": s.quantile(0.25),
                "median": s.median(),
                "p75": s.quantile(0.75),
                "max": s.max(),
                "skew": s.skew() if len(s) > 2 else np.nan,
                "kurtosis": s.kurtosis() if len(s) > 3 else np.nan,
            })

        stats["columns"][col] = col_stats

    return stats


def format_metrics_table(results: List[Dict[str, Any]], highlight_best: bool = True) -> str:
    """Format metrics as a rich-compatible table string.

    Args:
        results: List of result dictionaries
        highlight_best: Whether to highlight best values

    Returns:
        Formatted table string
    """
    if not results:
        return "No results to display."

    df = pd.DataFrame(results)

    # Find best values for each metric
    best = {}
    if highlight_best:
        for col in df.columns:
            if col == "k" or col == "n_samples":
                continue
            if "precision" in col or "recall" in col:
                best[col] = df[col].idxmax()
            else:
                best[col] = df[col].idxmin()

    lines = []
    lines.append("=" * 80)

    # Header
    header = f"{'K':>3} | {'Log Loss':>10} | {'Brier':>8} | {'Up Prec':>8} | {'Up Rec':>8} | {'Dn Prec':>8} | {'Dn Rec':>8} | {'N':>8}"
    lines.append(header)
    lines.append("-" * 80)

    for idx, row in df.iterrows():
        markers = []
        for col in ["log_loss", "brier", "up_precision", "up_recall", "down_precision", "down_recall"]:
            if col in best and best[col] == idx:
                markers.append(col)

        marker = " *" if markers else ""
        line = (
            f"{int(row['k']):>3} | "
            f"{row['log_loss']:>10.4f} | "
            f"{row['brier']:>8.4f} | "
            f"{row['up_precision']:>8.3f} | "
            f"{row['up_recall']:>8.3f} | "
            f"{row['down_precision']:>8.3f} | "
            f"{row['down_recall']:>8.3f} | "
            f"{int(row['n_samples']):>8}{marker}"
        )
        lines.append(line)

    lines.append("=" * 80)
    if highlight_best:
        lines.append("* indicates best value for that metric")

    return "\n".join(lines)


def format_prediction_summary(predictions_df: pd.DataFrame) -> Dict[str, Any]:
    """Compute summary statistics for predictions.

    Args:
        predictions_df: DataFrame with prediction columns

    Returns:
        Dictionary of summary statistics
    """
    summary = {
        "n_tickers": len(predictions_df),
        "asof_date": str(predictions_df["asof"].iloc[0]) if "asof" in predictions_df.columns else "N/A",
    }

    # Probability stats
    for col in ["p_down", "p_mid", "p_up"]:
        if col in predictions_df.columns:
            summary[f"{col}_mean"] = predictions_df[col].mean()
            summary[f"{col}_std"] = predictions_df[col].std()
            summary[f"{col}_min"] = predictions_df[col].min()
            summary[f"{col}_max"] = predictions_df[col].max()

    # Expected return stats
    if "mu_r_t1" in predictions_df.columns:
        summary["mu_return_mean"] = predictions_df["mu_r_t1"].mean()
        summary["mu_return_std"] = predictions_df["mu_r_t1"].std()
        summary["mu_return_min"] = predictions_df["mu_r_t1"].min()
        summary["mu_return_max"] = predictions_df["mu_r_t1"].max()

    # Signal distribution
    if "p_up" in predictions_df.columns and "p_down" in predictions_df.columns:
        signal = predictions_df["p_up"] - predictions_df["p_down"]
        summary["bullish_count"] = (signal > 0.1).sum()
        summary["neutral_count"] = ((signal >= -0.1) & (signal <= 0.1)).sum()
        summary["bearish_count"] = (signal < -0.1).sum()

    # Top movers
    if "p_up" in predictions_df.columns:
        top_up = predictions_df.nlargest(5, "p_up")[["ticker", "p_up"]].to_dict("records")
        summary["top_bullish"] = top_up

    if "p_down" in predictions_df.columns:
        top_down = predictions_df.nlargest(5, "p_down")[["ticker", "p_down"]].to_dict("records")
        summary["top_bearish"] = top_down

    return summary


def compute_walkforward_splits_stats(
    n_dates: int,
    min_train: int,
    test_block: int,
    step: int,
) -> Dict[str, Any]:
    """Compute statistics about walk-forward splits.

    Args:
        n_dates: Total number of dates
        min_train: Minimum training period
        test_block: Test block size
        step: Step size between splits

    Returns:
        Dictionary with split statistics
    """
    n_splits = 0
    train_end = min_train

    splits_info = []
    while train_end + test_block < n_dates - 2:
        splits_info.append({
            "split": n_splits + 1,
            "train_start": 0,
            "train_end": train_end,
            "test_start": train_end,
            "test_end": min(train_end + test_block, n_dates - 2),
            "train_days": train_end,
            "test_days": min(test_block, n_dates - 2 - train_end),
        })
        train_end += step
        n_splits += 1

    return {
        "n_splits": n_splits,
        "total_dates": n_dates,
        "min_train_days": min_train,
        "test_block_days": test_block,
        "step_days": step,
        "splits": splits_info,
        "total_test_predictions": sum(s["test_days"] for s in splits_info),
    }


def compute_class_balance(
    labels: List[int] | np.ndarray,
    class_names: List[str] = None,
) -> Dict[str, Any]:
    """Compute class balance statistics.

    Args:
        labels: Array of class labels
        class_names: Names for each class

    Returns:
        Dictionary with class balance info
    """
    if class_names is None:
        class_names = ["Down", "Mid", "Up"]

    labels = np.array(labels)
    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)

    balance = {
        "total": total,
        "classes": {},
    }

    for cls, count in zip(unique, counts):
        name = class_names[int(cls)] if int(cls) < len(class_names) else f"Class {cls}"
        balance["classes"][name] = {
            "count": int(count),
            "pct": count / total * 100,
        }

    # Imbalance ratio
    if len(counts) > 0:
        balance["imbalance_ratio"] = counts.max() / counts.min()

    return balance
