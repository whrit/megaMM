"""Chart generation functions for megaMM pipeline."""
from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# Set style defaults
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def _ensure_dir(path: Path) -> Path:
    """Ensure directory exists and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def plot_transition_matrix(
    A: torch.Tensor | np.ndarray,
    output_path: Path,
    title: str = "HMM Transition Matrix",
) -> Path:
    """Plot transition matrix as a heatmap.

    Args:
        A: Transition matrix (K x K)
        output_path: Path to save the figure
        title: Chart title

    Returns:
        Path to saved figure
    """
    if isinstance(A, torch.Tensor):
        A = A.cpu().numpy()

    k = A.shape[0]
    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        A,
        annot=True,
        fmt=".3f",
        cmap="Blues",
        square=True,
        xticklabels=[f"State {i}" for i in range(k)],
        yticklabels=[f"State {i}" for i in range(k)],
        vmin=0,
        vmax=1,
        ax=ax,
        cbar_kws={"label": "Transition Probability"},
    )
    ax.set_xlabel("To State", fontsize=12)
    ax.set_ylabel("From State", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")

    _ensure_dir(output_path.parent)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    output_path: Path,
    labels: List[str] = None,
    title: str = "Confusion Matrix",
) -> Path:
    """Plot confusion matrix.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        output_path: Path to save the figure
        labels: Class labels
        title: Chart title

    Returns:
        Path to saved figure
    """
    from sklearn.metrics import confusion_matrix

    if labels is None:
        labels = ["Down", "Mid", "Up"]

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Raw counts
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=axes[0],
    )
    axes[0].set_xlabel("Predicted", fontsize=12)
    axes[0].set_ylabel("Actual", fontsize=12)
    axes[0].set_title("Counts", fontsize=12)

    # Normalized
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2%",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=axes[1],
    )
    axes[1].set_xlabel("Predicted", fontsize=12)
    axes[1].set_ylabel("Actual", fontsize=12)
    axes[1].set_title("Normalized (Row %)", fontsize=12)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    _ensure_dir(output_path.parent)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_calibration(
    y_true: List[int],
    y_probs: List[Tuple[float, float, float]],
    output_path: Path,
    n_bins: int = 10,
    title: str = "Probability Calibration",
) -> Path:
    """Plot calibration curves for each class.

    Args:
        y_true: True labels (0, 1, 2)
        y_probs: List of (p_down, p_mid, p_up) tuples
        output_path: Path to save the figure
        n_bins: Number of bins for calibration
        title: Chart title

    Returns:
        Path to saved figure
    """
    y_true = np.array(y_true)
    probs = np.array(y_probs)
    class_names = ["Down", "Mid", "Up"]
    colors = ["#e74c3c", "#3498db", "#2ecc71"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for i, (ax, name, color) in enumerate(zip(axes, class_names, colors)):
        # Binary labels for this class
        y_binary = (y_true == i).astype(int)
        p_class = probs[:, i]

        # Bin the probabilities
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_indices = np.digitize(p_class, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)

        bin_true_freq = np.zeros(n_bins)
        bin_pred_mean = np.zeros(n_bins)
        bin_counts = np.zeros(n_bins)

        for b in range(n_bins):
            mask = bin_indices == b
            if mask.sum() > 0:
                bin_true_freq[b] = y_binary[mask].mean()
                bin_pred_mean[b] = p_class[mask].mean()
                bin_counts[b] = mask.sum()

        # Plot
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect")
        mask = bin_counts > 0
        ax.scatter(bin_pred_mean[mask], bin_true_freq[mask], c=color, s=bin_counts[mask] / 10 + 20, alpha=0.7)
        ax.plot(bin_pred_mean[mask], bin_true_freq[mask], color=color, alpha=0.5)
        ax.set_xlabel("Mean Predicted Probability", fontsize=11)
        ax.set_ylabel("Fraction of Positives", fontsize=11)
        ax.set_title(f"{name} Class", fontsize=12)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc="lower right")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    _ensure_dir(output_path.parent)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_walkforward_metrics(
    results_df: pd.DataFrame,
    output_path: Path,
    title: str = "Walk-Forward Evaluation Results",
) -> Path:
    """Plot walk-forward evaluation metrics comparison.

    Args:
        results_df: DataFrame with columns: k, log_loss, brier, up_precision, etc.
        output_path: Path to save the figure
        title: Chart title

    Returns:
        Path to saved figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Sort by k for consistent plotting
    df = results_df.sort_values("k")
    k_values = df["k"].values

    # Log Loss
    ax = axes[0, 0]
    bars = ax.bar(k_values.astype(str), df["log_loss"], color="#3498db", alpha=0.8)
    ax.set_xlabel("Number of States (K)", fontsize=11)
    ax.set_ylabel("Log Loss", fontsize=11)
    ax.set_title("Log Loss by K (lower is better)", fontsize=12)
    best_idx = df["log_loss"].idxmin()
    bars[df.index.get_loc(best_idx)].set_color("#2ecc71")

    # Brier Score
    ax = axes[0, 1]
    bars = ax.bar(k_values.astype(str), df["brier"], color="#9b59b6", alpha=0.8)
    ax.set_xlabel("Number of States (K)", fontsize=11)
    ax.set_ylabel("Brier Score", fontsize=11)
    ax.set_title("Brier Score by K (lower is better)", fontsize=12)
    best_idx = df["brier"].idxmin()
    bars[df.index.get_loc(best_idx)].set_color("#2ecc71")

    # Precision
    ax = axes[1, 0]
    x = np.arange(len(k_values))
    width = 0.35
    ax.bar(x - width/2, df["up_precision"], width, label="Up", color="#2ecc71", alpha=0.8)
    ax.bar(x + width/2, df["down_precision"], width, label="Down", color="#e74c3c", alpha=0.8)
    ax.set_xlabel("Number of States (K)", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_title("Tail Precision by K", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(k_values.astype(str))
    ax.legend()
    ax.set_ylim(0, 1)

    # Recall
    ax = axes[1, 1]
    ax.bar(x - width/2, df["up_recall"], width, label="Up", color="#2ecc71", alpha=0.8)
    ax.bar(x + width/2, df["down_recall"], width, label="Down", color="#e74c3c", alpha=0.8)
    ax.set_xlabel("Number of States (K)", fontsize=11)
    ax.set_ylabel("Recall", fontsize=11)
    ax.set_title("Tail Recall by K", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(k_values.astype(str))
    ax.legend()
    ax.set_ylim(0, 1)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    _ensure_dir(output_path.parent)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_prediction_distribution(
    predictions_df: pd.DataFrame,
    output_path: Path,
    title: str = "Prediction Distribution",
) -> Path:
    """Plot distribution of predictions.

    Args:
        predictions_df: DataFrame with p_down, p_mid, p_up columns
        output_path: Path to save the figure
        title: Chart title

    Returns:
        Path to saved figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Probability distributions
    ax = axes[0, 0]
    ax.hist(predictions_df["p_down"], bins=30, alpha=0.7, label="P(Down)", color="#e74c3c")
    ax.hist(predictions_df["p_mid"], bins=30, alpha=0.7, label="P(Mid)", color="#3498db")
    ax.hist(predictions_df["p_up"], bins=30, alpha=0.7, label="P(Up)", color="#2ecc71")
    ax.set_xlabel("Probability", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Probability Distributions", fontsize=12)
    ax.legend()

    # Stacked bar by ticker
    ax = axes[0, 1]
    df_sorted = predictions_df.sort_values("p_up", ascending=False).head(20)
    x = np.arange(len(df_sorted))
    ax.bar(x, df_sorted["p_down"], label="P(Down)", color="#e74c3c", alpha=0.8)
    ax.bar(x, df_sorted["p_mid"], bottom=df_sorted["p_down"], label="P(Mid)", color="#3498db", alpha=0.8)
    ax.bar(x, df_sorted["p_up"], bottom=df_sorted["p_down"] + df_sorted["p_mid"], label="P(Up)", color="#2ecc71", alpha=0.8)
    ax.set_xlabel("Ticker", fontsize=11)
    ax.set_ylabel("Probability", fontsize=11)
    ax.set_title("Probability Breakdown (Top 20 by P(Up))", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(df_sorted["ticker"], rotation=45, ha="right", fontsize=8)
    ax.legend(loc="upper right")

    # Expected return distribution
    ax = axes[1, 0]
    if "mu_r_t1" in predictions_df.columns:
        ax.hist(predictions_df["mu_r_t1"] * 100, bins=30, alpha=0.7, color="#9b59b6")
        ax.axvline(0, color="black", linestyle="--", alpha=0.5)
        ax.set_xlabel("Expected Return (%)", fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.set_title("Expected Next-Day Return Distribution", fontsize=12)

    # Scatter: Expected return vs P(Up) - P(Down)
    ax = axes[1, 1]
    if "mu_r_t1" in predictions_df.columns:
        signal = predictions_df["p_up"] - predictions_df["p_down"]
        ax.scatter(signal, predictions_df["mu_r_t1"] * 100, alpha=0.6, c=signal, cmap="RdYlGn")
        ax.axhline(0, color="black", linestyle="--", alpha=0.3)
        ax.axvline(0, color="black", linestyle="--", alpha=0.3)
        ax.set_xlabel("P(Up) - P(Down)", fontsize=11)
        ax.set_ylabel("Expected Return (%)", fontsize=11)
        ax.set_title("Signal vs Expected Return", fontsize=12)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    _ensure_dir(output_path.parent)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_feature_correlations(
    features_df: pd.DataFrame,
    output_path: Path,
    title: str = "Feature Correlations",
) -> Path:
    """Plot feature correlation matrix.

    Args:
        features_df: DataFrame with feature columns
        output_path: Path to save the figure
        title: Chart title

    Returns:
        Path to saved figure
    """
    corr = features_df.corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        square=True,
        ax=ax,
        cbar_kws={"label": "Correlation"},
        annot_kws={"size": 9},
    )
    ax.set_title(title, fontsize=14, fontweight="bold")

    _ensure_dir(output_path.parent)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_regime_timeline(
    dates: pd.DatetimeIndex,
    regimes: np.ndarray,
    returns: Optional[pd.Series] = None,
    output_path: Path = None,
    title: str = "Regime Timeline",
) -> Path:
    """Plot regime assignments over time.

    Args:
        dates: DatetimeIndex
        regimes: Array of regime assignments
        returns: Optional return series to overlay
        output_path: Path to save the figure
        title: Chart title

    Returns:
        Path to saved figure
    """
    k = int(regimes.max()) + 1
    colors = plt.cm.Set2(np.linspace(0, 1, k))

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={"height_ratios": [1, 3]})

    # Regime bar
    ax = axes[0]
    for i in range(k):
        mask = regimes == i
        ax.fill_between(dates, 0, 1, where=mask, alpha=0.8, color=colors[i], label=f"State {i}")
    ax.set_ylabel("Regime", fontsize=11)
    ax.set_yticks([])
    ax.legend(loc="upper right", ncol=k)
    ax.set_title(title, fontsize=14, fontweight="bold")

    # Returns overlay
    ax = axes[1]
    if returns is not None:
        cumret = (1 + returns).cumprod() - 1
        ax.plot(dates, cumret * 100, color="black", alpha=0.7, linewidth=0.8)
        ax.set_ylabel("Cumulative Return (%)", fontsize=11)
        ax.axhline(0, color="gray", linestyle="--", alpha=0.5)

    ax.set_xlabel("Date", fontsize=11)

    _ensure_dir(output_path.parent)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_training_convergence(
    improvements: List[float],
    times: List[float],
    output_path: Path,
    title: str = "Training Convergence",
) -> Path:
    """Plot training convergence curve.

    Args:
        improvements: List of improvement values per iteration
        times: List of iteration times
        output_path: Path to save the figure
        title: Chart title

    Returns:
        Path to saved figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    iterations = np.arange(1, len(improvements) + 1)

    # Improvement curve
    ax = axes[0]
    ax.semilogy(iterations, improvements, "b-", linewidth=2, alpha=0.8)
    ax.fill_between(iterations, improvements, alpha=0.2)
    ax.set_xlabel("Iteration", fontsize=11)
    ax.set_ylabel("Improvement (log scale)", fontsize=11)
    ax.set_title("Convergence", fontsize=12)
    ax.grid(True, alpha=0.3)

    # Cumulative time
    ax = axes[1]
    cumtime = np.cumsum(times)
    ax.plot(iterations, cumtime, "g-", linewidth=2, alpha=0.8)
    ax.set_xlabel("Iteration", fontsize=11)
    ax.set_ylabel("Cumulative Time (s)", fontsize=11)
    ax.set_title("Training Time", fontsize=12)
    ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    _ensure_dir(output_path.parent)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path
