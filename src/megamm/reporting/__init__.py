"""Reporting and visualization module for megaMM pipeline."""
from .charts import (
    plot_transition_matrix,
    plot_confusion_matrix,
    plot_calibration,
    plot_walkforward_metrics,
    plot_prediction_distribution,
    plot_feature_correlations,
    plot_regime_timeline,
    plot_training_convergence,
)
from .stats import (
    compute_summary_stats,
    format_metrics_table,
    format_prediction_summary,
)
from .console import (
    create_progress,
    log_section,
    log_stats_table,
    log_model_summary,
)

__all__ = [
    "plot_transition_matrix",
    "plot_confusion_matrix",
    "plot_calibration",
    "plot_walkforward_metrics",
    "plot_prediction_distribution",
    "plot_feature_correlations",
    "plot_regime_timeline",
    "plot_training_convergence",
    "compute_summary_stats",
    "format_metrics_table",
    "format_prediction_summary",
    "create_progress",
    "log_section",
    "log_stats_table",
    "log_model_summary",
]
