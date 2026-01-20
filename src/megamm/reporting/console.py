"""Console output utilities using Rich for megaMM pipeline."""
from __future__ import annotations
from typing import Any, Dict, List, Optional
from contextlib import contextmanager
import time

from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
)
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box

console = Console()


def create_progress(description: str = "Processing") -> Progress:
    """Create a progress bar with standard columns.

    Args:
        description: Default description for tasks

    Returns:
        Progress instance
    """
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    )


@contextmanager
def progress_context(description: str = "Processing", total: int = 100):
    """Context manager for progress tracking.

    Usage:
        with progress_context("Training", total=100) as update:
            for i in range(100):
                update(1)  # advance by 1
    """
    progress = create_progress()
    with progress:
        task_id = progress.add_task(description, total=total)

        def update(advance: int = 1, description: str = None):
            if description:
                progress.update(task_id, description=description, advance=advance)
            else:
                progress.update(task_id, advance=advance)

        yield update


def log_section(title: str, subtitle: str = None):
    """Print a section header.

    Args:
        title: Section title
        subtitle: Optional subtitle
    """
    console.print()
    if subtitle:
        console.print(Panel(
            f"[bold]{title}[/bold]\n[dim]{subtitle}[/dim]",
            border_style="blue",
            box=box.ROUNDED,
        ))
    else:
        console.print(Panel(
            f"[bold]{title}[/bold]",
            border_style="blue",
            box=box.ROUNDED,
        ))
    console.print()


def log_stats_table(
    stats: Dict[str, Any],
    title: str = "Statistics",
    columns: int = 2,
):
    """Print a statistics table.

    Args:
        stats: Dictionary of stat name -> value
        title: Table title
        columns: Number of columns (stat pairs per row)
    """
    table = Table(title=title, box=box.ROUNDED, show_header=True)

    for _ in range(columns):
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

    items = list(stats.items())
    rows = [items[i:i + columns] for i in range(0, len(items), columns)]

    for row in rows:
        row_data = []
        for name, value in row:
            row_data.append(str(name))
            if isinstance(value, float):
                row_data.append(f"{value:.4f}")
            else:
                row_data.append(str(value))
        # Pad if needed
        while len(row_data) < columns * 2:
            row_data.extend(["", ""])
        table.add_row(*row_data)

    console.print(table)


def log_model_summary(
    k: int,
    train_seconds: float,
    device: str,
    n_tickers: int,
    n_dates: int,
    transition_matrix: Optional[Any] = None,
):
    """Print a model training summary.

    Args:
        k: Number of states
        train_seconds: Training time in seconds
        device: Device used for training
        n_tickers: Number of tickers
        n_dates: Number of dates
        transition_matrix: Optional transition matrix to display
    """
    table = Table(title="Model Summary", box=box.ROUNDED)
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("States (K)", str(k))
    table.add_row("Training Time", f"{train_seconds:.2f}s")
    table.add_row("Device", device)
    table.add_row("Tickers", str(n_tickers))
    table.add_row("Time Steps", str(n_dates))
    table.add_row("Total Observations", f"{n_tickers * n_dates:,}")

    console.print(table)

    if transition_matrix is not None:
        import torch
        if isinstance(transition_matrix, torch.Tensor):
            A = transition_matrix.cpu().numpy()
        else:
            A = transition_matrix

        trans_table = Table(title="Transition Matrix", box=box.SIMPLE)
        trans_table.add_column("From \\ To", style="bold")
        for j in range(A.shape[1]):
            trans_table.add_column(f"S{j}", justify="right")

        for i in range(A.shape[0]):
            row = [f"S{i}"] + [f"{A[i, j]:.3f}" for j in range(A.shape[1])]
            trans_table.add_row(*row)

        console.print(trans_table)


def log_walkforward_progress(
    k: int,
    split: int,
    total_splits: int,
    train_days: int,
    test_days: int,
    elapsed: float = None,
):
    """Log walk-forward split progress.

    Args:
        k: Number of states
        split: Current split number
        total_splits: Total number of splits
        train_days: Training period days
        test_days: Test period days
        elapsed: Elapsed time for this split
    """
    time_str = f" ({elapsed:.1f}s)" if elapsed else ""
    console.print(
        f"  [blue]K={k}[/blue] Split [green]{split}/{total_splits}[/green]: "
        f"train={train_days} days, test={test_days} days{time_str}"
    )


def log_metrics_comparison(results: List[Dict[str, Any]], highlight_best: bool = True):
    """Print a metrics comparison table.

    Args:
        results: List of result dictionaries
        highlight_best: Whether to highlight best values
    """
    if not results:
        console.print("[yellow]No results to display[/yellow]")
        return

    import pandas as pd
    df = pd.DataFrame(results)

    # Find best values
    best = {}
    if highlight_best:
        for col in df.columns:
            if col in ("k", "n_samples"):
                continue
            if "precision" in col or "recall" in col:
                best[col] = df[col].idxmax()
            else:
                best[col] = df[col].idxmin()

    table = Table(title="Walk-Forward Results", box=box.ROUNDED, show_header=True)
    table.add_column("K", justify="center", style="bold")
    table.add_column("Log Loss", justify="right")
    table.add_column("Brier", justify="right")
    table.add_column("Up Prec", justify="right")
    table.add_column("Up Rec", justify="right")
    table.add_column("Dn Prec", justify="right")
    table.add_column("Dn Rec", justify="right")
    table.add_column("Samples", justify="right")

    for idx, row in df.iterrows():
        def fmt(col: str, val: float, is_pct: bool = False) -> str:
            if is_pct:
                s = f"{val:.1%}"
            else:
                s = f"{val:.4f}"
            if col in best and best[col] == idx:
                return f"[bold green]{s}[/bold green]"
            return s

        table.add_row(
            str(int(row["k"])),
            fmt("log_loss", row["log_loss"]),
            fmt("brier", row["brier"]),
            fmt("up_precision", row["up_precision"], True),
            fmt("up_recall", row["up_recall"], True),
            fmt("down_precision", row["down_precision"], True),
            fmt("down_recall", row["down_recall"], True),
            f"{int(row['n_samples']):,}",
        )

    console.print(table)

    # Summary
    best_k = int(df.loc[df["log_loss"].idxmin(), "k"])
    console.print(f"\n[bold]Best K by Log Loss:[/bold] [green]{best_k}[/green]")


def log_prediction_summary(summary: Dict[str, Any]):
    """Print a prediction summary.

    Args:
        summary: Dictionary from format_prediction_summary
    """
    console.print(Panel("[bold]Prediction Summary[/bold]", border_style="green"))

    # Basic info
    table = Table(box=box.SIMPLE, show_header=False)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("As-of Date", str(summary.get("asof_date", "N/A")))
    table.add_row("Tickers", str(summary.get("n_tickers", "N/A")))

    if "bullish_count" in summary:
        table.add_row("Bullish (signal > 0.1)", str(summary["bullish_count"]))
        table.add_row("Neutral", str(summary["neutral_count"]))
        table.add_row("Bearish (signal < -0.1)", str(summary["bearish_count"]))

    console.print(table)

    # Top movers
    if "top_bullish" in summary and summary["top_bullish"]:
        console.print("\n[bold cyan]Top Bullish:[/bold cyan]")
        for item in summary["top_bullish"]:
            console.print(f"  {item['ticker']}: P(Up) = {item['p_up']:.1%}")

    if "top_bearish" in summary and summary["top_bearish"]:
        console.print("\n[bold red]Top Bearish:[/bold red]")
        for item in summary["top_bearish"]:
            console.print(f"  {item['ticker']}: P(Down) = {item['p_down']:.1%}")


def log_data_summary(
    n_tickers: int,
    date_range: tuple,
    n_features: int,
    memory_mb: float,
):
    """Print a data summary.

    Args:
        n_tickers: Number of tickers
        date_range: Tuple of (start_date, end_date)
        n_features: Number of features
        memory_mb: Memory usage in MB
    """
    table = Table(title="Data Summary", box=box.ROUNDED)
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Tickers", str(n_tickers))
    table.add_row("Date Range", f"{date_range[0]} to {date_range[1]}")
    table.add_row("Features", str(n_features))
    table.add_row("Memory", f"{memory_mb:.2f} MB")

    console.print(table)


def log_success(message: str):
    """Print a success message."""
    console.print(f"[bold green]\u2714[/bold green] {message}")


def log_warning(message: str):
    """Print a warning message."""
    console.print(f"[bold yellow]\u26a0[/bold yellow] {message}")


def log_error(message: str):
    """Print an error message."""
    console.print(f"[bold red]\u2718[/bold red] {message}")


def log_info(message: str):
    """Print an info message."""
    console.print(f"[bold blue]\u2139[/bold blue] {message}")
