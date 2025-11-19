from pathlib import Path
import json
from typing import Optional, Sequence, Union

import pandas as pd
import matplotlib.pyplot as plt

PathLike = Union[str, Path]


def load_history(run_dir: PathLike) -> dict:
    """
    Load trainer_state.json saved at the root of a run directory.

    Parameters
    ----------
    run_dir : str or Path
        Path to the output_dir used by HuggingFace Trainer.

    Returns
    -------
    state : dict
        The deserialized trainer_state.
    """
    run_dir = Path(run_dir)
    state_path = run_dir / "trainer_state.json"
    if not state_path.exists():
        raise FileNotFoundError(f"trainer_state.json not found at {state_path}")

    with open(state_path, "r", encoding="UTF-8") as f:
        state = json.load(f)

    log_history = state.get("log_history", [])
    df = pd.DataFrame(log_history)

    # Sort by global step if present
    if "step" in df.columns:
        df = df.sort_values("step").reset_index(drop=True)
    return df

def plot_train_loss(df: pd.DataFrame, out_dir: PathLike) -> Optional[Path]:
    """
    Plot training loss vs. step and save as PNG.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame from history_to_df.
    out_dir : str or Path
        Directory where the plot will be saved.

    Returns
    -------
    path : Path or None
        Path to the saved image, or None if 'loss' column is absent.
    """
    if "loss" not in df.columns:
        return None

    out_dir = Path(out_dir)
    df_loss = df.dropna(subset=["loss"])
    if df_loss.empty:
        return None

    plt.figure()
    plt.plot(df_loss["step"], df_loss["loss"])
    plt.xlabel("Step")
    plt.ylabel("Training loss")
    plt.title("Training loss vs. step")
    plt.tight_layout()

    out_path = out_dir / "train_loss_vs_step.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path


def plot_eval_loss(df: pd.DataFrame, out_dir: PathLike) -> Optional[Path]:
    """
    Plot eval loss vs. step and save as PNG.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame from history_to_df.
    out_dir : str or Path
        Directory where the plot will be saved.

    Returns
    -------
    path : Path or None
        Path to the saved image, or None if 'eval_loss' column is absent.
    """
    if "eval_loss" not in df.columns:
        return None

    out_dir = Path(out_dir)
    df_eval = df.dropna(subset=["eval_loss"])
    if df_eval.empty:
        return None

    plt.figure()
    plt.plot(df_eval["step"], df_eval["eval_loss"], marker="o")
    plt.xlabel("Step")
    plt.ylabel("Eval loss")
    plt.title("Eval loss vs. step")
    plt.tight_layout()

    out_path = out_dir / "eval_loss_vs_step.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path


def plot_eval_metrics(
    df: pd.DataFrame,
    out_dir: PathLike,
    metric_names: Sequence[str] = ("accuracy", "precision", "recall", "f1", "mse", "mae", "rmse"),
) -> dict:
    """
    Plot one or more eval metrics vs. step and save each as PNG.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame from history_to_df.
    out_dir : str or Path
        Directory where plots will be saved.
    metric_names : list of str
        Names of eval metrics **without** the 'eval_' prefix, e.g. ['pearson', 'rmse'].

    Returns
    -------
    paths : dict
        Mapping metric_name -> Path (or None if not found).
    """
    out_dir = Path(out_dir)
    paths = {}

    for name in metric_names:
        col = f"eval_{name}"
        if col not in df.columns:
            paths[name] = None
            continue

        df_metric = df.dropna(subset=[col])
        if df_metric.empty:
            paths[name] = None
            continue

        plt.figure()
        plt.plot(df_metric["step"], df_metric[col], marker="o")
        plt.xlabel("Step")
        plt.ylabel(col)
        plt.title(f"{col} vs. step")
        plt.tight_layout()

        out_path = out_dir / f"{col}_vs_step.png"
        plt.savefig(out_path, dpi=200)
        plt.close()

        paths[name] = out_path

    return paths

def analyze_run(
    run_dir: PathLike,
    extra_eval_metrics: Optional[Sequence[str]] = None,
) -> dict:
    """
    High-level helper: load trainer_state.json, build history DataFrame,
    and generate standard plots (train loss, eval loss, and optional extra metrics).

    Parameters
    ----------
    run_dir : str or Path
        The output_dir for the run.
    extra_eval_metrics : list of str, optional
        Extra eval metrics (without 'eval_' prefix) to plot, e.g. ['pearson', 'rmse'].

    Returns
    -------
    result : dict
        {
            'df': DataFrame,
            'train_loss_path': Path or None,
            'eval_loss_path': Path or None,
            'extra_metric_paths': dict(metric_name -> Path or None)
        }
    """
    df = load_history(run_dir)

    train_loss_path = plot_train_loss(df, run_dir)
    eval_loss_path = plot_eval_loss(df, run_dir)

    extra_metric_paths = {}
    if extra_eval_metrics:
        extra_metric_paths = plot_eval_metrics(
            df,
            run_dir,
            extra_eval_metrics,
        )
    
    return {
        "df": df,
        "train_loss_path": train_loss_path,
        "eval_loss_path": eval_loss_path,
        "extra_metric_paths": extra_metric_paths,
    }