"""Evaluate a trained PPE detector and save real figures to figures/.

Ultralytics writes its own confusion matrix / PR curve plots into runs/, but the task
requires these as committed figures under figures/ rather than left in the
(gitignored) runs/ directory -- this script re-renders them there from the same
validation run, plus a per-class metrics table.

Run as a script:

    python -m src.evaluate --weights models/best.pt --split test --out-prefix flagship
"""

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from ultralytics import YOLO

from src.config import CLASS_NAMES, DATA_YAML, FIGURES_DIR, MODELS_DIR


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for an evaluation run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default=str(MODELS_DIR / "best.pt"))
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--out-prefix", default="flagship")
    return parser.parse_args()


def per_class_table(metrics) -> pd.DataFrame:
    """Build a per-class precision/recall/mAP50/mAP50-95 dataframe from val() results.

    Args:
        metrics: The ``DetMetrics`` object returned by ``YOLO.val()``.

    Returns:
        Dataframe indexed by class name.
    """
    box = metrics.box
    class_indices = box.ap_class_index
    rows = []
    for row_i, cls_idx in enumerate(class_indices):
        rows.append(
            {
                "class": CLASS_NAMES[cls_idx],
                "precision": box.p[row_i],
                "recall": box.r[row_i],
                "mAP50": box.ap50[row_i],
                "mAP50-95": box.ap[row_i],
            }
        )
    return pd.DataFrame(rows).set_index("class")


def plot_per_class_metrics(df: pd.DataFrame, out_path) -> None:
    """Save a grouped bar chart of precision/recall/mAP50 per class."""
    fig, ax = plt.subplots(figsize=(11, 5))
    df[["precision", "recall", "mAP50"]].plot(kind="bar", ax=ax)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title("Per-class detection performance")
    plt.xticks(rotation=30, ha="right")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_pr_curve(metrics, out_path) -> None:
    """Save a per-class precision-recall curve plot plus the mean across classes.

    Args:
        metrics: The ``DetMetrics`` object returned by ``YOLO.val()``.
        out_path: Where to save the figure.
    """
    box = metrics.box
    recall = box.px  # 1000-point recall axis, shared across classes
    precision = box.prec_values  # (n_classes_with_ap, 1000)

    fig, ax = plt.subplots(figsize=(8, 7))
    for row_i, cls_idx in enumerate(box.ap_class_index):
        ax.plot(recall, precision[row_i], linewidth=1, label=f"{CLASS_NAMES[cls_idx]} ({box.ap50[row_i]:.2f})")
    ax.plot(
        recall, precision.mean(axis=0), color="black", linewidth=3,
        label=f"mean ({box.map50:.2f})",
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.set_title("Precision-Recall curve by class (mAP50 in legend)")
    ax.legend(loc="lower left", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(confusion_matrix: np.ndarray, out_path) -> None:
    """Save a heatmap of the raw (non-normalized) confusion matrix.

    Ultralytics' confusion matrix has ``nc + 1`` rows/cols (extra row/col for
    "background" false positives/negatives).

    Args:
        confusion_matrix: ``(nc+1, nc+1)`` array from ``metrics.confusion_matrix.matrix``.
        out_path: Where to save the figure.
    """
    labels = CLASS_NAMES + ["background"]
    fig, ax = plt.subplots(figsize=(10, 9))
    sns.heatmap(
        confusion_matrix,
        annot=True,
        fmt=".0f",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        cbar=True,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix (raw counts)")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    """Evaluate a checkpoint, print a summary, and save figures + a metrics JSON."""
    args = parse_args()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)
    # plots=True is required for this Ultralytics version to actually populate
    # metrics.confusion_matrix -- it stays all-zero with plots=False. We still ignore
    # Ultralytics' own saved plots (written to the gitignored runs/ dir) and re-render
    # our own into figures/ below, per the task's requirement for committed figures.
    metrics = model.val(data=str(DATA_YAML), split=args.split, verbose=False, plots=True)

    df = per_class_table(metrics)
    plot_per_class_metrics(df, FIGURES_DIR / f"{args.out_prefix}_per_class_metrics.png")
    plot_pr_curve(metrics, FIGURES_DIR / f"{args.out_prefix}_pr_curve.png")
    plot_confusion_matrix(
        metrics.confusion_matrix.matrix, FIGURES_DIR / f"{args.out_prefix}_confusion_matrix.png"
    )

    summary = {
        "split": args.split,
        "mAP50": float(metrics.box.map50),
        "mAP50-95": float(metrics.box.map),
        "precision_mean": float(metrics.box.mp),
        "recall_mean": float(metrics.box.mr),
        "per_class": df.reset_index().to_dict(orient="records"),
    }
    out_json = MODELS_DIR / f"{args.out_prefix}_metrics.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{args.split} set: mAP50={metrics.box.map50:.3f}  mAP50-95={metrics.box.map:.3f}")
    print(df.round(3).to_string())
    print(f"\nSaved figures to {FIGURES_DIR}, metrics to {out_json}")


if __name__ == "__main__":
    main()
