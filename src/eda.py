"""Exploratory data analysis for the PPE dataset -- no trained model required.

Run as a script to regenerate all EDA figures into figures/:

    python -m src.eda
"""

import random
from collections import Counter
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.config import CLASS_NAMES, DATA_DIR, FIGURES_DIR, RANDOM_STATE

SPLIT_COLORS = {"train": "#4C72B0", "valid": "#55A868", "test": "#C44E52"}


def _iter_label_files(split: str):
    """Yield (image_path, label_path) pairs for a split."""
    label_dir = DATA_DIR / split / "labels"
    image_dir = DATA_DIR / split / "images"
    for label_path in sorted(label_dir.glob("*.txt")):
        yield image_dir / f"{label_path.stem}.jpg", label_path


def _parse_yolo_label(label_path: Path) -> list:
    """Parse a YOLO-format label file into (cls, x, y, w, h) tuples."""
    boxes = []
    for line in label_path.read_text().splitlines():
        if not line.strip():
            continue
        cls, x, y, w, h = line.split()
        boxes.append((int(cls), float(x), float(y), float(w), float(h)))
    return boxes


def class_counts_per_split() -> dict:
    """Count boxes per class for each split.

    Returns:
        Mapping of split name to a Counter of class name -> box count.
    """
    result = {}
    for split in ["train", "valid", "test"]:
        counts = Counter()
        for _, label_path in _iter_label_files(split):
            for cls, *_ in _parse_yolo_label(label_path):
                counts[CLASS_NAMES[cls]] += 1
        result[split] = counts
    return result


def plot_class_distribution(out_dir: Path = FIGURES_DIR) -> None:
    """Save a grouped bar chart of box counts per class, split by train/valid/test."""
    counts = class_counts_per_split()
    x = np.arange(len(CLASS_NAMES))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, split in enumerate(["train", "valid", "test"]):
        values = [counts[split].get(name, 0) for name in CLASS_NAMES]
        ax.bar(x + (i - 1) * width, values, width, label=split, color=SPLIT_COLORS[split])
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right")
    ax.set_ylabel("Box count")
    ax.set_title("Class distribution by split")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "class_distribution.png", dpi=150)
    plt.close(fig)


def plot_boxes_per_image(out_dir: Path = FIGURES_DIR) -> None:
    """Save a histogram of the number of annotated boxes per training image."""
    counts = []
    for _, label_path in _iter_label_files("train"):
        counts.append(len(_parse_yolo_label(label_path)))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(counts, bins=range(0, max(counts) + 2), color="#4C72B0", edgecolor="white")
    ax.set_xlabel("Boxes per image")
    ax.set_ylabel("Number of images")
    ax.set_title(f"Annotation density (train split, mean={np.mean(counts):.1f} boxes/image)")
    fig.tight_layout()
    fig.savefig(out_dir / "boxes_per_image.png", dpi=150)
    plt.close(fig)


def plot_sample_grid(split: str = "train", n: int = 6, out_dir: Path = FIGURES_DIR) -> None:
    """Save a grid of sample images with their ground-truth boxes drawn on.

    Args:
        split: Which split to sample from.
        n: Number of sample images.
        out_dir: Directory to save the figure into.
    """
    rng = random.Random(RANDOM_STATE)
    pairs = list(_iter_label_files(split))
    sample = rng.sample(pairs, min(n, len(pairs)))

    cols = 3
    rows = (len(sample) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = np.array(axes).reshape(-1)

    for ax, (img_path, label_path) in zip(axes, sample):
        img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        for cls, cx, cy, bw, bh in _parse_yolo_label(label_path):
            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)
            color = (255, 0, 0) if CLASS_NAMES[cls].startswith("NO-") else (0, 200, 0)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                img, CLASS_NAMES[cls], (x1, max(y1 - 5, 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
            )
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(img_path.name, fontsize=8)

    for ax in axes[len(sample):]:
        ax.axis("off")

    fig.suptitle(f"Sample annotated images ({split})", y=1.0)
    fig.tight_layout()
    fig.savefig(out_dir / f"sample_annotations_{split}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_image_size_distribution(out_dir: Path = FIGURES_DIR) -> None:
    """Save a scatter plot of image (width, height) across the training set."""
    sizes = []
    for img_path, _ in _iter_label_files("train"):
        img = cv2.imread(str(img_path))
        if img is not None:
            h, w = img.shape[:2]
            sizes.append((w, h))

    widths, heights = zip(*sizes)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(widths, heights, alpha=0.3, s=10, color="#4C72B0")
    ax.set_xlabel("Width (px)")
    ax.set_ylabel("Height (px)")
    ax.set_title(f"Image size distribution (train, n={len(sizes)})")
    fig.tight_layout()
    fig.savefig(out_dir / "image_size_distribution.png", dpi=150)
    plt.close(fig)


def run_all(out_dir: Path = FIGURES_DIR) -> None:
    """Generate and save every EDA figure."""
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_class_distribution(out_dir)
    plot_boxes_per_image(out_dir)
    plot_sample_grid("train", n=6, out_dir=out_dir)
    plot_image_size_distribution(out_dir)
    print(f"Saved EDA figures to {out_dir}")


if __name__ == "__main__":
    run_all()
