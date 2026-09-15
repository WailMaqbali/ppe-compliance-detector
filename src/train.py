"""Fine-tune a YOLOv8 model on the PPE compliance dataset.

Run as a script:

    python -m src.train --model yolov8s.pt --name flagship_yolov8s --out models/best.pt
    python -m src.train --model yolov8n.pt --name lite_yolov8n --out models/best_lite.pt

Trains on the GPU (device=0) when available, since this script is meant to run on the
training machine, not the inference/demo machine (see src/serve_utils.py for the
auto-detecting device logic used at inference time).
"""

import argparse
import shutil

import torch
from ultralytics import YOLO

from src.config import DATA_YAML, MODELS_DIR, RANDOM_STATE, RUNS_DIR


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for a training run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="yolov8s.pt", help="Base checkpoint to fine-tune.")
    parser.add_argument("--epochs", type=int, default=200, help="Max epochs.")
    parser.add_argument(
        "--patience", type=int, default=30, help="Early-stopping patience (epochs)."
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size.")
    parser.add_argument(
        "--batch", type=float, default=-1, help="Batch size; -1 = Ultralytics AutoBatch."
    )
    parser.add_argument("--name", default="train_run", help="Run name under runs/detect/.")
    parser.add_argument("--out", default=str(MODELS_DIR / "best.pt"), help="Where to copy best.pt.")
    return parser.parse_args()


def main() -> None:
    """Train a YOLOv8 model on the PPE dataset and copy the best checkpoint to models/."""
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. This training script targets the GPU training "
            "machine; run src/serve_utils.py-based inference elsewhere for CPU/MPS."
        )
    print(f"Training on GPU: {torch.cuda.get_device_name(0)}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)
    model.train(
        data=str(DATA_YAML),
        epochs=args.epochs,
        patience=args.patience,
        imgsz=args.imgsz,
        batch=args.batch,
        device=0,
        project=str(RUNS_DIR / "detect"),
        name=args.name,
        seed=RANDOM_STATE,
        verbose=False,
        plots=True,
    )

    best_weights = RUNS_DIR / "detect" / args.name / "weights" / "best.pt"
    shutil.copy(best_weights, args.out)
    print(f"Copied best checkpoint to {args.out}")


if __name__ == "__main__":
    main()
