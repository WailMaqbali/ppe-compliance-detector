"""Standalone live webcam PPE detection using OpenCV.

Run:

    python -m src.webcam_demo
    python -m src.webcam_demo --weights models/best_lite.pt --camera 0

Uses the lightweight checkpoint (models/best_lite.pt) by default, since smooth live
video matters more than squeezing out the last bit of mAP for this mode -- see the
README for the flagship-vs-lightweight tradeoff. Auto-detects CUDA / Apple Silicon MPS
/ CPU (src.serve_utils.auto_device), so this runs unchanged on the GPU training machine
or, e.g., a MacBook Air (M1) doing live inference.

Press 'q' to quit the window.
"""

import argparse
import time

import cv2

from src.config import LIGHTWEIGHT_MODEL_PATH
from src.serve_utils import annotate_image, auto_device, load_model


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the webcam demo."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default=str(LIGHTWEIGHT_MODEL_PATH))
    parser.add_argument("--camera", type=int, default=0, help="cv2.VideoCapture index.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    return parser.parse_args()


def main() -> None:
    """Open the webcam, run live detection on each frame, and display the annotated feed."""
    args = parse_args()

    device = auto_device()
    print(f"Loading {args.weights} on device: {device}")
    model = load_model(args.weights, device=device)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}.")

    print("Press 'q' to quit.")
    frame_count = 0
    fps_timer = time.time()
    fps = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read a frame from the camera; stopping.")
            break

        annotated, counts, violation = annotate_image(model, frame, conf=args.conf)

        frame_count += 1
        elapsed = time.time() - fps_timer
        if elapsed >= 1.0:
            fps = frame_count / elapsed
            frame_count = 0
            fps_timer = time.time()

        label = "VIOLATION" if violation else "OK"
        color = (0, 0, 255) if violation else (0, 200, 0)
        cv2.putText(
            annotated, f"{label} | FPS: {fps:.1f}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2,
        )

        cv2.imshow("PPE Compliance Detector (press q to quit)", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
