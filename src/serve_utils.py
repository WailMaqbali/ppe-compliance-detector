"""Device detection and inference helpers shared by the Gradio demo and webcam script.

Kept separate from src/train.py because this module is meant to run anywhere --
including a machine with no NVIDIA GPU at all (e.g. an Apple Silicon MacBook) -- so it
must never assume CUDA is available.
"""

from collections import Counter
from typing import Optional

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from src.config import CLASS_NAMES, VIOLATION_CLASSES


def auto_device() -> str:
    """Pick the best available inference device on the current machine.

    Returns:
        ``"cuda"`` if an NVIDIA GPU is available, ``"mps"`` on Apple Silicon, else
        ``"cpu"``. Ultralytics accepts all three as its ``device`` argument.
    """
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(weights_path: str, device: Optional[str] = None) -> YOLO:
    """Load a YOLO checkpoint onto the best available device.

    Args:
        weights_path: Path to a ``.pt`` checkpoint (e.g. ``models/best.pt``).
        device: Force a specific device string; auto-detected if ``None``.

    Returns:
        A ready-to-use ``ultralytics.YOLO`` model.
    """
    model = YOLO(weights_path)
    model.to(device or auto_device())
    return model


def summarize_detections(class_ids: list) -> dict:
    """Count detections per class name from a list of predicted class indices.

    Args:
        class_ids: Predicted class indices for one image/frame.

    Returns:
        Mapping of class name to detection count, for classes with at least one
        detection, most common first.
    """
    counts = Counter(CLASS_NAMES[i] for i in class_ids)
    return dict(counts.most_common())


def has_violation(class_ids: list) -> bool:
    """Whether any detected class represents a PPE compliance violation.

    Args:
        class_ids: Predicted class indices for one image/frame.

    Returns:
        True if any detection is one of ``src.config.VIOLATION_CLASSES``.
    """
    return any(CLASS_NAMES[i] in VIOLATION_CLASSES for i in class_ids)


def annotate_image(model: YOLO, image: np.ndarray, conf: float = 0.25) -> tuple:
    """Run detection on a single image and return the annotated result.

    Args:
        model: A loaded YOLO model.
        image: BGR image array (as read by OpenCV) or RGB (Ultralytics handles both
            consistently as long as it's used the same way in and out).
        conf: Confidence threshold for keeping a detection.

    Returns:
        Tuple of ``(annotated_image, class_counts, violation_flag)``.
    """
    results = model.predict(image, conf=conf, verbose=False)
    result = results[0]
    class_ids = result.boxes.cls.int().tolist() if result.boxes is not None else []
    annotated = result.plot()
    return annotated, summarize_detections(class_ids), has_violation(class_ids)


def annotate_video(
    model: YOLO, input_path: str, output_path: str, conf: float = 0.25
) -> dict:
    """Run detection frame-by-frame on a video and write an annotated copy.

    Args:
        model: A loaded YOLO model.
        input_path: Path to the source video file.
        output_path: Path to write the annotated video to (mp4).
        conf: Confidence threshold for keeping a detection.

    Returns:
        Aggregate class counts across every frame, most common first.
    """
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    total_counts = Counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        results = model.predict(frame, conf=conf, verbose=False)
        result = results[0]
        class_ids = result.boxes.cls.int().tolist() if result.boxes is not None else []
        total_counts.update(CLASS_NAMES[i] for i in class_ids)
        writer.write(result.plot())

    cap.release()
    writer.release()
    return dict(total_counts.most_common())
