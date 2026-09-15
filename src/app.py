"""Gradio demo: upload an image or video (or use a webcam) and get PPE detections back.

Run:

    python -m src.app

Uses the flagship checkpoint (models/best.pt) for image/video, matching the mAP figures
reported in the README. Auto-detects CUDA / Apple Silicon MPS / CPU -- see
src/serve_utils.auto_device -- so the same code runs on the training machine or a
MacBook Air (M1) with no changes.
"""

import tempfile
from pathlib import Path

import cv2
import gradio as gr
import numpy as np

from src.config import FLAGSHIP_MODEL_PATH
from src.serve_utils import annotate_image, annotate_video, auto_device, load_model

_model = None


def get_model():
    """Lazily load the flagship model once and reuse it across requests."""
    global _model
    if _model is None:
        if not FLAGSHIP_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"No trained model found at {FLAGSHIP_MODEL_PATH}. "
                "Run `python -m src.train` first."
            )
        _model = load_model(str(FLAGSHIP_MODEL_PATH))
        print(f"Loaded {FLAGSHIP_MODEL_PATH} on device: {auto_device()}")
    return _model


def predict_image(image: np.ndarray, conf: float):
    """Run detection on an uploaded (or webcam-captured) image.

    Args:
        image: RGB image array from Gradio.
        conf: Confidence threshold slider value.

    Returns:
        Tuple of (annotated RGB image, detection summary markdown string).
    """
    if image is None:
        return None, "Upload an image first."
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    annotated_bgr, counts, violation = annotate_image(get_model(), bgr, conf=conf)
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    return annotated_rgb, _format_summary(counts, violation)


def predict_video(video_path: str, conf: float):
    """Run detection on every frame of an uploaded video and return an annotated copy.

    Args:
        video_path: Path to the uploaded video file (provided by Gradio).
        conf: Confidence threshold slider value.

    Returns:
        Tuple of (path to annotated mp4, detection summary markdown string).
    """
    if video_path is None:
        return None, "Upload a video first."
    out_path = str(Path(tempfile.gettempdir()) / "ppe_annotated_output.mp4")
    counts = annotate_video(get_model(), video_path, out_path, conf=conf)
    return out_path, _format_summary(counts, violation=any(
        "NO-" in k for k in counts
    ))


def _format_summary(counts: dict, violation: bool) -> str:
    """Render a detection count dict as a short markdown summary."""
    if not counts:
        return "No PPE-related objects detected."
    lines = [f"- **{name}**: {n}" for name, n in counts.items()]
    header = "**PPE VIOLATION DETECTED**" if violation else "No violations detected."
    return header + "\n\n" + "\n".join(lines)


def build_app() -> gr.Blocks:
    """Construct the Gradio Blocks app with image, video, and webcam tabs."""
    with gr.Blocks(title="PPE Compliance Detector") as demo:
        gr.Markdown(
            "# PPE Compliance Detector\n"
            "Upload a construction-site image or video, or use your webcam, to detect "
            "hardhats, masks, safety vests, and violations (missing PPE). "
            "Portfolio project -- not a certified safety system, see MODEL_CARD.md."
        )
        conf_slider = gr.Slider(0.05, 0.9, value=0.25, step=0.05, label="Confidence threshold")

        with gr.Tab("Image upload"):
            with gr.Row():
                img_in = gr.Image(type="numpy", label="Upload image")
                img_out = gr.Image(type="numpy", label="Detections")
            img_summary = gr.Markdown()
            img_btn = gr.Button("Detect", variant="primary")
            img_btn.click(predict_image, [img_in, conf_slider], [img_out, img_summary])

        with gr.Tab("Video upload"):
            with gr.Row():
                vid_in = gr.Video(label="Upload video")
                vid_out = gr.Video(label="Annotated result")
            vid_summary = gr.Markdown()
            vid_btn = gr.Button("Detect", variant="primary")
            vid_btn.click(predict_video, [vid_in, conf_slider], [vid_out, vid_summary])

        with gr.Tab("Webcam"):
            gr.Markdown("Capture a frame from your webcam and run detection on it.")
            with gr.Row():
                cam_in = gr.Image(type="numpy", label="Webcam", sources=["webcam"])
                cam_out = gr.Image(type="numpy", label="Detections")
            cam_summary = gr.Markdown()
            cam_btn = gr.Button("Detect", variant="primary")
            cam_btn.click(predict_image, [cam_in, conf_slider], [cam_out, cam_summary])

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch()
