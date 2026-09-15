"""Tests for src/app.py.

Only checks that the Gradio Blocks graph builds correctly; model loading is lazy
(see src.app.get_model), so this does not require a trained checkpoint to exist.
"""

import gradio as gr

from src.app import build_app


def test_build_app_returns_blocks():
    demo = build_app()
    assert isinstance(demo, gr.Blocks)


def test_build_app_has_three_tabs():
    demo = build_app()
    tab_labels = [
        child.label
        for child in demo.blocks.values()
        if isinstance(child, gr.Tab)
    ]
    assert set(tab_labels) == {"Image upload", "Video upload", "Webcam"}
