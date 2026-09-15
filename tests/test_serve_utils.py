"""Tests for src/serve_utils.py."""

import torch

from src.serve_utils import auto_device, has_violation, summarize_detections


def test_auto_device_returns_valid_choice():
    device = auto_device()
    assert device in {"cuda", "mps", "cpu"}


def test_auto_device_matches_actual_availability():
    device = auto_device()
    if torch.cuda.is_available():
        assert device == "cuda"
    elif torch.backends.mps.is_available():
        assert device == "mps"
    else:
        assert device == "cpu"


def test_summarize_detections_counts_per_class():
    # 0=Hardhat, 5=Person, 5=Person, 2=NO-Hardhat
    counts = summarize_detections([0, 5, 5, 2])
    assert counts["Person"] == 2
    assert counts["Hardhat"] == 1
    assert counts["NO-Hardhat"] == 1


def test_summarize_detections_empty_input():
    assert summarize_detections([]) == {}


def test_summarize_detections_most_common_first():
    counts = summarize_detections([5, 5, 5, 0, 0, 8])
    assert list(counts.keys())[0] == "Person"


def test_has_violation_true_for_no_hardhat():
    assert has_violation([2]) is True  # NO-Hardhat


def test_has_violation_true_for_no_mask():
    assert has_violation([3]) is True  # NO-Mask


def test_has_violation_true_for_no_safety_vest():
    assert has_violation([4]) is True  # NO-Safety Vest


def test_has_violation_false_for_compliant_classes():
    assert has_violation([0, 1, 5, 7]) is False  # Hardhat, Mask, Person, Safety Vest


def test_has_violation_false_for_empty():
    assert has_violation([]) is False


def test_has_violation_mixed_detections():
    # A compliant worker (Hardhat) alongside a violation (NO-Mask) in the same frame.
    assert has_violation([0, 3]) is True
