"""Tests exercising real inference with the trained flagship model.

Skipped automatically if no trained model is present (models/ is gitignored; run
`python -m src.train` first).
"""

import cv2
import pytest

from src.config import CLASS_NAMES, DATA_DIR, FLAGSHIP_MODEL_PATH
from src.serve_utils import annotate_image, load_model

pytestmark = pytest.mark.skipif(
    not FLAGSHIP_MODEL_PATH.exists(),
    reason="No trained model found -- run `python -m src.train` first.",
)


@pytest.fixture(scope="module")
def model():
    return load_model(str(FLAGSHIP_MODEL_PATH))


@pytest.fixture(scope="module")
def sample_image_path():
    images = sorted((DATA_DIR / "test" / "images").glob("*.jpg"))
    assert images, "test split has no images"
    return images[0]


def test_annotate_image_returns_correct_shape(model, sample_image_path):
    img = cv2.imread(str(sample_image_path))
    annotated, counts, violation = annotate_image(model, img, conf=0.25)
    assert annotated.shape == img.shape


def test_annotate_image_detection_counts_use_known_class_names(model, sample_image_path):
    img = cv2.imread(str(sample_image_path))
    _, counts, _ = annotate_image(model, img, conf=0.25)
    for class_name in counts:
        assert class_name in CLASS_NAMES


def test_annotate_image_higher_conf_never_finds_more_detections(model, sample_image_path):
    img = cv2.imread(str(sample_image_path))
    _, low_conf_counts, _ = annotate_image(model, img, conf=0.1)
    _, high_conf_counts, _ = annotate_image(model, img, conf=0.8)
    assert sum(high_conf_counts.values()) <= sum(low_conf_counts.values())


def test_model_detects_something_on_a_real_test_image(model):
    # A weaker but still meaningful check across several images, in case any single
    # image happens to have no detections above threshold.
    images = sorted((DATA_DIR / "test" / "images").glob("*.jpg"))[:5]
    total_detections = 0
    for path in images:
        img = cv2.imread(str(path))
        _, counts, _ = annotate_image(model, img, conf=0.25)
        total_detections += sum(counts.values())
    assert total_detections > 0
