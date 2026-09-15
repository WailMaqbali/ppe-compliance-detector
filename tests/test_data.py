"""Tests validating the PPE dataset on disk.

Skipped automatically if data/ hasn't been set up yet (it's gitignored -- see
README.md for the clone command).
"""

import yaml
import pytest

from src.config import DATA_DIR, DATA_YAML

pytestmark = pytest.mark.skipif(
    not DATA_YAML.exists(), reason="data/ not set up -- see README.md for the clone command."
)


@pytest.fixture(scope="module")
def data_config():
    with open(DATA_YAML) as f:
        return yaml.safe_load(f)


def test_data_yaml_class_count(data_config):
    assert data_config["nc"] == 10
    assert len(data_config["names"]) == 10


def test_data_yaml_class_names_are_real_words(data_config):
    for name in data_config["names"]:
        assert any(c.isalpha() for c in name), f"suspicious class name: {name!r}"


@pytest.mark.parametrize(
    "split,expected_count", [("train", 2605), ("valid", 114), ("test", 82)]
)
def test_split_image_counts(split, expected_count):
    images = list((DATA_DIR / split / "images").glob("*.jpg"))
    assert len(images) == expected_count


@pytest.mark.parametrize("split", ["train", "valid", "test"])
def test_every_image_has_a_matching_label_file(split):
    images = {p.stem for p in (DATA_DIR / split / "images").glob("*.jpg")}
    labels = {p.stem for p in (DATA_DIR / split / "labels").glob("*.txt")}
    assert images == labels


@pytest.mark.parametrize("split", ["train", "valid", "test"])
def test_all_label_lines_are_well_formed(split, data_config):
    nc = data_config["nc"]
    for label_file in (DATA_DIR / split / "labels").glob("*.txt"):
        for line in label_file.read_text().splitlines():
            if not line.strip():
                continue
            parts = line.split()
            assert len(parts) == 5, f"{label_file}: expected 5 fields, got {len(parts)}"
            cls_id = int(parts[0])
            assert 0 <= cls_id < nc, f"{label_file}: class id {cls_id} out of range"
            for coord in parts[1:]:
                assert 0.0 <= float(coord) <= 1.0, f"{label_file}: coord {coord} out of [0,1]"
