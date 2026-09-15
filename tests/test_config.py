"""Tests for src/config.py."""

from src.config import CLASS_NAMES, VIOLATION_CLASSES


def test_class_names_count_matches_dataset():
    assert len(CLASS_NAMES) == 10


def test_class_names_are_real_words_not_corrupted():
    # Guards against the kind of corrupted-export class list this check is meant to catch.
    for name in CLASS_NAMES:
        assert name.strip() == name
        assert len(name) >= 2
        assert any(c.isalpha() for c in name)


def test_class_names_no_duplicates():
    assert len(CLASS_NAMES) == len(set(CLASS_NAMES))


def test_violation_classes_are_subset_of_class_names():
    assert VIOLATION_CLASSES.issubset(set(CLASS_NAMES))


def test_violation_classes_are_the_expected_three():
    assert VIOLATION_CLASSES == {"NO-Hardhat", "NO-Mask", "NO-Safety Vest"}
