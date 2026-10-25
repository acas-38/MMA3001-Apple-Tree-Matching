"""Deterministic numerical verification for MMA3001 matching features.

Uses synthetic image arrays rather than course data. These tests verify code
behaviour and known answers, NOT the physical accuracy of tree recognition.
"""
from pathlib import Path
import importlib.util
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code" / "30_compare_matching_methods.py"
spec = importlib.util.spec_from_file_location("matching_methods", SCRIPT)
matching = importlib.util.module_from_spec(spec)
spec.loader.exec_module(matching)


def test_global_features_known_values():
    """Verify mean, population standard deviation and zero percentage."""
    pixels = np.array([[0., 0.], [2., 2.]])
    np.testing.assert_allclose(matching.global_features(pixels), [1., 1., 50.])


def test_spatial_feature_vector_shape_and_order():
    """Each of nine spatial cells contributes three values."""
    pixels = np.zeros((6, 6), dtype=float)
    pixels[:2, :2] = 10
    features = matching.spatial_features(pixels)
    assert features.shape == (27,)
    np.testing.assert_allclose(features[:3], [10, 0, 0])
    np.testing.assert_allclose(features[3:6], [0, 0, 100])


def test_global_features_all_nonfinite():
    """Invalid-only input must yield explicit nonfinite features."""
    result = matching.global_features(np.array([[np.nan, np.inf]]))
    assert len(result) == 3
    assert np.isnan(result).all()


def test_comparison_perfect_identification():
    """Distinct synthetic feature vectors should match correctly."""
    winter = np.array([[0., 0.], [2., 4.], [5., 10.]])
    summer = winter.copy()
    ranks, predictions, valid, gallery_size = matching.compare(summer, winter)
    np.testing.assert_array_equal(ranks, [1, 1, 1])
    np.testing.assert_array_equal(predictions, [0, 1, 2])
    assert valid.all()
    assert gallery_size == 3


def test_query_with_missing_feature_counts_as_failure():
    """A missing query is not excluded from the evaluation denominator."""
    winter = np.array([[0., 1.], [10., 11.]])
    summer = np.array([[np.nan, 1.], [10., 11.]])
    ranks, predicted, valid, gallery_size = matching.compare(summer, winter)
    assert np.isnan(ranks[0])
    assert predicted[0] == -1
    assert not valid[0]
    assert ranks[1] == 1
    assert gallery_size == 2


def test_invalid_gallery_item_excluded_but_query_retained():
    """Unusable gallery entry is not a candidate, yet its query still counts."""
    winter = np.array([[0., 0.], [np.nan, 2.], [3., 3.]])
    summer = np.array([[0., 0.], [1., 1.], [3., 3.]])
    ranks, predictions, valid, gallery_size = matching.compare(summer, winter)
    assert gallery_size == 2
    assert predictions[1] != 1
    assert np.isnan(ranks[1])
    assert not valid[1]
    assert ranks[0] == 1 and ranks[2] == 1


def test_uninformative_constant_features_do_not_divide_by_zero():
    """Zero standard deviation features must not create invalid distances."""
    winter = np.ones((2, 3))
    summer = np.ones((2, 3))
    ranks, predictions, valid, gallery_size = matching.compare(summer, winter)
    assert gallery_size == 2
    assert np.isfinite(ranks).all()
    assert valid.all()
    # Stable tie handling always ranks gallery index zero first.
    np.testing.assert_array_equal(predictions, [0, 0])


def test_no_valid_gallery_graceful_failure():
    winter = np.full((2, 3), np.nan)
    summer = np.zeros((2, 3))
    ranks, predictions, valid, gallery_size = matching.compare(summer, winter)
    assert gallery_size == 0
    assert np.isnan(ranks).all()
    assert (predictions == -1).all()
    assert not valid.any()


def test_geometry_blank_image_returns_invalid_features():
    """Blank masks must not be represented as valid detected tree geometry."""
    result = matching.geometry_features(np.zeros((80, 80)))
    assert result.shape == (8,)
    assert np.isnan(result).all()


def test_geometry_synthetic_vertical_component():
    """Thick synthetic vertical component gives finite, bounded descriptors."""
    pixels = np.zeros((120, 120))
    pixels[10:110, 54:66] = 255
    result = matching.geometry_features(pixels)
    assert result.shape == (8,)
    assert np.isfinite(result).all()
    assert 0 < result[6] <= 1
    assert 0 < result[7] <= 1
