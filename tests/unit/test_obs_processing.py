# tests/unit/test_obs_processing.py
"""
Unit tests for observation processing functions.
"""

import numpy as np
import pytest

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.client.obs_processing import (
    motion_energy,
    motion_energy_scalar,
    object_signatures,
    cosine_sim,
    get_object_centroids,
    get_object_bboxes,
    count_objects,
)


@pytest.mark.unit
class TestMotionEnergy:
    """Tests for motion energy functions."""
    
    def test_motion_energy_identical_frames(self, sample_rgba):
        """Identical frames should have zero motion."""
        result = motion_energy(sample_rgba, sample_rgba)
        assert result.shape == sample_rgba.shape[:2]
        assert np.all(result == 0)
    
    def test_motion_energy_different_frames(self, sample_rgba):
        """Different frames should have non-zero motion."""
        frame2 = sample_rgba.copy()
        frame2[:100, :100, :3] = 255  # White square
        
        result = motion_energy(sample_rgba, frame2)
        assert result.shape == sample_rgba.shape[:2]
        assert np.sum(result) > 0
    
    def test_motion_energy_scalar_identical(self, sample_rgba):
        """Scalar motion for identical frames should be ~0."""
        result = motion_energy_scalar(sample_rgba, sample_rgba)
        assert result == 0.0
    
    def test_motion_energy_scalar_range(self, sample_rgba):
        """Scalar motion should be in [0, 1]."""
        frame2 = np.zeros_like(sample_rgba)
        frame2[:, :, 3] = 255
        
        result = motion_energy_scalar(sample_rgba, frame2)
        assert 0.0 <= result <= 1.0


@pytest.mark.unit
class TestObjectSignatures:
    """Tests for object signature extraction."""
    
    def test_extracts_signatures_for_objects(self, sample_rgba, sample_index):
        """Should extract signatures for each object."""
        sigs = object_signatures(sample_rgba, sample_index, min_pixels=100)
        
        assert isinstance(sigs, dict)
        # Should have signatures for objects 1, 2, 3
        assert len(sigs) >= 1
    
    def test_signature_shape(self, sample_rgba, sample_index):
        """Signatures should have correct shape (8 bins * 3 channels = 24)."""
        sigs = object_signatures(sample_rgba, sample_index, min_pixels=100)
        
        for sig in sigs.values():
            assert sig.shape == (24,)
            assert sig.dtype == np.float32
    
    def test_excludes_small_objects(self, sample_rgba, sample_index):
        """Should exclude objects with fewer pixels than threshold."""
        # Set very high threshold
        sigs = object_signatures(sample_rgba, sample_index, min_pixels=100000)
        assert len(sigs) == 0
    
    def test_excludes_background(self, sample_rgba, sample_index):
        """Should not include background (id=0)."""
        sigs = object_signatures(sample_rgba, sample_index)
        assert 0 not in sigs


@pytest.mark.unit
class TestCosineSimilarity:
    """Tests for cosine similarity function."""
    
    def test_identical_vectors(self):
        """Identical vectors should have similarity 1.0."""
        v = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        assert cosine_sim(v, v) == pytest.approx(1.0, rel=1e-5)
    
    def test_orthogonal_vectors(self):
        """Orthogonal vectors should have similarity 0."""
        v1 = np.array([1, 0, 0], dtype=np.float32)
        v2 = np.array([0, 1, 0], dtype=np.float32)
        assert cosine_sim(v1, v2) == pytest.approx(0.0, abs=1e-5)
    
    def test_opposite_vectors(self):
        """Opposite vectors should have similarity -1.0."""
        v1 = np.array([1, 2, 3], dtype=np.float32)
        v2 = np.array([-1, -2, -3], dtype=np.float32)
        assert cosine_sim(v1, v2) == pytest.approx(-1.0, rel=1e-5)


@pytest.mark.unit
class TestObjectCentroids:
    """Tests for object centroid extraction."""
    
    def test_extracts_centroids(self, sample_index):
        """Should extract centroid for each object."""
        centroids = get_object_centroids(sample_index)
        
        assert isinstance(centroids, dict)
        assert 1 in centroids
        assert 2 in centroids
        assert 3 in centroids
    
    def test_centroid_format(self, sample_index):
        """Centroids should be (x, y) integer tuples."""
        centroids = get_object_centroids(sample_index)
        
        for oid, (x, y) in centroids.items():
            assert isinstance(x, int)
            assert isinstance(y, int)
    
    def test_excludes_background(self, sample_index):
        """Should not include background."""
        centroids = get_object_centroids(sample_index)
        assert 0 not in centroids


@pytest.mark.unit
class TestObjectBboxes:
    """Tests for object bounding box extraction."""
    
    def test_extracts_bboxes(self, sample_index):
        """Should extract bounding box for each object."""
        bboxes = get_object_bboxes(sample_index)
        
        assert isinstance(bboxes, dict)
        assert len(bboxes) >= 1
    
    def test_bbox_format(self, sample_index):
        """Bboxes should be (x1, y1, x2, y2) tuples."""
        bboxes = get_object_bboxes(sample_index)
        
        for oid, (x1, y1, x2, y2) in bboxes.items():
            assert x1 <= x2
            assert y1 <= y2


@pytest.mark.unit
class TestCountObjects:
    """Tests for object counting."""
    
    def test_counts_objects(self, sample_index):
        """Should count objects correctly."""
        count = count_objects(sample_index, min_pixels=100)
        assert count == 3
    
    def test_respects_min_pixels(self, sample_index):
        """Should respect minimum pixel threshold."""
        count = count_objects(sample_index, min_pixels=100000)
        assert count == 0
