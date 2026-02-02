# tests/unit/test_labeler.py
"""
Unit tests for novelty labeling system.
"""

import numpy as np
import pytest

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.client.novelty_labeler import LabelMemory, NoveltyController


@pytest.mark.unit
class TestLabelMemory:
    """Tests for LabelMemory class."""
    
    def test_init_empty(self, temp_labels_file):
        """Should initialize with empty data."""
        mem = LabelMemory(path=temp_labels_file)
        assert len(mem.data["objects"]) == 0
    
    def test_init_with_existing(self, sample_labels_file):
        """Should load existing labels."""
        mem = LabelMemory(path=sample_labels_file)
        assert len(mem.data["objects"]) == 2
        assert mem.data["objects"][0]["label"] == "cube"
    
    def test_add_object(self, temp_labels_file):
        """Should add object and save."""
        mem = LabelMemory(path=temp_labels_file)
        sig = np.array([0.1] * 24, dtype=np.float32)
        
        mem.add_object("test_object", sig)
        
        assert len(mem.data["objects"]) == 1
        assert mem.data["objects"][0]["label"] == "test_object"
    
    def test_match_object_found(self, sample_labels_file):
        """Should match similar signatures."""
        mem = LabelMemory(path=sample_labels_file, sim_thresh=0.9)
        
        # Very similar to stored "cube" signature
        sig = np.array([0.1] * 24, dtype=np.float32)
        label, sim = mem.match_object(sig)
        
        assert label == "cube"
        assert sim >= 0.9
    
    def test_match_object_not_found(self, sample_labels_file):
        """Should return None for dissimilar signatures."""
        mem = LabelMemory(path=sample_labels_file, sim_thresh=0.99)
        
        # Create a signature that points in a different direction
        # The stored signatures are uniform [0.5]*24 and [0.3]*24
        # This alternating pattern will have lower cosine similarity
        sig = np.array([1.0, -1.0] * 12, dtype=np.float32)
        label, sim = mem.match_object(sig)
        
        # With high threshold and different direction, shouldn't match
        assert label is None or sim < 0.99
    
    def test_get_all_labels(self, sample_labels_file):
        """Should return all labels."""
        mem = LabelMemory(path=sample_labels_file)
        labels = mem.get_all_labels()
        
        assert "cube" in labels
        assert "sphere" in labels
    
    def test_clear(self, temp_labels_file):
        """Should clear all data."""
        mem = LabelMemory(path=temp_labels_file)
        sig = np.array([0.1] * 24, dtype=np.float32)
        mem.add_object("test", sig)
        
        mem.clear()
        
        assert len(mem.data["objects"]) == 0


@pytest.mark.unit
class TestNoveltyController:
    """Tests for NoveltyController class."""
    
    def test_update_returns_labels(self, temp_labels_file):
        """Update should return label dict."""
        mem = LabelMemory(path=temp_labels_file)
        ctrl = NoveltyController(mem, stable_steps=3)
        
        sigs = {1: np.array([0.1] * 24, dtype=np.float32)}
        labels = ctrl.update(sigs)
        
        assert isinstance(labels, dict)
        assert 1 in labels
    
    def test_unknown_objects_get_none(self, temp_labels_file):
        """Unknown objects should get None label."""
        mem = LabelMemory(path=temp_labels_file)
        ctrl = NoveltyController(mem, stable_steps=3)
        
        sigs = {1: np.array([0.1] * 24, dtype=np.float32)}
        labels = ctrl.update(sigs)
        
        assert labels[1] is None
    
    def test_known_objects_get_label(self, sample_labels_file):
        """Known objects should get matching label."""
        mem = LabelMemory(path=sample_labels_file, sim_thresh=0.9)
        ctrl = NoveltyController(mem, stable_steps=3)
        
        # Use signature similar to "cube"
        sigs = {1: np.array([0.1] * 24, dtype=np.float32)}
        labels = ctrl.update(sigs)
        
        assert labels[1] == "cube"
    
    def test_pending_prompts_after_stable(self, temp_labels_file):
        """Should have pending prompts after stable_steps."""
        mem = LabelMemory(path=temp_labels_file)
        ctrl = NoveltyController(mem, stable_steps=3)
        
        sig = np.array([0.5] * 24, dtype=np.float32)
        
        # Update multiple times
        for _ in range(5):
            ctrl.update({1: sig})
        
        assert ctrl.has_pending_prompts()
        assert ctrl.get_pending_count() == 1
    
    def test_no_pending_before_stable(self, temp_labels_file):
        """Should not have pending prompts before stable_steps."""
        mem = LabelMemory(path=temp_labels_file)
        ctrl = NoveltyController(mem, stable_steps=10)
        
        sig = np.array([0.5] * 24, dtype=np.float32)
        
        # Update just twice
        ctrl.update({1: sig})
        ctrl.update({1: sig})
        
        assert not ctrl.has_pending_prompts()
    
    def test_reset_clears_state(self, temp_labels_file):
        """Reset should clear tracking state."""
        mem = LabelMemory(path=temp_labels_file)
        ctrl = NoveltyController(mem, stable_steps=2)
        
        sig = np.array([0.5] * 24, dtype=np.float32)
        for _ in range(5):
            ctrl.update({1: sig})
        
        ctrl.reset()
        
        assert not ctrl.has_pending_prompts()
        assert ctrl.get_pending_count() == 0
