# src/client/obs_processing.py
"""
Observation processing utilities for visual learning.

Provides functions for:
    - Motion energy computation
    - Object signature extraction (RGB histograms)
    - Similarity metrics
"""

from typing import Dict, Tuple

import cv2
import numpy as np


def motion_energy(prev_rgba: np.ndarray, rgba: np.ndarray) -> np.ndarray:
    """
    Compute per-pixel motion energy between frames.
    
    Args:
        prev_rgba: Previous frame (H, W, 4) uint8
        rgba: Current frame (H, W, 4) uint8
        
    Returns:
        Motion energy map (H, W) uint8
    """
    prev = cv2.cvtColor(prev_rgba[:, :, :3], cv2.COLOR_RGB2GRAY)
    cur = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2GRAY)
    diff = cv2.absdiff(cur, prev)
    return diff


def motion_energy_scalar(prev_rgba: np.ndarray, rgba: np.ndarray) -> float:
    """
    Compute scalar motion energy between frames.
    
    Args:
        prev_rgba: Previous frame (H, W, 4) uint8
        rgba: Current frame (H, W, 4) uint8
        
    Returns:
        Normalized motion energy in [0, 1]
    """
    diff = motion_energy(prev_rgba, rgba)
    return float(np.mean(diff)) / 255.0


def object_signatures(
    rgba: np.ndarray, 
    index: np.ndarray, 
    min_pixels: int = 200,
    n_bins: int = 8
) -> Dict[int, np.ndarray]:
    """
    Extract visual signatures for each object in frame.
    
    Uses RGB histograms as simple but effective signatures.
    
    Args:
        rgba: RGBA image (H, W, 4) uint8
        index: Object index map (H, W) int32
        min_pixels: Minimum pixels for object to be included
        n_bins: Number of histogram bins per channel
        
    Returns:
        Dict mapping object_id -> signature vector (n_bins*3,) float32
    """
    sigs = {}
    rgb = rgba[:, :, :3]
    
    unique_ids = np.unique(index)
    
    for oid in unique_ids:
        # Skip background (id 0)
        if oid <= 0:
            continue
        
        mask = (index == oid)
        pixel_count = mask.sum()
        
        if pixel_count < min_pixels:
            continue
        
        # Extract pixels for this object
        pixels = rgb[mask]  # (N, 3)
        
        # Build histogram signature
        hist = []
        for c in range(3):
            h, _ = np.histogram(
                pixels[:, c], 
                bins=n_bins, 
                range=(0, 255), 
                density=True
            )
            hist.append(h)
        
        sig = np.concatenate(hist).astype(np.float32)
        sigs[int(oid)] = sig
    
    return sigs


def cosine_sim(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    """
    Compute cosine similarity between two vectors.
    
    Args:
        a: First vector
        b: Second vector
        eps: Small value to prevent division by zero
        
    Returns:
        Cosine similarity in [-1, 1]
    """
    na = np.linalg.norm(a) + eps
    nb = np.linalg.norm(b) + eps
    return float(np.dot(a, b) / (na * nb))


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute Euclidean distance between two vectors.
    
    Args:
        a: First vector
        b: Second vector
        
    Returns:
        Euclidean distance
    """
    return float(np.linalg.norm(a - b))


def get_object_centroids(index: np.ndarray) -> Dict[int, Tuple[int, int]]:
    """
    Get centroid (x, y) for each object in index map.
    
    Args:
        index: Object index map (H, W) int32
        
    Returns:
        Dict mapping object_id -> (x, y) centroid coordinates
    """
    centroids = {}
    
    unique_ids = np.unique(index)
    
    for oid in unique_ids:
        if oid <= 0:
            continue
        
        mask = (index == oid)
        ys, xs = np.where(mask)
        
        if len(xs) == 0:
            continue
        
        centroids[int(oid)] = (int(np.mean(xs)), int(np.mean(ys)))
    
    return centroids


def get_object_bboxes(index: np.ndarray) -> Dict[int, Tuple[int, int, int, int]]:
    """
    Get bounding boxes for each object in index map.
    
    Args:
        index: Object index map (H, W) int32
        
    Returns:
        Dict mapping object_id -> (x1, y1, x2, y2) bounding box
    """
    bboxes = {}
    
    unique_ids = np.unique(index)
    
    for oid in unique_ids:
        if oid <= 0:
            continue
        
        mask = (index == oid)
        ys, xs = np.where(mask)
        
        if len(xs) == 0:
            continue
        
        bboxes[int(oid)] = (
            int(np.min(xs)),
            int(np.min(ys)),
            int(np.max(xs)),
            int(np.max(ys))
        )
    
    return bboxes


def count_objects(index: np.ndarray, min_pixels: int = 100) -> int:
    """
    Count number of objects in index map.
    
    Args:
        index: Object index map (H, W) int32
        min_pixels: Minimum pixels for object to be counted
        
    Returns:
        Number of objects
    """
    count = 0
    unique_ids = np.unique(index)
    
    for oid in unique_ids:
        if oid <= 0:
            continue
        
        mask = (index == oid)
        if mask.sum() >= min_pixels:
            count += 1
    
    return count
