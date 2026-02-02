# tests/conftest.py
"""
Pytest fixtures and configuration for Blender+Monty tests.
"""

import json
import os
import tempfile
from unittest.mock import MagicMock

import numpy as np
import pytest


@pytest.fixture
def sample_rgba():
    """Generate sample RGBA image."""
    h, w = 240, 320
    rgba = np.random.randint(0, 255, (h, w, 4), dtype=np.uint8)
    rgba[:, :, 3] = 255  # Full opacity
    return rgba


@pytest.fixture
def sample_depth():
    """Generate sample depth map."""
    h, w = 240, 320
    depth = np.random.uniform(0.5, 10.0, (h, w)).astype(np.float32)
    return depth


@pytest.fixture
def sample_index():
    """Generate sample object index map with 3 objects."""
    h, w = 240, 320
    index = np.zeros((h, w), dtype=np.int32)
    
    # Object 1: top-left region
    index[20:80, 20:100] = 1
    
    # Object 2: center region
    index[100:180, 100:200] = 2
    
    # Object 3: bottom-right region
    index[180:230, 220:300] = 3
    
    return index


@pytest.fixture
def sample_observation(sample_rgba, sample_depth, sample_index):
    """Generate complete sample observation."""
    return {
        "rgba": sample_rgba,
        "depth": sample_depth,
        "index": sample_index,
    }


@pytest.fixture
def temp_labels_file():
    """Create temporary labels file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump({"objects": [], "actions": []}, f)
        path = f.name
    
    yield path
    
    # Cleanup
    if os.path.exists(path):
        os.remove(path)


@pytest.fixture
def sample_labels_file():
    """Create labels file with sample data."""
    data = {
        "objects": [
            {
                "label": "cube",
                "signature": [0.1] * 24,
                "created_at": 1234567890
            },
            {
                "label": "sphere",
                "signature": [0.2] * 24,
                "created_at": 1234567891
            }
        ],
        "actions": []
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(data, f)
        path = f.name
    
    yield path
    
    if os.path.exists(path):
        os.remove(path)


@pytest.fixture
def temp_output_dir():
    """Create temporary output directory."""
    path = tempfile.mkdtemp()
    yield path
    
    # Cleanup
    import shutil
    if os.path.exists(path):
        shutil.rmtree(path)


@pytest.fixture
def mock_blender_rpc():
    """Create mock BlenderRPC for testing without server."""
    mock = MagicMock()
    
    # Default responses
    mock.render.return_value = (
        np.random.randint(0, 255, (240, 320, 4), dtype=np.uint8),
        np.random.uniform(0.5, 10.0, (240, 320)).astype(np.float32),
        np.zeros((240, 320), dtype=np.int32),
    )
    
    mock.state.return_value = {
        "ok": True,
        "frame": 1,
        "camera": {
            "name": "Camera",
            "location": [0, -5, 2],
            "rotation_euler": [1.0, 0, 0],
        }
    }
    
    mock.reset.return_value = {"ok": True, "frame": 1}
    mock.step.return_value = {"ok": True, "frame": 2}
    
    mock.list_cameras.return_value = [
        {"name": "Camera", "type": "PERSP", "lens": 50},
        {"name": "TopCam", "type": "ORTHO", "ortho_scale": 10},
    ]
    
    return mock


# Markers for test categories
def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line(
        "markers", "unit: Unit tests (no external dependencies)"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests (may require server)"
    )
    config.addinivalue_line(
        "markers", "requires_blender: Tests that require Blender server running"
    )
