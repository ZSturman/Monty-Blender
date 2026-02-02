# tests/integration/test_server_connection.py
"""
Integration tests for Blender server connection.

These tests require the Blender server to be running.
Run with: pytest tests/integration/ -m integration
"""

import pytest
import numpy as np

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


@pytest.mark.integration
@pytest.mark.requires_blender
class TestServerConnection:
    """Tests for server connection."""
    
    @pytest.fixture
    def rpc_client(self):
        """Create RPC client connection."""
        from src.client.blender_rpc import BlenderRPC
        
        try:
            client = BlenderRPC(host="127.0.0.1", port=5555, timeout=5.0)
            yield client
            client.close()
        except ConnectionError:
            pytest.skip("Blender server not running")
    
    def test_connection(self, rpc_client):
        """Should connect to server."""
        assert rpc_client.sock is not None
    
    def test_render(self, rpc_client):
        """Should receive render data."""
        rgba, depth, index = rpc_client.render()
        
        assert rgba.shape[2] == 4  # RGBA
        assert rgba.dtype == np.uint8
        assert depth.dtype == np.float32
        assert index.dtype == np.int32
    
    def test_reset(self, rpc_client):
        """Should reset scene."""
        resp = rpc_client.reset(frame=1)
        
        assert resp["ok"] == True
        assert resp["frame"] == 1
    
    def test_step(self, rpc_client):
        """Should step simulation."""
        rpc_client.reset(frame=1)
        
        action = {
            "dpos": [0.0, 0.0, 0.0],
            "drot_euler": [0.0, 0.0, 0.01],
        }
        resp = rpc_client.step(n=1, action=action)
        
        assert resp["ok"] == True
        assert resp["frame"] == 2
    
    def test_state(self, rpc_client):
        """Should get state."""
        state = rpc_client.state()
        
        assert state["ok"] == True
        assert "frame" in state
        assert "camera" in state
    
    def test_list_cameras(self, rpc_client):
        """Should list cameras."""
        cameras = rpc_client.list_cameras()
        
        assert isinstance(cameras, list)
        assert len(cameras) >= 1
        assert "name" in cameras[0]


@pytest.mark.integration
@pytest.mark.requires_blender
class TestGymEnvironment:
    """Tests for Gym environment."""
    
    @pytest.fixture
    def env(self):
        """Create Gym environment."""
        from src.client.blender_gym_env import BlenderVisualEnv
        
        try:
            environment = BlenderVisualEnv(
                host="127.0.0.1",
                port=5555,
                max_steps=10,
            )
            yield environment
            environment.close()
        except ConnectionError:
            pytest.skip("Blender server not running")
    
    def test_reset(self, env):
        """Should reset environment."""
        obs, info = env.reset()
        
        assert "rgba" in obs
        assert "depth" in obs
        assert "index" in obs
        assert "frame" in info
    
    def test_step(self, env):
        """Should step environment."""
        env.reset()
        
        action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.02])
        obs, reward, terminated, truncated, info = env.step(action)
        
        assert "rgba" in obs
        assert "step" in info
        assert info["step"] == 1
    
    def test_termination_on_max_steps(self, env):
        """Should truncate after max_steps."""
        env.reset()
        
        action = np.zeros(6)
        
        for _ in range(15):  # More than max_steps=10
            obs, reward, terminated, truncated, info = env.step(action)
            if truncated:
                break
        
        assert truncated
        assert info["stop_reason"] == "max_steps"
