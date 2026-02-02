# tests/unit/test_camera_controller.py
"""
Unit tests for camera controller.
"""

import numpy as np
import pytest

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.client.camera import CameraController, MultiCameraController, MovementPreset


@pytest.mark.unit
class TestCameraController:
    """Tests for CameraController class."""
    
    def test_init_with_defaults(self):
        """Should initialize with default presets."""
        ctrl = CameraController()
        
        assert "orbit" in ctrl.presets
        assert "approach" in ctrl.presets
        assert "overview" in ctrl.presets
    
    def test_init_with_custom_presets(self):
        """Should accept custom presets from config."""
        custom = {
            "spin": {
                "description": "Fast spin",
                "pattern": [{"dpos": [0, 0, 0], "drot": [0, 0, 0.1]}],
                "duration": 50
            }
        }
        
        ctrl = CameraController(presets=custom)
        
        assert "spin" in ctrl.presets
        assert ctrl.presets["spin"].description == "Fast spin"
    
    def test_get_action_returns_dict(self):
        """get_action should return action dict."""
        ctrl = CameraController()
        ctrl.start_exploration()
        
        action = ctrl.get_action()
        
        assert "dpos" in action
        assert "drot_euler" in action
        assert len(action["dpos"]) == 3
        assert len(action["drot_euler"]) == 3
    
    def test_get_action_array_returns_numpy(self):
        """get_action_array should return numpy array."""
        ctrl = CameraController()
        ctrl.start_exploration()
        
        action = ctrl.get_action_array()
        
        assert isinstance(action, np.ndarray)
        assert action.shape == (6,)
        assert action.dtype == np.float32
    
    def test_scales_applied(self):
        """Should apply position and rotation scales."""
        ctrl = CameraController(position_scale=2.0, rotation_scale=0.5)
        ctrl.set_preset("orbit")  # orbit has rotation only
        
        action = ctrl.get_action()
        
        # Orbit preset has drot of 0.02, scaled by 0.5 = 0.01
        assert action["drot_euler"][2] == pytest.approx(0.01, rel=0.1)
    
    def test_phase_advancement(self):
        """Should advance to next phase after duration."""
        ctrl = CameraController()
        ctrl.exploration_sequence = ["approach", "orbit"]
        
        # Override duration for faster testing
        ctrl.presets["approach"].duration = 2
        ctrl.start_exploration()
        
        assert ctrl.current_phase_name == "approach"
        
        ctrl.get_action()
        ctrl.get_action()
        ctrl.get_action()  # Should trigger phase change
        
        assert ctrl.current_phase_name == "orbit"
    
    def test_exploration_completion(self):
        """Should mark exploration complete after all phases."""
        ctrl = CameraController()
        ctrl.exploration_sequence = ["orbit"]
        ctrl.presets["orbit"].duration = 2
        ctrl.start_exploration()
        
        assert not ctrl.is_exploration_complete
        
        ctrl.get_action()
        ctrl.get_action()
        ctrl.get_action()  # Completes
        
        assert ctrl.is_exploration_complete
    
    def test_set_preset(self):
        """set_preset should switch to single preset mode."""
        ctrl = CameraController()
        
        result = ctrl.set_preset("approach")
        
        assert result == True
        assert ctrl.current_phase_name == "approach"
    
    def test_set_preset_invalid(self):
        """set_preset should return False for invalid preset."""
        ctrl = CameraController()
        
        result = ctrl.set_preset("nonexistent")
        
        assert result == False
    
    def test_get_progress(self):
        """get_progress should return progress dict."""
        ctrl = CameraController()
        ctrl.start_exploration()
        
        progress = ctrl.get_progress()
        
        assert "phase" in progress
        assert "total_steps" in progress
        assert "is_complete" in progress
    
    def test_reset(self):
        """reset should clear state."""
        ctrl = CameraController()
        ctrl.start_exploration()
        
        for _ in range(10):
            ctrl.get_action()
        
        ctrl.reset()
        
        assert ctrl.total_steps == 0
        assert not ctrl.is_exploration_complete


@pytest.mark.unit
class TestMultiCameraController:
    """Tests for MultiCameraController class."""
    
    def test_init(self):
        """Should initialize with cameras."""
        switch_fn = lambda x: None
        ctrl = MultiCameraController(switch_fn, cameras=["Cam1", "Cam2"])
        
        assert ctrl.cameras == ["Cam1", "Cam2"]
        assert ctrl.current_camera == "Cam1"
    
    def test_next_camera(self):
        """next_camera should cycle through cameras."""
        calls = []
        switch_fn = lambda x: calls.append(x)
        ctrl = MultiCameraController(switch_fn, cameras=["Cam1", "Cam2", "Cam3"])
        
        ctrl.next_camera()
        assert ctrl.current_camera == "Cam2"
        
        ctrl.next_camera()
        assert ctrl.current_camera == "Cam3"
        
        ctrl.next_camera()
        assert ctrl.current_camera == "Cam1"  # Wraps around
    
    def test_switch_to(self):
        """switch_to should switch to specific camera."""
        calls = []
        switch_fn = lambda x: calls.append(x)
        ctrl = MultiCameraController(switch_fn, cameras=["Cam1", "Cam2", "Cam3"])
        
        result = ctrl.switch_to("Cam3")
        
        assert result == True
        assert ctrl.current_camera == "Cam3"
        assert "Cam3" in calls
    
    def test_switch_to_invalid(self):
        """switch_to should return False for invalid camera."""
        switch_fn = lambda x: None
        ctrl = MultiCameraController(switch_fn, cameras=["Cam1", "Cam2"])
        
        result = ctrl.switch_to("Invalid")
        
        assert result == False
    
    def test_auto_switch_on_interval(self):
        """Should auto-switch after interval."""
        calls = []
        switch_fn = lambda x: calls.append(x)
        ctrl = MultiCameraController(
            switch_fn, 
            cameras=["Cam1", "Cam2"],
            switch_interval=3
        )
        
        ctrl.on_step()  # 1
        ctrl.on_step()  # 2
        result = ctrl.on_step()  # 3 - should switch
        
        assert result == "Cam2"
