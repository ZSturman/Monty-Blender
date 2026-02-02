# src/client/camera.py
"""
Camera controller with movement presets and phase cycling.

Provides structured camera movements for comprehensive 3D scene exploration.
"""

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Any

import numpy as np


class MovementPhase(Enum):
    """Movement phase types."""
    APPROACH = auto()
    ORBIT = auto()
    OVERVIEW = auto()
    CUSTOM = auto()


@dataclass
class MovementStep:
    """Single movement step."""
    dpos: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    drot: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])


@dataclass
class MovementPreset:
    """Movement pattern preset."""
    name: str
    description: str
    pattern: List[MovementStep]
    duration: int  # Steps before switching to next phase
    
    @classmethod
    def from_dict(cls, name: str, data: Dict) -> "MovementPreset":
        """Create preset from config dict."""
        pattern = []
        for step in data.get("pattern", []):
            pattern.append(MovementStep(
                dpos=step.get("dpos", [0, 0, 0]),
                drot=step.get("drot", [0, 0, 0])
            ))
        
        return cls(
            name=name,
            description=data.get("description", ""),
            pattern=pattern if pattern else [MovementStep()],
            duration=data.get("duration", 100)
        )


class CameraController:
    """
    Controls camera movement through exploration phases.
    
    Cycles through movement presets to provide comprehensive 3D coverage:
    - Approach: Move toward the scene center
    - Orbit: Rotate around the scene
    - Overview: High-angle pan
    
    Supports:
    - Preset-based movements from config
    - Phase cycling for full exploration
    - Manual action override
    - Multi-camera switching via callback
    """
    
    # Default presets if not provided via config
    DEFAULT_PRESETS = {
        "orbit": MovementPreset(
            name="orbit",
            description="Rotate around the scene center",
            pattern=[MovementStep(dpos=[0, 0, 0], drot=[0, 0, 0.02])],
            duration=100
        ),
        "approach": MovementPreset(
            name="approach",
            description="Move toward the mesh center",
            pattern=[
                MovementStep(dpos=[0, 0.05, 0], drot=[0, 0, 0]),
                MovementStep(dpos=[0, 0.05, -0.01], drot=[0, 0, 0]),
            ],
            duration=50
        ),
        "overview": MovementPreset(
            name="overview",
            description="High angle pan across scene",
            pattern=[MovementStep(dpos=[0.02, 0, 0], drot=[0, 0, 0.01])],
            duration=80
        ),
    }
    
    # Default exploration sequence
    DEFAULT_SEQUENCE = ["approach", "orbit", "overview"]
    
    def __init__(
        self,
        presets: Optional[Dict[str, Dict]] = None,
        exploration_sequence: Optional[List[str]] = None,
        position_scale: float = 1.0,
        rotation_scale: float = 1.0,
    ):
        """
        Initialize camera controller.
        
        Args:
            presets: Dict of preset configs from YAML
            exploration_sequence: List of preset names for full exploration
            position_scale: Global scale for position movements
            rotation_scale: Global scale for rotation movements
        """
        self.position_scale = position_scale
        self.rotation_scale = rotation_scale
        
        # Load presets
        self.presets: Dict[str, MovementPreset] = {}
        
        # Start with defaults
        for name, preset in self.DEFAULT_PRESETS.items():
            self.presets[name] = preset
        
        # Override/add from config
        if presets:
            for name, data in presets.items():
                if isinstance(data, dict) and "pattern" in data:
                    self.presets[name] = MovementPreset.from_dict(name, data)
        
        # Exploration sequence
        self.exploration_sequence = exploration_sequence or self.DEFAULT_SEQUENCE
        
        # State
        self._current_phase_idx = 0
        self._step_in_phase = 0
        self._pattern_idx = 0
        self._total_steps = 0
        self._is_exploring = False
    
    @property
    def current_preset(self) -> MovementPreset:
        """Get current active preset."""
        if self._current_phase_idx >= len(self.exploration_sequence):
            # Fallback to orbit when exploration complete
            return self.presets.get("orbit", self.DEFAULT_PRESETS["orbit"])
        name = self.exploration_sequence[self._current_phase_idx]
        return self.presets.get(name, self.DEFAULT_PRESETS["orbit"])
    
    @property
    def current_phase_name(self) -> str:
        """Get name of current phase."""
        if self._current_phase_idx >= len(self.exploration_sequence):
            return "complete"
        return self.exploration_sequence[self._current_phase_idx]
    
    @property
    def total_steps(self) -> int:
        """Get total steps taken."""
        return self._total_steps
    
    @property
    def is_exploration_complete(self) -> bool:
        """Check if full exploration sequence is complete."""
        return (self._current_phase_idx >= len(self.exploration_sequence) and 
                not self._is_exploring)
    
    def start_exploration(self) -> None:
        """Start/restart the exploration sequence."""
        self._current_phase_idx = 0
        self._step_in_phase = 0
        self._pattern_idx = 0
        self._is_exploring = True
    
    def get_action(self) -> Dict[str, Any]:
        """
        Get next action for current exploration phase.
        
        Returns:
            Action dict with 'dpos' and 'drot_euler' keys
        """
        preset = self.current_preset
        
        # Get current movement step
        step = preset.pattern[self._pattern_idx]
        
        # Build action with scaling
        action = {
            "type": "camera_delta",
            "dpos": [v * self.position_scale for v in step.dpos],
            "drot_euler": [v * self.rotation_scale for v in step.drot],
        }
        
        # Advance state
        self._pattern_idx = (self._pattern_idx + 1) % len(preset.pattern)
        self._step_in_phase += 1
        self._total_steps += 1
        
        # Check for phase transition
        if self._step_in_phase >= preset.duration:
            self._advance_phase()
        
        return action
    
    def get_action_array(self) -> np.ndarray:
        """
        Get next action as numpy array for gym environment.
        
        Returns:
            6D action array [dx, dy, dz, droll, dpitch, dyaw]
        """
        action = self.get_action()
        dpos = action["dpos"]
        drot = action["drot_euler"]
        return np.array(dpos + drot, dtype=np.float32)
    
    def _advance_phase(self) -> None:
        """Advance to next phase in sequence."""
        self._current_phase_idx += 1
        self._step_in_phase = 0
        self._pattern_idx = 0
        
        if self._current_phase_idx >= len(self.exploration_sequence):
            self._is_exploring = False
            print(f"[CameraController] Exploration complete after {self._total_steps} steps")
        else:
            print(f"[CameraController] Phase: {self.current_phase_name}")
    
    def set_preset(self, preset_name: str) -> bool:
        """
        Set single preset mode (no sequence).
        
        Args:
            preset_name: Name of preset to use
            
        Returns:
            True if preset exists
        """
        if preset_name not in self.presets:
            return False
        
        self.exploration_sequence = [preset_name]
        self._current_phase_idx = 0
        self._step_in_phase = 0
        self._pattern_idx = 0
        self._is_exploring = True
        
        # Set duration to very high for continuous mode
        self.presets[preset_name].duration = 999999
        
        return True
    
    def get_progress(self) -> Dict[str, Any]:
        """
        Get exploration progress info.
        
        Returns:
            Dict with progress information
        """
        preset = self.current_preset
        
        return {
            "phase": self.current_phase_name,
            "phase_idx": self._current_phase_idx,
            "total_phases": len(self.exploration_sequence),
            "step_in_phase": self._step_in_phase,
            "phase_duration": preset.duration,
            "total_steps": self._total_steps,
            "is_complete": self.is_exploration_complete,
        }
    
    def reset(self) -> None:
        """Reset controller state."""
        self._current_phase_idx = 0
        self._step_in_phase = 0
        self._pattern_idx = 0
        self._total_steps = 0
        self._is_exploring = False


class MultiCameraController:
    """
    Manages multiple cameras with automatic switching.
    
    Can switch cameras:
    - On phase transitions
    - After N steps
    - On keyboard input
    """
    
    def __init__(
        self,
        switch_camera_fn,
        cameras: Optional[List[str]] = None,
        switch_on_phase: bool = True,
        switch_interval: int = 0,
    ):
        """
        Initialize multi-camera controller.
        
        Args:
            switch_camera_fn: Function to call to switch camera (takes camera name)
            cameras: List of camera names (auto-discovered if None)
            switch_on_phase: Switch camera when phase changes
            switch_interval: Switch every N steps (0 = disabled)
        """
        self.switch_camera_fn = switch_camera_fn
        self.cameras = cameras or []
        self.switch_on_phase = switch_on_phase
        self.switch_interval = switch_interval
        
        self._current_idx = 0
        self._steps_since_switch = 0
    
    def set_cameras(self, cameras: List[str]) -> None:
        """Set available cameras."""
        self.cameras = cameras
        self._current_idx = 0
    
    @property
    def current_camera(self) -> Optional[str]:
        """Get current camera name."""
        if not self.cameras:
            return None
        return self.cameras[self._current_idx]
    
    def next_camera(self) -> Optional[str]:
        """Switch to next camera."""
        if not self.cameras:
            return None
        
        self._current_idx = (self._current_idx + 1) % len(self.cameras)
        self._steps_since_switch = 0
        
        cam_name = self.cameras[self._current_idx]
        self.switch_camera_fn(cam_name)
        print(f"[MultiCamera] Switched to: {cam_name}")
        
        return cam_name
    
    def switch_to(self, camera_name: str) -> bool:
        """
        Switch to specific camera.
        
        Args:
            camera_name: Name of camera to switch to
            
        Returns:
            True if successful
        """
        if camera_name not in self.cameras:
            return False
        
        self._current_idx = self.cameras.index(camera_name)
        self._steps_since_switch = 0
        self.switch_camera_fn(camera_name)
        
        return True
    
    def on_step(self) -> Optional[str]:
        """
        Call each step to check for automatic switching.
        
        Returns:
            New camera name if switched, None otherwise
        """
        self._steps_since_switch += 1
        
        if self.switch_interval > 0:
            if self._steps_since_switch >= self.switch_interval:
                return self.next_camera()
        
        return None
    
    def on_phase_change(self, phase_name: str) -> Optional[str]:
        """
        Call when movement phase changes.
        
        Args:
            phase_name: Name of new phase
            
        Returns:
            New camera name if switched, None otherwise
        """
        if self.switch_on_phase and len(self.cameras) > 1:
            return self.next_camera()
        return None
