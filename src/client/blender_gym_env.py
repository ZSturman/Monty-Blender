# src/client/blender_gym_env.py
"""
OpenAI Gymnasium environment for Blender visual learning.

Wraps BlenderRPC to provide a standard Gym interface with
observation/action spaces, stepping, and termination conditions.
"""

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np

from .blender_rpc import BlenderRPC


class BlenderVisualEnv(gym.Env):
    """
    Gymnasium environment for Blender visual observations.
    
    Observation Space:
        Dict with:
            - rgba: (H, W, 4) uint8 array
            - depth: (H, W) float32 array  
            - index: (H, W) int32 array (object IDs)
    
    Action Space:
        Box(6,) float32 in [-1, 1]:
            [dx, dy, dz, droll, dpitch, dyaw]
        
        Actions are scaled by position_scale and rotation_scale
        before being applied to the camera.
    
    Termination:
        - max_steps reached
        - Motion energy below threshold for stable_frames consecutive frames
    """
    
    metadata = {"render_modes": ["human", "rgb_array"]}
    
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 5555,
        step_frames: int = 1,
        position_scale: float = 0.05,
        rotation_scale: float = 0.02,
        max_steps: int = 500,
        motion_threshold: float = 0.001,
        stable_frames: int = 30,
        min_steps: int = 50,
    ):
        """
        Initialize Blender visual environment.
        
        Args:
            host: Blender server host
            port: Blender server port
            step_frames: Frames to advance per step
            position_scale: Scale factor for position actions
            rotation_scale: Scale factor for rotation actions
            max_steps: Maximum steps before truncation
            motion_threshold: Motion energy threshold for early stopping
            stable_frames: Consecutive low-motion frames to trigger stop
            min_steps: Minimum steps before allowing early stop
        """
        super().__init__()
        
        self.host = host
        self.port = port
        self.step_frames = step_frames
        self.position_scale = position_scale
        self.rotation_scale = rotation_scale
        self.max_steps = max_steps
        self.motion_threshold = motion_threshold
        self.stable_frames = stable_frames
        self.min_steps = min_steps
        
        # Connect to server
        self.rpc = BlenderRPC(
            host=host, 
            port=port,
            position_scale=position_scale,
            rotation_scale=rotation_scale
        )
        
        # Get initial observation to determine shape
        rgba, depth, idx = self.rpc.render()
        h, w, _ = rgba.shape
        self._height = h
        self._width = w
        
        # Define spaces
        self.observation_space = gym.spaces.Dict({
            "rgba": gym.spaces.Box(0, 255, shape=(h, w, 4), dtype=np.uint8),
            "depth": gym.spaces.Box(0.0, np.inf, shape=(h, w), dtype=np.float32),
            "index": gym.spaces.Box(0, np.iinfo(np.int32).max, shape=(h, w), dtype=np.int32),
        })
        
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(6,), dtype=np.float32
        )
        
        # State tracking
        self._step_count = 0
        self._low_motion_count = 0
        self._prev_rgba = None
        self._motion_history = []
    
    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[Dict] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Reset environment to initial state.
        
        Args:
            seed: Random seed (unused, environment is deterministic)
            options: Optional dict with 'frame' key to reset to specific frame
            
        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed=seed)
        
        frame = 1
        if options and "frame" in options:
            frame = options["frame"]
        
        self.rpc.reset(frame=frame)
        
        # Reset state tracking
        self._step_count = 0
        self._low_motion_count = 0
        self._motion_history = []
        
        obs = self._get_obs()
        self._prev_rgba = obs["rgba"].copy()
        
        state = self.rpc.state()
        info = {
            "frame": state["frame"],
            "camera": state.get("camera", {}),
            "step": 0,
        }
        
        return obs, info
    
    def step(
        self, 
        action: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        Take action in environment.
        
        Args:
            action: 6D action array [dx, dy, dz, droll, dpitch, dyaw]
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        action = np.asarray(action, dtype=np.float32)
        
        # Build action dict
        act = {
            "type": "camera_delta",
            "dpos": action[:3].tolist(),
            "drot_euler": action[3:].tolist(),
        }
        
        # Step simulation
        self.rpc.step(n=self.step_frames, action=act)
        self._step_count += 1
        
        # Get observation
        obs = self._get_obs()
        
        # Calculate motion energy
        motion = self._compute_motion_energy(obs["rgba"])
        self._motion_history.append(motion)
        self._prev_rgba = obs["rgba"].copy()
        
        # Check termination conditions
        terminated = False
        truncated = False
        stop_reason = None
        
        # Max steps reached
        if self._step_count >= self.max_steps:
            truncated = True
            stop_reason = "max_steps"
        
        # Low motion detection (after min_steps)
        if motion < self.motion_threshold:
            self._low_motion_count += 1
        else:
            self._low_motion_count = 0
        
        if (self._step_count >= self.min_steps and 
            self._low_motion_count >= self.stable_frames):
            terminated = True
            stop_reason = "stable_view"
        
        # Build info
        state = self.rpc.state()
        info = {
            "frame": state["frame"],
            "camera": state.get("camera", {}),
            "step": self._step_count,
            "motion_energy": motion,
            "low_motion_count": self._low_motion_count,
            "stop_reason": stop_reason,
        }
        
        # Reward is 0 (this is primarily for observation, not RL)
        reward = 0.0
        
        return obs, reward, terminated, truncated, info
    
    def _get_obs(self) -> Dict[str, np.ndarray]:
        """Get current observation from renderer."""
        rgba, depth, idx = self.rpc.render()
        return {"rgba": rgba, "depth": depth, "index": idx}
    
    def _compute_motion_energy(self, rgba: np.ndarray) -> float:
        """
        Compute motion energy between current and previous frame.
        
        Returns normalized mean absolute difference.
        """
        if self._prev_rgba is None:
            return 1.0
        
        # Convert to grayscale and compute difference
        prev_gray = np.mean(self._prev_rgba[:, :, :3], axis=2)
        curr_gray = np.mean(rgba[:, :, :3], axis=2)
        
        diff = np.abs(curr_gray.astype(np.float32) - prev_gray.astype(np.float32))
        motion = np.mean(diff) / 255.0
        
        return float(motion)
    
    def get_motion_history(self) -> list:
        """Get history of motion energy values."""
        return self._motion_history.copy()
    
    def get_available_cameras(self) -> list:
        """Get list of available cameras in scene."""
        return self.rpc.list_cameras()
    
    def switch_camera(self, camera_name: str) -> bool:
        """Switch to specified camera."""
        return self.rpc.switch_camera(camera_name)
    
    def get_current_camera(self) -> str:
        """Get name of current camera."""
        return self.rpc.get_current_camera()
    
    def close(self) -> None:
        """Close environment and connection."""
        try:
            self.rpc.close()
        except Exception:
            pass
    
    def render(self) -> Optional[np.ndarray]:
        """
        Render current frame.
        
        Returns:
            RGB array if render_mode is 'rgb_array', None otherwise
        """
        obs = self._get_obs()
        return obs["rgba"][:, :, :3]  # Return RGB only
