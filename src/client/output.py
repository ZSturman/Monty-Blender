# src/client/output.py
"""
Output management for session data.

Handles organized saving of:
    - Session metadata
    - Rendered images
    - Depth maps
    - Labels
"""

import json
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional, Any

import numpy as np
from PIL import Image


@dataclass
class SessionMetadata:
    """Metadata for a session."""
    session_id: str
    started_at: str
    config: Dict[str, Any] = field(default_factory=dict)
    total_steps: int = 0
    cameras_used: List[str] = field(default_factory=list)
    objects_detected: int = 0
    labels_assigned: int = 0
    stop_reason: Optional[str] = None
    ended_at: Optional[str] = None


class OutputManager:
    """
    Manages organized output for observation sessions.
    
    Directory structure:
        output/
            {session_id}/
                images/
                    frame_0001.png
                    frame_0002.png
                    ...
                depth/
                    frame_0001.npy
                    ...
                labels.json
                session.json
    """
    
    def __init__(
        self,
        base_dir: str = "output",
        session_id: Optional[str] = None,
        save_images: bool = True,
        save_depth: bool = False,
        save_metadata: bool = True,
        image_format: str = "png",
    ):
        """
        Initialize output manager.
        
        Args:
            base_dir: Base output directory
            session_id: Session identifier (auto-generated if None)
            save_images: Whether to save RGB images
            save_depth: Whether to save depth maps
            save_metadata: Whether to save session metadata
            image_format: Image format ('png' or 'jpg')
        """
        self.base_dir = base_dir
        self.save_images = save_images
        self.save_depth = save_depth
        self.save_metadata = save_metadata
        self.image_format = image_format
        
        # Generate session ID
        if session_id is None:
            session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_id = session_id
        
        # Create directories
        self.session_dir = os.path.join(base_dir, session_id)
        self.images_dir = os.path.join(self.session_dir, "images")
        self.depth_dir = os.path.join(self.session_dir, "depth")
        
        os.makedirs(self.session_dir, exist_ok=True)
        if save_images:
            os.makedirs(self.images_dir, exist_ok=True)
        if save_depth:
            os.makedirs(self.depth_dir, exist_ok=True)
        
        # Initialize metadata
        self.metadata = SessionMetadata(
            session_id=session_id,
            started_at=datetime.now().isoformat(),
        )
        
        # Frame counter
        self._frame_count = 0
        
        print(f"[OutputManager] Session: {self.session_dir}")
    
    @property
    def labels_path(self) -> str:
        """Get path to labels.json."""
        return os.path.join(self.session_dir, "labels.json")
    
    @property
    def metadata_path(self) -> str:
        """Get path to session.json."""
        return os.path.join(self.session_dir, "session.json")
    
    def set_config(self, config: Dict[str, Any]) -> None:
        """Store config in metadata."""
        self.metadata.config = config
    
    def save_frame(
        self,
        rgba: np.ndarray,
        depth: Optional[np.ndarray] = None,
        index: Optional[np.ndarray] = None,
        labels: Optional[Dict[int, str]] = None,
        camera_name: Optional[str] = None,
        step: Optional[int] = None,
    ) -> Dict[str, str]:
        """
        Save frame data.
        
        Args:
            rgba: RGBA image (H, W, 4) uint8
            depth: Optional depth map (H, W) float32
            index: Optional object index map (not saved, for reference)
            labels: Optional current labels dict
            camera_name: Name of camera used
            step: Step number (auto-incremented if None)
            
        Returns:
            Dict with saved file paths
        """
        if step is None:
            step = self._frame_count
        self._frame_count = max(self._frame_count, step + 1)
        
        saved = {}
        
        # Save RGB image
        if self.save_images:
            filename = f"frame_{step:04d}.{self.image_format}"
            if camera_name:
                filename = f"{camera_name}_{step:04d}.{self.image_format}"
            
            path = os.path.join(self.images_dir, filename)
            img = Image.fromarray(rgba[:, :, :3])  # RGB only
            img.save(path)
            saved["image"] = path
        
        # Save depth
        if self.save_depth and depth is not None:
            filename = f"frame_{step:04d}.npy"
            if camera_name:
                filename = f"{camera_name}_{step:04d}.npy"
            
            path = os.path.join(self.depth_dir, filename)
            np.save(path, depth)
            saved["depth"] = path
        
        # Update metadata
        self.metadata.total_steps = max(self.metadata.total_steps, step + 1)
        if camera_name and camera_name not in self.metadata.cameras_used:
            self.metadata.cameras_used.append(camera_name)
        
        return saved
    
    def save_labels(self, labels_data: Dict) -> str:
        """
        Save labels to labels.json.
        
        Args:
            labels_data: Labels dict to save
            
        Returns:
            Path to saved file
        """
        with open(self.labels_path, "w") as f:
            json.dump(labels_data, f, indent=2)
        
        # Update metadata
        self.metadata.labels_assigned = len(labels_data.get("objects", []))
        
        return self.labels_path
    
    def save_session_metadata(
        self,
        stop_reason: Optional[str] = None,
        objects_detected: int = 0,
    ) -> str:
        """
        Save session metadata.
        
        Args:
            stop_reason: Why session ended
            objects_detected: Number of objects detected
            
        Returns:
            Path to saved file
        """
        self.metadata.ended_at = datetime.now().isoformat()
        self.metadata.stop_reason = stop_reason
        self.metadata.objects_detected = objects_detected
        
        with open(self.metadata_path, "w") as f:
            json.dump(asdict(self.metadata), f, indent=2)
        
        return self.metadata_path
    
    def get_frame_paths(self) -> List[str]:
        """Get list of all saved frame paths."""
        if not self.save_images:
            return []
        
        files = os.listdir(self.images_dir)
        files = [f for f in files if f.endswith(f".{self.image_format}")]
        files.sort()
        
        return [os.path.join(self.images_dir, f) for f in files]
    
    def get_summary(self) -> Dict[str, Any]:
        """Get session summary."""
        return {
            "session_id": self.session_id,
            "session_dir": self.session_dir,
            "total_frames": self._frame_count,
            "cameras_used": self.metadata.cameras_used,
            "labels_assigned": self.metadata.labels_assigned,
        }
    
    def finalize(
        self,
        stop_reason: str = "completed",
        objects_detected: int = 0,
        labels_data: Optional[Dict] = None,
    ) -> None:
        """
        Finalize session and save all metadata.
        
        Args:
            stop_reason: Why session ended
            objects_detected: Number of objects detected
            labels_data: Optional labels to save
        """
        if labels_data:
            self.save_labels(labels_data)
        
        if self.save_metadata:
            self.save_session_metadata(stop_reason, objects_detected)
        
        print(f"[OutputManager] Session finalized: {self.session_id}")
        print(f"[OutputManager] Total frames: {self._frame_count}")
        print(f"[OutputManager] Output: {self.session_dir}")
