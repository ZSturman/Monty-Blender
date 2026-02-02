# src/client/blender_rpc.py
"""
Blender RPC Client for communicating with Blender server.

Provides TCP socket communication with the Blender server, handling
rendering requests, camera control, and data decoding.
"""

import base64
import io
import json
import socket
import struct
import zlib
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from PIL import Image


class BlenderRPC:
    """
    RPC client for Blender server communication.
    
    Handles:
        - TCP socket connection
        - JSON-RPC message encoding/decoding
        - Image data decoding (PNG, zlib-compressed arrays)
        - Camera control commands
    """
    
    def __init__(self, host: str = "127.0.0.1", port: int = 5555, 
                 timeout: float = 30.0, position_scale: float = 1.0,
                 rotation_scale: float = 1.0):
        """
        Initialize connection to Blender server.
        
        Args:
            host: Server hostname
            port: Server port
            timeout: Socket timeout in seconds
            position_scale: Default scale for position movements
            rotation_scale: Default scale for rotation movements
        """
        self.host = host
        self.port = port
        self.timeout = timeout
        self.position_scale = position_scale
        self.rotation_scale = rotation_scale
        
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(timeout)
        
        try:
            self.sock.connect((host, port))
            print(f"[BlenderRPC] Connected to {host}:{port}")
        except ConnectionRefusedError:
            raise ConnectionError(
                f"Could not connect to Blender server at {host}:{port}. "
                "Make sure the server is running."
            )
    
    def close(self) -> None:
        """Close connection to server."""
        try:
            self._send({"cmd": "close"})
            self._recv()
        except Exception:
            pass
        finally:
            try:
                self.sock.close()
            except Exception:
                pass
    
    def reset(self, frame: int = 1) -> Dict[str, Any]:
        """
        Reset scene to specified frame.
        
        Args:
            frame: Frame number to reset to
            
        Returns:
            Response dict with 'ok' and 'frame' keys
        """
        self._send({"cmd": "reset", "frame": int(frame)})
        return self._recv()
    
    def step(self, n: int = 1, action: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Advance simulation by n frames and optionally apply action.
        
        Args:
            n: Number of frames to advance
            action: Camera action dict with 'dpos' and 'drot_euler' keys
            
        Returns:
            Response dict with 'ok' and 'frame' keys
        """
        msg = {
            "cmd": "step", 
            "n": int(n), 
            "action": action,
            "position_scale": self.position_scale,
            "rotation_scale": self.rotation_scale,
        }
        self._send(msg)
        return self._recv()
    
    def state(self) -> Dict[str, Any]:
        """
        Get current scene state.
        
        Returns:
            Response dict with 'frame' and 'camera' info
        """
        self._send({"cmd": "state"})
        return self._recv()
    
    def list_cameras(self) -> List[Dict[str, Any]]:
        """
        Get list of all cameras in scene.
        
        Returns:
            List of camera dicts with name, type, lens info
        """
        self._send({"cmd": "list_cameras"})
        resp = self._recv()
        if not resp.get("ok"):
            raise RuntimeError(resp.get("error", "Unknown error"))
        return resp.get("cameras", [])
    
    def switch_camera(self, camera_name: str) -> bool:
        """
        Switch active camera.
        
        Args:
            camera_name: Name of camera to switch to
            
        Returns:
            True if successful
        """
        self._send({"cmd": "switch_camera", "camera": camera_name})
        resp = self._recv()
        if not resp.get("ok"):
            raise RuntimeError(resp.get("error", f"Failed to switch to camera: {camera_name}"))
        return True
    
    def get_current_camera(self) -> str:
        """Get name of current active camera."""
        state = self.state()
        return state.get("camera", {}).get("name", "Unknown")
    
    def render(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Request render from server.
        
        Returns:
            Tuple of (rgba, depth, index) numpy arrays:
                - rgba: (H, W, 4) uint8 array
                - depth: (H, W) float32 array
                - index: (H, W) int32 array
        """
        self._send({"cmd": "render"})
        resp = self._recv()
        
        if not resp.get("ok"):
            raise RuntimeError(resp.get("error", "Render failed"))
        
        w = resp["width"]
        h = resp["height"]
        
        # Decode RGBA PNG
        rgba_png = base64.b64decode(resp["rgba_png_b64"])
        img = Image.open(io.BytesIO(rgba_png)).convert("RGBA")
        rgba = np.array(img, dtype=np.uint8)  # (H, W, 4)
        
        # Decode depth float32
        depth_blob = base64.b64decode(resp["depth_f32_zlib_b64"])
        depth = self._decode_float32_array(depth_blob, h, w)
        
        # Decode index int32
        idx_blob = base64.b64decode(resp["index_i32_zlib_b64"])
        index = self._decode_int32_array(idx_blob, h, w)
        
        return rgba, depth, index
    
    def _decode_float32_array(self, blob: bytes, h: int, w: int) -> np.ndarray:
        """Decode zlib-compressed float32 array."""
        raw = zlib.decompress(blob)
        arr = np.frombuffer(raw, dtype=np.float32)
        
        if arr.size == h * w:
            return arr.reshape(h, w)
        elif arr.size == h * w * 4:
            return arr.reshape(h, w, 4)[:, :, 0]
        else:
            raise ValueError(
                f"Unexpected buffer size {arr.size} for (h,w)=({h},{w}). "
                f"Expected {h*w} or {h*w*4}."
            )
    
    def _decode_int32_array(self, blob: bytes, h: int, w: int) -> np.ndarray:
        """Decode zlib-compressed int32 array."""
        raw = zlib.decompress(blob)
        arr = np.frombuffer(raw, dtype=np.int32)
        
        if arr.size == h * w:
            return arr.reshape(h, w)
        elif arr.size == h * w * 4:
            return arr.reshape(h, w, 4)[:, :, 0]
        else:
            raise ValueError(
                f"Unexpected buffer size {arr.size} for (h,w)=({h},{w}). "
                f"Expected {h*w} or {h*w*4}."
            )
    
    def _send(self, obj: Dict) -> None:
        """Send length-prefixed JSON message."""
        payload = json.dumps(obj).encode("utf-8")
        self.sock.sendall(struct.pack("!I", len(payload)) + payload)
    
    def _recv(self) -> Dict:
        """Receive length-prefixed JSON message."""
        header = self._recv_exact(4)
        (n,) = struct.unpack("!I", header)
        payload = self._recv_exact(n)
        return json.loads(payload.decode("utf-8"))
    
    def _recv_exact(self, n: int) -> bytes:
        """Receive exactly n bytes."""
        buf = b""
        while len(buf) < n:
            chunk = self.sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("Socket closed")
            buf += chunk
        return buf
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
