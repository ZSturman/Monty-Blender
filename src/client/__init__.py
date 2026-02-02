# src/client/__init__.py
"""
Blender+Monty Visual Agent Client

Components:
    - BlenderRPC: TCP/RPC communication with Blender server
    - BlenderVisualEnv: OpenAI Gym environment wrapper
    - CameraController: Movement presets and phase cycling
    - NoveltyController: Interactive labeling system
    - OutputManager: Session data organization
"""

from .blender_rpc import BlenderRPC
from .blender_gym_env import BlenderVisualEnv
from .camera import CameraController, MultiCameraController
from .novelty_labeler import LabelMemory, NoveltyController
from .obs_processing import (
    motion_energy,
    motion_energy_scalar,
    object_signatures,
    cosine_sim,
    get_object_centroids,
    count_objects,
)
from .output import OutputManager
from .config import Config, load_config, create_argument_parser

__all__ = [
    "BlenderRPC",
    "BlenderVisualEnv",
    "CameraController",
    "MultiCameraController",
    "LabelMemory",
    "NoveltyController",
    "OutputManager",
    "Config",
    "load_config",
    "create_argument_parser",
    "motion_energy",
    "motion_energy_scalar",
    "object_signatures",
    "cosine_sim",
    "get_object_centroids",
    "count_objects",
]
