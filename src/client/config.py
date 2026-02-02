# src/client/config.py
"""
Configuration management for Blender+Monty Visual Agent.

Loads settings from YAML config files with CLI override support.
"""

import os
import yaml
import argparse
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class NetworkConfig:
    host: str = "127.0.0.1"
    port: int = 5555


@dataclass
class RenderConfig:
    width: int = 320
    height: int = 240
    engine: str = "CYCLES"
    samples: int = 1
    gpu: bool = False
    gpu_device_type: str = "METAL"


@dataclass
class CameraConfig:
    default: str = "Camera"
    position_scale: float = 0.05
    rotation_scale: float = 0.02


@dataclass
class StoppingConfig:
    max_steps: int = 500
    motion_threshold: float = 0.001
    stable_frames: int = 30
    min_steps: int = 50


@dataclass
class LabelingConfig:
    similarity_threshold: float = 0.92
    min_pixels: int = 200
    stable_steps: int = 10
    prompt_timeout: int = 30
    auto_label_prefix: str = "unknown"


@dataclass
class OutputConfig:
    base_dir: str = "output"
    save_images: bool = True
    save_depth: bool = False
    save_metadata: bool = True
    image_format: str = "png"


@dataclass
class DisplayConfig:
    show_window: bool = True
    window_scale: float = 1.0
    show_motion_energy: bool = True
    show_depth: bool = False
    show_labels: bool = True
    show_status_overlay: bool = True
    fps_limit: int = 30


@dataclass
class Config:
    """Main configuration container."""
    network: NetworkConfig = field(default_factory=NetworkConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    stopping: StoppingConfig = field(default_factory=StoppingConfig)
    labeling: LabelingConfig = field(default_factory=LabelingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)
    movement_presets: Dict[str, Any] = field(default_factory=dict)
    cameras: List[Dict[str, str]] = field(default_factory=list)
    
    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        config = cls()
        
        if 'network' in data:
            config.network = NetworkConfig(**data['network'])
        if 'render' in data:
            config.render = RenderConfig(**data['render'])
        if 'camera' in data:
            config.camera = CameraConfig(**data['camera'])
        if 'stopping' in data:
            config.stopping = StoppingConfig(**data['stopping'])
        if 'labeling' in data:
            config.labeling = LabelingConfig(**data['labeling'])
        if 'output' in data:
            config.output = OutputConfig(**data['output'])
        if 'display' in data:
            config.display = DisplayConfig(**data['display'])
        if 'movement_presets' in data:
            config.movement_presets = data['movement_presets']
        if 'cameras' in data and data['cameras']:
            config.cameras = data['cameras']
        
        return config
    
    def apply_cli_overrides(self, args: argparse.Namespace) -> None:
        """Apply command-line argument overrides."""
        if hasattr(args, 'port') and args.port is not None:
            self.network.port = args.port
        if hasattr(args, 'host') and args.host is not None:
            self.network.host = args.host
        if hasattr(args, 'max_steps') and args.max_steps is not None:
            self.stopping.max_steps = args.max_steps
        if hasattr(args, 'output_dir') and args.output_dir is not None:
            self.output.base_dir = args.output_dir
        if hasattr(args, 'gpu') and args.gpu:
            self.render.gpu = True
        if hasattr(args, 'width') and args.width is not None:
            self.render.width = args.width
        if hasattr(args, 'height') and args.height is not None:
            self.render.height = args.height
        if hasattr(args, 'camera') and args.camera is not None:
            self.camera.default = args.camera
        if hasattr(args, 'no_display') and args.no_display:
            self.display.show_window = False


def get_default_config_path() -> str:
    """Get path to default config file."""
    # Look in several locations
    locations = [
        os.path.join(os.getcwd(), 'config', 'default.yaml'),
        os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'default.yaml'),
        os.path.expanduser('~/.config/blender-monty/config.yaml'),
    ]
    
    for path in locations:
        if os.path.exists(path):
            return os.path.abspath(path)
    
    return locations[0]  # Return first as default


def create_argument_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Blender+Monty Visual Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings
  python -m src.client.main
  
  # Run with custom port and max steps
  python -m src.client.main --port 5556 --max-steps 200
  
  # Run with GPU rendering
  python -m src.client.main --gpu
  
  # Run with custom config
  python -m src.client.main --config my_config.yaml
  
  # Run headless (no display window)
  python -m src.client.main --no-display
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default=None,
        help='Path to YAML configuration file'
    )
    
    parser.add_argument(
        '--port', '-p',
        type=int,
        default=None,
        help='Server port (overrides config)'
    )
    
    parser.add_argument(
        '--host',
        type=str,
        default=None,
        help='Server host (overrides config)'
    )
    
    parser.add_argument(
        '--max-steps', '-n',
        type=int,
        default=None,
        help='Maximum observation steps (overrides config)'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default=None,
        help='Output directory for session data'
    )
    
    parser.add_argument(
        '--gpu',
        action='store_true',
        help='Enable GPU rendering'
    )
    
    parser.add_argument(
        '--width',
        type=int,
        default=None,
        help='Render width'
    )
    
    parser.add_argument(
        '--height',
        type=int,
        default=None,
        help='Render height'
    )
    
    parser.add_argument(
        '--camera',
        type=str,
        default=None,
        help='Camera name to use'
    )
    
    parser.add_argument(
        '--preset',
        type=str,
        default='full_exploration',
        help='Movement preset name (orbit, approach, overview, full_exploration)'
    )
    
    parser.add_argument(
        '--no-display',
        action='store_true',
        help='Run without display window (headless)'
    )
    
    parser.add_argument(
        '--labels-file',
        type=str,
        default=None,
        help='Path to labels.json file'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    return parser


def load_config(args: Optional[argparse.Namespace] = None) -> Config:
    """Load configuration from file and apply CLI overrides."""
    if args is None:
        parser = create_argument_parser()
        args = parser.parse_args()
    
    # Determine config path
    config_path = args.config if hasattr(args, 'config') and args.config else get_default_config_path()
    
    # Load config
    if os.path.exists(config_path):
        config = Config.from_yaml(config_path)
        print(f"[Config] Loaded from: {config_path}")
    else:
        config = Config()
        print(f"[Config] Using defaults (no config file found)")
    
    # Apply CLI overrides
    config.apply_cli_overrides(args)
    
    return config
