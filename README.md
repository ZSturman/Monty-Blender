# Blender + Monty Visual Agent

A visual learning agent that interfaces with Blender for 3D scene observation, object detection, and interactive labeling.

## Overview

This project creates a bridge between **Blender 3D rendering** and a **visual learning agent**. The system:

- Renders RGBA, depth, and object index passes from Blender
- Provides an OpenAI Gym-compatible environment for RL workflows
- Detects objects and prompts for human labels
- Supports multiple cameras and movement presets
- Organizes session outputs (images, labels, metadata)

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Visual Agent (Python)                        │
├─────────────────────────────────────────────────────────────────────┤
│  main.py ───► BlenderVisualEnv ───► BlenderRPC ──┐                 │
│      │                                            │  TCP Socket     │
│      ├── CameraController (movement presets)     │  (JSON-RPC)     │
│      ├── NoveltyController (labeling)            │                 │
│      └── OutputManager (session data)            │                 │
└──────────────────────────────────────────────────│─────────────────┘
                                                   │
                                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      Blender Server                                  │
├─────────────────────────────────────────────────────────────────────┤
│  blender_server.py                                                   │
│      │                                                               │
│      ├── Render RGBA, Depth, Object Index                          │
│      ├── Camera control (move, rotate, switch)                      │
│      └── Scene management (frame, reset)                            │
│                                                                      │
│  simulation_scene.blend                                              │
│      └── 3D scene with objects and cameras                          │
└─────────────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Install Dependencies
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Start the Server
```bash
./start_server.sh
```

### 3. Run the Agent
```bash
python -m src.client.main
```

## Directory Structure

```
Blender+Monty/
├── config/
│   └── default.yaml          # Configuration settings
├── src/
│   ├── server/
│   │   └── blender_server.py # Blender RPC server
│   └── client/
│       ├── main.py           # Main entry point
│       ├── blender_rpc.py    # RPC client
│       ├── blender_gym_env.py # Gym environment
│       ├── camera.py         # Camera controller
│       ├── novelty_labeler.py # Labeling system
│       ├── obs_processing.py  # Vision utilities
│       ├── output.py         # Output manager
│       └── config.py         # Config loader
├── tests/
│   ├── unit/                 # Unit tests
│   └── integration/          # Integration tests
├── docs/
│   ├── SCENE_SETUP.md        # Scene configuration guide
│   └── RUNNING.md            # Running instructions
├── blender/
│   └── simulation_scene.blend # Default scene
├── output/                   # Session outputs
├── start_server.sh           # Server start script
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Features

### Stopping Conditions
The agent automatically stops when:
- **Max steps reached** - Configurable limit (default: 500)
- **Stable view detected** - Low motion for N consecutive frames
- **Exploration complete** - All movement phases finished

### Interactive Labeling
- Unknown objects trigger a label prompt after stable observation
- Labels are saved to `labels.json` for persistence
- Timeout auto-labels objects as `unknown_<id>`

### Movement Presets
- **orbit** - Rotate around the scene
- **approach** - Move toward objects
- **overview** - High-angle pan
- **full_exploration** - Combined sequence

### Multi-Camera Support
- Auto-discovers cameras in the scene
- Switch cameras via keyboard (1-9 keys)
- Configure automatic switching per phase

### GPU Rendering
Enable for faster rendering:
```bash
./start_server.sh --gpu
```

## Configuration

Edit `config/default.yaml` or use CLI arguments:

```bash
# Custom settings via CLI
python -m src.client.main \
  --port 5555 \
  --max-steps 200 \
  --preset orbit \
  --gpu
```

Key settings:
- `stopping.max_steps` - Maximum observations
- `stopping.motion_threshold` - Early stop threshold
- `labeling.similarity_threshold` - Label matching sensitivity
- `camera.position_scale` - Movement speed

## Documentation

- [Running Instructions](docs/RUNNING.md) - Complete usage guide
- [Scene Setup](docs/SCENE_SETUP.md) - Blender scene configuration

## Testing

```bash
# Unit tests
pytest tests/unit/ -v

# Integration tests (server must be running)
pytest tests/integration/ -m integration -v
```

## Outputs

Each session creates:
```
output/{session_id}/
├── images/          # Rendered frames
├── labels.json      # Object labels
└── session.json     # Session metadata
```

## Requirements

- Python 3.10+
- Blender 4.0+ (4.2+ recommended)
- numpy, gymnasium, pillow, opencv-python, pyyaml

## License

MIT License
