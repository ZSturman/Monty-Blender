# Running Instructions

Complete guide for running the Blender+Monty Visual Agent.

## Table of Contents
- [Quick Start](#quick-start)
- [Requirements](#requirements)
- [Installation](#installation)
- [Starting the Server](#starting-the-server)
- [Running the Agent](#running-the-agent)
- [CLI Reference](#cli-reference)
- [Configuration](#configuration)
- [Keyboard Controls](#keyboard-controls)
- [Troubleshooting](#troubleshooting)

---

## Quick Start

```bash
# Terminal 1: Start the Blender server
./start_server.sh

# Terminal 2: Run the agent
python -m src.client.main
```

---

## Requirements

### System
- macOS 10.15+, Linux, or Windows 10+
- Python 3.10+
- Blender 4.0+ (4.2+ recommended)

### Python Dependencies
```
numpy
gymnasium
pillow
opencv-python
pyyaml
```

### Optional
- OpenImageIO (bundled with Blender, improves EXR reading)
- pytest (for running tests)

---

## Installation

### 1. Clone/Download Project
```bash
cd /path/to/project
```

### 2. Create Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# or: .venv\Scripts\activate  # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Verify Blender Installation
```bash
# macOS
/Applications/Blender.app/Contents/MacOS/Blender --version

# Linux
blender --version
```

---

## Starting the Server

### Basic Start
```bash
./start_server.sh
```

### With Options
```bash
# Custom port
./start_server.sh --port 5556

# GPU rendering
./start_server.sh --gpu

# Higher resolution
./start_server.sh --width 640 --height 480

# Custom scene
./start_server.sh --scene path/to/scene.blend

# Combined
./start_server.sh --port 5555 --gpu --width 640 --height 480
```

### Server Options
| Option | Description | Default |
|--------|-------------|---------|
| `--port PORT` | Server port | 5555 |
| `--gpu` | Enable GPU rendering | disabled |
| `--width WIDTH` | Render width | 320 |
| `--height HEIGHT` | Render height | 240 |
| `--camera NAME` | Camera name | Camera |
| `--scene FILE` | Blender scene | blender/simulation_scene.blend |

### Expected Output
```
==============================================
Blender+Monty Server
==============================================

Scene:      blender/simulation_scene.blend
Port:       5555
Resolution: 320x240
Camera:     Camera
GPU:        disabled

Starting server...

[BlenderServer] Loaded: src/server/blender_server.py
[BlenderServer] VERSION: v3.0.0-2026-02-01
[BlenderServer] Assigned indices to 3 mesh objects:
  [1] Cube
  [2] Sphere  
  [3] Plane
[BlenderServer] Scene configured: 320x240, GPU=False
[BlenderServer] Listening on 127.0.0.1:5555
```

---

## Running the Agent

### Basic Run
```bash
python -m src.client.main
```

### With Options
```bash
# Custom port (must match server)
python -m src.client.main --port 5555

# Limit observations
python -m src.client.main --max-steps 100

# Use specific config
python -m src.client.main --config config/my_config.yaml

# Movement preset
python -m src.client.main --preset orbit

# Headless mode (no display)
python -m src.client.main --no-display

# Combined
python -m src.client.main --port 5555 --max-steps 200 --preset approach
```

---

## CLI Reference

### Agent Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--config FILE` | `-c` | Config file path | config/default.yaml |
| `--port PORT` | `-p` | Server port | 5555 |
| `--host HOST` | | Server host | 127.0.0.1 |
| `--max-steps N` | `-n` | Max observations | 500 |
| `--output-dir DIR` | `-o` | Output directory | output |
| `--gpu` | | Enable GPU rendering | disabled |
| `--width WIDTH` | | Render width | 320 |
| `--height HEIGHT` | | Render height | 240 |
| `--camera NAME` | | Camera name | Camera |
| `--preset NAME` | | Movement preset | full_exploration |
| `--no-display` | | Headless mode | false |
| `--labels-file FILE` | | Labels JSON path | (session dir) |
| `--verbose` | `-v` | Verbose output | false |

### Movement Presets

| Preset | Description |
|--------|-------------|
| `orbit` | Rotate around scene center |
| `approach` | Move toward mesh center |
| `overview` | High-angle pan |
| `full_exploration` | Combined sequence: approach → orbit → overview |

---

## Configuration

### Config File Location
Default: `config/default.yaml`

### Key Settings

```yaml
# Network
network:
  host: "127.0.0.1"
  port: 5555

# Render
render:
  width: 320
  height: 240
  gpu: false

# Stopping conditions
stopping:
  max_steps: 500           # Stop after N steps
  motion_threshold: 0.001  # Stop when motion below this
  stable_frames: 30        # Consecutive low-motion frames
  min_steps: 50           # Minimum steps before early stop

# Labeling
labeling:
  similarity_threshold: 0.92  # Match threshold
  stable_steps: 10           # Frames before prompting
  prompt_timeout: 30         # Seconds to wait for input

# Output
output:
  base_dir: "output"
  save_images: true
  save_depth: false
```

### Override Precedence
1. CLI arguments (highest priority)
2. Config file
3. Defaults (lowest priority)

---

## Keyboard Controls

During agent execution:

| Key | Action |
|-----|--------|
| `q` | Quit |
| `1-9` | Switch to camera N |
| `l` | Force label prompt for unknown objects |

---

## Output Structure

Each session creates a directory:
```
output/
  20260201_143052/           # Session ID (timestamp)
    images/
      Camera_0001.png
      Camera_0002.png
      ...
    labels.json              # Object labels
    session.json             # Session metadata
```

### Session Metadata
```json
{
  "session_id": "20260201_143052",
  "started_at": "2026-02-01T14:30:52",
  "ended_at": "2026-02-01T14:35:12",
  "total_steps": 245,
  "cameras_used": ["Camera", "TopCam"],
  "objects_detected": 3,
  "labels_assigned": 2,
  "stop_reason": "stable_view"
}
```

---

## Running Tests

### Unit Tests
```bash
# All unit tests
pytest tests/unit/ -v

# Specific test file
pytest tests/unit/test_labeler.py -v
```

### Integration Tests (requires server)
```bash
# Start server first, then:
pytest tests/integration/ -m integration -v
```

---

## Troubleshooting

### "Connection refused"
**Cause:** Server not running or wrong port
**Fix:**
1. Start server: `./start_server.sh`
2. Verify port matches: `--port 5555`

### "Camera not found"
**Cause:** Camera name doesn't match scene
**Fix:**
1. Open `.blend` file in Blender
2. Check exact camera name
3. Use `--camera "ExactName"`

### "No module named 'src'"
**Cause:** Running from wrong directory
**Fix:**
```bash
cd /path/to/Blender+Monty
python -m src.client.main
```

### Black/empty renders
**Cause:** Scene lighting or camera issue
**Fix:**
1. Open scene in Blender
2. Add lighting if missing
3. Verify camera isn't inside an object

### Slow rendering
**Cause:** CPU rendering
**Fix:**
1. Enable GPU: `./start_server.sh --gpu`
2. Ensure GPU drivers are installed

### Display window not showing
**Cause:** OpenCV issue or headless mode
**Fix:**
1. Install OpenCV: `pip install opencv-python`
2. Don't use `--no-display` flag
3. On remote servers, use X forwarding or VNC

### Labels not persisting
**Cause:** Session-specific labels
**Fix:**
1. Use `--labels-file labels.json` to use a fixed path
2. Or copy labels from session output
