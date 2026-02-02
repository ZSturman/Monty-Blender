# Blender + Monty Integration - FIXED for Blender 5.0.1

## What Was Fixed

The original code had issues with Blender 5.0.1's compositor API changes in background mode:
1. **Render Result passes not populated** in background mode
2. **Compositor node API instability** when using `scene.compositing_node_group`
3. **Multi-layer EXR format not available** in Blender 5.0.1

## Solution

Created `monty_blender_server_v2.py` which:
- Renders **3 times** with different compositor setups (RGB, Depth, Index)
- Uses PNG for RGB (reliable) and EXR for depth/index passes
- Sets up minimal compositor nodes for each pass
- Has proper error handling and fallbacks

## How to Run

### Terminal 1: Start Server
```bash
cd /Users/zacharysturman/Downloads/PORTFOLIO/_Tests/Blender+Monty

/Applications/Blender.app/Contents/MacOS/Blender \
  -b blender/simulation_scene.blend \
  --python blender/monty_blender_server_v2.py -- \
  --host 127.0.0.1 --port 5555 --width 320 --height 240
```

Wait for this message:
```
[BlenderServer] Listening on 127.0.0.1:5555
```

### Terminal 2: Test Connection
```bash
cd /Users/zacharysturman/Downloads/PORTFOLIO/_Tests/Blender+Monty
source .venv/bin/activate
python test_simple.py
```

If that works, try the full demo:
```bash
python python/run_demo.py
```

## Expected Output

**Server** should show:
```
[BlenderServer] Loaded script: ...
[BlenderServer] VERSION: v2-simplified-2026-02-01
[BlenderServer] OpenImageIO available
[BlenderServer] Assigned indices to 1 mesh objects
[BlenderServer] Scene configured: 320x240, passes enabled
[BlenderServer] Listening on 127.0.0.1:5555
[BlenderServer] Client connected: ('127.0.0.1', XXXXX)
[BlenderServer] Rendering RGB to: /tmp/monty_blender_out/rgb_000001.png
[BlenderServer] RGB loaded: 320x240, 307200 floats
[BlenderServer] Rendering depth to: /tmp/monty_blender_out/depth_000001.exr
[BlenderServer] Depth loaded: 76800 floats
[BlenderServer] Rendering index to: /tmp/monty_blender_out/index_000001.exr
[BlenderServer] Index loaded: 76800 floats
```

**Client** should show:
```
Connecting to 127.0.0.1:5555...
✓ Connected
Sending render command...
✓ Command sent
Waiting for response...
✓ Render successful!
  Width: 320
  Height: 240
  RGBA PNG size: XXXX bytes (base64)
  Depth data size: XXXX bytes (base64)
  Index data size: XXXX bytes (base64)
✓ All tests passed!
```

## Notes

- **Camera name**: Changed from "MontyCam" to "Camera" (the actual camera in simulation_scene.blend)
- **Performance**: Rendering 3 times is slower but more reliable than compositor nodes in Blender 5.0.1 background mode
- **OpenImageIO**: Available in Blender 5.0.1, used for reading EXR files
- **Fallbacks**: If depth/index rendering fails, uses dummy data instead of crashing

## Troubleshooting

If you see errors about missing passes:
- The compositor might not be setting up correctly
- Check that depth/index EXR files are created in `/tmp/monty_blender_out/`
- Depth and Index may show as "dummy data" warnings but RGB should always work

If the server crashes on startup:
- Make sure no other process is using port 5555
- Check that Camera object exists in simulation_scene.blend

## Original vs New

**Original** (`monty_blender_server.py`):
- Tried to use compositor nodes + File Output
- Failed due to Blender 5.0.1 API changes

**New** (`monty_blender_server_v2.py`):
- Renders multiple times with different setups
- More reliable but slightly slower
- Works with Blender 5.0.1 in background mode
