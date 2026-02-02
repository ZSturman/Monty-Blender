# Scene Setup Guide

This guide explains how to set up your Blender scene for use with the Monty Visual Agent.

## Table of Contents
- [Adding/Removing Objects](#addingremoving-objects)
- [Object Pass Indices](#object-pass-indices)
- [Camera Setup](#camera-setup)
- [Multiple Cameras](#multiple-cameras)
- [Lighting](#lighting)
- [Best Practices](#best-practices)

---

## Adding/Removing Objects

### Adding Objects
1. Open your `.blend` file in Blender
2. Add objects using `Shift+A` (Add menu)
3. Position and scale objects as needed
4. Save the file

**Supported object types:**
- Mesh objects (cubes, spheres, imported models, etc.)
- All mesh objects are automatically indexed on server start

### Removing Objects
1. Select the object in Blender
2. Press `X` or `Delete`
3. Confirm deletion
4. Save the file

**Note:** The server will re-index objects each time it starts, so adding/removing objects just requires restarting the server.

---

## Object Pass Indices

Object pass indices allow the system to identify which object each pixel belongs to.

### Automatic Assignment
By default, the server automatically assigns pass indices to all mesh objects at startup:
- Index 0 = Background
- Index 1 = First mesh object
- Index 2 = Second mesh object
- etc.

### Manual Assignment
To control the order or assign specific indices:

1. Select an object in Blender
2. Go to **Object Properties** (orange square icon)
3. Expand the **Relations** section
4. Set **Pass Index** to your desired value

**Important:** 
- Use values starting from 1 (0 is reserved for background)
- Each object should have a unique index
- Manual indices override automatic assignment

### Verifying Indices
When the server starts, it prints the assigned indices:
```
[BlenderServer] Assigned indices to 3 mesh objects:
  [1] Cube
  [2] Sphere
  [3] Plane
```

---

## Camera Setup

### Basic Camera Properties
1. Select the camera in Blender
2. Go to **Object Data Properties** (camera icon)
3. Configure:
   - **Type:** Perspective, Orthographic, or Panoramic
   - **Focal Length:** Lens focal length in mm (perspective only)
   - **Orthographic Scale:** View size for orthographic cameras

### Positioning the Camera
1. Select the camera
2. Use `G` to move, `R` to rotate
3. Press `Numpad 0` to view through the camera
4. Use `Ctrl+Alt+Numpad 0` to align camera to current view

### Recommended Camera Settings

| Use Case | Type | Focal Length | Notes |
|----------|------|--------------|-------|
| Overview | Perspective | 35mm | Good general view |
| Close-up | Perspective | 50-85mm | Detail inspection |
| Wide scene | Perspective | 18-24mm | See entire scene |
| Orthographic | Ortho | - | No perspective distortion |
| Top-down | Ortho/Persp | 35mm | Position above scene |

---

## Multiple Cameras

The system supports multiple cameras for different viewpoints.

### Adding Cameras
1. Press `Shift+A` → Camera
2. Name the camera descriptively (e.g., "CloseUpCam", "TopCam")
3. Position and configure the camera
4. Save the file

### Camera Naming Convention
Use descriptive names for easy identification:
- `Camera` - Main/default camera
- `CloseUpCam` - Close-up view
- `WideCam` - Wide angle view
- `TopCam` - Top-down view
- `SideCam` - Side view

### Switching Cameras at Runtime
In the agent:
- Press `1-9` keys to switch between cameras
- Or configure automatic switching in `config/default.yaml`

### Listing Available Cameras
The server command `list_cameras` returns all cameras:
```json
{
  "cameras": [
    {"name": "Camera", "type": "PERSP", "lens": 50},
    {"name": "CloseUpCam", "type": "PERSP", "lens": 85},
    {"name": "TopCam", "type": "ORTHO", "ortho_scale": 10}
  ]
}
```

---

## Lighting

Proper lighting ensures good visual observations.

### Recommended Setup
1. **Key Light:** Main light source (Sun or Area light)
2. **Fill Light:** Secondary softer light (Area or Point)
3. **HDRI Environment:** For realistic ambient lighting

### Adding an HDRI
1. Go to **World Properties**
2. Set **Surface** to "Background"
3. Set **Color** to "Environment Texture"
4. Load an HDRI image

### Lighting for Object Detection
- Ensure consistent lighting across the scene
- Avoid harsh shadows that obscure objects
- Use ambient occlusion for depth perception

---

## Best Practices

### Scene Organization
1. **Name objects clearly** - Names appear in logs and help debugging
2. **Use collections** - Group related objects
3. **Set origins properly** - Center object origins for predictable behavior

### Performance Tips
1. **Low poly for testing** - Use simplified geometry during development
2. **Optimize textures** - Use appropriate resolution textures
3. **Disable unused features** - Turn off motion blur, DOF for speed

### Render Settings
The server configures these automatically, but for testing in Blender:
- Use **Cycles** engine with 1 sample for speed
- Enable **GPU rendering** if available
- Set resolution to match config (default: 320x240)

### Checklist Before Running
- [ ] Scene saved as `.blend` file
- [ ] At least one camera exists
- [ ] Objects are mesh type
- [ ] Lighting is adequate
- [ ] No render-blocking errors

---

## Troubleshooting

### "Camera not found"
- Ensure camera exists and is named correctly
- Check spelling matches exactly (case-sensitive)

### Objects not detected
- Verify object is mesh type (not curve, text, etc.)
- Check object has enough pixels (min 200 by default)
- Ensure object is visible in render

### Dark/black renders
- Add lighting to the scene
- Check camera isn't inside an object
- Verify render settings

### Low object index values
- Background is always index 0
- Increase `min_pixels` threshold if small objects are detected as background
