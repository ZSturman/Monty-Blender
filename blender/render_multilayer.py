"""
Replacement for _render_and_extract that uses multi-layer EXR.
This avoids the Render Result issue in Blender 5.0.1 background mode.
"""

import bpy
import os
import tempfile
import time
from array import array

def _render_and_extract_multilayer(width, height):
    """Render using multi-layer EXR which includes all passes.
    
    Returns:
      (w, h, rgba_float_pixels, depth_float_pixels, index_float_pixels)
    """
    scene = bpy.context.scene
    view_layer = bpy.context.view_layer
    
    # Configure render engine
    try:
        scene.render.engine = "CYCLES"
        scene.cycles.device = "CPU"
        scene.cycles.samples = 1
        scene.cycles.preview_samples = 1
        scene.cycles.use_adaptive_sampling = False
    except Exception:
        scene.render.engine = "BLENDER_EEVEE"
    
    # Resolution
    scene.render.resolution_x = int(width)
    scene.render.resolution_y = int(height)
    scene.render.resolution_percentage = 100
    
    # Ensure passes are enabled
    view_layer.use_pass_z = True
    view_layer.use_pass_object_index = True
    view_layer.use_pass_combined = True
    
    # Output setup
    out_dir = os.path.join(tempfile.gettempdir(), "monty_blender_out")
    os.makedirs(out_dir, exist_ok=True)
    
    frame = int(scene.frame_current)
    exr_path = os.path.join(out_dir, f"render_{frame:06d}.exr")
    
    # Multi-layer EXR format
    scene.render.image_settings.file_format = "OPEN_EXR_MULTILAYER"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "32"
    scene.render.image_settings.exr_codec = "ZIP"
    scene.render.filepath = exr_path
    
    # Render
    print(f"[BlenderServer] Rendering to: {exr_path}")
    bpy.context.view_layer.update()
    bpy.ops.render.render(write_still=True)
    
    if not os.path.exists(exr_path):
        raise RuntimeError(f"EXR not created: {exr_path}")
    
    print(f"[BlenderServer] EXR created successfully")
    
    # Load and read EXR
    try:
        import OpenImageIO as oiio
    except ImportError:
        # Fallback: use Blender's image loader
        return _read_exr_with_blender(exr_path, width, height)
    
    # Use OpenImageIO for better control
    inp = oiio.ImageInput.open(exr_path)
    if inp is None:
        raise RuntimeError(f"Failed to open EXR: {exr_path}")
    
    spec = inp.spec()
    w, h = spec.width, spec.height
    channels = [ch.name for ch in spec.channelnames]
    
    print(f"[BlenderServer] EXR size: {w}x{h}, channels: {channels}")
    
    # Read all pixels (interleaved channels)
    pixels = inp.read_image(format="float")
    inp.close()
    
    if pixels is None:
        raise RuntimeError("Failed to read EXR pixels")
    
    # Extract passes from channels
    # Format: "ViewLayer.PassName.Channel" or sometimes just "PassName.Channel"
    def find_channels(pass_name, channel_suffix):
        """Find channel indices for a pass."""
        result = []
        for suffix in channel_suffix:
            for i, ch in enumerate(channels):
                if pass_name in ch and ch.endswith(suffix):
                    result.append(i)
                    break
        return result
    
    # Find Combined/RGBA channels
    rgba_indices = find_channels("Combined", ["R", "G", "B", "A"])
    if len(rgba_indices) != 4:
        # Try without ViewLayer prefix
        rgba_indices = [i for i, ch in enumerate(channels) if any(ch.endswith(s) for s in [".R", ".G", ".B", ".A"])][:4]
    
    # Find Depth channel
    depth_indices = find_channels("Depth", ["V", "Z", "Y"])
    if not depth_indices:
        depth_indices = [i for i, ch in enumerate(channels) if "Depth" in ch or ch.endswith(".Z")][:1]
    
    # Find IndexOB channel  
    index_indices = find_channels("IndexOB", ["X", "R"])
    if not index_indices:
        index_indices = [i for i, ch in enumerate(channels) if "Index" in ch][:1]
    
    print(f"[BlenderServer] Channel indices - RGBA: {rgba_indices}, Depth: {depth_indices}, Index: {index_indices}")
    
    if len(rgba_indices) < 3:
        raise RuntimeError(f"Could not find RGBA channels. Available: {channels}")
    if not depth_indices:
        raise RuntimeError(f"Could not find Depth channel. Available: {channels}")
    if not index_indices:
        raise RuntimeError(f"Could not find Index channel. Available: {channels}")
    
    # Extract data
    num_channels = len(channels)
    npx = w * h
    
    rgba_data = array("f", [0.0]) * (npx * 4)
    depth_data = array("f", [0.0]) * npx
    index_data = array("f", [0.0]) * npx
    
    for i in range(npx):
        base_idx = i * num_channels
        # RGBA
        for j in range(min(4, len(rgba_indices))):
            rgba_data[i * 4 + j] = float(pixels[base_idx + rgba_indices[j]])
        # Depth
        depth_data[i] = float(pixels[base_idx + depth_indices[0]])
        # Index
        index_data[i] = float(pixels[base_idx + index_indices[0]])
    
    return w, h, rgba_data, depth_data, index_data


def _read_exr_with_blender(exr_path, expected_w, expected_h):
    """Fallback: read EXR using Blender's image loader."""
    print("[BlenderServer] Using Blender image loader for EXR")
    
    img = bpy.data.images.load(exr_path, check_existing=False)
    try:
        w, h = img.size
        if w != expected_w or h != expected_h:
            print(f"[BlenderServer] WARNING: EXR size {w}x{h} != expected {expected_w}x{expected_h}")
        
        # Blender loads multi-layer EXR but only shows Combined pass in pixels
        # We need to access render passes differently
        # This is a limitation - we can only get Combined reliably
        pixels = array("f", img.pixels[:])
        
        # For depth and index, we'd need to parse the EXR metadata
        # For now, return dummy data
        npx = w * h
        depth_data = array("f", [1.0]) * npx  # Dummy depth
        index_data = array("f", [0.0]) * npx  # Dummy index
        
        print("[BlenderServer] WARNING: Depth and Index not available via Blender loader")
        
        return w, h, pixels, depth_data, index_data
    finally:
        bpy.data.images.remove(img)
