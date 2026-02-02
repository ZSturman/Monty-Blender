"""Test rendering to multi-layer EXR and reading passes."""
import bpy
import os
import tempfile
import OpenImageIO as oiio

scene = bpy.context.scene
vl = bpy.context.view_layer

# Enable passes
vl.use_pass_z = True
vl.use_pass_object_index = True

# Configure rendering
scene.render.engine = "CYCLES"
scene.cycles.device = "CPU"
scene.cycles.samples = 1
scene.render.resolution_x = 64
scene.render.resolution_y = 64

# Multi-layer EXR output
out_dir = tempfile.gettempdir()
exr_path = os.path.join(out_dir, "test_multilayer.exr")

scene.render.image_settings.file_format = "OPEN_EXR"
scene.render.image_settings.color_mode = "RGBA"
scene.render.image_settings.color_depth = "32"
# Enable multi-layer in EXR
if hasattr(scene.render.image_settings, 'use_zbuffer'):
    scene.render.image_settings.use_zbuffer = True
if hasattr(scene.render.image_settings, 'exr_codec'):
    scene.render.image_settings.exr_codec = "ZIP"
scene.render.filepath = exr_path

print(f"\n=== RENDERING TO MULTI-LAYER EXR ===")
print(f"Output: {exr_path}")
bpy.ops.render.render(write_still=True)

if os.path.exists(exr_path):
    print(f"✓ EXR created: {os.path.getsize(exr_path)} bytes")
    
    print(f"\n=== READING WITH OIIO ===")
    inp = oiio.ImageInput.open(exr_path)
    if inp:
        spec = inp.spec()
        print(f"Size: {spec.width}x{spec.height}")
        print(f"Channels: {spec.nchannels}")
        print(f"Channel names: {spec.channelnames}")  # channelnames is already a list of strings
        
        # Read all pixels
        pixels = inp.read_image(format="float")
        inp.close()
        
        if pixels:
            print(f"✓ Read {len(pixels)} float values")
            print(f"Expected: {spec.width * spec.height * spec.nchannels}")
        else:
            print("✗ Failed to read pixels")
    else:
        print("✗ Failed to open EXR with OIIO")
else:
    print("✗ EXR not created")
