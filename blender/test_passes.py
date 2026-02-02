#!/usr/bin/env python3
"""Test script to run inside Blender to check what passes are available."""
import bpy

scene = bpy.context.scene
view_layer = bpy.context.view_layer

print("\n=== VIEW LAYER PASS SETTINGS ===")
print(f"use_pass_z: {view_layer.use_pass_z}")
print(f"use_pass_object_index: {view_layer.use_pass_object_index}")

# Try to enable them
view_layer.use_pass_z = True
view_layer.use_pass_object_index = True

print(f"\nAfter enabling:")
print(f"use_pass_z: {view_layer.use_pass_z}")
print(f"use_pass_object_index: {view_layer.use_pass_object_index}")

# Configure rendering
scene.render.engine = "CYCLES"
scene.cycles.device = "CPU"
scene.cycles.samples = 1
scene.render.resolution_x = 64
scene.render.resolution_y = 64

print(f"\n=== RENDERING ===")
bpy.ops.render.render()

print(f"\n=== CHECKING RENDER RESULT ===")
rr = bpy.data.images.get("Render Result")
if rr is None:
    print("ERROR: No Render Result found!")
else:
    print(f"Render Result found: size={rr.size}")
    print(f"Has pixels: {hasattr(rr, 'pixels')}")
    if hasattr(rr, 'pixels'):
        try:
            pixels = rr.pixels[:]
            print(f"Pixels length: {len(pixels)} (expected: {rr.size[0] * rr.size[1] * 4})")
        except Exception as e:
            print(f"Error reading pixels: {e}")
    
    layers = getattr(rr, "layers", None)
    if layers:
        print(f"Number of layers: {len(layers)}")
        for i, layer in enumerate(layers):
            print(f"\nLayer {i}: {layer.name}")
            passes = getattr(layer, "passes", None)
            if passes:
                print(f"  Number of passes: {len(passes)}")
                for p in passes:
                    print(f"    - {p.name} (channels: {p.channels})")
            else:
                print(f"  No passes attribute")
    else:
        print("No layers found")

# Try accessing view layer directly
print(f"\n=== VIEW LAYER AFTER RENDER ===")
vl = bpy.context.view_layer
print(f"View layer name: {vl.name}")
print(f"use_pass_z: {vl.use_pass_z}")
print(f"use_pass_object_index: {vl.use_pass_object_index}")
