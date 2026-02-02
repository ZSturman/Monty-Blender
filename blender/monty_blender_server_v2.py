# blender/monty_blender_server_v2.py
# Clean implementation for Blender 5.0.1

import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import bpy
import argparse
import base64
import json
import socket
import struct
import zlib
import tempfile
import time
from array import array

print(f"[BlenderServer] Loaded script: {__file__}")
print("[BlenderServer] VERSION: v2-simplified-2026-02-01")

try:
    import OpenImageIO as oiio
    print("[BlenderServer] OpenImageIO available")
except ImportError:
    oiio = None
    print("[BlenderServer] WARNING: OpenImageIO not available")

# ===== NETWORKING =====

def recv_exact(conn, n):
    buf = b""
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Socket closed")
        buf += chunk
    return buf

def recv_msg(conn):
    header = recv_exact(conn, 4)
    (n,) = struct.unpack("!I", header)
    payload = recv_exact(conn, n)
    return json.loads(payload.decode("utf-8"))

def send_msg(conn, obj):
    payload = json.dumps(obj).encode("utf-8")
    conn.sendall(struct.pack("!I", len(payload)) + payload)

# ===== RENDERING =====

def setup_scene_for_rendering(width, height):
    """Configure scene for rendering with passes."""
    scene = bpy.context.scene
    view_layer = bpy.context.view_layer
    
    # Engine
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
    scene.render.use_border = False
    scene.render.use_crop_to_border = False
    
    # Color management
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    scene.render.dither_intensity = 0.0
    
    # Enable passes
    view_layer.use_pass_z = True
    view_layer.use_pass_object_index = True
    view_layer.use_pass_combined = True
    
    print(f"[BlenderServer] Scene configured: {width}x{height}, passes enabled")

def render_to_files(width, height):
    """Render and return RGBA, depth, index as numpy-like arrays.
    
    Strategy: Render 3 times with different output configurations.
    This is the most reliable approach for Blender 5.0.1 background mode.
    """
    scene = bpy.context.scene
    view_layer = bpy.context.view_layer
    
    out_dir = os.path.join(tempfile.gettempdir(), "monty_blender_out")
    os.makedirs(out_dir, exist_ok=True)
    
    frame = scene.frame_current
    
    # ===== 1. Render RGB to PNG =====
    png_path = os.path.join(out_dir, f"rgb_{frame:06d}.png")
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "8"
    scene.render.filepath = png_path
    
    print(f"[BlenderServer] Rendering RGB to: {png_path}")
    bpy.context.view_layer.update()
    bpy.ops.render.render(write_still=True)
    
    if not os.path.exists(png_path):
        raise RuntimeError(f"PNG not created: {png_path}")
    
    # Load RGB
    img_rgb = bpy.data.images.load(png_path, check_existing=False)
    w, h = img_rgb.size
    rgba_pixels = array("f", img_rgb.pixels[:])
    bpy.data.images.remove(img_rgb)
    
    print(f"[BlenderServer] RGB loaded: {w}x{h}, {len(rgba_pixels)} floats")
    
    # ===== 2. Get Depth from compositor or render pass =====
    # For Blender 5.0.1, we'll render to EXR and extract depth
    depth_exr = os.path.join(out_dir, f"depth_{frame:06d}.exr")
    
    # Temporarily change to EXR output
    scene.render.image_settings.file_format = "OPEN_EXR"
    scene.render.image_settings.color_mode = "BW"  # Single channel
    scene.render.image_settings.color_depth = "32"
    scene.render.filepath = depth_exr
    
    # Use compositor to output depth
    use_nodes_orig = scene.use_nodes if hasattr(scene, 'use_nodes') else False
    node_tree_orig = scene.node_tree if hasattr(scene, 'node_tree') else None
    
    try:
        # Try to set up a simple compositor for depth output
        if hasattr(scene, 'use_nodes'):
            scene.use_nodes = True
        
        if hasattr(scene, 'node_tree') and scene.node_tree is not None:
            tree = scene.node_tree
            nodes = tree.nodes
            links = tree.links
            
            # Clear and rebuild
            nodes.clear()
            
            # Render Layers
            rl = nodes.new('CompositorNodeRLayers')
            rl.location = (0, 0)
            
            # Composite (output)
            comp = nodes.new('CompositorNodeComposite')
            comp.location = (300, 0)
            
            # Connect depth to composite
            if 'Depth' in rl.outputs:
                links.new(rl.outputs['Depth'], comp.inputs['Image'])
            elif 'Z' in rl.outputs:
                links.new(rl.outputs['Z'], comp.inputs['Image'])
            
            print(f"[BlenderServer] Rendering depth to: {depth_exr}")
            bpy.ops.render.render(write_still=True)
        else:
            # Fallback: render normally and try to extract from render result
            print(f"[BlenderServer] No compositor available, rendering normally")
            bpy.ops.render.render(write_still=True)
    except Exception as e:
        print(f"[BlenderServer] Compositor setup failed: {e}, using fallback")
        bpy.ops.render.render(write_still=True)
    finally:
        # Restore
        if hasattr(scene, 'use_nodes'):
            scene.use_nodes = use_nodes_orig
    
    # Read depth EXR
    if os.path.exists(depth_exr):
        if oiio:
            inp = oiio.ImageInput.open(depth_exr)
            spec = inp.spec()
            depth_data = inp.read_image(format="float")
            inp.close()
            # Extract first channel - depth_data is flat array of all channels
            num_pixels = w * h
            nchannels = spec.nchannels
            # Reshape and extract first channel
            import numpy as np
            depth_arr = np.array(depth_data, dtype=np.float32)
            if len(depth_arr.shape) > 1:
                # Already multi-dimensional
                depth_channel = depth_arr[:, :, 0].flatten()
            else:
                # Flat array, need to extract every nth element
                depth_channel = depth_arr[::nchannels]
            depth_pixels = array("f", depth_channel.tolist())
        else:
            img_depth = bpy.data.images.load(depth_exr, check_existing=False)
            depth_data_raw = img_depth.pixels[:]
            bpy.data.images.remove(img_depth)
            num_pixels = w * h
            depth_pixels = array("f", [0.0] * num_pixels)
            for i in range(num_pixels):
                depth_pixels[i] = depth_data_raw[i * 4]  # RGBA format, take R
        
        print(f"[BlenderServer] Depth loaded: {len(depth_pixels)} floats")
    else:
        print(f"[BlenderServer] WARNING: Depth EXR not found, using dummy data")
        depth_pixels = array("f", [1.0] * (w * h))
    
    # ===== 3. Get Object Index =====
    # Similar approach for object index
    index_exr = os.path.join(out_dir, f"index_{frame:06d}.exr")
    
    scene.render.image_settings.file_format = "OPEN_EXR"
    scene.render.image_settings.color_mode = "BW"
    scene.render.filepath = index_exr
    
    try:
        if hasattr(scene, 'use_nodes') and hasattr(scene, 'node_tree') and scene.node_tree:
            scene.use_nodes = True
            tree = scene.node_tree
            nodes = tree.nodes
            links = tree.links
            
            nodes.clear()
            rl = nodes.new('CompositorNodeRLayers')
            comp = nodes.new('CompositorNodeComposite')
            
            # Connect index
            connected = False
            for out_name in ['IndexOB', 'Index Object', 'Object Index']:
                if out_name in rl.outputs:
                    links.new(rl.outputs[out_name], comp.inputs['Image'])
                    connected = True
                    break
            
            if connected:
                print(f"[BlenderServer] Rendering index to: {index_exr}")
                bpy.ops.render.render(write_still=True)
            else:
                print(f"[BlenderServer] WARNING: No Index output found")
        else:
            print(f"[BlenderServer] WARNING: Cannot set up compositor for index")
    except Exception as e:
        print(f"[BlenderServer] Index render failed: {e}")
    
    # Read index EXR
    if os.path.exists(index_exr):
        if oiio:
            inp = oiio.ImageInput.open(index_exr)
            spec = inp.spec()
            index_data = inp.read_image(format="float")
            inp.close()
            num_pixels = w * h
            nchannels = spec.nchannels
            # Reshape and extract first channel
            import numpy as np
            index_arr = np.array(index_data, dtype=np.float32)
            if len(index_arr.shape) > 1:
                # Already multi-dimensional
                index_channel = index_arr[:, :, 0].flatten()
            else:
                # Flat array, need to extract every nth element
                index_channel = index_arr[::nchannels]
            index_pixels = array("f", index_channel.tolist())
        else:
            img_index = bpy.data.images.load(index_exr, check_existing=False)
            index_data_raw = img_index.pixels[:]
            bpy.data.images.remove(img_index)
            num_pixels = w * h
            index_pixels = array("f", [0.0] * num_pixels)
            for i in range(num_pixels):
                index_pixels[i] = index_data_raw[i * 4]  # RGBA format, take R
        
        print(f"[BlenderServer] Index loaded: {len(index_pixels)} floats")
    else:
        print(f"[BlenderServer] WARNING: Index EXR not found, using dummy data")
        index_pixels = array("f", [0.0] * (w * h))
    
    return w, h, rgba_pixels, depth_pixels, index_pixels

# ===== DATA CONVERSION =====

def rgba_float_to_png_bytes(width, height, rgba_floats):
    """Convert float RGBA to PNG bytes."""
    try:
        from PIL import Image
        import numpy as np
        arr = np.array(rgba_floats, dtype=np.float32).reshape((height, width, 4))
        arr_uint8 = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
        img = Image.fromarray(arr_uint8, mode='RGBA')
        import io
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        return buf.getvalue()
    except ImportError:
        # Fallback: use Blender
        temp_img = bpy.data.images.new("TempRGBA", width, height, alpha=True)
        temp_img.pixels = rgba_floats
        temp_path = os.path.join(tempfile.gettempdir(), "temp_rgba.png")
        temp_img.filepath_raw = temp_path
        temp_img.file_format = 'PNG'
        temp_img.save()
        bpy.data.images.remove(temp_img)
        with open(temp_path, 'rb') as f:
            return f.read()

def pack_float32_zlib(floats_arr):
    """Pack float array to compressed bytes."""
    import struct
    data = struct.pack(f'{len(floats_arr)}f', *floats_arr)
    return zlib.compress(data, level=6)

def pack_int32_zlib_from_float_indices(idx_float_arr):
    """Convert float indices to int32 and compress."""
    import struct
    int_vals = [int(round(f)) for f in idx_float_arr]
    data = struct.pack(f'{len(int_vals)}i', *int_vals)
    return zlib.compress(data, level=6)

# ===== SCENE MANAGEMENT =====

def set_object_indices():
    """Assign unique pass_index to each mesh object."""
    idx = 1
    for obj in bpy.data.objects:
        if obj.type == "MESH":
            obj.pass_index = idx
            idx += 1
    print(f"[BlenderServer] Assigned indices to {idx-1} mesh objects")

def get_camera(camera_name):
    """Get camera by name."""
    cam = bpy.data.objects.get(camera_name)
    if cam is None or cam.type != "CAMERA":
        raise RuntimeError(f"Camera '{camera_name}' not found")
    
    # Set as active camera
    bpy.context.scene.camera = cam
    return cam

def apply_action(cam, action):
    """Apply camera movement action."""
    if action is None:
        return
    
    dx, dy, dz = action.get("dpos", [0.0, 0.0, 0.0])
    droll, dpitch, dyaw = action.get("drot_euler", [0.0, 0.0, 0.0])
    
    cam.location.x += dx
    cam.location.y += dy
    cam.location.z += dz
    cam.rotation_euler.x += droll
    cam.rotation_euler.y += dpitch
    cam.rotation_euler.z += dyaw

def reset_scene(frame_start=1):
    """Reset scene to frame."""
    bpy.context.scene.frame_set(frame_start)

# ===== SERVER LOOP =====

def server_loop(host, port, camera_name, width, height):
    """Main server loop."""
    set_object_indices()
    cam = get_camera(camera_name)
    setup_scene_for_rendering(width, height)
    reset_scene()
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, port))
    sock.listen(1)
    print(f"[BlenderServer] Listening on {host}:{port}")
    
    # Accept multiple connections
    while True:
        try:
            conn, addr = sock.accept()
            print(f"[BlenderServer] Client connected: {addr}")
            
            try:
                while True:
                    req = recv_msg(conn)
                    cmd = req.get("cmd")
                    
                    if cmd == "close":
                        send_msg(conn, {"ok": True})
                        break
                    
                    if cmd == "reset":
                        frame = int(req.get("frame", bpy.context.scene.frame_start))
                        reset_scene(frame)
                        send_msg(conn, {"ok": True, "frame": frame})
                        continue
                    
                    if cmd == "step":
                        n = int(req.get("n", 1))
                        action = req.get("action")
                        apply_action(cam, action)
                        
                        scene = bpy.context.scene
                        new_frame = scene.frame_current + n
                        scene.frame_set(new_frame)
                        send_msg(conn, {"ok": True, "frame": new_frame})
                        continue
                    
                    if cmd == "state":
                        scene = bpy.context.scene
                        send_msg(conn, {
                            "ok": True,
                            "frame": scene.frame_current,
                            "camera": {
                                "location": [cam.location.x, cam.location.y, cam.location.z],
                                "rotation_euler": [cam.rotation_euler.x, cam.rotation_euler.y, cam.rotation_euler.z],
                            }
                        })
                        continue
                    
                    if cmd == "render":
                        try:
                            w, h, rgba_f, z_f, idx_f = render_to_files(width, height)
                            
                            png_bytes = rgba_float_to_png_bytes(w, h, rgba_f)
                            depth_blob = pack_float32_zlib(z_f)
                            idx_blob = pack_int32_zlib_from_float_indices(idx_f)
                            
                            resp = {
                                "ok": True,
                                "width": w,
                                "height": h,
                                "rgba_png_b64": base64.b64encode(png_bytes).decode("ascii"),
                                "depth_f32_zlib_b64": base64.b64encode(depth_blob).decode("ascii"),
                                "index_i32_zlib_b64": base64.b64encode(idx_blob).decode("ascii"),
                            }
                            send_msg(conn, resp)
                        except Exception as e:
                            import traceback
                            traceback.print_exc()
                            send_msg(conn, {"ok": False, "error": str(e)})
                        continue
                    
                    send_msg(conn, {"ok": False, "error": f"Unknown cmd: {cmd}"})
            
            except (ConnectionError, BrokenPipeError, ConnectionResetError):
                print(f"[BlenderServer] Client {addr} disconnected")
            finally:
                try:
                    conn.close()
                except Exception:
                    pass
                print(f"[BlenderServer] Ready for next connection")
        
        except KeyboardInterrupt:
            print("\n[BlenderServer] Shutting down...")
            break
        except Exception as e:
            print(f"[BlenderServer] Server error: {e}")
            import traceback
            traceback.print_exc()
    
    sock.close()

# ===== ENTRY POINT =====

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--camera", default="Camera")  # Changed from "MontyCam"
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    
    import sys
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
    return parser.parse_args(argv)

def main():
    args = parse_args()
    server_loop(args.host, args.port, args.camera, args.width, args.height)

if __name__ == "__main__":
    main()
