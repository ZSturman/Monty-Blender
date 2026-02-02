# src/server/blender_server.py
# Blender Server for Monty Visual Agent
# Requires Blender 4.0+ with Python 3.10+

"""
Blender RPC Server for visual learning agents.

Provides render, camera control, and multi-camera switching via TCP socket.
Run inside Blender:
    blender -b scene.blend --python src/server/blender_server.py -- [options]
"""

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
import platform
from array import array

VERSION = "v3.0.0-2026-02-01"
print(f"[BlenderServer] Loaded: {__file__}")
print(f"[BlenderServer] VERSION: {VERSION}")

# Check for OpenImageIO
try:
    import OpenImageIO as oiio
    print("[BlenderServer] OpenImageIO available")
except ImportError:
    oiio = None
    print("[BlenderServer] WARNING: OpenImageIO not available")


# =============================================================================
# NETWORKING
# =============================================================================

def recv_exact(conn, n):
    """Receive exactly n bytes from connection."""
    buf = b""
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Socket closed")
        buf += chunk
    return buf


def recv_msg(conn):
    """Receive length-prefixed JSON message."""
    header = recv_exact(conn, 4)
    (n,) = struct.unpack("!I", header)
    payload = recv_exact(conn, n)
    return json.loads(payload.decode("utf-8"))


def send_msg(conn, obj):
    """Send length-prefixed JSON message."""
    payload = json.dumps(obj).encode("utf-8")
    conn.sendall(struct.pack("!I", len(payload)) + payload)


# =============================================================================
# GPU SETUP
# =============================================================================

def setup_gpu_rendering(device_type=None):
    """
    Configure GPU rendering for Cycles.
    
    Args:
        device_type: 'METAL' (macOS), 'CUDA', 'OPTIX' (NVIDIA), 'HIP' (AMD)
                    If None, auto-detects based on platform.
    """
    scene = bpy.context.scene
    
    if scene.render.engine != "CYCLES":
        print("[BlenderServer] GPU rendering requires Cycles engine")
        return False
    
    # Auto-detect device type if not specified
    if device_type is None:
        system = platform.system()
        if system == "Darwin":  # macOS
            device_type = "METAL"
        else:
            device_type = "CUDA"  # Default to CUDA for Linux/Windows
    
    try:
        # Set Cycles to use GPU
        scene.cycles.device = "GPU"
        
        # Configure compute device preferences
        prefs = bpy.context.preferences.addons['cycles'].preferences
        prefs.compute_device_type = device_type
        
        # Refresh device list
        prefs.get_devices()
        
        # Enable all available devices
        enabled_count = 0
        for device in prefs.devices:
            if device.type != 'CPU':
                device.use = True
                enabled_count += 1
                print(f"[BlenderServer] Enabled GPU: {device.name}")
        
        if enabled_count == 0:
            print(f"[BlenderServer] No {device_type} GPU found, falling back to CPU")
            scene.cycles.device = "CPU"
            return False
        
        print(f"[BlenderServer] GPU rendering enabled: {device_type} ({enabled_count} devices)")
        return True
        
    except Exception as e:
        print(f"[BlenderServer] GPU setup failed: {e}, using CPU")
        scene.cycles.device = "CPU"
        return False


# =============================================================================
# SCENE SETUP
# =============================================================================

def setup_scene_for_rendering(width, height, use_gpu=False, gpu_device_type=None):
    """Configure scene for rendering with passes."""
    scene = bpy.context.scene
    view_layer = bpy.context.view_layer
    
    # Engine setup
    try:
        scene.render.engine = "CYCLES"
        scene.cycles.samples = 1
        scene.cycles.preview_samples = 1
        scene.cycles.use_adaptive_sampling = False
        
        if use_gpu:
            setup_gpu_rendering(gpu_device_type)
        else:
            scene.cycles.device = "CPU"
    except Exception as e:
        print(f"[BlenderServer] Cycles setup failed: {e}, using EEVEE")
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
    
    print(f"[BlenderServer] Scene configured: {width}x{height}, GPU={use_gpu}")


def set_object_indices():
    """Assign unique pass_index to each mesh object."""
    idx = 1
    objects_indexed = []
    for obj in bpy.data.objects:
        if obj.type == "MESH":
            obj.pass_index = idx
            objects_indexed.append((idx, obj.name))
            idx += 1
    print(f"[BlenderServer] Assigned indices to {idx-1} mesh objects:")
    for oid, name in objects_indexed:
        print(f"  [{oid}] {name}")
    return objects_indexed


# =============================================================================
# CAMERA MANAGEMENT
# =============================================================================

def list_cameras():
    """Get list of all cameras in scene."""
    cameras = []
    for obj in bpy.data.objects:
        if obj.type == "CAMERA":
            cam_data = obj.data
            cameras.append({
                "name": obj.name,
                "type": cam_data.type,  # PERSP, ORTHO, PANO
                "lens": cam_data.lens if cam_data.type == 'PERSP' else None,
                "ortho_scale": cam_data.ortho_scale if cam_data.type == 'ORTHO' else None,
                "location": [obj.location.x, obj.location.y, obj.location.z],
                "rotation": [obj.rotation_euler.x, obj.rotation_euler.y, obj.rotation_euler.z],
            })
    return cameras


def get_camera(camera_name):
    """Get camera by name and set as active."""
    cam = bpy.data.objects.get(camera_name)
    if cam is None or cam.type != "CAMERA":
        available = [c["name"] for c in list_cameras()]
        raise RuntimeError(f"Camera '{camera_name}' not found. Available: {available}")
    
    bpy.context.scene.camera = cam
    return cam


def switch_camera(camera_name):
    """Switch active camera to specified name."""
    cam = get_camera(camera_name)
    print(f"[BlenderServer] Switched to camera: {camera_name}")
    return cam


def apply_action(cam, action, position_scale=1.0, rotation_scale=1.0):
    """
    Apply camera movement action with scaling.
    
    Args:
        cam: Blender camera object
        action: Dict with 'dpos' [dx, dy, dz] and 'drot_euler' [rx, ry, rz]
        position_scale: Scale factor for position deltas
        rotation_scale: Scale factor for rotation deltas
    """
    if action is None:
        return
    
    dpos = action.get("dpos", [0.0, 0.0, 0.0])
    drot = action.get("drot_euler", [0.0, 0.0, 0.0])
    
    # Apply scaled position changes
    cam.location.x += dpos[0] * position_scale
    cam.location.y += dpos[1] * position_scale
    cam.location.z += dpos[2] * position_scale
    
    # Apply scaled rotation changes
    cam.rotation_euler.x += drot[0] * rotation_scale
    cam.rotation_euler.y += drot[1] * rotation_scale
    cam.rotation_euler.z += drot[2] * rotation_scale


def reset_scene(frame_start=1):
    """Reset scene to specified frame."""
    bpy.context.scene.frame_set(frame_start)


# =============================================================================
# RENDERING
# =============================================================================

def render_to_files(width, height):
    """
    Render scene and return RGBA, depth, index data.
    
    Uses separate renders for reliability with Blender 5.0+ background mode.
    Returns: (width, height, rgba_floats, depth_floats, index_floats)
    """
    scene = bpy.context.scene
    
    out_dir = os.path.join(tempfile.gettempdir(), "monty_blender_out")
    os.makedirs(out_dir, exist_ok=True)
    
    frame = scene.frame_current
    
    # ===== 1. Render RGB to PNG =====
    png_path = os.path.join(out_dir, f"rgb_{frame:06d}.png")
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "8"
    scene.render.filepath = png_path
    
    bpy.context.view_layer.update()
    bpy.ops.render.render(write_still=True)
    
    if not os.path.exists(png_path):
        raise RuntimeError(f"PNG not created: {png_path}")
    
    # Load RGB
    img_rgb = bpy.data.images.load(png_path, check_existing=False)
    w, h = img_rgb.size
    rgba_pixels = array("f", img_rgb.pixels[:])
    bpy.data.images.remove(img_rgb)
    
    # ===== 2. Render Depth =====
    depth_pixels = render_pass_to_array(w, h, frame, out_dir, "depth", ["Depth", "Z"])
    
    # ===== 3. Render Object Index =====
    index_pixels = render_pass_to_array(w, h, frame, out_dir, "index", 
                                        ["IndexOB", "Index Object", "Object Index"])
    
    return w, h, rgba_pixels, depth_pixels, index_pixels


def render_pass_to_array(width, height, frame, out_dir, pass_name, output_names):
    """
    Render a specific pass and return as float array.
    
    Args:
        width, height: Image dimensions
        frame: Current frame number
        out_dir: Temp output directory
        pass_name: Name for output file
        output_names: List of possible Render Layer output names to try
    
    Returns:
        array of floats
    """
    scene = bpy.context.scene
    exr_path = os.path.join(out_dir, f"{pass_name}_{frame:06d}.exr")
    
    scene.render.image_settings.file_format = "OPEN_EXR"
    scene.render.image_settings.color_mode = "BW"
    scene.render.image_settings.color_depth = "32"
    scene.render.filepath = exr_path
    
    use_nodes_orig = getattr(scene, 'use_nodes', False)
    
    try:
        scene.use_nodes = True
        tree = scene.node_tree
        nodes = tree.nodes
        links = tree.links
        
        nodes.clear()
        
        rl = nodes.new('CompositorNodeRLayers')
        rl.location = (0, 0)
        
        comp = nodes.new('CompositorNodeComposite')
        comp.location = (300, 0)
        
        # Try to connect the requested output
        connected = False
        for out_name in output_names:
            if out_name in rl.outputs:
                links.new(rl.outputs[out_name], comp.inputs['Image'])
                connected = True
                break
        
        if connected:
            bpy.ops.render.render(write_still=True)
        
    except Exception as e:
        print(f"[BlenderServer] {pass_name} render failed: {e}")
    finally:
        scene.use_nodes = use_nodes_orig
    
    # Read EXR file
    num_pixels = width * height
    
    if os.path.exists(exr_path):
        if oiio:
            inp = oiio.ImageInput.open(exr_path)
            spec = inp.spec()
            data = inp.read_image(format="float")
            inp.close()
            
            import numpy as np
            arr = np.array(data, dtype=np.float32)
            
            if len(arr.shape) > 1:
                channel = arr[:, :, 0].flatten()
            else:
                nchannels = spec.nchannels
                channel = arr[::nchannels]
            
            return array("f", channel.tolist())
        else:
            img = bpy.data.images.load(exr_path, check_existing=False)
            raw = img.pixels[:]
            bpy.data.images.remove(img)
            
            pixels = array("f", [0.0] * num_pixels)
            for i in range(num_pixels):
                pixels[i] = raw[i * 4]  # RGBA format, take R
            return pixels
    
    print(f"[BlenderServer] WARNING: {pass_name} EXR not found, using dummy data")
    return array("f", [1.0 if pass_name == "depth" else 0.0] * num_pixels)


# =============================================================================
# DATA ENCODING
# =============================================================================

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
        # Fallback using Blender
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
    data = struct.pack(f'{len(floats_arr)}f', *floats_arr)
    return zlib.compress(data, level=6)


def pack_int32_zlib_from_float_indices(idx_float_arr):
    """Convert float indices to int32 and compress."""
    int_vals = [int(round(f)) for f in idx_float_arr]
    data = struct.pack(f'{len(int_vals)}i', *int_vals)
    return zlib.compress(data, level=6)


# =============================================================================
# SERVER LOOP
# =============================================================================

def server_loop(host, port, camera_name, width, height, 
                use_gpu=False, gpu_device_type=None,
                position_scale=1.0, rotation_scale=1.0):
    """
    Main server loop handling client connections.
    
    Commands:
        render: Render current frame, return RGBA/depth/index
        step: Advance frame and optionally apply camera action
        reset: Reset to specified frame
        state: Get current frame and camera transform
        list_cameras: Get all cameras in scene
        switch_camera: Switch active camera
        close: Close connection
    """
    # Scene setup
    objects_indexed = set_object_indices()
    cam = get_camera(camera_name)
    setup_scene_for_rendering(width, height, use_gpu, gpu_device_type)
    reset_scene()
    
    # Socket setup
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, port))
    sock.listen(1)
    
    print(f"[BlenderServer] Listening on {host}:{port}")
    print(f"[BlenderServer] Camera: {camera_name}")
    print(f"[BlenderServer] Resolution: {width}x{height}")
    print(f"[BlenderServer] GPU: {use_gpu}")
    
    while True:
        try:
            conn, addr = sock.accept()
            print(f"[BlenderServer] Client connected: {addr}")
            
            try:
                while True:
                    req = recv_msg(conn)
                    cmd = req.get("cmd")
                    
                    # === CLOSE ===
                    if cmd == "close":
                        send_msg(conn, {"ok": True})
                        break
                    
                    # === RESET ===
                    if cmd == "reset":
                        frame = int(req.get("frame", bpy.context.scene.frame_start))
                        reset_scene(frame)
                        send_msg(conn, {"ok": True, "frame": frame})
                        continue
                    
                    # === STEP ===
                    if cmd == "step":
                        n = int(req.get("n", 1))
                        action = req.get("action")
                        
                        # Get scales from request or use defaults
                        pos_scale = req.get("position_scale", position_scale)
                        rot_scale = req.get("rotation_scale", rotation_scale)
                        
                        apply_action(cam, action, pos_scale, rot_scale)
                        
                        scene = bpy.context.scene
                        new_frame = scene.frame_current + n
                        scene.frame_set(new_frame)
                        send_msg(conn, {"ok": True, "frame": new_frame})
                        continue
                    
                    # === STATE ===
                    if cmd == "state":
                        scene = bpy.context.scene
                        send_msg(conn, {
                            "ok": True,
                            "frame": scene.frame_current,
                            "camera": {
                                "name": cam.name,
                                "location": [cam.location.x, cam.location.y, cam.location.z],
                                "rotation_euler": [cam.rotation_euler.x, cam.rotation_euler.y, cam.rotation_euler.z],
                            }
                        })
                        continue
                    
                    # === LIST CAMERAS ===
                    if cmd == "list_cameras":
                        cameras = list_cameras()
                        send_msg(conn, {"ok": True, "cameras": cameras})
                        continue
                    
                    # === SWITCH CAMERA ===
                    if cmd == "switch_camera":
                        new_camera = req.get("camera")
                        if not new_camera:
                            send_msg(conn, {"ok": False, "error": "No camera name provided"})
                            continue
                        try:
                            cam = switch_camera(new_camera)
                            send_msg(conn, {"ok": True, "camera": new_camera})
                        except RuntimeError as e:
                            send_msg(conn, {"ok": False, "error": str(e)})
                        continue
                    
                    # === RENDER ===
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
                    
                    # === UNKNOWN COMMAND ===
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


# =============================================================================
# ENTRY POINT
# =============================================================================

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Blender RPC Server")
    parser.add_argument("--host", default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=5555, help="Server port")
    parser.add_argument("--camera", default="Camera", help="Camera name")
    parser.add_argument("--width", type=int, default=320, help="Render width")
    parser.add_argument("--height", type=int, default=240, help="Render height")
    parser.add_argument("--gpu", action="store_true", help="Enable GPU rendering")
    parser.add_argument("--gpu-type", default=None, 
                       help="GPU device type: METAL, CUDA, OPTIX, HIP")
    parser.add_argument("--position-scale", type=float, default=1.0,
                       help="Position movement scale factor")
    parser.add_argument("--rotation-scale", type=float, default=1.0,
                       help="Rotation movement scale factor")
    
    # Handle Blender's argument passing
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
    
    return parser.parse_args(argv)


def main():
    """Entry point when run as Blender script."""
    args = parse_args()
    server_loop(
        host=args.host,
        port=args.port,
        camera_name=args.camera,
        width=args.width,
        height=args.height,
        use_gpu=args.gpu,
        gpu_device_type=args.gpu_type,
        position_scale=args.position_scale,
        rotation_scale=args.rotation_scale,
    )


if __name__ == "__main__":
    main()
