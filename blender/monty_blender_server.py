# blender/monty_blender_server.py

import os
import sys

# Ensure Blender can import sibling modules when this script is run via an absolute path.
_SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import bpy
print(f"[BlenderServer] Loaded script: {__file__}")
print("[BlenderServer] VERSION: pass-fallback-2026-02-01")

# Try importing OpenImageIO (oiio) for EXR reading
try:
    import OpenImageIO as oiio
except Exception:
    oiio = None


import argparse
import base64
import json
import socket
import struct
import zlib
from array import array
import tempfile

from compositor_setup import set_object_indices, setup_passes_and_viewers

# --------------------------
# Networking helpers
# --------------------------
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

def _render_and_extract(width, height):
    """Render RGBA + depth + object index in Blender headless using disk outputs.

    Background/headless builds can return an empty in-memory Render Result (0x0). To avoid that,
    render to disk:
      - RGB(A): scene.render.filepath (PNG)
      - Depth + Object Index: compositor File Output nodes (OPEN_EXR)

    Returns:
      (w, h, rgba_float_pixels, depth_float_pixels, index_float_pixels)
      where rgba_float_pixels is array('f') length w*h*4 in [0..1].
    """
    import time
    import glob

    scene = bpy.context.scene
    view_layer = bpy.context.view_layer

    # Force a known engine in headless mode.
    try:
        scene.render.engine = "CYCLES"
    except Exception:
        scene.render.engine = "BLENDER_EEVEE"

    # Cycles: force CPU + low samples for speed
    if getattr(scene.render, "engine", "") == "CYCLES" and hasattr(scene, "cycles"):
        try:
            scene.cycles.device = "CPU"
        except Exception:
            pass
        for attr, val in (
            ("samples", 1),
            ("preview_samples", 1),
            ("use_adaptive_sampling", False),
        ):
            try:
                setattr(scene.cycles, attr, val)
            except Exception:
                pass

    # Resolution / safety knobs
    scene.render.resolution_x = int(width)
    scene.render.resolution_y = int(height)
    try:
        scene.render.resolution_percentage = 100
    except Exception:
        pass
    for attr in ("use_border", "use_crop_to_border"):
        if hasattr(scene.render, attr):
            try:
                setattr(scene.render, attr, False)
            except Exception:
                pass

    # Ensure we have an active camera.
    if getattr(scene, "camera", None) is None or getattr(scene.camera, "type", None) != "CAMERA":
        for obj in bpy.data.objects:
            if getattr(obj, "type", None) == "CAMERA":
                scene.camera = obj
                break

    # Deterministic color
    try:
        scene.view_settings.view_transform = "Standard"
        scene.view_settings.exposure = 0.0
        scene.view_settings.gamma = 1.0
    except Exception:
        pass
    try:
        scene.render.dither_intensity = 0.0
    except Exception:
        pass

    # Ensure depth + object index passes are enabled
    try:
        view_layer.use_pass_z = True
        view_layer.use_pass_object_index = True
    except Exception:
        pass

    # Disk output locations
    out_dir = os.path.join(tempfile.gettempdir(), "monty_blender_out")
    os.makedirs(out_dir, exist_ok=True)

    frame = int(scene.frame_current)
    
    # STRATEGY: Render to multi-layer EXR (includes all passes), then extract
    # This works reliably in Blender 5.0.1 background mode
    exr_path = os.path.join(out_dir, f"render_{frame:06d}.exr")
    
    # Configure multi-layer EXR output with all passes
    scene.render.image_settings.file_format = "OPEN_EXR_MULTILAYER"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "32"
    scene.render.image_settings.exr_codec = "ZIP"  # Compression
    scene.render.filepath = exr_path

    t0 = time.time()

    # Render (writes multi-layer EXR with Combined, Depth, IndexOB passes)
    try:
        bpy.context.view_layer.update()
    except Exception:
        pass
    
    print(f"[BlenderServer] Rendering to multi-layer EXR: {exr_path}")
    bpy.ops.render.render(write_still=True)

    if not os.path.exists(exr_path):
        raise RuntimeError(f"Multi-layer EXR was not written: {exr_path}")
    
    print(f"[BlenderServer] Multi-layer EXR created: {exr_path}")

    # --- Read passes from multi-layer EXR ---
    def _read_exr_pass(path, pass_name):
        """Read a specific pass from multi-layer EXR."""
        if oiio is None:
            raise RuntimeError("OpenImageIO not available - cannot read multi-layer EXR")
        
        inp = oiio.ImageInput.open(path)
        if inp is None:
            raise RuntimeError(f"Failed to open EXR: {path}")
        
        spec = inp.spec()
        w, h = spec.width, spec.height
        
        # List available channels
        channels = [ch.name for ch in spec.channelnames]
        print(f"[BlenderServer] EXR channels: {channels}")
        
        # Find the channel for this pass
        # Multi-layer EXR format: ViewLayer.PassName.Channel
        # e.g., "ViewLayer.Combined.R", "ViewLayer.Depth.V", "ViewLayer.IndexOB.X"
        target_channel = None
        for ch in channels:
            if pass_name in ch:
                target_channel = ch
                break
        
        if target_channel is None:
            inp.close()
            raise RuntimeError(f"Pass '{pass_name}' not found in EXR. Available: {channels}")
        
        print(f"[BlenderServer] Reading pass '{pass_name}' from channel '{target_channel}'")
        
        # Read just that channel
        pixels = inp.read_image(format="float")
        inp.close()
        
        if pixels is None:
            raise RuntimeError(f"Failed to read pixels from {path}")
        
        # Extract the specific channel
        # pixels is a flat array of all channels: [R,G,B,A, R,G,B,A, ...]
        num_channels = len(channels)
        channel_idx = channels.index(target_channel)
        
        # Extract every num_channels-th value starting at channel_idx
        pass_data = array("f", [0.0]) * (w * h)
        for i in range(w * h):
            pass_data[i] = float(pixels[i * num_channels + channel_idx])
        
        return w, h, pass_data
    
    # Read Combined (RGBA) pass
    w0, h0, combined_data = _read_exr_pass(exr_path, "Combined")
    
    # Convert combined_data (single channel per pixel) to RGBA
    # Actually, we need to read all 4 channels for Combined
    # Let me fix this...
        w0, h0 = rr_img.size
        if int(w0) <= 0 or int(h0) <= 0:
            raise RuntimeError(f"Loaded PNG has invalid size ({w0},{h0}): {png_path}")
        pix = array("f", rr_img.pixels[:])
    finally:
        try:
            if rr_img is not None:
                bpy.data.images.remove(rr_img)
        except Exception:
            pass

    # Ensure pixel length matches
    expected = int(w0) * int(h0) * 4
    if len(pix) != expected:
        raise RuntimeError(f"PNG pixels length {len(pix)} != expected {expected} for size ({w0},{h0})")

    # --- Find the newest EXR outputs for depth + index ---
    def _pick_latest(patterns):
        candidates = []
        for pat in patterns:
            # Support recursive patterns using **
            try:
                candidates.extend(glob.glob(pat, recursive=True))
            except TypeError:
                # Older glob without recursive kw
                candidates.extend(glob.glob(pat))

        candidates = [p for p in candidates if os.path.isfile(p)]

        # Prefer files created/updated after we started this render
        recent = [p for p in candidates if os.path.getmtime(p) >= (t0 - 0.5)]
        pool = recent if recent else candidates
        if not pool:
            return None
        return max(pool, key=lambda p: os.path.getmtime(p))

    depth_path = _pick_latest([
        os.path.join(depth_dir, "*.exr"),
        os.path.join(depth_dir, "**", "*.exr"),
        os.path.join(out_dir, "**", "depth*.exr"),
        os.path.join(out_dir, "**", "*depth*.exr"),
    ])
    index_path = _pick_latest([
        os.path.join(index_dir, "*.exr"),
        os.path.join(index_dir, "**", "*.exr"),
        os.path.join(out_dir, "**", "index*.exr"),
        os.path.join(out_dir, "**", "*index*.exr"),
    ])

    # --- Fallback: try reading passes directly from the Render Result (in-memory) ---
    def _try_read_render_pass_first_channel(pass_name_candidates, w, h):
        """Attempt to read a render pass from bpy.data.images['Render Result'].

        Returns array('f') length w*h if found, else None.
        """
        rr = bpy.data.images.get("Render Result")
        if rr is None:
            print(f"[BlenderServer] DEBUG: No 'Render Result' image found")
            return None
        layers = getattr(rr, "layers", None)
        if not layers or len(layers) == 0:
            print(f"[BlenderServer] DEBUG: Render Result has no layers")
            return None
        layer0 = layers[0]
        passes = getattr(layer0, "passes", None)
        if passes is None:
            print(f"[BlenderServer] DEBUG: Layer 0 has no passes attribute")
            return None

        # Build a list of available pass names (debug + matching)
        try:
            available = [p.name for p in passes]
            print(f"[BlenderServer] DEBUG: Available passes in Render Result: {available}")
        except Exception as e:
            print(f"[BlenderServer] DEBUG: Failed to list passes: {e}")
            available = []

        target = None
        # Exact match first
        for cand in pass_name_candidates:
            for p in passes:
                try:
                    if p.name == cand:
                        target = p
                        break
                except Exception:
                    continue
            if target is not None:
                break

        # Case-insensitive / fuzzy match
        if target is None:
            wanted = [c.lower() for c in pass_name_candidates]
            for p in passes:
                try:
                    n = p.name.lower()
                except Exception:
                    continue
                if n in wanted:
                    target = p
                    break

        if target is None:
            # Common fuzzy variants
            for p in passes:
                try:
                    n = p.name.lower()
                except Exception:
                    continue
                if any(k in n for k in ("depth", " z")) and any(c.lower() in ("depth", "z") for c in pass_name_candidates):
                    target = p
                    break
                if "index" in n and any("index" in c.lower() for c in pass_name_candidates):
                    target = p
                    break

        if target is None:
            return None

        rect = getattr(target, "rect", None)
        if rect is None:
            return None

        # `rect` is typically w*h*4 floats; take channel 0
        try:
            rect_list = list(rect)
        except Exception:
            return None

        npx = int(w) * int(h)
        if len(rect_list) == npx:
            return array("f", rect_list)
        if len(rect_list) >= npx * 4:
            out = array("f", [0.0]) * npx
            for i in range(npx):
                out[i] = float(rect_list[i * 4 + 0])
            return out
        return None

    # If compositor EXRs aren't present, attempt Render Result fallback.
    z_pixels_fallback = None
    idx_pixels_fallback = None
    if depth_path is None or index_path is None:
        try:
            z_pixels_fallback = _try_read_render_pass_first_channel(
                ["Depth", "Z"], w0, h0
            )
            idx_pixels_fallback = _try_read_render_pass_first_channel(
                ["IndexOB", "Index Object", "Object Index", "Index Ob", "Index"], w0, h0
            )
        except Exception:
            z_pixels_fallback = None
            idx_pixels_fallback = None

    # If we still have nothing, print a quick directory dump to aid debugging.
    if (depth_path is None and z_pixels_fallback is None) or (index_path is None and idx_pixels_fallback is None):
        try:
            import glob as _glob
            exrs = _glob.glob(os.path.join(out_dir, "**", "*.exr"), recursive=True)
            print("[BlenderServer] DEBUG: EXR candidates under out_dir:", exrs[:20], "(total", len(exrs), ")")
        except Exception:
            pass

    if depth_path is None and z_pixels_fallback is None:
        raise RuntimeError(
            f"Depth not available: no EXR under {depth_dir} (or {out_dir}) and Render Result has no Depth/Z pass."
        )
    if index_path is None and idx_pixels_fallback is None:
        raise RuntimeError(
            f"Index not available: no EXR under {index_dir} (or {out_dir}) and Render Result has no Object Index pass."
        )

    # --- Read EXR using OpenImageIO if available, else via bpy image load ---
    def _read_exr_first_channel(path, w, h):
        # Returns array('f') length w*h (float32)
        if oiio is not None:
            inp = oiio.ImageInput.open(path)
            if inp is None:
                raise RuntimeError(f"OpenImageIO failed to open EXR: {path}")
            spec = inp.spec()
            cw, ch, c = int(spec.width), int(spec.height), int(spec.nchannels)
            pixels = inp.read_image("float")
            inp.close()
            if pixels is None:
                raise RuntimeError(f"OpenImageIO failed to read EXR pixels: {path}")
            if cw != int(w) or ch != int(h):
                raise RuntimeError(f"EXR size mismatch {cw}x{ch} != expected {w}x{h}: {path}")
            # pixels is a flat sequence length w*h*c
            if c <= 0:
                raise RuntimeError(f"EXR has no channels: {path}")
            # Take channel 0. Note: in the OIIO Python bindings, `read_image()` may return
            # a NumPy array (H,W,C) rather than a flat Python list.
            try:
                import numpy as _np
            except Exception:
                _np = None

            if _np is not None:
                arr = _np.asarray(pixels)
                # Normalize to shape (H*W, C)
                if arr.ndim == 3:
                    # (H, W, C)
                    if arr.shape[0] != ch or arr.shape[1] != cw:
                        # Some builds might return (W, H, C)
                        if arr.shape[0] == cw and arr.shape[1] == ch:
                            arr = _np.transpose(arr, (1, 0, 2))
                        else:
                            raise RuntimeError(f"Unexpected EXR array shape {arr.shape} for {cw}x{ch}x{c}: {path}")
                    flat = arr.reshape(-1, arr.shape[2])
                elif arr.ndim == 2:
                    # (H*W, C)
                    flat = arr
                elif arr.ndim == 1:
                    # flat sequence length H*W*C
                    flat = arr.reshape(-1, c)
                else:
                    raise RuntimeError(f"Unexpected EXR array ndim {arr.ndim} shape {arr.shape}: {path}")

                if flat.shape[1] < 1:
                    raise RuntimeError(f"EXR has no channels after reshape: {path}")
                ch0 = flat[:, 0].astype(_np.float32, copy=False)
                return array("f", ch0.ravel().tolist())

            # No numpy: fall back to treating pixels as a flat sequence
            out = array("f", [0.0]) * (int(w) * int(h))
            for i in range(int(w) * int(h)):
                out[i] = float(pixels[i * c + 0])
            return out

        # Fallback: let Blender load it
        img = None
        try:
            img = bpy.data.images.load(path, check_existing=False)
            cw, ch = img.size
            if int(cw) != int(w) or int(ch) != int(h):
                raise RuntimeError(f"EXR size mismatch {cw}x{ch} != expected {w}x{h}: {path}")
            p = img.pixels[:]
            # Blender images are RGBA-like flat buffers even for EXR; take first channel
            n = int(w) * int(h)
            if len(p) < n:
                raise RuntimeError(f"EXR pixel buffer too small ({len(p)}) for {n} pixels: {path}")
            out = array("f", [0.0]) * n
            if len(p) == n:
                # single channel
                out = array("f", p)
            else:
                # assume at least 4 channels
                for i in range(n):
                    out[i] = float(p[i * 4 + 0])
            return out
        finally:
            try:
                if img is not None:
                    bpy.data.images.remove(img)
            except Exception:
                pass

    if depth_path is not None:
        z_pixels = _read_exr_first_channel(depth_path, w0, h0)
    else:
        z_pixels = z_pixels_fallback

    if index_path is not None:
        idx_float = _read_exr_first_channel(index_path, w0, h0)
    else:
        idx_float = idx_pixels_fallback

    # idx_float is float values like 1.0, 2.0... keep as float array for packing path
    return (int(w0), int(h0), pix, z_pixels, idx_float)

def _rgba_float_to_png_bytes(width, height, rgba_floats):
    """Convert float RGBA [0..1] to PNG bytes.

    Blender's RNA for `Image.pixels` can be picky about accepting non-list sequences in some
    versions/builds (e.g. `array('f')`). Use `foreach_set` for reliable bulk assignment.
    """
    # Ensure we have a flat float sequence of length width*height*4
    expected_len = int(width) * int(height) * 4
    if rgba_floats is None:
        raise ValueError("rgba_floats is None")

    # `array('f')` supports len() and buffer protocol; Blender sometimes rejects direct assignment.
    n = len(rgba_floats)
    if n != expected_len:
        raise ValueError(
            f"RGBA float buffer has unexpected length {n} (expected {expected_len}) for (w,h)=({width},{height})"
        )

    name = "_TMP_RGBA_OUT"
    if name in bpy.data.images:
        img = bpy.data.images[name]
        # Ensure dimensions match
        try:
            if tuple(img.size) != (int(width), int(height)):
                img.scale(int(width), int(height))
        except Exception:
            # If scale is unavailable or fails, recreate
            try:
                bpy.data.images.remove(img)
            except Exception:
                pass
            img = bpy.data.images.new(name=name, width=int(width), height=int(height), alpha=True)
    else:
        img = bpy.data.images.new(name=name, width=int(width), height=int(height), alpha=True)

    # Assign pixels robustly
    try:
        img.pixels.foreach_set(rgba_floats)
    except Exception:
        # Fallback: force a plain Python list
        img.pixels = list(rgba_floats)

    # Save to temp file and read bytes back
    import tempfile, os
    fd, path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    img.filepath_raw = path
    img.file_format = "PNG"
    img.save()

    with open(path, "rb") as f:
        data = f.read()
    try:
        os.remove(path)
    except OSError:
        pass
    return data

def _pack_float32_zlib(floats_arr):
    # floats_arr is array('f') float32
    raw = floats_arr.tobytes()
    return zlib.compress(raw, level=3)

def _pack_int32_zlib_from_float_indices(idx_float_arr):
    # idx pass often comes as float values like 1.0, 2.0...
    # Convert to int32.
    import math
    ints = array("i", [int(math.floor(x + 0.5)) for x in idx_float_arr])
    return zlib.compress(ints.tobytes(), level=3)

# --------------------------
# Simulation / action helpers
# --------------------------
def get_camera(camera_name):
    cam = bpy.data.objects.get(camera_name)
    if cam is None:
        bpy.ops.object.camera_add()
        cam = bpy.context.object
        cam.name = camera_name
    bpy.context.scene.camera = cam
    return cam

def apply_action(cam, action):
    """
    action: dict
      { "type": "camera_delta",
        "dpos": [dx, dy, dz],
        "drot_euler": [droll, dpitch, dyaw] }
    """
    if action is None:
        return
    if action.get("type") != "camera_delta":
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
    bpy.context.scene.frame_set(frame_start)

def server_loop(host, port, camera_name, width, height):
    set_object_indices()
    try:
        setup_passes_and_viewers()
    except Exception as e:
        print(f"[BlenderServer] WARNING: compositor setup failed: {e}")
    cam = get_camera(camera_name)
    reset_scene()

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind((host, port))
    sock.listen(1)
    print(f"[BlenderServer] Listening on {host}:{port}")

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
                w, h, rgba_f, z_f, idx_f = _render_and_extract(width, height)

                png_bytes = _rgba_float_to_png_bytes(w, h, rgba_f)
                depth_blob = _pack_float32_zlib(z_f)
                idx_blob = _pack_int32_zlib_from_float_indices(idx_f)

                resp = {
                    "ok": True,
                    "width": w,
                    "height": h,
                    "rgba_png_b64": base64.b64encode(png_bytes).decode("ascii"),
                    "depth_f32_zlib_b64": base64.b64encode(depth_blob).decode("ascii"),
                    "index_i32_zlib_b64": base64.b64encode(idx_blob).decode("ascii"),
                }
                send_msg(conn, resp)
                continue

            send_msg(conn, {"ok": False, "error": f"Unknown cmd: {cmd}"})

    finally:
        try:
            conn.close()
        except Exception:
            pass
        sock.close()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--camera", default="MontyCam")
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)

    # Blender passes args; user args after '--'
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