# blender/compositor_setup.py
import bpy
import os
import tempfile

def setup_passes_and_viewers():
    scene = bpy.context.scene

    # Enable compositor evaluation.
    try:
        scene.render.use_compositing = True
    except Exception:
        pass

    # Debug: what compositor-related attributes exist on this build?
    scene_attrs = {
        "has_use_nodes": hasattr(scene, "use_nodes"),
        "has_node_tree": hasattr(scene, "node_tree"),
        "has_compositor_node_tree": hasattr(scene, "compositor_node_tree"),
        "has_compositing_node_group": hasattr(scene, "compositing_node_group"),
    }
    print("[BlenderServer] Scene compositor attrs:", scene_attrs)

    # For Blender 5.0.1, use multi-layer EXR output instead of compositor nodes
    # This avoids crashes and properly exports all passes in background mode
    
    print("[BlenderServer] Blender 5.0.1: Using multi-layer EXR output for passes")
    
    # Enable render passes at the view layer level
    view_layer = bpy.context.view_layer
    view_layer.use_pass_z = True
    view_layer.use_pass_object_index = True
    view_layer.use_pass_combined = True
    
    print(f"[BlenderServer] Enabled passes: Z={view_layer.use_pass_z}, Index={view_layer.use_pass_object_index}")

    # Render Layers node
    rl = nodes.new(type="CompositorNodeRLayers")
    rl.location = (0, 0)

    def _pick_output(node, preferred_names):
        """Return the first matching output socket by name (robust across Blender versions)."""
        # First try exact matches
        for name in preferred_names:
            if name in node.outputs:
                return node.outputs[name]
        # Then try case-insensitive matches
        preferred_lower = [n.lower() for n in preferred_names]
        for sock in node.outputs:
            if sock.name.lower() in preferred_lower:
                return sock
        # Finally try fuzzy contains matches for object index
        for sock in node.outputs:
            n = sock.name.lower()
            if "index" in n and ("object" in n or n.endswith("ob") or " ob" in n):
                return sock
        return None

    image_out = _pick_output(rl, ["Image", "Combined"])
    depth_out = _pick_output(rl, ["Depth", "Z"])  # some UIs label Z as Depth
    index_ob_out = _pick_output(rl, ["IndexOB", "Index Object", "Object Index", "Index Ob"])

    if image_out is None:
        raise RuntimeError(f"Render Layers node has no Image/Combined output. Outputs: {[s.name for s in rl.outputs]}")
    if depth_out is None:
        raise RuntimeError(f"Render Layers node has no Depth/Z output. Outputs: {[s.name for s in rl.outputs]}")
    if index_ob_out is None:
        raise RuntimeError(
            "Render Layers node has no Object Index output. "
            "Ensure View Layer > Passes > Data > Indexes > Object Index is enabled. "
            f"Outputs: {[s.name for s in rl.outputs]}"
        )

    # Output node to force compositor evaluation in headless/background mode.
    # Some Blender 5 builds appear to not register CompositorNodeComposite, so fall back to a File Output RGB node.
    comp = None
    for t in ("CompositorNodeComposite", "CompositorNodeCompositeLegacy", "CompositorNodeCompositeOutput"):
        try:
            comp = nodes.new(type=t)
            comp.location = (400, 300)
            break
        except Exception:
            comp = None

    if comp is None:
        print("[BlenderServer] WARNING: No Composite node type available; using File Output RGB to force compositor eval")

    # Viewer nodes (one per output)
    viewer_rgb = nodes.new(type="CompositorNodeViewer")
    viewer_rgb.location = (400, 150)
    viewer_rgb.name = "ViewerRGB"
    viewer_rgb.label = "ViewerRGB"
    if hasattr(viewer_rgb, "use_alpha"):
        viewer_rgb.use_alpha = True

    viewer_z = nodes.new(type="CompositorNodeViewer")
    viewer_z.location = (400, 0)
    viewer_z.name = "ViewerZ"
    viewer_z.label = "ViewerZ"
    if hasattr(viewer_z, "use_alpha"):
        viewer_z.use_alpha = True

    viewer_idx = nodes.new(type="CompositorNodeViewer")
    viewer_idx.location = (400, -150)
    viewer_idx.name = "ViewerIndex"
    viewer_idx.label = "ViewerIndex"
    if hasattr(viewer_idx, "use_alpha"):
        viewer_idx.use_alpha = True

    # File Output nodes (reliable in -b background mode; Viewer images may be None)
    out_dir = os.path.join(tempfile.gettempdir(), "monty_blender_out")
    os.makedirs(out_dir, exist_ok=True)

    # Use dedicated subdirectories per output node because Blender 5's
    # CompositorNodeOutputFile API can vary (slot naming, paths, etc.).
    rgb_dir = os.path.join(out_dir, "rgb")
    depth_dir = os.path.join(out_dir, "depth")
    index_dir = os.path.join(out_dir, "index")
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(index_dir, exist_ok=True)

    file_rgb = None
    if comp is None:
        file_rgb = nodes.new(type="CompositorNodeOutputFile")
        file_rgb.location = (650, 150)
        file_rgb.name = "FileOutRGB"
        file_rgb.label = "FileOutRGB"

        # Configure output directory
        if hasattr(file_rgb, "directory"):
            # Some builds require a trailing path separator
            file_rgb.directory = rgb_dir + os.sep
        elif hasattr(file_rgb, "base_path"):
            file_rgb.base_path = rgb_dir
        elif hasattr(file_rgb, "path"):
            file_rgb.path = rgb_dir

        # Configure filename/prefix
        if hasattr(file_rgb, "file_name"):
            file_rgb.file_name = "rgb_"

        # Configure format
        try:
            file_rgb.format.file_format = "PNG"
            file_rgb.format.color_mode = "RGBA"
            file_rgb.format.color_depth = "8"
        except Exception:
            pass

        print("[BlenderServer] FileOutRGB props:", {
            "has_directory": hasattr(file_rgb, "directory"),
            "has_file_name": hasattr(file_rgb, "file_name"),
            "inputs": [s.name for s in getattr(file_rgb, "inputs", [])],
        })

    file_z = nodes.new(type="CompositorNodeOutputFile")
    file_z.location = (650, 0)
    file_z.name = "FileOutZ"
    file_z.label = "FileOutZ"
    print("[BlenderServer] FileOutZ props:", {
        "has_base_path": hasattr(file_z, "base_path"),
        "has_directory": hasattr(file_z, "directory"),
        "has_path": hasattr(file_z, "path"),
        "has_file_name": hasattr(file_z, "file_name"),
        "has_file_slots": hasattr(file_z, "file_slots"),
        "has_slots": hasattr(file_z, "slots"),
        "has_layer_slots": hasattr(file_z, "layer_slots"),
        "inputs": [s.name for s in getattr(file_z, "inputs", [])],
    })

    # Blender 5 API changed OUTPUT_FILE properties.
    # - base_path was renamed (e.g. directory)
    # - file_slots may be renamed/removed
    if hasattr(file_z, "base_path"):
        file_z.base_path = depth_dir
    elif hasattr(file_z, "directory"):
        file_z.directory = depth_dir + os.sep
    elif hasattr(file_z, "path"):
        file_z.path = depth_dir

    # Configure filename/prefix
    if hasattr(file_z, "file_name"):
        file_z.file_name = "depth_"

    slots = None
    if hasattr(file_z, "file_slots"):
        slots = file_z.file_slots
    elif hasattr(file_z, "slots"):
        slots = file_z.slots
    elif hasattr(file_z, "layer_slots"):
        slots = file_z.layer_slots

    if slots is not None and len(slots) > 0:
        try:
            # keep a single slot when API allows
            while len(slots) > 1:
                slots.remove(slots[-1])
        except Exception:
            pass
        try:
            slots[0].path = "depth_"
        except Exception:
            pass

    try:
        file_z.format.file_format = "OPEN_EXR"
        file_z.format.color_mode = "RGB"
        file_z.format.color_depth = "32"
    except Exception:
        pass

    file_idx = nodes.new(type="CompositorNodeOutputFile")
    file_idx.location = (650, -150)
    file_idx.name = "FileOutIndex"
    file_idx.label = "FileOutIndex"
    print("[BlenderServer] FileOutIndex props:", {
        "has_base_path": hasattr(file_idx, "base_path"),
        "has_directory": hasattr(file_idx, "directory"),
        "has_path": hasattr(file_idx, "path"),
        "has_file_name": hasattr(file_idx, "file_name"),
        "has_file_slots": hasattr(file_idx, "file_slots"),
        "has_slots": hasattr(file_idx, "slots"),
        "has_layer_slots": hasattr(file_idx, "layer_slots"),
        "inputs": [s.name for s in getattr(file_idx, "inputs", [])],
    })

    if hasattr(file_idx, "base_path"):
        file_idx.base_path = index_dir
    elif hasattr(file_idx, "directory"):
        file_idx.directory = index_dir + os.sep
    elif hasattr(file_idx, "path"):
        file_idx.path = index_dir

    if hasattr(file_idx, "file_name"):
        file_idx.file_name = "index_"

    slots = None
    if hasattr(file_idx, "file_slots"):
        slots = file_idx.file_slots
    elif hasattr(file_idx, "slots"):
        slots = file_idx.slots
    elif hasattr(file_idx, "layer_slots"):
        slots = file_idx.layer_slots

    if slots is not None and len(slots) > 0:
        try:
            while len(slots) > 1:
                slots.remove(slots[-1])
        except Exception:
            pass
        try:
            slots[0].path = "index_"
        except Exception:
            pass

    try:
        file_idx.format.file_format = "OPEN_EXR"
        file_idx.format.color_mode = "RGB"
        file_idx.format.color_depth = "32"
    except Exception:
        pass

    # Connect passes to viewers
    links.new(image_out, viewer_rgb.inputs["Image"])
    if comp is not None and "Image" in comp.inputs:
        links.new(image_out, comp.inputs["Image"])
    elif file_rgb is not None:
        links.new(image_out, file_rgb.inputs[0])
    links.new(depth_out, viewer_z.inputs["Image"])
    links.new(index_ob_out, viewer_idx.inputs["Image"])
    links.new(depth_out, file_z.inputs[0])
    links.new(index_ob_out, file_idx.inputs[0])

    # Force update (Blender 5.0+ compositor node groups don't expose tree.update()).
    # Touch a scene setting to ensure depsgraph picks up the compositor changes.
    scene.frame_set(scene.frame_current)

def set_object_indices():
    """
    Assign each mesh object a unique pass_index for segmentation.
    You can override these in your .blend if you want.
    """
    idx = 1
    for obj in bpy.data.objects:
        if obj.type == "MESH":
            obj.pass_index = idx
            idx += 1