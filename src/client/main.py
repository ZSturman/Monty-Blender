# src/client/main.py
"""
Main entry point for Blender+Monty Visual Agent.

Run with:
    python -m src.client.main [options]
    
Or:
    python src/client/main.py [options]
"""

import os
import sys
import time
from typing import Dict, Optional

import cv2
import numpy as np

from .config import load_config, create_argument_parser, Config
from .blender_gym_env import BlenderVisualEnv
from .camera import CameraController, MultiCameraController
from .novelty_labeler import LabelMemory, NoveltyController
from .obs_processing import motion_energy, object_signatures, get_object_centroids
from .output import OutputManager


def draw_labels(
    frame: np.ndarray, 
    index: np.ndarray, 
    labels: Dict[int, Optional[str]]
) -> np.ndarray:
    """
    Draw object labels on frame.
    
    Args:
        frame: RGB image (H, W, 3)
        index: Object index map (H, W)
        labels: Dict mapping object_id -> label
        
    Returns:
        Annotated frame
    """
    out = frame.copy()
    
    for oid, label in labels.items():
        if not label:
            continue
        
        mask = (index == oid)
        ys, xs = np.where(mask)
        
        if len(xs) == 0:
            continue
        
        x0, y0 = int(xs.mean()), int(ys.mean())
        
        # Draw background rectangle for readability
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(out, (x0-2, y0-th-2), (x0+tw+2, y0+2), (0, 0, 0), -1)
        
        # Draw label text
        cv2.putText(out, label, (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    return out


def draw_status_overlay(
    frame: np.ndarray,
    step: int,
    max_steps: int,
    camera_name: str,
    phase_name: str,
    motion: float,
    object_count: int,
    labels: Dict[int, Optional[str]],
) -> np.ndarray:
    """
    Draw status overlay on frame.
    
    Args:
        frame: Image to draw on
        step: Current step
        max_steps: Maximum steps
        camera_name: Current camera
        phase_name: Current movement phase
        motion: Motion energy value
        object_count: Number of detected objects
        labels: Current labels
        
    Returns:
        Frame with overlay
    """
    out = frame.copy()
    h, w = out.shape[:2]
    
    # Semi-transparent background for status bar
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
    out = cv2.addWeighted(overlay, 0.5, out, 0.5, 0)
    
    # Status text
    y = 20
    cv2.putText(out, f"Step: {step}/{max_steps}", (10, y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    y += 20
    cv2.putText(out, f"Camera: {camera_name} | Phase: {phase_name}", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    y += 20
    cv2.putText(out, f"Motion: {motion:.4f} | Objects: {object_count}", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Show known labels
    known = [l for l in labels.values() if l]
    if known:
        y += 20
        label_text = f"Labels: {', '.join(known[:5])}"
        if len(known) > 5:
            label_text += f" (+{len(known)-5} more)"
        cv2.putText(out, label_text, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    return out


def run_agent(config: Config, args) -> None:
    """
    Run the visual agent main loop.
    
    Args:
        config: Configuration object
        args: Command-line arguments
    """
    print(f"\n{'='*60}")
    print("Blender+Monty Visual Agent")
    print(f"{'='*60}\n")
    
    # Initialize output manager
    output = OutputManager(
        base_dir=config.output.base_dir,
        save_images=config.output.save_images,
        save_depth=config.output.save_depth,
        save_metadata=config.output.save_metadata,
        image_format=config.output.image_format,
    )
    
    # Initialize environment
    print(f"[Main] Connecting to server at {config.network.host}:{config.network.port}")
    try:
        env = BlenderVisualEnv(
            host=config.network.host,
            port=config.network.port,
            position_scale=config.camera.position_scale,
            rotation_scale=config.camera.rotation_scale,
            max_steps=config.stopping.max_steps,
            motion_threshold=config.stopping.motion_threshold,
            stable_frames=config.stopping.stable_frames,
            min_steps=config.stopping.min_steps,
        )
    except ConnectionError as e:
        print(f"\n[Error] {e}")
        print("\nMake sure the Blender server is running:")
        print("  ./start_server.sh")
        sys.exit(1)
    
    # Initialize camera controller
    camera_ctrl = CameraController(
        presets=config.movement_presets,
        position_scale=config.camera.position_scale,
        rotation_scale=config.camera.rotation_scale,
    )
    
    # Set movement preset if specified
    preset_name = getattr(args, 'preset', 'full_exploration')
    if preset_name != 'full_exploration':
        camera_ctrl.set_preset(preset_name)
    else:
        camera_ctrl.start_exploration()
    
    # Initialize multi-camera support
    cameras = env.get_available_cameras()
    camera_names = [c["name"] for c in cameras]
    print(f"[Main] Available cameras: {camera_names}")
    
    multi_cam = MultiCameraController(
        switch_camera_fn=env.switch_camera,
        cameras=camera_names,
        switch_on_phase=len(camera_names) > 1,
    )
    
    # Initialize labeling system
    labels_path = args.labels_file if hasattr(args, 'labels_file') and args.labels_file else output.labels_path
    label_mem = LabelMemory(path=labels_path, sim_thresh=config.labeling.similarity_threshold)
    novelty = NoveltyController(
        label_mem,
        stable_steps=config.labeling.stable_steps,
        prompt_timeout=config.labeling.prompt_timeout,
        auto_label_prefix=config.labeling.auto_label_prefix,
    )
    
    # Reset environment
    obs, info = env.reset()
    prev_rgba = obs["rgba"].copy()
    
    print(f"[Main] Starting observation loop (max {config.stopping.max_steps} steps)")
    print(f"[Main] Press 'q' to quit, '1-9' to switch cameras, 'l' to force label prompt\n")
    
    current_labels = {}
    step = 0
    stop_reason = None
    
    try:
        while True:
            # Get action from camera controller
            action = camera_ctrl.get_action_array()
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            step = info["step"]
            
            rgba = obs["rgba"]
            depth = obs["depth"]
            index = obs["index"]
            
            # Get motion energy
            motion = info.get("motion_energy", 0.0)
            
            # Extract object signatures and update labels
            sigs = object_signatures(rgba, index, min_pixels=config.labeling.min_pixels)
            current_labels = novelty.update(sigs)
            
            # Count objects
            object_count = len(sigs)
            
            # Save frame if enabled
            if config.output.save_images:
                output.save_frame(
                    rgba=rgba,
                    depth=depth if config.output.save_depth else None,
                    index=index,
                    labels=current_labels,
                    camera_name=multi_cam.current_camera,
                    step=step,
                )
            
            # Display
            if config.display.show_window:
                # Prepare visualization
                vis_frame = rgba[:, :, :3].copy()
                
                # Draw labels on objects
                if config.display.show_labels:
                    vis_frame = draw_labels(vis_frame, index, current_labels)
                
                # Add status overlay
                if config.display.show_status_overlay:
                    vis_frame = draw_status_overlay(
                        vis_frame,
                        step=step,
                        max_steps=config.stopping.max_steps,
                        camera_name=multi_cam.current_camera or config.camera.default,
                        phase_name=camera_ctrl.current_phase_name,
                        motion=motion,
                        object_count=object_count,
                        labels=current_labels,
                    )
                
                # Motion energy visualization
                if config.display.show_motion_energy:
                    mot = motion_energy(prev_rgba, rgba)
                    mot_color = cv2.applyColorMap(mot, cv2.COLORMAP_JET)
                    
                    # Stack views horizontally
                    stacked = np.hstack([
                        cv2.cvtColor(vis_frame, cv2.COLOR_RGB2BGR),
                        mot_color
                    ])
                else:
                    stacked = cv2.cvtColor(vis_frame, cv2.COLOR_RGB2BGR)
                
                # Show window
                cv2.imshow("Blender+Monty Visual Agent", stacked)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    stop_reason = "user_quit"
                    break
                elif key == ord('l'):
                    # Force label prompt
                    if novelty.has_pending_prompts():
                        new_labels = novelty.prompt_for_unknowns(interactive=True)
                        current_labels.update(new_labels)
                elif ord('1') <= key <= ord('9'):
                    # Switch camera
                    cam_idx = key - ord('1')
                    if cam_idx < len(camera_names):
                        multi_cam.switch_to(camera_names[cam_idx])
            
            prev_rgba = rgba.copy()
            
            # Check stopping conditions
            if terminated:
                stop_reason = info.get("stop_reason", "stable_view")
                print(f"\n[Main] Stopped: {stop_reason}")
                break
            
            if truncated:
                stop_reason = info.get("stop_reason", "max_steps")
                print(f"\n[Main] Truncated: {stop_reason}")
                break
            
            # Check exploration completion
            if camera_ctrl.is_exploration_complete:
                stop_reason = "exploration_complete"
                print(f"\n[Main] Exploration sequence complete")
                break
            
            # FPS limiting
            if config.display.fps_limit > 0:
                time.sleep(1.0 / config.display.fps_limit)
    
    except KeyboardInterrupt:
        stop_reason = "interrupted"
        print("\n[Main] Interrupted by user")
    
    finally:
        # Prompt for any pending labels before closing
        if novelty.has_pending_prompts():
            print(f"\n[Main] {novelty.get_pending_count()} objects waiting for labels")
            new_labels = novelty.prompt_for_unknowns(interactive=True)
            current_labels.update(new_labels)
        
        # Save final state
        output.finalize(
            stop_reason=stop_reason or "unknown",
            objects_detected=len(sigs) if 'sigs' in dir() else 0,
            labels_data=label_mem.data,
        )
        
        # Cleanup
        env.close()
        cv2.destroyAllWindows()
    
    print(f"\n{'='*60}")
    print("Session Complete")
    print(f"{'='*60}")
    summary = output.get_summary()
    print(f"  Session ID: {summary['session_id']}")
    print(f"  Total frames: {summary['total_frames']}")
    print(f"  Cameras used: {', '.join(summary['cameras_used']) or 'Default'}")
    print(f"  Labels assigned: {summary['labels_assigned']}")
    print(f"  Output: {summary['session_dir']}")
    print()


def main():
    """Main entry point."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    config = load_config(args)
    
    if args.verbose:
        print(f"[Config] Network: {config.network.host}:{config.network.port}")
        print(f"[Config] Render: {config.render.width}x{config.render.height}, GPU={config.render.gpu}")
        print(f"[Config] Stopping: max_steps={config.stopping.max_steps}")
    
    run_agent(config, args)


if __name__ == "__main__":
    main()
