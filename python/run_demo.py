# python/run_demo.py
import numpy as np
import cv2
from src/client/blender_gym_env import BlenderVisualEnv
from obs_processing import motion_energy, object_signatures
from novelty_labeler import LabelMemory, NoveltyController

def draw_labels(frame, index, labels):
    out = frame.copy()
    for oid, label in labels.items():
        if not label:
            continue
        mask = (index == oid)
        ys, xs = np.where(mask)
        if len(xs) == 0:
            continue
        x0, y0 = int(xs.mean()), int(ys.mean())
        cv2.putText(out, f"{label}", (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    return out

def main():
    env = BlenderVisualEnv(host="127.0.0.1", port=5556, step_frames=1)
    obs, info = env.reset()

    mem = LabelMemory(path="labels.json", sim_thresh=0.92)
    novelty = NoveltyController(mem, stable_steps=15)

    prev_rgba = obs["rgba"]

    try:
        while True:
            # Simple gentle orbit motion (you can replace with a policy)
            action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.02], dtype=np.float32)
            obs, _, _, _, info = env.step(action)

            rgba = obs["rgba"]
            depth = obs["depth"]
            index = obs["index"]
            
            print("rgba min/max:", rgba.min(), rgba.max(), "shape", rgba.shape)
            print("depth min/max:", float(np.nanmin(depth)), float(np.nanmax(depth)))
            print("index unique (first 20):", np.unique(index)[:20])

            # discovery + labeling
            sigs = object_signatures(rgba, index)
            labels = novelty.update_and_maybe_ask(sigs)

            # visualize
            mot = motion_energy(prev_rgba, rgba)
            mot_color = cv2.applyColorMap(mot, cv2.COLORMAP_JET)
            vis = draw_labels(rgba[:, :, :3], index, labels)

            stacked = np.hstack([
                cv2.cvtColor(vis, cv2.COLOR_RGB2BGR),
                mot_color
            ])

            cv2.imshow("Monty View (left) + Motion Energy (right)", stacked)
            prev_rgba = rgba

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    finally:
        env.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()