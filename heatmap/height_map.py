import argparse
import warnings
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
import os
import torch
from transformers import pipeline
import cv2

# loads depth map from cache or runs the model using Depth Anything V2 from huggingface
def get_depth(image_path):
    cache = f"{Path(image_path).stem}_depth_norm.npy"
    if os.path.exists(cache):
        return np.load(cache).astype(np.float32)

    img = Image.open(image_path).convert("RGB")
    pipe = pipeline(
        task="depth-estimation",
        model="depth-anything/Depth-Anything-V2-Large-hf",
        device=0 if torch.cuda.is_available() else -1,
    )

    out = pipe(img)
    depth = np.array(out["depth"], dtype=np.float32)

    depth_normalized = (depth-depth.min()) / (depth.max()-depth.min()+1e-10)

    np.save(cache, depth_normalized)
    return depth_normalized


# using Otsu's thresholding to find the cutoff for binary elevation mask. Doesn't seem to work well.
def get_mask(depth_normalized):
    depth = (depth_normalized * 255).astype(np.uint8)
    thresh_val, raw = cv2.threshold(depth, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = raw.astype(bool)

    return mask, thresh_val / 255.0


# builds height map from reference points
def build_height_map(depth_normalized, refs):
    if not refs:
        return None

    pairs = []
    for x, y, h in refs:
        d = float(depth_normalized[y, x])
        pairs.append((d, h))

    dArray = np.array([p[0] for p in pairs])
    hArray = np.array([p[1] for p in pairs])

    if len(pairs) == 1:
        scale = hArray[0] / dArray[0]
        return (depth_normalized * scale).astype(np.float32)
    else:
        a, b = np.polyfit(dArray, hArray, 1)
        return (a * depth_normalized + b).astype(np.float32)


def query(depth_normalized, height_map, px, py):
    h, w = depth_normalized.shape
    px = max(0, min(px, w - 1))
    py = max(0, min(py, h - 1))
    result = {"pixel": (px, py), "rel": float(depth_normalized[py, px])}
    if height_map is not None:
        result["feet"] = float(height_map[py, px])
    return result


def show_result(image_path, depth_normalized, mask, thresh, height_map, refs, interactive, out_path):
    img = Image.open(image_path).convert("RGB")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    #input image display
    axes[0].imshow(img)
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    #heatmap display
    im1 = axes[1].imshow(depth_normalized, cmap="plasma")
    axes[1].set_title("Depth Heatmap")
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    #elevation mask display
    axes[2].imshow(img)
    axes[2].imshow(mask, alpha=0.5, cmap="Reds")
    axes[2].set_title("Elevation Mask")
    axes[2].axis("off")

    #reference points display
    for x, y, h in refs:
        for ax in axes:
            ax.plot(x, y, "y*", markersize=10, markeredgecolor="black")
        axes[0].annotate(f"{h}ft", xy=(x, y), xytext=(5, -10),textcoords="offset points",fontsize=8, fontweight="bold")

    plt.suptitle("Height Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if interactive:
        markers = []

        def onclick(event):
            if event.inaxes is None or event.button != 1:
                return
            
            px, py = int(event.xdata), int(event.ydata)
            q = query(depth_normalized, height_map, px, py)

            if "feet" in q:
                label = f"({px},{py})  rel={q['rel']:.3f}  ~{q['feet']:.1f}ft"
            else:
                label = f"({px},{py})  rel={q['rel']:.3f}"
            print(f"  {label}")

            for m in markers:
                m.remove()
            markers.clear()
            for ax in axes:
                markers.append(ax.plot(px, py, "cx", markersize=10)[0])

            fig.canvas.draw_idle()

        fig.canvas.mpl_connect("button_press_event", onclick)

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()


def save(mask, height_map, image_path):
    s = Path(image_path).stem
    np.save(f"{s}_mask.npy", mask)
    if height_map is not None:
        np.save(f"{s}_height_map.npy", height_map)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--ref", nargs=3, type=float, metavar=("X", "Y", "HEIGHT"),
                        action="append", default=[])
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()

    refs = [(int(x), int(y), h) for x, y, h in args.ref]
    s = Path(args.image).stem
    out = f"{s}_height_result.png"

    if refs:
        for x, y, h in refs:
            print(f"  ref: ({x}, {y}) = {h}ft\n")
    
    depth_normalized = get_depth(args.image)
    mask, thresh = get_mask(depth_normalized)
    print(f"otsu threshold: {thresh:.3f}  ({mask.mean()*100:.1f}% elevated)")

    height_map = build_height_map(depth_normalized, refs)

    show_result(args.image, depth_normalized, mask, thresh, height_map,
                refs, args.interactive, out)
    save(mask, height_map, args.image)
