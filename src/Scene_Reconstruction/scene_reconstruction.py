import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from skimage.morphology import binary_closing, binary_erosion, disk, remove_small_objects


# Basic array and image utilities

def ensure_2d(array, name="array"):
    """Return a 2D version of an array.

    If the input has three channels, the first channel is used. This is useful
    for masks saved as RGB images or arrays.
    """
    arr = np.asarray(array)

    if arr.ndim == 3:
        return arr[:, :, 0]

    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D or 3D, got shape {arr.shape}.")

    return arr


def resize_to_shape(array, target_shape, interpolation=cv2.INTER_AREA):
    """Resize a 2D or 3D array to match target_shape=(H, W)."""
    H, W = target_shape[:2]

    if array.shape[:2] == (H, W):
        return array

    return cv2.resize(array, (W, H), interpolation=interpolation)


def normalize_depth_input(arr,
                          convention="closer_higher",
                          invert=False,
                          clip_percentiles=(1, 99)):
    """Normalize a raw depth/disparity map into a 0-1 height map.

    Parameters
    ----------
    arr : array-like
        Raw depth or disparity map.
    convention : str
        "closer_higher" means larger input values already correspond to larger
        reconstructed heights. "closer_lower" flips the normalized map.
    invert : bool
        Optional extra inversion after the convention is applied.
    clip_percentiles : tuple
        Percentiles used to remove extreme outliers before normalization.
    """
    a = ensure_2d(arr, name="depth_input").astype(float)
    a[~np.isfinite(a)] = np.nan

    lo, hi = np.nanpercentile(a, clip_percentiles)
    a = np.clip(a, lo, hi)
    a = (a - lo) / max(hi - lo, 1e-12)

    if convention == "closer_lower":
        a = 1.0 - a
    elif convention != "closer_higher":
        raise ValueError("convention must be 'closer_higher' or 'closer_lower'.")

    if invert:
        a = 1.0 - a

    a[~np.isfinite(a)] = 0.0
    return a


def prepare_height_map(height_map, smooth=False, sigma=1.0, smooth_method="gaussian"):
    """Convert a height map to a clean 2D float array and optionally smooth it."""
    hm = ensure_2d(height_map, name="height_map").astype(float)
    hm[~np.isfinite(hm)] = 0.0

    if smooth:
        if smooth_method == "gaussian":
            hm = ndimage.gaussian_filter(hm, sigma=sigma)
        elif smooth_method == "median":
            size = max(3, int(2 * sigma + 1))
            hm = ndimage.median_filter(hm, size=size)
        else:
            raise ValueError("smooth_method must be 'gaussian' or 'median'.")

    return hm


def load_rgb_image(path, target_shape=None):
    """Load an RGB image and optionally resize it to target_shape=(H, W)."""
    img = cv2.imread(str(path))

    if img is None:
        raise FileNotFoundError(f"Could not read RGB image: {path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if target_shape is not None:
        img = resize_to_shape(img, target_shape, interpolation=cv2.INTER_AREA)

    return img


def load_mask(path, target_shape=None, normalize=True):
    """Load a mask from .npy or image format.

    The returned mask is a 2D float array. If normalize=True, masks stored in
    0-255 format are scaled to 0-1.
    """
    path = Path(path)

    if path.suffix.lower() == ".npy":
        mask = np.load(path)
        mask = ensure_2d(mask, name="mask")
    else:
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Could not read mask: {path}")

    mask = mask.astype(float)

    if normalize and mask.size > 0 and mask.max() > 1.0:
        mask = mask / 255.0

    if target_shape is not None:
        mask = resize_to_shape(mask, target_shape, interpolation=cv2.INTER_NEAREST)

    return mask


def prepare_binary_mask(mask, target_shape=None, threshold=0.5, min_size=30, closing_radius=2):
    """Convert a raw mask into a cleaned boolean mask."""
    m = ensure_2d(mask, name="mask").astype(float)

    if m.size > 0 and m.max() > 1.0:
        m = m / 255.0

    if target_shape is not None:
        m = resize_to_shape(m, target_shape, interpolation=cv2.INTER_NEAREST)

    binary = m > threshold

    if closing_radius is not None and closing_radius > 0:
        binary = binary_closing(binary, footprint=disk(closing_radius))

    if min_size is not None and min_size > 0:
        binary = remove_small_objects(binary, min_size=min_size)

    return binary


# Reconstruction

def reconstruct_terrain_from_height_map(height_map,
                                        mask=None,
                                        min_height=0.0,
                                        min_size=30,
                                        mask_threshold=0.5,
                                        smooth=False,
                                        sigma=1.0,
                                        sample_step=8,
                                        smooth_method="gaussian",
                                        use_otsu=False):
    """
    Terrain reconstruction with optional mask guidance.

    If mask is None, the full height map is reconstructed.
    If mask is provided, reconstruction is restricted to the cleaned mask region.
    """
    hm = prepare_height_map(height_map)

    if smooth:
        if smooth_method == "gaussian":
            hm = ndimage.gaussian_filter(hm, sigma=sigma)
        elif smooth_method == "median":
            size = max(3, int(2 * sigma + 1))
            hm = ndimage.median_filter(hm, size=size)
        else:
            raise ValueError("smooth_method must be 'gaussian' or 'median'.")

    if use_otsu:
        try:
            from skimage.filters import threshold_otsu
            nonzero = hm[hm > 0]
            if nonzero.size:
                min_height = float(threshold_otsu(nonzero))
        except Exception:
            pass

    terrain_mask = hm > min_height

    if mask is not None:
        cleaned_mask = prepare_binary_mask(
            mask,
            threshold=mask_threshold,
            min_size=min_size
        )
        terrain_mask = np.logical_and(terrain_mask, cleaned_mask)
    else:
        cleaned_mask = None

    H, W = hm.shape
    rs, cs = np.mgrid[0:H:sample_step, 0:W:sample_step]

    sampled = terrain_mask[rs, cs]
    rs = rs[sampled]
    cs = cs[sampled]
    zs = hm[rs, cs]

    if rs.size:
        point_cloud = np.column_stack([cs, rs, zs]).astype(float)
    else:
        point_cloud = np.zeros((0, 3), dtype=float)

    return {
        "mode": "masked_terrain" if mask is not None else "terrain",
        "binary_mask": terrain_mask,
        "input_mask": cleaned_mask,
        "object_height_map": hm * terrain_mask,
        "point_cloud": point_cloud,
    }


def colorize_point_cloud(point_cloud, rgb_image):
    """Assign each point the RGB value at its original image coordinate."""
    if len(point_cloud) == 0:
        return np.zeros((0, 3), dtype=np.uint8)

    H, W = rgb_image.shape[:2]
    cols = np.clip(point_cloud[:, 0].astype(int), 0, W - 1)
    rows = np.clip(point_cloud[:, 1].astype(int), 0, H - 1)

    return rgb_image[rows, cols]


def apply_texture(results, rgb_image):
    """Attach aligned RGB image and point-cloud colors to reconstruction results."""
    rgb = resize_to_shape(rgb_image, results["object_height_map"].shape, interpolation=cv2.INTER_AREA)
    results["rgb_image"] = rgb

    if "point_cloud" in results:
        results["point_cloud_colors"] = colorize_point_cloud(results["point_cloud"], rgb)

    return results

# Visualization

def save_figure(fig, save_path=None, dpi=300, show=True):
    """Save and/or display a Matplotlib figure."""
    if save_path is not None:
        folder = os.path.dirname(save_path)
        if folder:
            os.makedirs(folder, exist_ok=True)

        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved figure to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_2d_diagnostics(height_map=None,
                        binary_mask=None,
                        object_height_map=None,
                        title_prefix="",
                        save_path=None,
                        show=True):
    """Plot intermediate 2D arrays used in reconstruction."""
    items = []

    if height_map is not None:
        items.append((prepare_height_map(height_map), "Height Map", "viridis"))

    if binary_mask is not None:
        items.append((binary_mask, "Reconstruction Mask", "gray"))

    if object_height_map is not None:
        items.append((object_height_map, "Masked Height Map", "magma"))

    if not items:
        print("No diagnostic images to plot.")
        return

    fig, axes = plt.subplots(1, len(items), figsize=(5 * len(items), 5))

    if len(items) == 1:
        axes = [axes]

    for ax, (img, title, cmap) in zip(axes, items):
        ax.imshow(img, cmap=cmap)
        ax.set_title(f"{title_prefix}{title}")
        ax.axis("off")

    plt.tight_layout()
    save_figure(fig, save_path=save_path, show=show)


def plot_point_cloud(points,
                     colors=None,
                     color_by_height=True,
                     z_exaggeration=300,
                     point_size=8,
                     alpha=0.95,
                     title="Textured Point Cloud",
                     save_path=None,
                     show=True):
    """Plot a 3D point cloud."""
    if len(points) == 0:
        print("Point cloud is empty.")
        return

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2] * z_exaggeration

    if colors is not None:
        ax.scatter(
            x, y, z,
            c=colors / 255.0,
            s=point_size,
            alpha=alpha,
            linewidths=0
        )
    elif color_by_height:
        sc = ax.scatter(
            x, y, z,
            c=z,
            cmap="terrain",
            s=point_size,
            alpha=alpha,
            linewidths=0
        )
        plt.colorbar(sc, ax=ax, shrink=0.6, label="Relative height")
    else:
        ax.scatter(x, y, z, s=point_size, alpha=alpha, linewidths=0)

    x_range = max(np.ptp(x), 1)
    y_range = max(np.ptp(y), 1)
    z_range = max(np.ptp(z), 1e-6)

    ax.set_box_aspect([x_range, y_range, z_range])
    ax.invert_yaxis()
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z exaggerated")
    ax.set_title(title)

    plt.tight_layout()
    save_figure(fig, save_path=save_path, show=show)


def plot_textured_surface(height_map,
                          rgb_image,
                          mask=None,
                          crop_to_mask=False,
                          crop_padding=20,
                          mask_erode=0,
                          downsample=4,
                          upsample=2,
                          smooth_sigma=0.9,
                          texture_blur=0.35,
                          z_exaggeration=None,
                          title="Textured 3D Surface",
                          elev=38,
                          azim=-55,
                          save_path=None,
                          show=True):

    hm = prepare_height_map(height_map).astype(float)
    rgb = resize_to_shape(rgb_image, hm.shape, interpolation=cv2.INTER_AREA)

    # Prepare mask at full resolution first
    m_full = None
    if mask is not None:
        m_full = prepare_binary_mask(
            mask, target_shape=hm.shape,
            threshold=0.5, min_size=0, closing_radius=0,
        )

    # Crop to mask bounding box before any smoothing/upsampling
    if m_full is not None and crop_to_mask and m_full.any():
        rows = np.where(np.any(m_full, axis=1))[0]
        cols = np.where(np.any(m_full, axis=0))[0]
        r0, r1 = max(rows[0] - crop_padding, 0), min(rows[-1] + crop_padding + 1, hm.shape[0])
        c0, c1 = max(cols[0] - crop_padding, 0), min(cols[-1] + crop_padding + 1, hm.shape[1])
        hm = hm[r0:r1, c0:c1]
        rgb = rgb[r0:r1, c0:c1]
        m_full = m_full[r0:r1, c0:c1]

    # Smooth geometry on real heights (no zero-ramp at the boundary now)
    if smooth_sigma and smooth_sigma > 0:
        hm = ndimage.gaussian_filter(hm, sigma=smooth_sigma)

    if texture_blur and texture_blur > 0:
        rgb = ndimage.gaussian_filter(rgb.astype(float), sigma=(texture_blur, texture_blur, 0))
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    if upsample and upsample > 1:
        H, W = hm.shape
        hm = cv2.resize(hm, (W * upsample, H * upsample), interpolation=cv2.INTER_CUBIC)
        rgb = cv2.resize(rgb, (W * upsample, H * upsample), interpolation=cv2.INTER_LINEAR)
        if m_full is not None:
            m_full = cv2.resize(m_full.astype(np.uint8), (W * upsample, H * upsample),
                                interpolation=cv2.INTER_NEAREST).astype(bool)

    if downsample and downsample > 1:
        hm = hm[::downsample, ::downsample]
        rgb = rgb[::downsample, ::downsample]
        if m_full is not None:
            m_full = m_full[::downsample, ::downsample]

    # Erode the mask slightly to drop the one-pixel boundary smear left by smoothing
    if m_full is not None and mask_erode and mask_erode > 0:
        m_full = binary_erosion(m_full, footprint=disk(mask_erode))

    if m_full is not None:
        hm = np.where(m_full, hm, np.nan)

    H, W = hm.shape
    X, Y = np.meshgrid(np.arange(W), np.arange(H))

    finite = hm[np.isfinite(hm)]
    if finite.size == 0:
        print("Masked surface is empty.")
        return

    height_range = max(np.ptp(finite), 1e-6)
    if z_exaggeration is None:
        z_exaggeration = 0.35 * max(W, H) / height_range

    Z = hm * z_exaggeration
    facecolors = rgb.astype(float) / 255.0
    if m_full is not None:
        facecolors = np.dstack([facecolors, m_full.astype(float)])  # alpha = mask

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, facecolors=facecolors,
                    rstride=1, cstride=1, linewidth=0, antialiased=True, shade=False)

    z_span = np.ptp(Z[np.isfinite(Z)])
    ax.set_box_aspect([W, H, max(z_span, 1)])
    ax.invert_yaxis()
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title(title)
    plt.tight_layout()
    save_figure(fig, save_path=save_path, show=show)


def run_reconstruction_pipeline(
    name="louvre",
    depth_path="inputs/louvreCroppedSatellite_depth_norm.npy",
    rgb_path="inputs/louvreCroppedSatellite.png",
    mask_path="inputs/mask_louvreCroppedSatellite.npy",
    output_dir="outputs",
    depth_convention="closer_higher",
    min_height=-1e-6,
    min_size=30,
    mask_threshold=0.5,
    smooth=True,
    smooth_method="gaussian",
    sigma=1.0,
    sample_step=1,
    z_exaggeration=140,
    show=False,
    make_plots=True,
):
    """Run the full terrain + masked-terrain reconstruction pipeline.

    This function is meant to replace the old script-only main block so the
    pipeline can be imported and called from a Jupyter notebook.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_depth = np.load(depth_path)
    height_map = normalize_depth_input(raw_depth, convention=depth_convention)
    rgb = load_rgb_image(rgb_path, target_shape=height_map.shape)

    if mask_path is not None:
        mask = load_mask(mask_path, target_shape=height_map.shape)
    else:
        mask = None

    # Full terrain reconstruction
    results_full = reconstruct_terrain_from_height_map(
        height_map=height_map,
        mask=None,
        min_height=min_height,
        min_size=min_size,
        mask_threshold=mask_threshold,
        smooth=smooth,
        smooth_method=smooth_method,
        sigma=sigma,
        sample_step=sample_step,
    )
    results_full = apply_texture(results_full, rgb)

    # Mask-focused terrain reconstruction, if a mask was provided
    if mask is not None:
        results_mask_focus = reconstruct_terrain_from_height_map(
            height_map=height_map,
            mask=mask,
            min_height=min_height,
            min_size=min_size,
            mask_threshold=mask_threshold,
            smooth=smooth,
            smooth_method=smooth_method,
            sigma=sigma,
            sample_step=sample_step,
        )
        results_mask_focus = apply_texture(results_mask_focus, rgb)
    else:
        results_mask_focus = None

    output_paths = {}

    if make_plots:
        # Diagnostics for masked reconstruction if available; otherwise use full reconstruction.
        diagnostics_results = results_mask_focus if results_mask_focus is not None else results_full
        diagnostics_path = output_dir / f"{name}_masked_diagnostics.png"
        plot_2d_diagnostics(
            height_map=height_map,
            binary_mask=diagnostics_results["binary_mask"],
            object_height_map=diagnostics_results["object_height_map"],
            title_prefix=f"{name} ",
            save_path=str(diagnostics_path),
            show=show,
        )
        output_paths["diagnostics"] = diagnostics_path

        # Full textured surface
        full_surface_path = output_dir / f"{name}_full_textured_surface.png"
        plot_textured_surface(
            height_map=results_full["object_height_map"],
            rgb_image=rgb,
            mask=None,
            downsample=4,
            upsample=2,
            smooth_sigma=0.9,
            texture_blur=0.35,
            z_exaggeration=z_exaggeration,
            title=f"{name} - Full Textured Surface",
            elev=38,
            azim=-55,
            save_path=str(full_surface_path),
            show=show,
        )
        output_paths["full_textured_surface"] = full_surface_path

        # Mask-focused textured surface
        if results_mask_focus is not None:
            mask_surface_path = output_dir / f"{name}_mask_textured_surface.png"
            plot_textured_surface(
                height_map=height_map,
                rgb_image=rgb,
                mask=results_mask_focus["binary_mask"],
                crop_to_mask=True,        # ← add
                crop_padding=30,          # ← add
                mask_erode=2,             # ← add
                downsample=2,
                upsample=2,
                smooth_sigma=0.9,
                texture_blur=0.35,
                z_exaggeration=z_exaggeration,
                title=f"{name} - Mask-Focused Textured Surface",
                elev=38,
                azim=-55,
                save_path=str(mask_surface_path),
                show=show,
            )
            output_paths["mask_textured_surface"] = mask_surface_path

        # Full textured point cloud
        point_cloud_path = output_dir / f"{name}_full_point_cloud.png"
        plot_point_cloud(
            results_full["point_cloud"],
            colors=results_full.get("point_cloud_colors"),
            z_exaggeration=z_exaggeration,
            point_size=3,
            alpha=0.95,
            title=f"{name} - Full Textured Point Cloud",
            save_path=str(point_cloud_path),
            show=show,
        )
        output_paths["full_point_cloud"] = point_cloud_path

    print("Full mode:", results_full["mode"])
    if results_mask_focus is not None:
        print("Mask mode:", results_mask_focus["mode"])
    print("Height map shape:", height_map.shape)
    print("RGB shape:", rgb.shape)
    print("Full point cloud size:", len(results_full["point_cloud"]))
    if results_mask_focus is not None:
        print("Masked point cloud size:", len(results_mask_focus["point_cloud"]))
    print("Full has point cloud colors:", "point_cloud_colors" in results_full)
    if results_mask_focus is not None:
        print("Masked has point cloud colors:", "point_cloud_colors" in results_mask_focus)
    print("Saved outputs to:", output_dir)

    return {
        "height_map": height_map,
        "rgb": rgb,
        "mask": mask,
        "results_full": results_full,
        "results_mask_focus": results_mask_focus,
        "output_dir": output_dir,
        "output_paths": output_paths,
    }


if __name__ == "__main__":
    run_reconstruction_pipeline(
        name="mtRushmore",
        depth_path="inputs/mtRushmore_depth_norm.npy",
        rgb_path="inputs/rgb_rushmore.png",
        mask_path="inputs/mask_rgb_rushmore.npy",
        output_dir="outputs",
        min_size=30,
        mask_threshold=0.5,
        smooth=True,
        smooth_method="gaussian",
        sigma=1.0,
        sample_step=1,
        z_exaggeration=220,
        make_plots=True,
        show=False
    )
