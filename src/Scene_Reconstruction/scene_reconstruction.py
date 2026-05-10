import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from scipy.ndimage import binary_dilation
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


# GSD calibration utilities

def measure_pixel_length(image_path):
    """Open an image and let the user click two points; return pixel distance.

    Use this for the Google Earth GSD workflow: measure a real-world feature
    in Google Earth (in feet or meters), then call this to measure the same
    feature in pixels on your satellite image, then divide.
    """
    img = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.imshow(img)
    ax.set_title("Click two endpoints of your reference feature, then close window")
    pts = plt.ginput(2, timeout=0)
    plt.close(fig)
    if len(pts) != 2:
        raise RuntimeError("Need exactly 2 clicks.")
    (x1, y1), (x2, y2) = pts
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5


def compute_gsd(real_world_length, pixel_length):
    """Compute ground sampling distance from a measured reference.

    Parameters
    ----------
    real_world_length : float
        Distance measured in Google Earth (in feet, meters, etc).
    pixel_length : float
        Same distance measured in pixels on the satellite image.

    Returns
    -------
    float
        Ground sampling distance in (real-world units) per pixel. When passed
        as xy_scale to the plotting functions, X and Y will be in the same
        units as the height map's Z, giving metrically faithful renders.
    """
    if pixel_length <= 0:
        raise ValueError("pixel_length must be positive.")
    return real_world_length / pixel_length


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
                     z_exaggeration=None,
                     xy_scale=None,
                     point_size=8,
                     alpha=0.95,
                     title="Textured Point Cloud",
                     save_path=None,
                     show=True):
    """Plot a 3D point cloud.

    If xy_scale is provided (ground sampling distance), X and Y are scaled
    so all three axes share real-world units; Z is left raw. Otherwise the
    pixel coordinates are kept and Z is exaggerated by an auto-scaling
    heuristic so the rendered scene has visually sensible proportions.
    """
    if len(points) == 0:
        print("Point cloud is empty.")
        return

    x = points[:, 0].astype(float)
    y = points[:, 1].astype(float)
    z_raw = points[:, 2]

    if xy_scale is not None:
        x = x * xy_scale
        y = y * xy_scale
        z_exaggeration = 1.0
    elif z_exaggeration is None:
        x_extent = max(np.ptp(x), 1)
        y_extent = max(np.ptp(y), 1)
        z_extent = max(np.ptp(z_raw), 1e-6)
        z_exaggeration = 0.35 * max(x_extent, y_extent) / z_extent

    z = z_raw * z_exaggeration

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

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
    ax.set_zlabel("Z" if xy_scale is not None else "Z exaggerated")
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
                          xy_scale=None,
                          title="Textured 3D Surface",
                          elev=38,
                          azim=-55,
                          save_path=None,
                          show=True):
    """Plot a textured 3D surface from a height map.

    If xy_scale is provided (ground sampling distance), X and Y are scaled
    so all three axes share real-world units; Z is left raw. Otherwise the
    pixel coordinates are kept and Z is exaggerated by an auto-scaling
    heuristic so the rendered scene has visually sensible proportions.
    """
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

    # Track effective xy_scale so it stays consistent through resizing
    effective_xy_scale = xy_scale

    if upsample and upsample > 1:
        H, W = hm.shape
        hm = cv2.resize(hm, (W * upsample, H * upsample), interpolation=cv2.INTER_CUBIC)
        rgb = cv2.resize(rgb, (W * upsample, H * upsample), interpolation=cv2.INTER_LINEAR)
        if m_full is not None:
            m_full = cv2.resize(m_full.astype(np.uint8), (W * upsample, H * upsample),
                                interpolation=cv2.INTER_NEAREST).astype(bool)
        if effective_xy_scale is not None:
            effective_xy_scale = effective_xy_scale / upsample

    if downsample and downsample > 1:
        hm = hm[::downsample, ::downsample]
        rgb = rgb[::downsample, ::downsample]
        if m_full is not None:
            m_full = m_full[::downsample, ::downsample]
        if effective_xy_scale is not None:
            effective_xy_scale = effective_xy_scale * downsample

    # Erode the mask slightly to drop the one-pixel boundary smear left by smoothing
    if m_full is not None and mask_erode and mask_erode > 0:
        m_full = binary_erosion(m_full, footprint=disk(mask_erode))

    if m_full is not None:
        hm = np.where(m_full, hm, np.nan)

    H, W = hm.shape

    # Build coordinate grids — scale by effective GSD if provided for metric output
    if effective_xy_scale is not None:
        X, Y = np.meshgrid(np.arange(W) * effective_xy_scale,
                           np.arange(H) * effective_xy_scale)
    else:
        X, Y = np.meshgrid(np.arange(W), np.arange(H))

    finite = hm[np.isfinite(hm)]
    if finite.size == 0:
        print("Masked surface is empty.")
        return

    height_range = max(np.ptp(finite), 1e-6)

    # Choose Z scaling
    if effective_xy_scale is not None:
        # Metric mode: X/Y already in real units, leave Z raw
        z_exaggeration = 1.0
    elif z_exaggeration is None:
        # Heuristic mode: exaggerate Z to make scene visually balanced
        z_exaggeration = 0.35 * max(W, H) / height_range

    Z = hm * z_exaggeration
    facecolors = rgb.astype(float) / 255.0
    if m_full is not None:
        facecolors = np.dstack([facecolors, m_full.astype(float)])  # alpha = mask

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, facecolors=facecolors,
                    rstride=1, cstride=1, linewidth=0, antialiased=True, shade=False)

    z_span = np.ptp(Z[np.isfinite(Z)]) if np.isfinite(Z).any() else 1.0
    ax.set_box_aspect([
        max(np.ptp(X), 1),
        max(np.ptp(Y), 1),
        max(z_span, 1),
    ])
    ax.invert_yaxis()
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z" if effective_xy_scale is not None else "Z exaggerated")
    ax.set_title(title)
    plt.tight_layout()
    save_figure(fig, save_path=save_path, show=show)


def run_reconstruction_pipeline(
    name,
    height_path,
    rgb_path,
    mask_path,
    output_dir="outputs",
    min_height=-1e-6,
    min_size=30,
    mask_threshold=0.5,
    smooth=True,
    smooth_method="gaussian",
    sigma=1.0,
    sample_step=1,
    z_exaggeration=None,
    xy_scale=None,
    show=False,
    make_plots=True,
):
    """Run the full terrain + masked-terrain reconstruction pipeline.

    Parameters
    ----------
    xy_scale : float or None
        Ground sampling distance (real-world units per pixel). If provided,
        plots are rendered metrically — X, Y, and Z all share the same units.
        If None, plots fall back to a percentile-based auto-scale heuristic.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    height_map = np.load(height_path)
    height_map = prepare_height_map(height_map)
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
            height_map=height_map,
            rgb_image=rgb,
            mask=None,
            downsample=4,
            upsample=2,
            smooth_sigma=0.9,
            texture_blur=0.35,
            z_exaggeration=z_exaggeration,
            xy_scale=xy_scale,
            title=f"{name} - Full Textured Surface",
            elev=38,
            azim=-55,
            save_path=str(full_surface_path),
            show=show,
        )
        output_paths["full_textured_surface"] = full_surface_path

        # Mask-focused textured surface — full scene texture, masked geometry
        if results_mask_focus is not None:
            mask_scene_path = output_dir / f"{name}_mask_scene_textured_surface.png"

            pre_smoothed = ndimage.gaussian_filter(height_map, sigma=0.9)
            m = results_mask_focus["binary_mask"]

            ring = binary_dilation(m, iterations=3) & ~m
            if ring.any():
                base = float(np.median(pre_smoothed[ring]))
            else:
                base = float(pre_smoothed[m].min()) if m.any() else 0.0

            masked_height = np.where(m, pre_smoothed, base)

            plot_textured_surface(
                height_map=masked_height,
                rgb_image=rgb,
                mask=None,
                crop_to_mask=False,
                mask_erode=0,
                downsample=4,
                upsample=2,
                smooth_sigma=0,
                texture_blur=0.35,
                z_exaggeration=None,
                xy_scale=xy_scale,
                title=f"{name} - Mask-Focused Geometry on Full Scene",
                elev=38,
                azim=-55,
                save_path=str(mask_scene_path),
                show=show,
            )
            output_paths["mask_scene_textured_surface"] = mask_scene_path

        # Full textured point cloud
        point_cloud_path = output_dir / f"{name}_full_point_cloud.png"
        plot_point_cloud(
            results_full["point_cloud"],
            colors=results_full.get("point_cloud_colors"),
            z_exaggeration=z_exaggeration,
            xy_scale=xy_scale,
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
    if xy_scale is not None:
        print(f"Metric mode: GSD = {xy_scale:.3f} units/pixel")
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
    # After clicking the two endpoints of your reference feature, the script prints the pixel
    # length; combine it with your Google Earth distance to get GSD.
    pixel_length = measure_pixel_length("inputs/eastMittenButteSatellite.png")
    print(f"Pixel length: {pixel_length:.1f}")
    gsd = compute_gsd(real_world_length=370, pixel_length=pixel_length) # ft or m depending on height_map
    print(f"GSD: {gsd:.3f} ft/pixel")

    run_reconstruction_pipeline(
        name="EastMittenButte",
        height_path="inputs/eastMittenButteSatellite_height_map.npy",
        rgb_path="inputs/eastMittenButteSatellite.png",
        mask_path="inputs/mask_rgb_eastMittenButteSatellite.npy",
        output_dir="outputs",
        min_size=30,
        mask_threshold=0.5,
        smooth=True,
        smooth_method="gaussian",
        sigma=1.0,
        sample_step=1,
        z_exaggeration=None,
        xy_scale=gsd,         # set to None for auto-scale heuristic
        make_plots=True,
        show=False,
    )
