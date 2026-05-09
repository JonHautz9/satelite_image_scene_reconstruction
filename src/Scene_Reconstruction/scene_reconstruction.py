import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from skimage.morphology import binary_closing, disk, remove_small_objects

def ensure_2d(array, name="array"):
    arr = np.asarray(array)
    if arr.ndim == 3:
        return arr[:, :, 0]
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D or 3D, got shape {arr.shape}.")
    return arr

def normalize_depth_input(arr, convention="closer_higher", invert=False, clip_percentiles=(1, 99)):
    """
    Normalize an arbitrary depth/disparity map into a 0-1 height map used by the
    rest of the pipeline.

    convention:
      closer_higher -> larger values already mean closer / higher
      closer_lower  -> larger values mean farther, so flip after normalization
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

# Mask + Height Processing

def prepare_binary_mask(mask, threshold=0.5, min_size=30, closing_radius=2):
    # Convert an input mask into a cleaned binary mask.

    mask = ensure_2d(mask, name="mask")
    binary = mask > threshold
    binary = binary_closing(binary, footprint=disk(closing_radius))
    binary = remove_small_objects(binary, min_size=min_size)
    return binary



def prepare_height_map(height_map, smooth=False, sigma=1.0):
    # Prepare a height map by converting it to 2D and optionally smoothing it.
    
    hm = ensure_2d(height_map, name="height_map").astype(float)
    hm[~np.isfinite(hm)] = 0.0

    if smooth:
        hm = ndimage.gaussian_filter(hm, sigma=sigma)

    return hm

def prepare_generic_inputs(raw_height_map,
                           rgb_image=None,
                           convention="closer_higher",
                           invert=False,
                           clip_percentiles=(1, 99),
                           smooth=False,
                           sigma=1.0):
    """
    Generic preprocessing entry point for arbitrary depth/disparity inputs.
    """
    hm = normalize_depth_input(
        raw_height_map,
        convention=convention,
        invert=invert,
        clip_percentiles=clip_percentiles,
    )
    hm = prepare_height_map(hm, smooth=smooth, sigma=sigma)

    rgb = None
    if rgb_image is not None:
        rgb = align_rgb_to_height(rgb_image, hm)

    return hm, rgb

# Object extraction

def label_connected_regions(binary_mask):
    labeled, num_objects = ndimage.label(binary_mask)
    return labeled, num_objects


def extract_objects_from_mask(binary_mask, min_size=30):
    labeled_mask, num_objects = label_connected_regions(binary_mask)
    object_data = []

    for obj_id in range(1, num_objects + 1):
        obj_pixels = labeled_mask == obj_id
        num_pixels = int(np.sum(obj_pixels))

        if num_pixels < min_size:
            continue

        coords = np.argwhere(obj_pixels)
        y_coords = coords[:, 0]
        x_coords = coords[:, 1]

        bbox = {
            "min_row": int(np.min(y_coords)),
            "max_row": int(np.max(y_coords)),
            "min_col": int(np.min(x_coords)),
            "max_col": int(np.max(x_coords)),
        }

        centroid = {
            "row": float(np.mean(y_coords)),
            "col": float(np.mean(x_coords)),
        }

        object_data.append({
            "id": obj_id,
            "mask": obj_pixels,
            "bbox": bbox,
            "centroid": centroid,
            "num_pixels": num_pixels,
        })

    filtered_labeled = np.zeros_like(labeled_mask)
    relabeled_objects = []

    for new_id, obj in enumerate(object_data, start=1):
        filtered_labeled[obj["mask"]] = new_id
        obj["id"] = new_id
        relabeled_objects.append(obj)

    return filtered_labeled, relabeled_objects


def assign_heights_to_objects(object_data, height_map, method="avg"):
    hm = prepare_height_map(height_map)

    for obj in object_data:
        values = hm[obj["mask"]]
        values = values[np.isfinite(values)]

        if len(values) == 0:
            rep_height = 0.0
        elif method == "avg":
            rep_height = float(np.mean(values))
        elif method == "max":
            rep_height = float(np.max(values))
        elif method == "median":
            rep_height = float(np.median(values))
        else:
            raise ValueError("method must be 'avg', 'max', or 'median'.")

        obj["height"] = rep_height

    return object_data


def objects_to_height_map(image_shape, object_data):
    H, W = image_shape[:2]
    object_height_map = np.zeros((H, W), dtype=float)

    for obj in object_data:
        object_height_map[obj["mask"]] = obj.get("height", 0.0)

    return object_height_map


# Reconstruction modes

def reconstruct_from_roof_mask(roof_mask, default_height=10.0, min_size=30, roof_threshold=0.5):
    roof_binary = prepare_binary_mask(
        roof_mask,
        threshold=roof_threshold,
        min_size=min_size
    )
    labeled_roofs, object_data = extract_objects_from_mask(roof_binary, min_size=min_size)

    for obj in object_data:
        obj["height"] = float(default_height)

    object_height_map = objects_to_height_map(roof_binary.shape, object_data)

    return {
        "mode": "roof_only",
        "binary_mask": roof_binary,
        "labeled_objects": labeled_roofs,
        "object_data": object_data,
        "object_height_map": object_height_map,
    }


def reconstruct_from_height_map(height_map, min_size=30, height_threshold=None, threshold_percentile=75):
    hm = prepare_height_map(height_map)

    nonzero = hm[hm > 0]
    if height_threshold is None:
        if len(nonzero) == 0:
            height_threshold = 0.0
        else:
            height_threshold = np.percentile(nonzero, threshold_percentile)

    elevated_mask = hm > height_threshold
    elevated_mask = remove_small_objects(elevated_mask, min_size=min_size)

    labeled_regions, object_data = extract_objects_from_mask(elevated_mask, min_size=min_size)
    object_data = assign_heights_to_objects(object_data, hm, method="avg")
    object_height_map = objects_to_height_map(hm.shape, object_data)

    return {
        "mode": "height_only",
        "binary_mask": elevated_mask,
        "labeled_objects": labeled_regions,
        "object_data": object_data,
        "object_height_map": object_height_map,
        "height_threshold": float(height_threshold),
    }


def reconstruct_terrain_from_height_map(height_map,
                                        min_height=0.0,
                                        smooth=False,
                                        sigma=1.0,
                                        sample_step=8,
                                        smooth_method="gaussian",
                                        use_otsu=False):
    """
    Terrain reconstruction that preserves local surface variation.
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
        "mode": "terrain",
        "binary_mask": terrain_mask,
        "labeled_objects": None,
        "object_data": [],
        "object_height_map": hm * terrain_mask,
        "point_cloud": point_cloud,
    }


def reconstruct_combined(height_map, roof_mask, vertical_mask=None,
                         min_size=30, roof_threshold=0.5,
                         height_method="avg", use_vertical_refinement=False):
    hm = prepare_height_map(height_map)
    roof_binary = prepare_binary_mask(
        roof_mask,
        threshold=roof_threshold,
        min_size=min_size
    )

    if use_vertical_refinement and vertical_mask is not None:
        vm = prepare_binary_mask(
            vertical_mask,
            threshold=roof_threshold,
            min_size=min_size
        )
        roof_binary = np.logical_or(roof_binary, vm)
        roof_binary = remove_small_objects(roof_binary, min_size=min_size)

    labeled_roofs, object_data = extract_objects_from_mask(roof_binary, min_size=min_size)
    object_data = assign_heights_to_objects(object_data, hm, method=height_method)
    object_height_map = objects_to_height_map(hm.shape, object_data)

    return {
        "mode": "combined",
        "binary_mask": roof_binary,
        "labeled_objects": labeled_roofs,
        "object_data": object_data,
        "object_height_map": object_height_map,
    }

def reconstruct_combined_surface(height_map, roof_mask, vertical_mask=None,
                                 min_size=30, roof_threshold=0.5,
                                 use_vertical_refinement=False):
    hm = prepare_height_map(height_map)
    roof_binary = prepare_binary_mask(
        roof_mask,
        threshold=roof_threshold,
        min_size=min_size
    )

    if use_vertical_refinement and vertical_mask is not None:
        vm = prepare_binary_mask(
            vertical_mask,
            threshold=roof_threshold,
            min_size=min_size
        )
        roof_binary = np.logical_or(roof_binary, vm)
        roof_binary = remove_small_objects(roof_binary, min_size=min_size)

    masked_height_map = hm * roof_binary

    return {
        "mode": "combined_surface",
        "binary_mask": roof_binary,
        "labeled_objects": None,
        "object_data": [],
        "object_height_map": masked_height_map,
    }

# 3D visualization helpers

def build_voxel_scene(object_height_map, height_scale=1.0):
    if height_scale <= 0:
        raise ValueError("height_scale must be positive.")

    hm = np.asarray(object_height_map, dtype=float)
    scaled = np.maximum(0, np.round(hm / height_scale).astype(int))
    Z = int(scaled.max()) + 1 if scaled.size else 1
    voxels = np.arange(Z)[None, None, :] < scaled[:, :, None]
    return voxels


def build_point_cloud(object_height_map, sample_step=1):
    hm = np.asarray(object_height_map, dtype=float)
    points = []

    H, W = hm.shape
    for r in range(0, H, sample_step):
        for c in range(0, W, sample_step):
            z = hm[r, c]
            if z > 0:
                points.append([c, r, z])

    if len(points) == 0:
        return np.zeros((0, 3), dtype=float)

    return np.array(points, dtype=float)


def get_object_heights_from_clicks(selected_points, labeled_objects, object_data):
    object_lookup = {obj["id"]: obj for obj in object_data}
    H, W = labeled_objects.shape
    heights = []

    for x, y in selected_points:
        if not (0 <= x < W and 0 <= y < H):
            heights.append(None)
            continue

        obj_id = labeled_objects[y, x]
        if obj_id == 0:
            heights.append(None)
        else:
            heights.append(object_lookup[obj_id]["height"])

    return heights

# Texture / RGB Decoration 

def load_rgb_image(path, target_shape=None):
    """Load RGB and optionally resize to match a (H, W) target."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if target_shape is not None:
        H, W = target_shape[:2]
        if img.shape[:2] != (H, W):
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    return img


def align_rgb_to_height(rgb, height_map):
    """Resize RGB to match height_map's (H, W). No-op if already aligned."""
    H, W = height_map.shape[:2]
    if rgb.shape[:2] == (H, W):
        return rgb
    return cv2.resize(rgb, (W, H), interpolation=cv2.INTER_AREA)


def colorize_point_cloud(point_cloud, rgb_image):
    """Sample RGB at each point's (x, y) -> (N, 3) uint8 colors.

    point_cloud columns are (x=col, y=row, z=height).
    """
    if len(point_cloud) == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    H, W = rgb_image.shape[:2]
    cs = np.clip(point_cloud[:, 0].astype(int), 0, W - 1)
    rs = np.clip(point_cloud[:, 1].astype(int), 0, H - 1)
    return rgb_image[rs, cs]


def colorize_objects(object_data, rgb_image, method="median"):
    """Assign each object a representative RGB sampled from its mask."""
    for obj in object_data:
        pixels = rgb_image[obj["mask"]]
        if len(pixels) == 0:
            obj["color"] = np.array([128, 128, 128], dtype=np.uint8)
            continue

        if method == "median":
            obj["color"] = np.median(pixels, axis=0).astype(np.uint8)
        elif method == "avg":
            obj["color"] = np.mean(pixels, axis=0).astype(np.uint8)
        else:
            raise ValueError("method must be 'median' or 'avg'.")

    return object_data



def apply_texture(results, rgb_image, height_map=None,
                  texture_objects=True, texture_point_cloud=True):
    ref = height_map if height_map is not None else results.get("object_height_map")
    if ref is None:
        raise ValueError("A reference height or mask is required to align the RGB image.")

    rgb = align_rgb_to_height(rgb_image, ref)

    if texture_point_cloud and "point_cloud" in results:
        results["point_cloud_colors"] = colorize_point_cloud(results["point_cloud"], rgb)

    if texture_objects and results.get("object_data"):
        results["object_data"] = colorize_objects(results["object_data"], rgb)

    results["rgb_image"] = rgb
    return results

# Visualization

def plot_2d_diagnostics(height_map=None, binary_mask=None, labeled_objects=None,
                        object_height_map=None, title_prefix="",
                        save_path=None, show=True):
    items = []

    if height_map is not None:
        items.append((prepare_height_map(height_map), "Height Map", "viridis"))
    if binary_mask is not None:
        items.append((binary_mask, "Binary Object Mask", "gray"))
    if labeled_objects is not None:
        items.append((labeled_objects, "Labeled Objects", "nipy_spectral"))
    if object_height_map is not None:
        items.append((object_height_map, "Object Height Map", "magma"))

    n = len(items)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, (img, title, cmap) in zip(axes, items):
        ax.imshow(img, cmap=cmap)
        ax.set_title(f"{title_prefix}{title}")
        ax.axis("off")

    plt.tight_layout()
    save_figure(fig, save_path=save_path, show=show)


def plot_voxel_scene(voxels, face_colors=None):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    if face_colors is not None:
        ax.voxels(voxels, facecolors=face_colors, edgecolor="k")
    else:
        ax.voxels(voxels, edgecolor="k")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title("3D Voxel Reconstruction")
    plt.tight_layout()
    plt.show()


def plot_point_cloud(points, colors=None, color_by_height=True,
                     z_exaggeration=300,
                     point_size=8,
                     alpha=0.95,
                     title="3D Point Cloud Reconstruction",
                     save_path=None,
                     show=True):
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
        ax.scatter(
            x, y, z,
            s=point_size,
            alpha=alpha,
            linewidths=0
        )

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


def plot_terrain_surface(height_map, downsample=4, z_exaggeration=None,
                         cmap="terrain", title="Terrain Surface"):
    hm = prepare_height_map(height_map)
    hm_ds = hm[::downsample, ::downsample]
    H, W = hm_ds.shape
    X, Y = np.meshgrid(np.arange(W) * downsample, np.arange(H) * downsample)

    z_range = max(np.ptp(hm_ds), 1e-6)
    if z_exaggeration is None:
        z_exaggeration = 0.4 * max(W * downsample, H * downsample) / z_range

    fig = plt.figure(figsize=(13, 9))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(
        X,
        Y,
        hm_ds * z_exaggeration,
        cmap=cmap,
        linewidth=0,
        antialiased=True
    )
    ax.set_box_aspect([W * downsample, H * downsample, z_range * z_exaggeration])
    ax.invert_yaxis()
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z (exaggerated)")
    ax.set_title(title)
    plt.colorbar(surf, ax=ax, shrink=0.6, label="Relative height")
    plt.tight_layout()
    plt.show()


def plot_textured_surface(height_map, rgb_image, mask=None,
                          downsample=1,
                          upsample=2,
                          smooth_sigma=1.2,
                          texture_blur=0.6,
                          z_exaggeration=None,
                          title="Textured Terrain Surface",
                          elev=40, azim=-60,
                          save_path=None,
                        show=True):

    # Prepare inputs
    hm = prepare_height_map(height_map).astype(float)
    rgb = align_rgb_to_height(rgb_image, hm)

    # Optional smoothing of geometry
    if smooth_sigma is not None and smooth_sigma > 0:
        hm = ndimage.gaussian_filter(hm, sigma=smooth_sigma)

    # Optional mask handling
    if mask is not None:
        m = align_rgb_to_height(mask.astype(np.uint8), hm).astype(bool)
        # Instead of cutting harshly at the mask, keep outside near zero
        hm = np.where(m, hm, 0.0)

    # Optional texture smoothing
    if texture_blur is not None and texture_blur > 0:
        rgb = ndimage.gaussian_filter(rgb.astype(float), sigma=(texture_blur, texture_blur, 0))
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    # Upsample for smoother rendering
    if upsample > 1:
        H, W = hm.shape
        new_W = W * upsample
        new_H = H * upsample

        hm = cv2.resize(hm, (new_W, new_H), interpolation=cv2.INTER_CUBIC)
        rgb = cv2.resize(rgb, (new_W, new_H), interpolation=cv2.INTER_LINEAR)

    # Downsample for plotting if desired
    hm = hm[::downsample, ::downsample]
    rgb = rgb[::downsample, ::downsample]

    H, W = hm.shape
    X, Y = np.meshgrid(np.arange(W), np.arange(H))

    z_range = max(np.max(hm) - np.min(hm), 1e-6)
    if z_exaggeration is None:
        z_exaggeration = 0.35 * max(W, H) / z_range

    Z = hm * z_exaggeration
    facecolors = rgb.astype(float) / 255.0

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(
        X, Y, Z,
        facecolors=facecolors,
        rstride=1,
        cstride=1,
        linewidth=0,
        antialiased=True,
        shade=False
    )

    ax.set_box_aspect([W, H, max(np.ptp(Z), 1)])
    ax.invert_yaxis()
    ax.view_init(elev=elev, azim=azim)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)

    plt.tight_layout()
    save_figure(fig, save_path=save_path, show=show)


# Reconstruction Scene Function

def reconstruct_scene(height_map=None,
                      roof_mask=None,
                      vertical_mask=None,
                      rgb_image=None,
                      mode="combined",
                      min_size=30,
                      roof_threshold=0.5,
                      default_height=10.0,
                      height_method="avg",
                      height_threshold=None,
                      threshold_percentile=75,
                      use_vertical_refinement=False,
                      build_representation="voxels",
                      height_scale=1.0,
                      point_cloud_sample_step=1,
                      terrain_sample_step=8,
                      terrain_smooth=False,
                      terrain_sigma=1.0,
                      terrain_smooth_method="gaussian",
                      terrain_use_otsu=False,
                      apply_rgb_texture=False,
                      texture_objects=True,
                      texture_point_cloud=True,
                      plot_results=True):
    """
    Master reconstruction function.

    Modes:
    - combined
    - roof_only
    - height_only
    - terrain
    """
    if mode == "combined":
        if height_map is None or roof_mask is None:
            raise ValueError("combined mode requires both height_map and roof_mask.")
        results = reconstruct_combined(
            height_map=height_map,
            roof_mask=roof_mask,
            vertical_mask=vertical_mask,
            min_size=min_size,
            roof_threshold=roof_threshold,
            height_method=height_method,
            use_vertical_refinement=use_vertical_refinement,
        )

    elif mode == "roof_only":
        if roof_mask is None:
            raise ValueError("roof_only mode requires roof_mask.")
        results = reconstruct_from_roof_mask(
            roof_mask=roof_mask,
            default_height=default_height,
            min_size=min_size,
            roof_threshold=roof_threshold,
        )

    elif mode == "height_only":
        if height_map is None:
            raise ValueError("height_only mode requires height_map.")
        results = reconstruct_from_height_map(
            height_map=height_map,
            min_size=min_size,
            height_threshold=height_threshold,
            threshold_percentile=threshold_percentile,
        )

    elif mode == "terrain":
        if height_map is None:
            raise ValueError("terrain mode requires height_map.")
        results = reconstruct_terrain_from_height_map(
            height_map=height_map,
            min_height=height_threshold if height_threshold is not None else 0.0,
            smooth=terrain_smooth,
            sigma=terrain_sigma,
            sample_step=terrain_sample_step,
            smooth_method=terrain_smooth_method,
            use_otsu=terrain_use_otsu,
        )
    
    elif mode == "combined_surface":
        if height_map is None or roof_mask is None:
            raise ValueError("combined_surface mode requires both height_map and roof_mask.")
        results = reconstruct_combined_surface(
            height_map=height_map,
            roof_mask=roof_mask,
            vertical_mask=vertical_mask,
            min_size=min_size,
            roof_threshold=roof_threshold,
            use_vertical_refinement=use_vertical_refinement,
        )

    else:
        raise ValueError("mode must be 'combined', 'combined_surface', 'roof_only', 'height_only', or 'terrain'.")

    object_height_map = results["object_height_map"]

    if build_representation in ("voxels", "both"):
        results["voxels"] = build_voxel_scene(object_height_map, height_scale=height_scale)

    if build_representation in ("point_cloud", "both"):
        if "point_cloud" not in results:
            results["point_cloud"] = build_point_cloud(
                object_height_map,
                sample_step=point_cloud_sample_step
            )

    if apply_rgb_texture:
        if rgb_image is None:
            raise ValueError("apply_rgb_texture=True requires rgb_image.")
        results = apply_texture(
            results,
            rgb_image=rgb_image,
            height_map=height_map,
            texture_objects=texture_objects,
            texture_point_cloud=texture_point_cloud,
        )

    if plot_results:
        plot_2d_diagnostics(
            height_map=height_map,
            binary_mask=results.get("binary_mask"),
            labeled_objects=results.get("labeled_objects"),
            object_height_map=results.get("object_height_map"),
            title_prefix=f"[{results['mode']}] ",
        )

        if build_representation == "voxels":
            plot_voxel_scene(results["voxels"])
        elif build_representation == "point_cloud":
            plot_point_cloud(
                results["point_cloud"],
                colors=results.get("point_cloud_colors")
            )
        elif build_representation == "both":
            plot_voxel_scene(results["voxels"])
            plot_point_cloud(
                results["point_cloud"],
                colors=results.get("point_cloud_colors")
            )

    return results

def save_figure(fig, save_path=None, dpi=300, show=True):
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved figure to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

if __name__ == "__main__":
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    raw = np.load("inputs/mtRushmore_depth_norm.npy")
    height_map = normalize_depth_input(raw, convention="closer_higher")
    rgb = load_rgb_image("inputs/rgb_rushmore.png", target_shape=height_map.shape)

    # raw = np.load("inputs/foellingerSatellite_depth_norm.npy")
    # height_map = normalize_depth_input(raw, convention="closer_higher")
    # rgb = load_rgb_image("inputs/foellinger_satellite.png", target_shape=height_map.shape)

    # raw = np.load("inputs/bigBenCropped2Satellite_depth_norm.npy")
    # height_map = normalize_depth_input(raw, convention="closer_higher")
    # rgb = load_rgb_image("inputs/bigBenCropped2Satellite.png", target_shape=height_map.shape)

    # raw = np.load("inputs/eastMittenButteSatellite_depth_norm.npy")
    # height_map = normalize_depth_input(raw, convention="closer_higher")
    # rgb = load_rgb_image("inputs/eastMittenButteSatellite.png", target_shape=height_map.shape)

    # raw = np.load("inputs/gizaZoomedSatellite_depth_norm.npy")
    # height_map = normalize_depth_input(raw, convention="closer_higher")
    # rgb = load_rgb_image("inputs/GizaZoomedSatellite.png", target_shape=height_map.shape)

    # raw = np.load("inputs/louvreCroppedSatellite_depth_norm.npy")
    # height_map = normalize_depth_input(raw, convention="closer_higher")
    # rgb = load_rgb_image("inputs/louvreCroppedSatellite.png", target_shape=height_map.shape)


    results = reconstruct_scene(
        height_map=height_map,
        rgb_image=rgb,
        mode="terrain",
        terrain_smooth=True,
        terrain_smooth_method="gaussian",
        terrain_sigma=1.0,
        terrain_sample_step=1,
        height_threshold=-1e-6,
        build_representation="point_cloud",
        apply_rgb_texture=True,
        texture_point_cloud=True,
        plot_results=False
    )

    plot_2d_diagnostics(
        height_map=height_map,
        binary_mask=results["binary_mask"],
        object_height_map=results["object_height_map"],
        title_prefix=" ",
        save_path=os.path.join(output_dir, "mtRushmore_diagnostics.png"),
        show=False
    )

    plot_textured_surface(
        height_map=results["object_height_map"],
        rgb_image=rgb,
        mask=None,
        downsample=4,
        upsample=2,
        smooth_sigma=0.9,
        texture_blur=0.35,
        z_exaggeration=220,
        title="improved textured 3D surface",
        elev=38,
        azim=-55,
        save_path=os.path.join(output_dir, "mtRushmore_textured_surface.png"),
        show=False
    )

    plot_point_cloud(
        results["point_cloud"],
        colors=results.get("point_cloud_colors"),
        z_exaggeration=220,
        point_size=6,
        alpha=0.95,
        title="Foellinger — filled textured point cloud",
        save_path=os.path.join(output_dir, "mtRushmore_textured_point_cloud.png"),
        show=False
    )

    print("Height map shape:", height_map.shape)
    print("RGB shape:", rgb.shape)
    print("Point cloud size:", len(results["point_cloud"]))
    print("Has point cloud colors:", "point_cloud_colors" in results)
