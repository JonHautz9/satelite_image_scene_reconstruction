import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy import ndimage
from skimage.morphology import remove_small_objects, binary_closing, disk

def ensure_2d(array, name="array"):
    arr = np.asarray(array)
    if arr.ndim == 3:
        return arr[:, :, 0]
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D or 3D, got shape {arr.shape}.")
    return arr

def normalize_depth_input(arr, convention="closer_higher", invert=False,
                          clip_percentiles=(1, 99)):
    """
    Normalize an arbitrary depth/disparity map into a 0 to 1 'higher = closer'
    height map for the rest of the pipeline.

    convention:
      'closer_higher' — bigger value already means closer (e.g. disparity, MiDaS)
      'closer_lower'  — bigger value means farther (e.g. metric depth in meters)
    invert: force-flip after the convention rule
    clip_percentiles: robust min/max to ignore outliers
    """
    a = ensure_2d(arr).astype(float)
    a[~np.isfinite(a)] = np.nan

    lo, hi = np.nanpercentile(a, clip_percentiles)
    a = np.clip(a, lo, hi)
    a = (a - lo) / max(hi - lo, 1e-12)

    if convention == "closer_lower":
        a = 1.0 - a
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


# object extraction

def label_connected_regions(binary_mask):
    # Label connected components in a binary mask.
    labeled, num_objects = ndimage.label(binary_mask)
    return labeled, num_objects



def extract_objects_from_mask(binary_mask, min_size=30):
    """
    Extract connected objects from a binary mask.

    Returns
    -------
    labeled_mask : np.ndarray
        Integer-labeled image.
    object_data : list of dict
        Each dict contains object ID, mask, bbox, centroid, and pixel count.
    """
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

    # Re-label so IDs match filtered object_data in order
    filtered_labeled = np.zeros_like(labeled_mask)
    relabeled_objects = []

    for new_id, obj in enumerate(object_data, start=1):
        filtered_labeled[obj["mask"]] = new_id
        obj["id"] = new_id
        relabeled_objects.append(obj)

    return filtered_labeled, relabeled_objects



def assign_heights_to_objects(object_data, height_map, method="avg"):
    """
    Assign a representative height to each extracted object.

    Parameters
    ----------
    object_data : list of dict
        Objects with pixel masks.
    height_map : np.ndarray
        2D height map.
    method : str
        'avg', 'max', or 'median'.

    Returns
    object_data : list of dict
        Updated objects with assigned height values.
    """
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
    # Convert object footprints and assigned heights into a 2D object-height map.
    H, W = image_shape[:2]
    object_height_map = np.zeros((H, W), dtype=float)

    for obj in object_data:
        object_height_map[obj["mask"]] = obj.get("height", 0.0)

    return object_height_map


# Reconstruction modes

def reconstruct_from_roof_mask(roof_mask, default_height=10.0, min_size=30, roof_threshold=0.5):
    # Simple reconstruction using only the roof mask.
    # Every connected roof object is assigned the same default height.
    # This is useful when the height map is unavailable or unreliable.
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
    # Simple reconstruction using only the height map.
    # Elevated regions are detected by thresholding the height map. A threshold can
    # be provided directly, or estimated from a percentile of the nonzero heights.

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
    hm = prepare_height_map(height_map)

    if smooth:
        if smooth_method == "gaussian":
            hm = ndimage.gaussian_filter(hm, sigma=sigma)
        elif smooth_method == "median":
            # edge-preserving poor-man's bilateral; size ~ 2*sigma+1
            size = max(3, int(2 * sigma + 1))
            hm = ndimage.median_filter(hm, size=size)
        else:
            raise ValueError("smooth_method must be 'gaussian' or 'median'.")

    if use_otsu:
        from skimage.filters import threshold_otsu
        nonzero = hm[hm > 0]
        if nonzero.size:
            min_height = float(threshold_otsu(nonzero))

    terrain_mask = hm > min_height

    H, W = hm.shape
    rs, cs = np.mgrid[0:H:sample_step, 0:W:sample_step]
    sampled = terrain_mask[rs, cs]
    rs, cs = rs[sampled], cs[sampled]
    zs = hm[rs, cs]
    point_cloud = np.column_stack([cs, rs, zs]).astype(float) if rs.size else np.zeros((0, 3))

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
    """
    Combined reconstruction using both roof mask and height map.

    The roof mask defines the object footprints.
    The height map provides the representative height for each object.
    The vertical mask can optionally be used as a weak refinement cue.
    """
    hm = prepare_height_map(height_map)
    roof_binary = prepare_binary_mask(roof_mask, threshold=roof_threshold,
                                      min_size=min_size)

    if use_vertical_refinement and vertical_mask is not None:
        vm = prepare_binary_mask(vertical_mask, threshold=roof_threshold,
                                 min_size=min_size)
        # Weak refinement: union roof and vertical regions, but keep roof as the
        # main footprint definition. This can help in natural scenes or noisy masks.
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


# 3D Visualization

def build_voxel_scene(object_height_map, height_scale=1.0):
    if height_scale <= 0:
        raise ValueError("height_scale must be positive.")
    height_map = np.asarray(object_height_map, dtype=float)
    scaled = np.maximum(0, np.round(height_map / height_scale).astype(int))
    Z = int(scaled.max()) + 1 if scaled.size else 1
    # broadcast: voxel (r,c,z) is True iff z < scaled[r,c]
    return np.arange(Z)[None, None, :] < scaled[:, :, None]

def build_point_cloud(object_height_map, sample_step=1):
    # Convert the object-height map into a simple point cloud.
    # Each nonzero pixel becomes one 3D point: (x, y, z).
    # This is lighter than a voxel grid and useful for quick visualization.

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


# User-Selected Height Queries

def get_object_heights_from_clicks(selected_points, labeled_objects, object_data):
    """
    Given user-selected (x, y) points, return the height of the object clicked.

    This is more stable than reading the value from a single pixel, because the
    height returned is the representative height of the full connected object.
    """
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

# Visualization

def plot_2d_diagnostics(height_map=None, binary_mask=None, labeled_objects=None,
                        object_height_map=None, title_prefix=""):

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
    plt.show()


def plot_voxel_scene(voxels):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.voxels(voxels, edgecolor="k")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Voxel Reconstruction")
    plt.tight_layout()
    plt.show()


# def plot_point_cloud(points):
#     if len(points) == 0:
#         print("Point cloud is empty.")
#         return

#     fig = plt.figure(figsize=(10, 8))
#     ax = fig.add_subplot(111, projection="3d")
#     ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=3)
#     ax.set_xlabel("X")
#     ax.set_ylabel("Y")
#     ax.set_zlabel("Z")
#     ax.set_title("3D Point Cloud Reconstruction")
#     plt.tight_layout()
#     plt.show()

def plot_point_cloud(points, z_exaggeration=None, color_by_height=True):
    if len(points) == 0:
        print("Point cloud is empty.")
        return

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    if color_by_height:
        sc = ax.scatter(x, y, z, c=z, cmap="terrain", s=3)
        plt.colorbar(sc, ax=ax, shrink=0.6, label="Relative height")
    else:
        ax.scatter(x, y, z, s=3)

    # auto-pick exaggeration so Z is visible relative to XY extent
    x_range = max(np.ptp(x), 1)
    y_range = max(np.ptp(y), 1)
    z_range = max(np.ptp(z), 1e-6)
    if z_exaggeration is None:
        z_exaggeration = 0.4 * max(x_range, y_range) / z_range
    ax.set_box_aspect([x_range, y_range, z_range * z_exaggeration])
    ax.invert_yaxis()  # image coords: Y grows downward

    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title("3D Point Cloud Reconstruction")
    plt.tight_layout()
    plt.show()

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
    surf = ax.plot_surface(X, Y, hm_ds * z_exaggeration,
                           cmap=cmap, linewidth=0, antialiased=True)
    ax.set_box_aspect([W * downsample, H * downsample,
                       z_range * z_exaggeration])
    ax.invert_yaxis()
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z (exaggerated)")
    ax.set_title(title)
    plt.colorbar(surf, ax=ax, shrink=0.6, label="Relative height")
    plt.tight_layout()
    plt.show()

# Reconstruction Scene Function

def reconstruct_scene(height_map=None,
                      roof_mask=None,
                      vertical_mask=None,
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
                      terrain_sample_step=8,
                      terrain_smooth=False,
                      terrain_sigma=1.0,
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
        )

    else:
        raise ValueError("mode must be 'combined', 'roof_only', 'height_only', or 'terrain'.")

    object_height_map = results["object_height_map"]

    if build_representation in ("voxels", "both"):
        results["voxels"] = build_voxel_scene(object_height_map, height_scale=height_scale)

    if build_representation in ("point_cloud", "both"):
        if "point_cloud" not in results:
            results["point_cloud"] = build_point_cloud(object_height_map)

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
            plot_point_cloud(results["point_cloud"])
        elif build_representation == "both":
            plot_voxel_scene(results["voxels"])
            plot_point_cloud(results["point_cloud"])

    return results


if __name__ == "__main__":
    raw = np.load("mtRushmore_depth_norm.npy")
    height_map = normalize_depth_input(raw, convention="closer_higher")

    results = reconstruct_scene(
        height_map=height_map,
        mode="terrain",
        height_threshold=0.0,
        terrain_sample_step=6,
        terrain_smooth=True,
        terrain_sigma=1.5,
        build_representation="point_cloud",
        plot_results=True, 
    )

    # plot_terrain_surface(height_map, downsample=4, title="Mt. Rushmore — terrain surface")
    # plot_point_cloud(results["point_cloud"])

    print("Height map shape:", height_map.shape)
    print("Height range:", float(height_map.min()), "→", float(height_map.max()))
    print("Point cloud size:", len(results["point_cloud"]))
