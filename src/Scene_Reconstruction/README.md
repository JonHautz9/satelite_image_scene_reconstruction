# Scene Reconstruction Pipeline

Generates textured 3D reconstructions from a satellite image, a height map,
and an optional object mask. Produces diagnostic plots, textured surfaces,
and a textured point cloud, with optional metric scaling via ground
sampling distance (GSD).

## Files
- `scene_reconstruction.py` — main pipeline (importable functions + script entry point)
- `inputs/` — height maps (`.npy`), RGB satellite images (`.png`), masks (`.npy`)
- `outputs/` — generated diagnostic plots, textured surfaces, and point clouds

## Run
The `__main__` block runs the East Mitten Butte example by default. Edit the
`run_reconstruction_pipeline(...)` call at the bottom of the file to switch
to a different dataset.

## Dependencies
- numpy
- scipy
- opencv-python
- scikit-image
- matplotlib

## Inputs
For each scene, the pipeline expects three files (mask is optional):
- A height map saved as a `.npy` array (2D, real elevations or normalized [0, 1])
- An RGB satellite image (`.png` or any format OpenCV can read)
- A binary mask saved as a `.npy` array, with the subject region as nonzero pixels

## Outputs
For a scene named `<NAME>`, the pipeline writes:
1. `<NAME>_masked_diagnostics.png` — 2D diagnostic plot (height map, reconstruction mask, masked height map)
2. `<NAME>_full_textured_surface.png` — full 3D surface with texture across the whole canvas
3. `<NAME>_mask_scene_textured_surface.png` — mask-focused geometry on a flat ground plane set to the local terrain elevation around the subject
4. `<NAME>_full_point_cloud.png` — textured 3D point cloud of the full scene

## Scaling modes
The renderer supports two ways to size the 3D output:

**Auto-scale (default, `xy_scale=None`):** keeps X/Y as pixel coordinates and
picks a `z_exaggeration` factor from the ratio of pixel dimensions to height
range. Works on any height map (normalized [0, 1] or real elevations) without
manual tuning. Visually sensible but not metrically accurate.

**Metric (`xy_scale=GSD`):** scales X/Y by the ground sampling distance so
all three axes share real-world units. Produces metrically faithful renders
with true geometric proportions. Requires measuring GSD once per dataset.

To measure GSD:
1. Pick a feature visible in both Google Earth and your satellite image
   (e.g., a road segment, building edge).
2. Measure its real-world length in Google Earth.
3. Call `measure_pixel_length(image_path)` to click the same feature's
   endpoints in the satellite image and get its pixel length.
4. Compute `gsd = compute_gsd(real_world_length, pixel_length)`.
5. Pass `xy_scale=gsd` to `run_reconstruction_pipeline(...)`.
