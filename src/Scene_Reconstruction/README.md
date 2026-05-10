# Scene Reconstruction Pipeline

## Files
- scene_reconstruction.py: main pipeline
- inputs/: height maps (.npy), RGB images (.png), masks (.npy)
- outputs/: generated diagnostic plots, point clouds, textured surfaces

## Run
python scene_reconstruction.py

## Dependencies
numpy, scipy, opencv-python, scikit-image, matplotlib

## What it does
Loads a height map + RGB satellite image + optional mask, produces:
1. 2D diagnostic plot (height map, mask, masked height map)
2. Full textured 3D surface
3. Mask-focused 3D surface with full scene texture
4. Textured point cloud
