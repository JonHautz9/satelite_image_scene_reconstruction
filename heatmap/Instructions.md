# Setup Instructions

## 1. Install dependencies

```bash
pip install -r requirements.txt
```

## 2. Run

```bash
# Basic
python height_map.py your_image.png

# Interactive click-to-query mode
python height_map.py your_image.png --interactive

# With a known reference building height, can also add multiple points
python height_map.py your_image.png --interactive --ref X Y HEIGHT
```

## 3. Saved files

height_map.npy can be loaded and accessed heights[y, x] to get the height at that point. This is saved if reference points are provided.

mask.npy,depth_norm.npy are also indexed [y, x]. depth_norm is the output of the Depth Anything V2 model. Mask tries to get a mask of elevated items but isn't very accurate