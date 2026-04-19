# from scissors.feature_extraction import Scissors
from PIL import Image
import numpy as np
from scissors.gui import run_demo
import os
from scissors.feature_extraction import Scissors
from shapely.geometry import Point, Polygon

image_directory = "input_images"
input_image = 'rgb_rushmore.png'
input_image_path = os.path.join(os.path.dirname(__file__), image_directory, input_image)

def get_inside_points(path, im):
    h, w = im.shape
    # meshgrid is often more intuitive than indices here
    y, x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    points_xy = np.stack([x.ravel(), y.ravel()], axis=1)

    # Create polygon
    poly = Polygon(path)

    # This can  be slow; vectorized techniques (e.g., rasterio.features.rasterize)
    # are better for large polygons.
    mask = [poly.contains(Point(p)) for p in points_xy]
    inside = points_xy[mask]
    return inside

def scissors_extraction(im):
    scissors = Scissors(im)

    seed_x, seed_y = ...
    free_x, free_y = ...
    # Calculate the optimal path between two points
    # Implements the live wire segmentation algorithm
    path = scissors.find_path(seed_x, seed_y, free_x, free_y)
    return path

# Use this to save new file so it can be read by intelligent scissors
def rgb_save(im_file):
    img = Image.open(im_file).convert("RGB")
    img.save(os.path.join(os.path.dirname(__file__), image_directory, "rgb_" + input_image))

def demo():    
    if not os.path.exists(input_image_path):
        print(f"Error: File '{input_image_path}' not found in {os.getcwd()}")
    else:
        run_demo(input_image_path)

def main():
    demo()
    # rgb_save(input_image_path)

if __name__ == "__main__":
    main()