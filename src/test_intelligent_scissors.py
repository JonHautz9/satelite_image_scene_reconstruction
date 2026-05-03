# from scissors.feature_extraction import Scissors
from PIL import Image
import cv2
import numpy as np
from scissors.gui import get_segment_path
import os
from scissors.feature_extraction import Scissors
from shapely.geometry import Point, Polygon

INPUT_IMAGE_DIRECTORY = "input_images"

def get_inside_mask(segment_path, im):   
    coords = [(p[0], p[1]) for p in segment_path]
    poly = Polygon(coords)

    int_coords = np.array(poly.exterior.coords).astype(np.int32)

    mask = np.zeros(im.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [int_coords], 1)
    mask = mask.astype(bool)
    return mask

def scissors_extraction(im):
    scissors = Scissors(im)

    seed_x, seed_y = ...
    free_x, free_y = ...
    # Calculate the optimal path between two points
    # Implements the live wire segmentation algorithm
    path = scissors.find_path(seed_x, seed_y, free_x, free_y)
    return path

# Use this to save new file so it can be read by intelligent scissors
def rgb_save(im_file_path, input_image_fn):
    img = Image.open(im_file_path).convert("RGB")
    img.save(os.path.join(os.path.dirname(__file__), INPUT_IMAGE_DIRECTORY, "rgb_" + input_image_fn))

def get_segment(input_image_path):    
    if not os.path.exists(input_image_path):
        print(f"Error: File '{input_image_path}' not found in {os.getcwd()}")
    else:
        path = get_segment_path(input_image_path)
        # print(f"paths: {path}")
    return path

def main():
    input_image = 'rgb_rushmore.png'
    input_image_path = os.path.join(os.path.dirname(__file__), INPUT_IMAGE_DIRECTORY, input_image)
    # rgb_save(input_image_path, input_image)

    segment_path = get_segment(input_image_path)
    im = cv2.imread(input_image_path)
    print(f"im.shape: {im.shape}")
    inside_mask = get_inside_mask(segment_path, im)
    save_ready_mask = (inside_mask * 255).astype(np.uint8)

    print(f"inside_mask type: {type(inside_mask)}")
    cv2.imwrite(os.path.join(os.path.dirname(__file__), "mask_" + input_image), save_ready_mask)
    # img.save()

    

if __name__ == "__main__":
    main()