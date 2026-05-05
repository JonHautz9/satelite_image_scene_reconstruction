# from scissors.feature_extraction import Scissors
from PIL import Image
import cv2
import numpy as np
from scissors.gui import get_segment_path
import os
from skimage.draw import polygon2mask
import matplotlib.pyplot as plt

INPUT_IMAGE_DIRECTORY = "input_images"

def get_inside_mask(segment_path, im):   
    segment_path_arr = np.flip(np.array(segment_path))
    mask = polygon2mask(im.shape[:2], segment_path_arr)
    mask = mask.astype(bool)
    # The mask generated can have large unintended gaps
    # These two loops fill the gaps between the first
    # and last instance of 1 in a row or column.
    # NOTE: This may alter the shape of some complex 
    # segements.
    for row in mask:
        first, last = get_first_last_one_index(row)
        if first != None and last != None:
            row[first:last] = 1
    for row in mask.T:
        first, last = get_first_last_one_index(row)
        if first != None and last != None:
            row[first:last] = 1
    return mask

def get_first_last_one_index(arr):
    """Get the first and last index in an array where the 
    value is equal to one.

    Args:
        arr array: array to find first and last one index

    Returns:
        tuple(int): The first and last one index of the array.
    """
    indices = np.where(arr == 1)[0]
    lowest_index, highest_index = None, None

    if indices.size > 0:
        lowest_index = indices[0]      # First occurrence
        highest_index = indices[-1]    # Last occurrence
    return (lowest_index, highest_index)

# Use this to save new file so it can be read by intelligent scissors
def rgb_save(im_file_path, input_image_fn):
    img = Image.open(im_file_path).convert("RGB")
    img.save(os.path.join(os.path.dirname(__file__), INPUT_IMAGE_DIRECTORY, "rgb_" + input_image_fn))

def get_segment(input_image_path):    
    if not os.path.exists(input_image_path):
        print(f"Error: File '{input_image_path}' not found in {os.getcwd()}")
    else:
        path = get_segment_path(input_image_path)
    return path

def main():
    input_image = 'rgb_rushmore.png'
    input_image_path = os.path.join(os.path.dirname(__file__), INPUT_IMAGE_DIRECTORY, input_image)
    # rgb_save(input_image_path, input_image)

    segment_path = get_segment(input_image_path)
    im = cv2.imread(input_image_path)
    inside_mask = get_inside_mask(segment_path, im)
    plt.imshow(inside_mask)
    plt.show()
    save_ready_mask = (inside_mask * 255).astype(np.uint8)

    print(f"inside_mask type: {type(inside_mask)}")
    cv2.imwrite(os.path.join(os.path.dirname(__file__), "mask_" + input_image), save_ready_mask)
   

if __name__ == "__main__":
    main()