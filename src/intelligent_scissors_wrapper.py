from PIL import Image
import cv2
import numpy as np
from scissors.gui import get_segment_path
import os
from skimage.draw import polygon2mask
import matplotlib.pyplot as plt
import argparse

INPUT_IMAGE_DIRECTORY = "input_images"
MASKS_DIRECTORY = "masks"

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

def remove_outside_path(mask, sorted_segment_path):
    segment_lines = []
    prev_row = -1
    for t in sorted_segment_path:
        if t[0] == prev_row:
            continue
        segment_line = [(t, p) for p in sorted_segment_path if p[0] == t[0] and not np.array_equal(p, t)]
        if segment_line and len(segment_line) > 1:
            segment_lines.append(segment_line[-1])
        prev_row = t[0]
    for l in segment_lines:
        mask[l[0][0]][:l[0][0]] = 0 # set row up to segment line to 0
        mask[l[0][0]][l[1][1]:] = 0 # set row after segment line to 0

# Use this to save new file so it can be read by intelligent scissors
def rgb_save(im_file_path, input_image_fn):
    img = Image.open(im_file_path).convert("RGB")
    save_path = os.path.join(os.path.dirname(__file__), INPUT_IMAGE_DIRECTORY, "rgb_" + input_image_fn)
    img.save(save_path)
    print(f"Saved RGB image to: {save_path}")
    return save_path, "rgb_" + input_image_fn

def get_segments(input_image_path):
    segment_paths = []
    if not os.path.exists(input_image_path):
        print(f"Error: File '{input_image_path}' not found in {os.getcwd()}")
    else:
        segment_paths = get_segment_path(input_image_path)
    return segment_paths

def generate_segment_mask(input_image_path, input_image_fn):
    segment_paths = get_segments(input_image_path)
    im = cv2.imread(input_image_path)
    inside_masks = []
    for segment_path in segment_paths:
        inside_masks.append(get_inside_mask(segment_path, im))
    inside_mask = np.sum(inside_masks, axis=0)
    save_ready_mask = (inside_mask * 255).astype(np.uint8)
    save_path = os.path.join(os.path.dirname(__file__), MASKS_DIRECTORY, "mask_" + os.path.splitext(input_image_fn)[0])
    np.save(f"{save_path}.npy", save_ready_mask)
    print(f"Saved segment mask image to: {save_path}.npy")

    plt.imshow(inside_mask)
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Create segments of an image to generate a mask of.")
    parser.add_argument("image_path", help="The path to the image file")
    parser.add_argument("--alpha", action="store_true", help="Include this flag to save the output")
    args = parser.parse_args()
    input_image_path = args.image_path
    input_image_fn = os.path.basename(input_image_path)
    if args.alpha:
        input_image_path, input_image_fn = rgb_save(input_image_path, input_image_fn)
    generate_segment_mask(input_image_path, input_image_fn)

if __name__ == "__main__":
    main()