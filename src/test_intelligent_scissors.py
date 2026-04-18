# from scissors.feature_extraction import Scissors
from PIL import Image
import numpy as np
from scissors.gui import run_demo
import os
from scissors.feature_extraction import Scissors


image_directory = "input_images"
input_image = 'uiuc_rgb_image.png'
input_image_path = os.path.join(os.path.dirname(__file__), image_directory, input_image)


def scissors_extraction():
    # image = ...
    # scissors = Scissors(image)

    # seed_x, seed_y = ...
    # free_x, free_y = ...
    # path = scissors.find_path(seed_x, seed_y, free_x, free_y)
    pass

def rgb_save(im_file):
    img = Image.open(im_file).convert("RGB")
    img.save("rgb_" + im_file)

def demo():    
    if not os.path.exists(input_image_path):
        print(f"Error: File '{input_image_path}' not found in {os.getcwd()}")
    else:
        run_demo(input_image_path)

def main():
    demo()

if __name__ == "__main__":
    main()