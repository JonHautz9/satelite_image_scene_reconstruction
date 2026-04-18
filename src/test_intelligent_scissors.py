# from scissors.feature_extraction import Scissors
from PIL import Image
import numpy as np
from scissors.gui import run_demo
import os

file_name = 'uiuc_rgb_image.png'
# img = Image.open(file_name).convert("RGB")
# img.save("uiuc_rgb_image.png")
if not os.path.exists(file_name):
    print(f"Error: File '{file_name}' not found in {os.getcwd()}")
else:
    run_demo(file_name)
# img = Image.open(file_name).convert('RGB')
# scissors = Scissors(np.asarray(img))

