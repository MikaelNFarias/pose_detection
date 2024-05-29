import os
import sys

from PIL import Image
def add_background_to_image(image_path,background_path):

    image = Image.open(image_path).convert("RGBA")
    background_image = Image.open(background_path).convert("RGBA")

    combined = Image.alpha_composite(background_image,image)