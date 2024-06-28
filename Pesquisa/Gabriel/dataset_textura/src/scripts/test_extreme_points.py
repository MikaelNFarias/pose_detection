import os
import sys
import re
import json
from PIL import Image, ImageDraw

sys.path.append("..")

from src.utils import extract_numeration
from src.directories import TRAIN_ANNOTATIONS_DIR,TRAIN_RENDER_ANNOTATION_DIR, TRAIN_OUTPUT_DIR


def load_sample(image_frontal_path: str,
                image_side_path: str,
                annotations_path: str):

    numeration = extract_numeration(image_frontal_path)
    print(numeration)
    frontal = Image.open(image_frontal_path)
    side = Image.open(image_side_path)

    
    render_annotation_frontal = os.path.join(TRAIN_RENDER_ANNOTATION_DIR,f"train_render_{numeration}_frontal_N_1.json")

    render_annotation_side = os.path.join(TRAIN_RENDER_ANNOTATION_DIR, f"train_render_{numeration}_side_N_1.json")

    with open(render_annotation_frontal) as f:
        j = json.load(f)
    j = j['projections']['extremes']
    

    point_size = 5
    draw = ImageDraw.Draw(frontal) 
    for key, value in j.items():
        leftmost = tuple(map(int, value['leftmost']))
        rightmost = tuple(map(int, value['rightmost']))
        draw.ellipse([leftmost[0] - point_size, leftmost[1] - point_size, leftmost[0] + point_size, leftmost[1] + point_size], fill='red')
        draw.ellipse([rightmost[0] - point_size, rightmost[1] - point_size, rightmost[0] + point_size, rightmost[1] + point_size], fill='blue')



    frontal.show()


load_sample(
    image_frontal_path=os.path.join(TRAIN_OUTPUT_DIR, f"frontal/train_000000_frontal_N_1_render.png"),
    image_side_path=os.path.join(TRAIN_OUTPUT_DIR, "side/train_000000_side_N_1_render.png"),
    annotations_path=TRAIN_ANNOTATIONS_DIR

)
    