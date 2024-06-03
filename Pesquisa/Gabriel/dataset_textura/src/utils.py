from PIL import Image
import numpy as np
import smplx as SMPL
import logging
import re
from typing import List,Any,Optional


def load_texture_image(texture_image_path):
    with Image.open(texture_image_path) as texture_image:
        return np.asarray(texture_image.convert("RGB")).astype(np.float32)

def initialize_smpl(smpl_model_path, gender='male', model_type='smpl'):
    return SMPL.create(
        model_path=smpl_model_path,
        gender=gender,
        model_type=model_type
    )


def extract_numeration(texture_image_path: str) -> Optional[str]:
    match = re.search(r'\d+', texture_image_path)
    if match:
        number_part = match.group(0)
        return number_part
    return None

def setup_logger(debug):
    # Create a logger
    logger = logging.getLogger(__name__)
    # Set the level based on the debug parameter
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    
    # Create a console handler
    ch = logging.StreamHandler()
    # Set the level for the handler
    ch.setLevel(logging.DEBUG if debug else logging.INFO)
    
    # Create a formatter and set it for the handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    
    # Add the handler to the logger
    if not logger.handlers:
        logger.addHandler(ch)
    
    return logger