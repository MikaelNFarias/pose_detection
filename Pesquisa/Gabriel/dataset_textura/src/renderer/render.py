import trimesh
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import vedo
import os
from tqdm import tqdm
import sys
import torch
from render_mesh import render_mesh_textured
import smplx as SMPL
from pytorch3d.io import load_obj
import logging
# Load the OBJ file
import cv2 as cv
import re
from typing import List, Any, Optional, Sequence, Union, Dict,Tuple


sys.path.append('../')

from src.utils import *
from src.measurer.landmark_definitions import SMPL_LANDMARK_INDICES


def render(texture_image_path: str,
           smpl_uv_map_path: str,
           obj_mesh_path: str,
           output_path: str,
           view: str,
           at: Sequence[float],
           debug: bool = False,
           dataset: str = 'skeletex',
           image_size: int = 128,
           sample_number: int = None,
           background_image_path: str | None = None,
           eye_position: Sequence[float] = None,
           fov: float = 60.0,
           landmarks = SMPL_LANDMARK_INDICES,
           draw: bool = False,
    ) -> None:

    """

    :param texture_image_path:
    :param smpl_model_path:
    :param smpl_model_type:
    :param smpl_uv_map_path:
    :param obj_mesh_path:
    :param output_path:
    :param gender:
    :param views:
    :param debug:
    :param dataset:
    :param image_size:
    :param cam_dist:
    :param background_image_path:
    :param anti_aliasing:
    :return: None
    """
    logger: logging.Logger = setup_logger(debug)
    texture_image = load_texture_image(texture_image_path)

    background_image = None

    if background_image_path is not None:
        background_image = cv.imread(background_image_path)
        background_image = cv.resize(background_image, dsize=(image_size, image_size), interpolation=cv.INTER_CUBIC)

    ##get the numeration on texture image file
    file_numeration: str | None = extract_numeration(obj_mesh_path)
    

    obj_verts, obj_facets, obj_aux = load_obj(obj_mesh_path) # load obj verts, facets and aux

    at_aux = at


    _, smpl_faces, smpl_aux = load_obj(smpl_uv_map_path)
    smpl_verts_uvs = smpl_aux.verts_uvs[None, ...]  # (1, F, 3)
    smpl_faces_uvs = smpl_faces.textures_idx[None, ...]  # (1, F, 3)



    if view.lower() not in ('frontal','right','side','left','back'):
        raise ValueError(f'View {view} not recognized')

    file_name: str = f"{dataset}_{file_numeration}_{view}_{sample_number}_render.png"

    logger.info(f"""Rendering {file_name} image ({image_size},{image_size})
                    with texture image {texture_image_path}
                    and smpl uv map {smpl_uv_map_path}
                    and obj mesh {obj_mesh_path}
                    and output path {output_path}
                    and camera_oriented to {at_aux}
                    and orientation {view} \n""")

    render_mesh_textured(
            verts=obj_verts,
            textures=texture_image,
            verts_uvs=smpl_verts_uvs,
            faces_uvs=smpl_faces_uvs,
            faces_vertices=obj_facets.verts_idx,
            image_size=(image_size,image_size),  # image resolution  # camera position
            output_path=os.path.join(output_path,view),
            output_filename=file_name,
            at=at,
            background=background_image,
            eye_position=eye_position,
            fov=fov,
            landmarks_idx=landmarks,
            draw=draw
        )

