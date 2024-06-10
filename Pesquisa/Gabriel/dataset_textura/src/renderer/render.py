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


def render(texture_image_path: str,
           smpl_model_path: str,
           smpl_model_type: str,
           smpl_uv_map_path: str,
           obj_mesh_path: str,
           output_path: str,
           gender: str = 'female',
           views: List[str] = ('frontal', 'side'),
           debug: bool = False,
           dataset: str = 'skeletex',
           image_size: int = 128,
           cam_dist: float = 1.0,
           background_image_path: str | None = None,
           anti_aliasing=False,
           x_axis_weight = 1.0,
           y_axis_weight = 1.0) -> Dict[str,Any]:

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
    #smpl = initialize_smpl(smpl_model_path,gender)

    obj_verts, obj_facets, obj_aux = load_obj(obj_mesh_path)
    at = obj_verts.mean(dim=0)
    at_aux = at.tolist()


    #obj_faces_verts  = obj_facets.textures_idx[None, ...]
    # (1, F, 3)

    ## get the verts and faces from obj and uv map from smpl

    _, smpl_faces, smpl_aux = load_obj(smpl_uv_map_path)
    smpl_verts_uvs = smpl_aux.verts_uvs[None, ...]  # (1, F, 3)
    smpl_faces_uvs = smpl_faces.textures_idx[None, ...]  # (1, F, 3)

    #rotation dict
    rotation_dict: Dict[str, int] = {
        'frontal': 90,
        'side': 0,
        'back': 180
    }  # TODO : fix rotation dict

    RETURN_DATA = {
        "verts": obj_verts,
        "file_numeration": file_numeration,
        "dataset": dataset,
    }

    for m in tqdm(views):
        if m.lower() not in ['frontal', 'side', 'back']:
            logger.error(f"Invalid view {m}")
            continue

        file_name: str = f"{dataset}_{file_numeration}_{m}_render.png"

        logger.info(f"""Rendering {file_name} image ({image_size},{image_size})
                    with texture image {texture_image_path}
                    and smpl model {smpl_model_path}
                    and smpl uv map {smpl_uv_map_path}
                    and obj mesh {obj_mesh_path}
                    and output path {output_path}
                    and orientation {m}""")

        render_mesh_textured(
            verts=obj_verts,
            textures=texture_image,
            verts_uvs=smpl_verts_uvs,
            faces_uvs=smpl_faces_uvs,
            faces_vertices=obj_facets.verts_idx,
            image_size=image_size,  # image resolution  # camera position
            mesh_rot=0,  # mesh rotation in Y axis in degrees
            output_path=os.path.join(output_path,m),
            output_filename=file_name,
            azimut=rotation_dict[m],
            at=obj_verts.mean(dim=0),
            cam_dist=cam_dist,
            background=background_image,
            anti_aliasing=anti_aliasing,
            x_axis_weight=1.0,
            y_axis_weight=1.0,
        )
        eye_position = [
            at_aux[0] + x_axis_weight * (cam_dist * np.cos(np.deg2rad(rotation_dict[m]))),
            at_aux[1] + y_axis_weight * (cam_dist * np.sin(np.deg2rad(rotation_dict[m]))),
            at_aux[2],
        ]
        RETURN_DATA[f"eye_position_{m}"] = eye_position

    RETURN_DATA["at"] = at_aux
    RETURN_DATA["cam_dist"] = cam_dist
    RETURN_DATA['up'] = (0,0,1)



    return RETURN_DATA
