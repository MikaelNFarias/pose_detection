import trimesh
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import vedo
import os
import torch
from render_mesh import render_mesh_textured
import smplx as SMPL
from pytorch3d.io import load_obj
import logging
# Load the OBJ file
import cv2 as cv
import re
from utils import *
from typing import List,Any,Optional,Sequence, Union, Dict




def render(texture_image_path: str,
           smpl_model_path: str,
           smpl_model_type: str,
           smpl_uv_map_path: str,
           obj_mesh_path: str,
           output_path: str,
           gender: str = 'male',
           mode: List[str] = ['frontal','side'],
           debug: bool = False,
           dataset: str = 'skeletex',
           cam_pos: Sequence = torch.tensor([2,0.35,0]),
           image_size: int = 1024,
           cam_dist: float = 1.0,
           background_image_path: str | None = None) -> None:
    
    """
    Render the texture image on the mesh

    Args:
        texture_image_path: Path to the texture image
        smpl_model_path: Path to the SMPL model
        smpl_model_type: Type of the SMPL model
        smpl_uv_map_path: Path to the SMPL UV map
        obj_mesh_path: Path to the OBJ mesh
        output_path: Path to save the rendered image
        debug: Enable debug mode
        dataset: Dataset name
    
    Returns:
        None """


    logger: logging.Logger = setup_logger(debug)
    texture_image = load_texture_image(texture_image_path)

    if background_image_path is not None:
        background_image = cv.imread(background_image_path)
        background_image = cv.resize(background_image,dsize=(image_size,image_size),interpolation=cv.INTER_CUBIC)

    ##get the numeration on texture image file
    file_numeration: str | None  = extract_numeration(obj_mesh_path)
    #smpl = initialize_smpl(smpl_model_path,gender)

    obj_verts,obj_facets,obj_aux = load_obj(obj_mesh_path)
    #obj_faces_verts  = obj_facets.textures_idx[None, ...]
      # (1, F, 3)


    ## get the verts and faces from obj and uv map from smpl


    _,smpl_faces,smpl_aux = load_obj(smpl_uv_map_path)
    smpl_verts_uvs = smpl_aux.verts_uvs[None, ...]  # (1, F, 3)
    smpl_faces_uvs = smpl_faces.textures_idx[None, ...]  # (1, F, 3)

    #rotation dict
    rotation_dict: Dict[str,int] = { 
        'frontal': 90,
        'side': 0,
        'back': 180
    } # TODO : fix rotation dict

    for m in mode:
        if m.lower() not in ['frontal','side','back']:
            logger.error(f"Invalid mode {m}")
            continue
        
        file_name: str = f"{dataset}_{file_numeration}_{m}_render.png"

        logger.debug(f"""Rendering {file_name} image ({image_size},{image_size})
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
            image_size=image_size,  # image resolution
            cam_pos=cam_pos,  # camera position
            mesh_rot=0,  # mesh rotation in Y axis in degrees
            output_path=output_path,
            output_filename=file_name,
            azimut=rotation_dict[m],
            at = obj_verts.mean(dim=0),
            cam_dist = cam_dist,
            background = background_image if background_image_path is not None else None,
        )


