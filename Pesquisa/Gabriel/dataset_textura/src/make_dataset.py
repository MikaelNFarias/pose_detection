from pathlib import Path
import os
import json
import sys
import numpy as np
import random
from time import time
from typing import List,Any,Tuple,Dict
from glob import glob
sys.path.append('renderer')
sys.path.append('measurer')
import measurer.measure as ms
from measurer.measurement_definitions import STANDARD_LABELS
import renderer.render as rd
from utils import *


from directiories import *



logger = setup_logger(__name__)
@timer_function
def make_dataset(meshes_path: str,
                 textures_path: str,
                 dataset: str = 'skeletex',
                 output_folder: str = OUTPUT_DIR,
                 background_folder: str = None,
                 stop_after: int = 10,
                ) -> None:

    smpl_model_path = SMPL_DIR
    meshes_path: str = os.path.join(meshes_path, dataset)
    textures : List[str] = glob(os.path.join(textures_path, '*.png'))
    meshes: List[str] = sorted(glob(os.path.join(meshes_path, '*.obj')))
    backgrounds : List[str] = glob(os.path.join(background_folder,'*.jpeg')) + \
        glob(os.path.join(background_folder,'*.jpg')) + \
        glob(os.path.join(background_folder,'*.png'))


    for idx,mesh in enumerate(meshes):
        if idx >= stop_after:
            break
        texture =  random.choice(textures)
        background_image = random.choice(backgrounds)

        CAMERA_DISTANCES = np.linspace(2,3.8,15)
        try:
            for _ in range(3):
                cam_dist = random.choice(CAMERA_DISTANCES)
                render_data = rd.render(texture_image_path=texture,
                    smpl_model_path=smpl_model_path,
                    smpl_model_type='smpl',
                    smpl_uv_map_path=os.path.join(SAMPLE_DATA_DIR,'smpl_uv.obj'),
                    obj_mesh_path=mesh,
                    output_path=output_folder,
                    gender='female',
                    cam_dist=cam_dist,
                    background_image_path=background_image,
                )
        except Exception as e:
            logger.error(f"Erro ao renderizar {mesh}")
            logger.error(e)
            continue


    logger.info(f"""Finalizado a renderização do {mesh}""")

    obj_verts = render_data['verts']
    measurer = ms.MeasureBody('smpl')
    measurer.from_verts(verts = obj_verts)
    measurement_names = measurer.all_possible_measurements
    measurer.measure(measurement_names)
    measurer.label_measurements(STANDARD_LABELS)
    measurements = measurer.measurements
    plane_data = measurer.planes_info

    del render_data['verts']
    save_measurements_to_json(
        measurements=measurements,
        json_file_path=os.path.join(output_folder,'annotations'),
        texture = texture,
        background= background_image,
        render_data = render_data,
        plane_data = plane_data
    )








if __name__ == "__main__":
    make_dataset(meshes_path=MESHES_DIR,
                 textures_path=TEXTURES_DIR,
                 background_folder=BACKGROUNDS_DIR,
                 stop_after=2)


    ## cam_dist -> sai do nome da imagem
    ## at -> sai da anotação (media dos vertices)
    ## eye_position = [at[0] + x_axis_weight * (cam_dist * np.cos(np.deg2rad(rotation_dict[m]))),
    ##                at[1] + y_axis_weight * (cam_dist * np.sin(np.deg2rad(rotation_dict[m]))),
    ##                at[2]]