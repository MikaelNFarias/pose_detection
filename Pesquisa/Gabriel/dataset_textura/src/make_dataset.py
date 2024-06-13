from pathlib import Path
import os
import itertools
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


from directories import *



logger = setup_logger(__name__)
@timer_function
def make_dataset(meshes_path: str,
                 textures_path: str,
                 dataset: str = 'skeletex',
                 output_folder: str = OUTPUT_DIR,
                 background_folder: str = BACKGROUNDS_DIR,
                 smpl_model_path: str = SMPL_DIR,
                 stop_after: int = 10
        )-> None:


    smpl_model_path = smpl_model_path
    background_folder = os.path.join(background_folder,'train')
    meshes_path: str = os.path.join(meshes_path,'train')
    textures : List[str] = glob(os.path.join(textures_path, '*.png'))
    meshes: List[str] = sorted(glob(os.path.join(meshes_path, '*.obj')))
    backgrounds : List[str] = glob(os.path.join(background_folder,'*.jpeg')) + \
        glob(os.path.join(background_folder,'*.jpg')) + \
        glob(os.path.join(background_folder,'*.png'))
    

    # combinations = list(itertools.product(meshes,textures,backgrounds))
    # CAMERA_DISTANCES = np.linspace(2,3.8,15)

    # for _ in range(N):
    #     for combo in combinations:
    #         params ={
    #             'mesh': combo[0],
    #             'texture': combo[1],
    #             'background': combo[2],
    #             'cam_dist': random.choice(CAMERA_DISTANCES),
    #         }

    for idx_mesh,mesh in enumerate(meshes):
        if idx_mesh == stop_after:
            break
        
        texture =  random.choice(textures)
        background_image = random.choice(backgrounds)
        file_numeration = extract_numeration(mesh.split("/")[-1])
        CAMERA_DISTANCES = np.linspace(2,3.8,15)

        try:
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

        finally:
            try:
                if render_data is not None:
                    camera_data = render_data.copy()
                    camera_data = format_floats(camera_data)
                    del camera_data['verts']
                    save_to_json(os.path.join(CAMERA_ANNOTATION_DIR,f"{dataset}_{file_numeration}_{cam_dist:.4f}"),
                                 render_data = camera_data,
                                 background = background_image.split("/")[-1],
                                 texture = texture.split("/")[-1],)
            except Exception as e:
                logger.error(f"Erro ao salvar anotações de camera {mesh}")
                logger.error(e)


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

        save_to_json(
            json_file_path=os.path.join(MEASUREMENTS_ANNOTATION_DIR,f"{dataset}_{file_numeration}"),
            measurements_data=format_floats(measurements),
            file_numeration=file_numeration,
        )


        save_to_json(
            json_file_path=os.path.join(PLANE_ANNOTATION_DIR,f"{dataset}_{file_numeration}"),
            plane_data = format_floats(plane_data),
            file_numeration=file_numeration,
        )








if __name__ == "__main__":
    make_dataset(meshes_path=os.path.join(MESHES_DIR,'skeletex'),
                 textures_path=os.path.join(TEXTURES_DIR,'train'),
                 background_folder=BACKGROUNDS_DIR,
                 stop_after=1)


    #  TODO :
    # VARIAR VALORES (X,Y,Z) DO AT NUMA FAIXA PRE ESTBELECIDA [x]
    # VARIAR DISTÂNCIA DA CÂMERA [x]
    # VARIAR AZIMUT E ELEVAÇÃO [x]

    # VARIAR DISTANCIA FOCAL
    # VARIAR COEFICIENTE DE DISTORCAO