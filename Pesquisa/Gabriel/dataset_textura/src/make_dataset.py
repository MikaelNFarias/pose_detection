from pathlib import Path
import os
import json
import sys
import numpy as np
import random
sys.path.append('renderer')
sys.path.append('measurer')

import measurer.measure as ms
from measurer.measurement_definitions import STANDARD_LABELS
import renderer.render as rd
from utils import *

from typing import List,Any,Tuple,Dict
from glob import glob

def save_measurements_to_json(measurements: Dict[Any,Any],
                              json_file_path: str,
                              file_numeration: int,
                              texture: str,
                              background: str) -> None:
    json_file_path = os.path.join(json_file_path,'annotations')
    data_json = {
        "numero_malha": file_numeration,
        "texture": texture,
        "background": background,
        "medidas_antropometricas": measurements,
    }


    with open(json_file_path,'w+') as f:
        json.dump(data_json,f)


logger = setup_logger(debug=False)
def make_dataset(meshes_path: str,
                 textures_path: str,
                 dataset: str = 'skeletex',
                 output_folder = '../data/output',
                 background_folder: str = None,
                 stop_after: int = 10) -> None:


    smpl_model_path = "../sample_data/SMPL/models"
    meshes_path: str = os.path.join(meshes_path, dataset)
    textures : List[str] = glob(os.path.join(textures_path, '*.png'))
    meshes: List[str] = sorted(glob(os.path.join(meshes_path, '*.obj')))

    for idx,mesh in enumerate(meshes):

        if idx >= stop_after:
            break
        texture =  random.choice(textures)
        obj_verts, file_numeration = rd.render(texture_image_path=texture,
                  smpl_model_path=smpl_model_path,
                  smpl_model_type='smpl',
                  smpl_uv_map_path='../sample_data/smpl_uv.obj',
                  obj_mesh_path=mesh,
                  output_path=output_folder,
                  gender='female',
                  cam_dist=2.0,
                  )
        logger.info(f"""Finalizado a renderização do {mesh}""")

        measurer = ms.MeasureBody('smpl')
        measurer.from_verts(verts = obj_verts)

        measurement_names = measurer.all_possible_measurements
        measurer.measure(measurement_names)
        measurer.label_measurements(STANDARD_LABELS)

        measurements = measurer.measurements

        save_measurements_to_json(
            measurements=measurements,
            json_file_path=os.path.join(output_folder, f'{dataset}_{file_numeration}.json'),
            file_numeration=file_numeration,
            texture = texture,
            background= None # TODO precisa implementar funcao de adicionar background
        )
        print("finalizado")





if __name__ == "__main__":
    make_dataset(meshes_path='../data/meshes',
                 textures_path='../data/textures',
                 stop_after=1)
