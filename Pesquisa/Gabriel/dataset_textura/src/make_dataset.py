from pathlib import Path
import os
import json
import sys
import numpy as np
import random
sys.path.append('renderer')
sys.path.append('measurer')
from time import time
import measurer.measure as ms
from measurer.measurement_definitions import STANDARD_LABELS
import renderer.render as rd
from utils import *

from typing import List,Any,Tuple,Dict
from glob import glob

from directiories import *

print(DATA_DIR)
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
def timer_function(func):
    def wrapper(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        print(f"Function {func.__name__!r} executed in {(t2-t1):.4f}s")
        return  result
    return wrapper
def save_measurements_to_json(measurements: Dict[Any,Any],
                              json_file_path: str,
                              texture: str,
                              background: str,
                              unit='cm',
                              render_data= None,
                              plane_data = None,
                              **kwargs) -> None:

    file_numeration = render_data['file_numeration']
    json_file_path = os.path.join(json_file_path,f"{render_data['dataset']}_{file_numeration}_annotation.json")
    data_json = {
        "textura": texture,
        "background": background,
        "medidas_antropometricas": measurements,
        'unidade':unit,

    }
    if render_data is not None:
        ##del render_data['file_numeration']
        render_data = convert_numpy_to_list(render_data)
        data_json.update(render_data)
    if plane_data is not None:
        plane_data = convert_numpy_to_list(plane_data)
        data_json.update(plane_data)



    with open(json_file_path,'w+') as f:
        json.dump(data_json,f, indent=4,cls=NumpyEncoder)

    print(f"Arquivo JSON {json_file_path} salvo com sucesso")


logger = setup_logger(__name__)
@timer_function
def make_dataset(meshes_path: str,
                 textures_path: str,
                 dataset: str = 'skeletex',
                 output_folder: str = '../data/output',
                 background_folder: str = None,
                 stop_after: int = 10,
                ) -> None:

    smpl_model_path = "../sample_data/SMPL/models"
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

        CAMERA_DISTANCES = np.linspace(1.85,3.8,15)
        try:
            for i in range(0,len(CAMERA_DISTANCES),5):
                cam_dist = CAMERA_DISTANCES[i]
                render_data = rd.render(texture_image_path=texture,
                    smpl_model_path=smpl_model_path,
                    smpl_model_type='smpl',
                    smpl_uv_map_path='../sample_data/smpl_uv.obj',
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
                 stop_after=1)


    ## cam_dist -> sai do nome da imagem
    ## at -> sai da anotação (media dos vertices)
    ## eye_position = [at[0] + x_axis_weight * (cam_dist * np.cos(np.deg2rad(rotation_dict[m]))),
    ##                at[1] + y_axis_weight * (cam_dist * np.sin(np.deg2rad(rotation_dict[m]))),
    ##                at[2]]