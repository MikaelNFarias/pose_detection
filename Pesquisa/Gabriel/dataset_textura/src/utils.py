from PIL import Image
import numpy as np
import smplx as SMPL
import logging
import re
from typing import List,Any,Optional,Dict
from time import time
import json
import sys
import numpy as np
from scipy.spatial import ConvexHull
import os
import torch
import argparse



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

def setup_logger(context):
    # Create a logger
    logger = logging.getLogger(context)
    # Set the level based on the debug parameter
    logger.setLevel(logging.INFO)
    
    # Create a console handler
    ch = logging.StreamHandler()
    # Set the level for the handler
    ch.setLevel(logging.INFO)
    
    # Create a formatter and set it for the handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    
    # Add the handler to the logger
    if not logger.handlers:
        logger.addHandler(ch)
    
    return logger



def load_face_segmentation(path: str):
    '''
    Load face segmentation which defines for each body model part
    the faces that belong to it.
    :param path: str - path to json file with defined face segmentation
    '''

    try:
        with open(path, 'r') as f:
            face_segmentation = json.load(f)
    except FileNotFoundError:
        sys.exit(f"No such file - {path}")

    return face_segmentation


def convex_hull_from_3D_points(slice_segments: np.ndarray):
    '''
    Cretes convex hull from 3D points
    :param slice_segments: np.ndarray, dim N x 2 x 3 representing N 3D segments

    Returns:
    :param slice_segments_hull: np.ndarray, dim N x 2 x 3 representing N 3D segments
                                that form the convex hull
    '''

    # stack all points in N x 3 array
    merged_segment_points = np.concatenate(slice_segments)
    unique_segment_points = np.unique(merged_segment_points,
                                      axis=0)

    # points lie in plane -- find which ax of x,y,z is redundant
    redundant_plane_coord = np.argmin(np.max(unique_segment_points, axis=0) -
                                      np.min(unique_segment_points, axis=0))
    non_redundant_coords = [x for x in range(3) if x != redundant_plane_coord]

    # create convex hull
    hull = ConvexHull(unique_segment_points[:, non_redundant_coords])
    segment_point_hull_inds = hull.simplices.reshape(-1)

    slice_segments_hull = unique_segment_points[segment_point_hull_inds]
    slice_segments_hull = slice_segments_hull.reshape(-1, 2, 3)

    return slice_segments_hull


def filter_body_part_slices(slice_segments: np.ndarray,
                            sliced_faces: np.ndarray,
                            measurement_name: str,
                            circumf_2_bodypart: dict,
                            face_segmentation: dict
                            ):
    '''
    Remove segments that are not in the appropriate body part
    for the given measurement.
    :param slice_segments: np.ndarray - (N,2,3) for N segments
                                        represented as two 3D points
    :param sliced_faces: np.ndarray - (N,) representing the indices of the
                                        faces
    :param measurement_name: str - name of the measurement
    :param circumf_2_bodypart: dict - dict mapping measurement to body part
    :param face_segmentation: dict - dict mapping body part to all faces belonging
                                    to it

    Return:
    :param slice_segments: np.ndarray (K,2,3) where K < N, for K segments
                            represented as two 3D points that are in the
                            appropriate body part
    '''

    if measurement_name in circumf_2_bodypart.keys():

        body_parts = circumf_2_bodypart[measurement_name]

        if isinstance(body_parts, list):
            body_part_faces = [face_index for body_part in body_parts
                               for face_index in face_segmentation[body_part]]
        else:
            body_part_faces = face_segmentation[body_parts]

        N_sliced_faces = sliced_faces.shape[0]

        keep_segments = []
        for i in range(N_sliced_faces):
            if sliced_faces[i] in body_part_faces:
                keep_segments.append(i)

        return slice_segments[keep_segments]

    else:
        return slice_segments


def point_segmentation_to_face_segmentation(
        point_segmentation: dict,
        faces: np.ndarray,
        save_as: str = None):
    """
    :param point_segmentation: dict - dict mapping body part to
                                      all points belonging to it
    :param faces: np.ndarray - (N,3) representing the indices of the faces
    :param save_as: str - optional path to save face segmentation as json
    """

    import json
    from tqdm import tqdm
    from collections import Counter

    # create body parts to index mapping
    mapping_bp2ind = dict(zip(point_segmentation.keys(),
                              range(len(point_segmentation.keys()))))
    mapping_ind2bp = {v: k for k, v in mapping_bp2ind.items()}

    # assign each face to body part index
    faces_segmentation = np.zeros_like(faces)
    for i, face in tqdm(enumerate(faces)):
        for bp_name, bp_indices in point_segmentation.items():
            bp_label = mapping_bp2ind[bp_name]

            for k in range(3):
                if face[k] in bp_indices:
                    faces_segmentation[i, k] = bp_label

    # for each face, assign the most common body part
    face_segmentation_final = np.zeros(faces_segmentation.shape[0])
    for i, f in enumerate(faces_segmentation):
        c = Counter(list(f))
        face_segmentation_final[i] = c.most_common()[0][0]

    # create dict with body part as key and faces as values
    face_segmentation_dict = {k: [] for k in mapping_bp2ind.keys()}
    for i, fff in enumerate(face_segmentation_final):
        face_segmentation_dict[mapping_ind2bp[int(fff)]].append(i)

    # save face segmentation
    if save_as:
        with open(save_as, 'w') as f:
            json.dump(face_segmentation_dict, f)

    return face_segmentation_dict



def convert_numpy_to_list(dictionary: dict):

    for key, value in dictionary.items():
        if isinstance(value,np.ndarray):
            dictionary[key] = value.tolist()

    return dictionary

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (float, np.float32, np.float64)):
            return format(obj, '.4f')

        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        return super().default(obj)

    
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
                              camera_data = None,
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

    data_json = format_floats(data_json)

    with open(json_file_path,'w+') as f:
        json.dump(data_json,f, indent=4,cls=NumpyEncoder)

    print(f"Arquivo JSON {json_file_path} salvo com sucesso")

def save_to_json(json_file_path,render_data = None,
                 plane_data=None,
                 measurement_data=None,
                 cam_data=None,
                 **kwargs) -> None:
    
    json_file_path = f"{json_file_path}.json"
    data_json = {}
    ## add kwargs to data_json
    if kwargs:
        data_json.update(kwargs)
    if measurement_data is not None:
        data_json.update(measurement_data)
    if render_data is not None:
        data_json.update(render_data)
    if plane_data is not None:
        data_json.update(plane_data)
    if cam_data is not None:
        data_json.update(cam_data)
    
    with open(json_file_path,'w+') as f:
        json.dump(data_json,f, indent=4,cls=NumpyEncoder)

def format_floats(obj, precision=4):
    if isinstance(obj, float):
        return float(format(obj, f'.{precision}f'))
    elif isinstance(obj, np.ndarray):
        return np.array([format_floats(e) for e in obj.tolist()])
    elif isinstance(obj, list):
        return [format_floats(e) for e in obj]
    elif isinstance(obj, dict):
        return {k: format_floats(v) for k, v in obj.items()}
    elif isinstance(obj, tuple):
        return tuple(format_floats(e) for e in obj)
    elif isinstance(obj,torch.Tensor):
        return torch.tensor([format_floats(e.item()) for e in obj.flatten()]).reshape(obj.shape)
    return obj