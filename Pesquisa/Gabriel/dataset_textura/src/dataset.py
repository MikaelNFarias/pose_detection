import glob
import os
import random
import json
from pathlib import Path
import numpy as np
from typing import List, Tuple, Dict, Any, Sequence
import itertools
import shutil
import sys
import argparse

# Import custom modules
sys.path.append('renderer')
sys.path.append('measurer')
import renderer.render as rd
import measurer.measure as ms
from measurer.measurement_definitions import STANDARD_LABELS
from utils import *
from directories import *

from pytorch3d.io import load_obj

try:
    logger = setup_logger(__name__)
except Exception as e:
    print(e)
    print("[ERRO]: Logger não pode ser iniciado")
    sys.exit(1)


class DatasetGenerator:
    def __init__(self,
                 meshes_dir: str,
                 textures_dir: str,
                 backgrounds_dir: str,
                 focal_distances: Tuple[float, float],
                 radial_distortion_coeffs: np.ndarray,
                 train_output_dir: str,
                 test_output_dir: str,
                 train_schema_path: str,
                 test_schema_path: str,
                 dataset: str = "skeletex",
                 image_size: int = 1024,
                 at_original: Sequence[float] = (.5, .8, 1.2),
                 noise_at: Sequence[float] = (0.2, 0.2, 0.05),
                 cam_dist_range: np.ndarray = np.linspace(2, 3.8, 15),
                 x_weight_range: np.ndarray = np.linspace(.8, 1.5, 5),
                 y_weight_range: np.ndarray = np.linspace(.8, 1.5, 5),):

        self.meshes_dir = os.path.join(meshes_dir, dataset)
        self.textures_dir = textures_dir
        self.backgrounds_dir = backgrounds_dir

        self.focal_distances = focal_distances
        self.radial_distortion_coeffs = radial_distortion_coeffs
        self.image_size = image_size

        self.train_output_dir = train_output_dir
        self.test_output_dir = test_output_dir
        self.train_schema_path = Path(train_schema_path)
        self.test_schema_path = Path(test_schema_path)

        self.noise_at = noise_at
        self.at_original = np.array(at_original)
        self.ROTATION_DICT: Dict[str, float] = {
            'frontal': 90.0,
            'left': 0.0,
            'back': 270,
            'right': 180,
        }
        self.VIEWS = ('frontal', 'right','left')

        self.cam_dist_range = cam_dist_range
        self.x_weight_range = x_weight_range
        self.y_weight_range = y_weight_range
        self.z_weight_range = y_weight_range

        self.verts = None
        self.faces = None
        self.aux = None

    def generate_schemes(self, N: int,stop_after=None) -> None:
        if True:
            train_meshes_dir = os.path.join(self.meshes_dir, 'train')
            test_meshes_dir = os.path.join(self.meshes_dir, 'test')
            train_backgrounds_dir = os.path.join(self.backgrounds_dir, 'train')
            test_backgrounds_dir = os.path.join(self.backgrounds_dir, 'test')
            train_textures_dir = os.path.join(self.textures_dir, 'train')
            test_textures_dir = os.path.join(self.textures_dir, 'test')

            train_meshes = glob.glob(os.path.join(train_meshes_dir, '*.obj'))
            test_meshes = glob.glob(os.path.join(test_meshes_dir, '*.obj'))
            train_backgrounds = glob.glob(os.path.join(train_backgrounds_dir, '*.png')) + \
                                glob.glob(os.path.join(train_backgrounds_dir, '*.jpg')) + \
                                glob.glob(os.path.join(train_backgrounds_dir, '*.jpeg'))

            test_backgrounds = glob.glob(os.path.join(test_backgrounds_dir, '*.png')) + glob.glob(
                os.path.join(test_backgrounds_dir, "*.jpg")) + glob.glob(os.path.join(test_backgrounds_dir, "*.jpeg"))
            train_textures = glob.glob(os.path.join(train_textures_dir, "*.png")) + glob.glob(
                os.path.join(train_textures_dir, "*.jpg")) + glob.glob(os.path.join(train_textures_dir, "*.jpeg"))
            test_textures = glob.glob(os.path.join(test_textures_dir, "*.png")) + glob.glob(
                os.path.join(test_textures_dir, "*.jpg")) + glob.glob(os.path.join(test_textures_dir, "*.jpeg"))

            # train_combinations = list(
            #     itertools.product(train_meshes, train_textures, train_backgrounds))
            # test_combinations = list(
            #     itertools.product(test_meshes, test_textures, test_backgrounds))

            train_scheme = self._generate_render_scheme(meshes=train_meshes,
                                                        textures=train_textures,
                                                        backgrounds=train_backgrounds,
                                                        N=N,
                                                        stop_after=stop_after)
            test_scheme = self._generate_render_scheme(meshes=test_meshes,
                                                       textures=test_textures,
                                                       backgrounds=test_backgrounds,
                                                       N=N,
                                                       stop_after=stop_after)

            with open(self.train_schema_path, 'w') as f:
                json.dump(train_scheme, f, indent=4,cls=NumpyEncoder)

            with open(self.test_schema_path, 'w') as f:
                json.dump(test_scheme, f, indent=4,cls=NumpyEncoder)


    def _random_biased_value(min_val=0.7, max_val=1.5, mean=1, std_dev=0.2):
        value = np.random.normal(mean, std_dev)
        value = np.clip(value, min_val, max_val)
        return value


    def _generate_render_scheme(self,
                                meshes: List[str],
                                textures: List[str],
                                backgrounds: List[str],
                                N: int,
                                stop_after=None) -> List[Dict[str, Any]]:
        scheme = []
        for idx_mesh, mesh in enumerate(meshes):
            if stop_after is not None and idx_mesh == stop_after:
                break
            file_numeration = extract_numeration(mesh)
            for _ in range(N):
                cam_dist = random.choice(self.cam_dist_range)
                x_weight = np.random.choice(self.x_weight_range, p=[0.05, 0.15, 0.6, 0.15, 0.05])
                y_weight = np.random.choice(self.y_weight_range, p=[0.05, 0.15, 0.6, 0.15, 0.05])
                z_weight = np.random.choice(self.y_weight_range, p=[0.05, 0.05, 0.8, 0.05, 0.05])
                noise = np.random.normal(0, self.noise_at)
                at = np.where(np.random.rand(self.at_original.size) < 0.6, self.at_original, self.at_original + noise)
                texture = random.choice(textures)
                background = random.choice(backgrounds)
                focal_distance = random.uniform(*self.focal_distances)
                radial_distortion = [random.uniform(*self.radial_distortion_coeffs[i]) for i in range(3)]

                set_views = set()
                for view in self.VIEWS:
                    if view not in set_views:
                        if view == 'left' or view == 'right':
                            view = np.random.choice(['left','right'])
                            set_views.update(['right','left'])

                        eye_position = np.array([
                            at[0] + x_weight * cam_dist * np.cos(np.deg2rad(self.ROTATION_DICT[view])),
                            at[1] + y_weight * cam_dist * np.sin(np.deg2rad(self.ROTATION_DICT[view])),
                            at[2] + z_weight
                        ])
                        set_views.add(view)
                        
                    scheme.append({
                        'mesh': mesh,
                        'texture': texture,
                        'background': background,
                        'view': view if view.lower() not in ("left","right") else 'side',
                        'eye_position': eye_position,
                        'at': at,
                        'focal_distance': focal_distance,
                        'radial_distortion': radial_distortion,
                        'file_numeration': file_numeration,
                        'saved': False
                    })

        return scheme

    @staticmethod
    def _load_mesh(self, mesh_path):
        verts,faces,aux = load_obj(mesh_path)
        return verts, faces , aux

    def measure_mesh(self):
        pass

    def render_samples(self, dataset_type: str) -> None:
        if dataset_type == 'train':
            schema_path = self.train_schema_path
            output_dir = self.train_output_dir
        elif dataset_type == 'test':
            schema_path = self.test_schema_path
            output_dir = self.test_output_dir
        else:
            raise ValueError("Invalid dataset type. Use ['train'] or ['test'].")

        if not schema_path.exists():
            raise FileNotFoundError(f"The schema file {schema_path} does not exist.")

        with open(schema_path, 'r') as f:
            scheme = json.load(f)

        for sample in scheme:
            if not sample['saved']:
                try:
                    rd.render(
                        texture_image_path=sample['texture'],
                        smpl_uv_map_path=os.path.join(SAMPLE_DATA_DIR,'smpl_uv.obj'),
                        obj_mesh_path=sample['mesh'],
                        output_path=output_dir,
                        cam_dist=sample['eye_position'][2],  # assuming Z as the distance for simplicity
                        background_image_path=sample['background'],
                        eye_position=sample['eye_position'],
                        image_size=self.image_size,
                        at=sample['at'],
                        view=sample['view']
                    )

                    # Save render data
                    #file_numeration = extract_numeration(Path(sample['model']).name)

                    save_to_json(os.path.join(TRAIN_RENDER_ANNOTATION_DIR,f"{dataset_type}_{sample['file_numeration']}_{sample['view']}"),
                                 render_data=sample,
                                 background=Path(sample['background']).name,
                                 texture=Path(sample['texture']).name)

                    # Mark as saved
                    sample['saved'] = True

                except Exception as e:
                    logger.error(f"Error rendering {sample['mesh']}")
                    logger.error(e)

        # Save updated schema
        with open(schema_path, 'w') as f:
            json.dump(scheme, f, indent=4,cls=NumpyEncoder)


# Exemplo de uso
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Generate dataset')
    parser.add_argument('--meshes_dir', type=str, default=MESHES_DIR, help='Path to meshes directory')
    parser.add_argument('--textures_dir', type=str, default=TEXTURES_DIR, help='Path to textures directory')
    parser.add_argument('--backgrounds_dir', type=str, default=BACKGROUNDS_DIR, help='Path to backgrounds directory')
    parser.add_argument('-N',type=int,default=1,help='Number of samples to generate')

    parser.add_argument('--dataset-type', type=str, default='train', help='Dataset type')
    args = parser.parse_args()


    FOCAL_DISTANCES = (1.0, 1.5)
    RADIAL_DISTORTION_COEFFS = np.array([[0.1, 0.3], [0.1, 0.3], [0.1, 0.3]])

    if args.dataset_type == 'train':
        try:
            dataset_generator = DatasetGenerator(
                meshes_dir=MESHES_DIR,
                textures_dir=TEXTURES_DIR,
                backgrounds_dir=BACKGROUNDS_DIR,
                focal_distances=FOCAL_DISTANCES,
                radial_distortion_coeffs=RADIAL_DISTORTION_COEFFS,
                train_schema_path=os.path.join(TRAIN_SCHEMA_DIR, 'train_schema.json'),
                test_schema_path=os.path.join(TEST_SCHEMA_DIR, 'test_schema.json'),
                train_output_dir=TRAIN_OUTPUT_DIR,
                test_output_dir=TEST_OUTPUT_DIR,
            )
            dataset_generator.generate_schemes(N=args.N)
            dataset_generator.render_samples(dataset_type=args.dataset_type)

        except Exception as e:
            logger.error(e)
            sys.exit(1)

    elif args.dataset_type == 'test':
        try:
            dataset_generator = DatasetGenerator(
                meshes_dir=MESHES_DIR,
                textures_dir=TEXTURES_DIR,
                backgrounds_dir=BACKGROUNDS_DIR,
                focal_distances=FOCAL_DISTANCES,
                radial_distortion_coeffs=RADIAL_DISTORTION_COEFFS,
                train_schema_path=os.path.join(TRAIN_SCHEMA_DIR, 'train_schema.json'),
                test_schema_path=os.path.join(TEST_SCHEMA_DIR, 'test_schema.json'),
                train_output_dir=TRAIN_OUTPUT_DIR,
                test_output_dir=TEST_OUTPUT_DIR,
            )
            dataset_generator.generate_schemes(N=args.N)
            dataset_generator.render_samples(dataset_type=args.dataset_type)
        except Exception as e:
            logger.error(e)
            sys.exit(1)
            
    else:
        logger.error("Invalid dataset type. Use ['train'] or ['test'].")
        sys.exit(1)


    
