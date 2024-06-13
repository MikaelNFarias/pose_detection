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
                 eye_position_ranges: np.ndarray,
                 at_ranges: np.ndarray,
                 focal_distances: Tuple[float, float],
                 radial_distortion_coeffs: np.ndarray,
                 train_schema_path: str,
                 test_schema_path: str,
                 dataset: str = "skeletex",
                 at_original: Sequence[float] = (.5, .8, 1.2),
                 noise_at: Sequence[float] = (0.2, 0.2, 0.05),
                 cam_dist_range: np.ndarray = np.linspace(2, 3.8, 15),
                 x_weight_range: np.ndarray = np.linspace(.8, 1.5, 5),
                 y_weight_range: np.ndarray = np.linspace(.8, 1.5, 5),):

        self.meshes_dir = os.path.join(meshes_dir, dataset)
        self.textures_dir = textures_dir
        self.backgrounds_dir = backgrounds_dir
        self.eye_position_ranges = eye_position_ranges
        self.at_ranges = at_ranges
        self.focal_distances = focal_distances
        self.radial_distortion_coeffs = radial_distortion_coeffs
        self.train_schema_path = Path(train_schema_path)
        self.test_schema_path = Path(test_schema_path)
        self.noise_at = noise_at
        self.at_original = np.array(at_original)
        self.ROTATION_DICT: Dict[str, float] = {
            'frontal': 90.0,
            'side': 0.0
        }
        self.VIEWS = ('frontal', 'side')

        self.cam_dist_range = cam_dist_range
        self.x_weight_range = x_weight_range
        self.y_weight = y_weight_range
        self.z_weight = y_weight_range

    def generate_schemes(self, N: int) -> None:
        if not self.train_schema_path.exists() or not self.test_schema_path.exists():
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

            train_combinations = list(
                itertools.product(train_meshes, train_textures, train_backgrounds))
            test_combinations = list(
                itertools.product(test_meshes, test_textures, test_backgrounds))

            train_scheme = self._generate_render_scheme(train_combinations, N)
            test_scheme = self._generate_render_scheme(test_combinations, N)

            with open(self.train_schema_path, 'w') as f:
                json.dump(train_scheme, f, indent=4)

            with open(self.test_schema_path, 'w') as f:
                json.dump(test_scheme, f, indent=4)

    def _random_biased_value(self, min_val=0.7, max_val=1.5, mean=1, std_dev=0.2):
        value = np.random.normal(mean, std_dev)
        value = np.clip(value, min_val, max_val)
        return value

    @staticmethod
    def _generate_render_scheme(self, meshes, textures, N: int) -> List[Dict[str, Any]]:
        scheme = []
        for idx_mesh, mesh in enumerate(meshes):
            for _ in range(N):
                cam_dist = random.choice(self.cam_dist_range)
                x_weight = np.random.choice(self.x_weight_range, p=[0.05, 0.15, 0.6, 0.15, 0.05])
                y_weight = np.random.choice(self.y_weight_range, p=[0.05, 0.15, 0.6, 0.15, 0.05])
                z_weight = np.random.choice(self.y_weight_range, p=[0.05, 0.05, 0.8, 0.05, 0.05])
                noise = np.random.normal(0, self.noise_at)
                at = np.where(np.random.rand(self.at_original.size()) < 0.6, self.at_original, self.at_original + noise)
                texture = random.choice(textures)
                background = random.choice(textures)
                focal_distance = random.uniform(*self.focal_distances)
                radial_distortion = [random.uniform(*self.radial_distortion_coeffs[i]) for i in range(3)]

                for view in self.VIEWS:
                    eye_position = np.array([
                        at[0] + x_weight * cam_dist * np.cos(np.deg2rad(self.ROTATION_DICT[view])),
                        at[1] + y_weight * cam_dist * np.sin(np.deg2rad(self.ROTATION_DICT[view])),
                        at[2] + z_weight
                    ])
                    scheme.append({
                        'mesh': mesh,
                        'texture': texture,
                        'background': background,
                        'view': self.VIEWS,
                        'eye_position': eye_position,
                        'at': at,
                        'focal_distance': focal_distance,
                        'radial_distortion': radial_distortion,
                        'saved': False
                    })

        return scheme

    def _load_mesh(self, mesh_path):
        return load_obj(mesh_path)

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
                    render_data = rd.render(
                        texture_image_path=sample['texture'],
                        smpl_mesh_path=sample['mesh'],
                        smpl_mesh_type='smpl',
                        smpl_uv_map_path=str(SAMPLE_DATA_DIR / 'smpl_uv.obj'),
                        obj_mesh_path=sample['model'],
                        output_path=output_dir,
                        gender='female',
                        cam_dist=sample['eye_position'][2],  # assuming Z as the distance for simplicity
                        background_image_path=sample['background'],
                    )

                    # Save render data
                    file_numeration = extract_numeration(Path(sample['model']).name)
                    camera_data = format_floats(render_data)
                    del camera_data['verts']
                    save_to_json(os.path.join(CAMERA_ANNOTATION_DIR,
                                              f"{dataset_type}_{file_numeration}_{sample['eye_position'][2]:.4f}"),
                                 render_data=camera_data,
                                 background=Path(sample['background']).name,
                                 texture=Path(sample['texture']).name)

                    # Mark as saved
                    sample['saved'] = True

                except Exception as e:
                    logger.error(f"Error rendering {sample['model']}")
                    logger.error(e)

        # Save updated schema
        with open(schema_path, 'w') as f:
            json.dump(scheme, f, indent=4)


# Exemplo de uso
if __name__ == "__main__":
    eye_position_ranges = np.array([[1, 2], [1, 2], [2, 3.8]])
    at_ranges = np.array([[-1, 1], [-1, 1], [-1, 1]])
    focal_distances = (1.0, 1.5)
    radial_distortion_coeffs = np.array([[0.1, 0.3], [0.1, 0.3], [0.1, 0.3]])

    dataset_generator = DatasetGenerator(
        meshes_dir=MESHES_DIR,
        textures_dir=TEXTURES_DIR,
        backgrounds_dir=BACKGROUNDS_DIR,
        eye_position_ranges=eye_position_ranges,
        at_ranges=at_ranges,
        focal_distances=focal_distances,
        radial_distortion_coeffs=radial_distortion_coeffs,
        train_schema_path=os.path.join(TRAIN_SCHEMA_DIR, 'train_schema.json'),
        test_schema_path=os.path.join(TEST_SCHEMA_DIR, 'test_schema.json')
    )

    dataset_generator.generate_schemes(N=10)
    dataset_generator.render_samples(dataset_type='train')
    dataset_generator.render_samples(dataset_type='test')
