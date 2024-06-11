from directiories import *
from utils import *
from typing import List,Any,Tuple,Dict


class DatasetRenderer:
    def __init__(self,
                 meshes_path: str,
                 textures_path: str,
                 dataset: str = 'skeletex',
                 output_folder: str = OUTPUT_DIR,
                 background_folder: str = BACKGROUNDS_DIR,
                 smpl_model_path: str = SMPL_DIR,
                ) -> None:
        
        self.meshes_path = os.path.join(meshes_path, dataset)
        self.textures_path = textures_path
        self.dataset = dataset
        self.output_folder = output_folder
        self.background_folder = background_folder
        self.smpl_model_path = smpl_model_path

        self
        ...
    
    def __len__(self):
        return len(self.data)
    
    def render(self):
        pass

    def measure(self,):
        pass