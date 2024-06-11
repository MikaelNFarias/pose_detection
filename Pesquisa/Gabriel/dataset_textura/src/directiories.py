from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.absolute()
DATA_DIR = ROOT_DIR / 'data'
EDA_DIR = ROOT_DIR / 'eda'
SAMPLE_DATA_DIR = ROOT_DIR / 'sample_data'
SRC_DIR = ROOT_DIR / 'src'

####

MEASURER_DIR = SRC_DIR / 'measurer'
RENDERER_DIR = SRC_DIR / 'renderer'
SCRIPTS_DIR = SRC_DIR / 'scripts'

####

TEXTURES_DIR = DATA_DIR / 'textures'
MESHES_DIR = DATA_DIR / 'meshes'
BACKGROUNDS_DIR = DATA_DIR / 'background'
OUTPUT_DIR = DATA_DIR / 'output'