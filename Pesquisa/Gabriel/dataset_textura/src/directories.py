from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.absolute()
DATA_DIR = ROOT_DIR / 'data'
EDA_DIR = ROOT_DIR / 'eda'
SAMPLE_DATA_DIR = ROOT_DIR / 'sample_data'
SMPL_DIR = SAMPLE_DATA_DIR / 'SMPL' / 'models'
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
###
ANNOTATION_DIR = OUTPUT_DIR / 'annotations'
CAMERA_ANNOTATION_DIR = ANNOTATION_DIR / 'camera'
MEASUREMENTS_ANNOTATION_DIR = ANNOTATION_DIR / 'measurements'
PLANE_ANNOTATION_DIR = ANNOTATION_DIR / 'plane'
####

## TRAIN DIRECTORIES AND TEST ##
TRAIN_MESHES_DIR = MESHES_DIR / 'skeletex' / 'train'
TEST_MESHES_DIR = MESHES_DIR / 'skeletex' / 'test'

TRAIN_BACKGROUND_DIR = BACKGROUNDS_DIR / 'train'
TEST_BACKGROUND_DIR = BACKGROUNDS_DIR / 'test'

TRAIN_TEXTURES_DIR = TEXTURES_DIR / 'train'
TEST_TEXTURES_DIR = TEXTURES_DIR / 'test'

### SCHEMAS

TRAIN_SCHEMA_DIR = DATA_DIR / "schemas" / "train"
TEST_SCHEMA_DIR = DATA_DIR / "schemas" / "test"