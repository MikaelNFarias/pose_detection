from landmark_definitions import *
from joint_definitions import *

STANDARD_LABELS = {
    'A': 'head_circumference',
    'B': 'neck_circumference',
    'C': 'shoulder_to_crotch_height',
    'D': 'chest_circumference',
    'E': 'waist_circumference',
    'F': 'hip_circumference',
    'G': 'wrist_right_circumference',
    'H': 'bicep_right_circumference',
    'I': 'forearm_right_circumference',
    'J': 'arm_right_length',
    'K': 'inside_leg_height',
    'L': 'thigh_left_circumference',
    'M': 'calf_left_circumference',
    'N': 'ankle_left_circumference',
    'O': 'shoulder_breadth',
    'P': 'height'
}


class MeasurementType():
    CIRCUMFERENCE = "circumference"
    LENGTH = "length"


MEASUREMENT_TYPES = {
    "height": MeasurementType.LENGTH,
    "head_circumference": MeasurementType.CIRCUMFERENCE,
    "neck_circumference": MeasurementType.CIRCUMFERENCE,
    "shoulder_to_crotch_height": MeasurementType.LENGTH,
    "chest_circumference": MeasurementType.CIRCUMFERENCE,
    "waist_circumference": MeasurementType.CIRCUMFERENCE,
    "hip_circumference": MeasurementType.CIRCUMFERENCE,

    "wrist_right_circumference": MeasurementType.CIRCUMFERENCE,
    "bicep_right_circumference": MeasurementType.CIRCUMFERENCE,
    "forearm_right_circumference": MeasurementType.CIRCUMFERENCE,
    "arm_right_length": MeasurementType.LENGTH,
    "inside_leg_height": MeasurementType.LENGTH,
    "thigh_left_circumference": MeasurementType.CIRCUMFERENCE,
    "calf_left_circumference": MeasurementType.CIRCUMFERENCE,
    "ankle_left_circumference": MeasurementType.CIRCUMFERENCE,
    "shoulder_breadth": MeasurementType.LENGTH,
}


class SMPLMeasurementDefinitions():
    '''
    Definition of SMPL measurements.

    To add a new measurement:
    1. add it to the measurement_types dict and set the type:
       LENGTH or CIRCUMFERENCE
    2. depending on the type, define the measurement in LENGTHS or
       CIRCUMFERENCES dict
       - LENGTHS are defined using 2 landmarks - the measurement is
                found with distance between landmarks
       - CIRCUMFERENCES are defined with landmarks and joints - the
                measurement is found by cutting the SMPL model with the
                plane defined by a point (landmark point) and normal (
                vector connecting the two joints)
    3. If the body part is a CIRCUMFERENCE, a possible issue that arises is
       that the plane cutting results in multiple body part slices. To alleviate
       that, define the body part where the measurement should be located in
       CIRCUMFERENCE_TO_BODYPARTS dict. This way, only slice in that body part is
       used for finding the measurement. The body parts are defined by the SMPL
       face segmentation.
    '''

    possible_measurements = MEASUREMENT_TYPES.keys()

    LENGTHS = {"height":
                   (SMPL_LANDMARK_INDICES["HEAD_TOP"],
                    SMPL_LANDMARK_INDICES["HEELS"]
                    ),
               "shoulder_to_crotch_height":
                   (SMPL_LANDMARK_INDICES["SHOULDER_TOP"],
                    SMPL_LANDMARK_INDICES["INSEAM_POINT"]
                    ),
               "arm_left_length":
                   (SMPL_LANDMARK_INDICES["LEFT_SHOULDER"],
                    SMPL_LANDMARK_INDICES["LEFT_WRIST"]
                    ),
               "arm_right_length":
                   (SMPL_LANDMARK_INDICES["RIGHT_SHOULDER"],
                    SMPL_LANDMARK_INDICES["RIGHT_WRIST"]
                    ),
               "inside_leg_height":
                   (SMPL_LANDMARK_INDICES["LOW_LEFT_HIP"],
                    SMPL_LANDMARK_INDICES["LEFT_ANKLE"]
                    ),
               "shoulder_breadth":
                   (SMPL_LANDMARK_INDICES["LEFT_SHOULDER"],
                    SMPL_LANDMARK_INDICES["RIGHT_SHOULDER"]
                    ),
               }

    # defined with landmarks and joints
    # landmarks are defined with indices of the smpl model points
    # normals are defined with joint names of the smpl model
    CIRCUMFERENCES = {
        "head_circumference": {"LANDMARKS": ["HEAD_LEFT_TEMPLE"],
                               "JOINTS": ["pelvis", "spine3"]},

        "neck_circumference": {"LANDMARKS": ["NECK_ADAM_APPLE"],
                               "JOINTS": ["spine2", "head"]},

        "chest_circumference": {"LANDMARKS": ["LEFT_NIPPLE", "RIGHT_NIPPLE"],
                                "JOINTS": ["pelvis", "spine3"]},

        "waist_circumference": {"LANDMARKS": ["BELLY_BUTTON", "BACK_BELLY_BUTTON"],
                                "JOINTS": ["pelvis", "spine3"]},

        "hip_circumference": {"LANDMARKS": ["PUBIC_BONE"],
                              "JOINTS": ["pelvis", "spine3"]},

        "wrist_right_circumference": {"LANDMARKS": ["RIGHT_WRIST"],
                                      "JOINTS": ["right_wrist", "right_hand"]},

        "bicep_right_circumference": {"LANDMARKS": ["RIGHT_BICEP"],
                                      "JOINTS": ["right_shoulder", "right_elbow"]},

        "forearm_right_circumference": {"LANDMARKS": ["RIGHT_FOREARM"],
                                        "JOINTS": ["right_elbow", "right_wrist"]},

        "thigh_left_circumference": {"LANDMARKS": ["LEFT_THIGH"],
                                     "JOINTS": ["pelvis", "spine3"]},

        "calf_left_circumference": {"LANDMARKS": ["LEFT_CALF"],
                                    "JOINTS": ["pelvis", "spine3"]},

        "ankle_left_circumference": {"LANDMARKS": ["LEFT_ANKLE"],
                                     "JOINTS": ["pelvis", "spine3"]},

    }

    CIRCUMFERENCE_TO_BODYPARTS = {
        "head_circumference": "head",
        "neck_circumference": "neck",
        "chest_circumference": ["spine1", "spine2"],
        "waist_circumference": ["hips", "spine"],
        "hip_circumference": "hips",
        "wrist_right_circumference": ["rightHand", "rightForeArm"],
        "bicep_right_circumference": "rightArm",
        "forearm_right_circumference": "rightForeArm",
        "thigh_left_circumference": "leftUpLeg",
        "calf_left_circumference": "leftLeg",
        "ankle_left_circumference": "leftLeg",
    }


class SMPLXMeasurementDefinitions():
    '''
    Definition of SMPLX measurements.

    To add a new measurement:
    1. add it to the measurement_types dict and set the type:
       LENGTH or CIRCUMFERENCE
    2. depending on the type, define the measurement in LENGTHS or
       CIRCUMFERENCES dict
       - LENGTHS are defined using 2 landmarks - the measurement is
                found with distance between landmarks
       - CIRCUMFERENCES are defined with landmarks and joints - the
                measurement is found by cutting the SMPLX model with the
                plane defined by a point (landmark point) and normal (
                vector connecting the two joints)
    3. If the body part is a CIRCUMFERENCE, a possible issue that arises is
       that the plane cutting results in multiple body part slices. To alleviate
       that, define the body part where the measurement should be located in
       CIRCUMFERENCE_TO_BODYPARTS dict. This way, only slice in that body part is
       used for finding the measurement. The body parts are defined by the SMPL
       face segmentation.
    '''

    possible_measurements = MEASUREMENT_TYPES.keys()

    LENGTHS = {"height":
                   (SMPLX_LANDMARK_INDICES["HEAD_TOP"],
                    SMPLX_LANDMARK_INDICES["HEELS"]
                    ),
               "shoulder_to_crotch_height":
                   (SMPLX_LANDMARK_INDICES["SHOULDER_TOP"],
                    SMPLX_LANDMARK_INDICES["INSEAM_POINT"]
                    ),
               "arm_left_length":
                   (SMPLX_LANDMARK_INDICES["LEFT_SHOULDER"],
                    SMPLX_LANDMARK_INDICES["LEFT_WRIST"]
                    ),
               "arm_right_length":
                   (SMPLX_LANDMARK_INDICES["RIGHT_SHOULDER"],
                    SMPLX_LANDMARK_INDICES["RIGHT_WRIST"]
                    ),
               "inside_leg_height":
                   (SMPLX_LANDMARK_INDICES["LOW_LEFT_HIP"],
                    SMPLX_LANDMARK_INDICES["LEFT_ANKLE"]
                    ),
               "shoulder_breadth":
                   (SMPLX_LANDMARK_INDICES["LEFT_SHOULDER"],
                    SMPLX_LANDMARK_INDICES["RIGHT_SHOULDER"]
                    ),
               }

    # defined with landmarks and joints
    # landmarks are defined with indices of the smpl model points
    # normals are defined with joint names of the smpl model
    CIRCUMFERENCES = {
        "head_circumference": {"LANDMARKS": ["HEAD_LEFT_TEMPLE"],
                               "JOINTS": ["pelvis", "spine3"]},

        "neck_circumference": {"LANDMARKS": ["NECK_ADAM_APPLE"],
                               "JOINTS": ["spine1", "spine3"]},

        "chest_circumference": {"LANDMARKS": ["LEFT_NIPPLE", "RIGHT_NIPPLE"],
                                "JOINTS": ["pelvis", "spine3"]},

        "waist_circumference": {"LANDMARKS": ["BELLY_BUTTON", "BACK_BELLY_BUTTON"],
                                "JOINTS": ["pelvis", "spine3"]},

        "hip_circumference": {"LANDMARKS": ["PUBIC_BONE"],
                              "JOINTS": ["pelvis", "spine3"]},

        "wrist_right_circumference": {"LANDMARKS": ["RIGHT_WRIST"],
                                      "JOINTS": ["right_wrist", "right_elbow"]},  # different from SMPL

        "bicep_right_circumference": {"LANDMARKS": ["RIGHT_BICEP"],
                                      "JOINTS": ["right_shoulder", "right_elbow"]},

        "forearm_right_circumference": {"LANDMARKS": ["RIGHT_FOREARM"],
                                        "JOINTS": ["right_elbow", "right_wrist"]},

        "thigh_left_circumference": {"LANDMARKS": ["LEFT_THIGH"],
                                     "JOINTS": ["pelvis", "spine3"]},

        "calf_left_circumference": {"LANDMARKS": ["LEFT_CALF"],
                                    "JOINTS": ["pelvis", "spine3"]},

        "ankle_left_circumference": {"LANDMARKS": ["LEFT_ANKLE"],
                                     "JOINTS": ["pelvis", "spine3"]},

    }

    CIRCUMFERENCE_TO_BODYPARTS = {
        "head_circumference": "head",
        "neck_circumference": "neck",
        "chest_circumference": ["spine1", "spine2"],
        "waist_circumference": ["hips", "spine"],
        "hip_circumference": "hips",
        "wrist_right_circumference": ["rightHand", "rightForeArm"],
        "bicep_right_circumference": "rightArm",
        "forearm_right_circumference": "rightForeArm",
        "thigh_left_circumference": "leftUpLeg",
        "calf_left_circumference": "leftLeg",
        "ankle_left_circumference": "leftLeg",
    }