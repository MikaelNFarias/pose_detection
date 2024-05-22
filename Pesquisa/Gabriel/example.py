import pywavefront
import numpy as np
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from body.measurement import Body3D

current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(current_dir, 'data')

def main():
    person = pywavefront.Wavefront(
        os.path.join(data_dir, 'ANSURI_fem_0005.obj'),
        create_materials=True,
        collect_faces=True
    )
    faces = np.array(person.mesh_list[0].faces)
    vertices = np.array(person.vertices)

    body = Body3D(vertices, faces)

    body_measurements = body.getMeasurements()

    height = body.height()
    weight = body.weight()
    shoulder_2d, shoulder_location, shoulder_length = body.shoulder()
    chest_2d, chest_location, chest_length = body.chest()
    hip_2d, hip_location, hip_length = body.hip()
    waist_2d, waist_location, waist_length = body.waist()
    thigh_2d, thigh_location, thigh_length = body.thighOutline()
    outer_leg_length = body.outerLeg()
    inner_leg_length = body.innerLeg()
    neck_2d, neck_location, neck_length = body.neck()
    neck_hip_length = body.neckToHip()


    dicionario = {
        "height": body_measurements[1],
        "weight": body_measurements[0],
        "shoulder": body_measurements[2],
        "chest": body_measurements[3],
        "hip": body_measurements[4],
        "waist": body_measurements[5],
        "thigh": body_measurements[6],
        "outer_leg": body_measurements[7],
        "inner_leg": body_measurements[8],
        "neck": body_measurements[9],
        "neck_hip": body_measurements[10]
    }

    print(dicionario)

if __name__ == '__main__':
    main()