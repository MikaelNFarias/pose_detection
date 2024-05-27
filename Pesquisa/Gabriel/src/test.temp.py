import numpy as np
import os
import sys
from utils import *

from avatar import Avatar


def test_avatar():
    ## Import info: measurements data
    gender, measurements = load_input_data(data_file='test.csv')
    measurements = np.array(measurements).transpose()

    body = Avatar(measurements, gender)

    #input_meas21 = body.predict()

    ## Create 3D avatar
    #body.create_obj_file()

    ## Extract measurements from the 3D avatar
    output_meas21 = body.measure()

    return_data = {
        "input_meas21": input_meas21,
        "output_meas21": output_meas21
    }


if __name__ == "__main__":
    test_avatar()