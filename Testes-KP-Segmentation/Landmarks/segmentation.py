from tflite_support.task import vision
from tflite_support.task import core
from tflite_support.task import processor

import matplotlib.pyplot as plt
import numpy as np
import os

"""
Segmeation with tflite: https://www.tensorflow.org/lite/inference_with_metadata/task_library/image_segmenter#run_inference_in_python
DeepLab tensorflow    : https://www.tensorflow.org/lite/examples/segmentation/overview

TFHub (deeplab): https://www.kaggle.com/models?id=72,287,103,292,54,283&tfhub-redirect=true

https://www.kaggle.com/models/spsayakpaul/deeplabv3-xception65
- Slow and not precise

https://www.kaggle.com/models/tensorflow/deeplabv3
- metadata: Trained on COCO dataset. Good result but low resolution.

https://www.kaggle.com/models/spsayakpaul/deeplabv3-mobilenetv2/tfLite/int8
- ade20k (mobilenetv2_ade20k_train.tflite): Good. Has higher resolution (530x530), which leads to more noise.
- default (mobilenetv2_coco_voc_trainval.tflite): Doesn't work (asks for more metadata).
- dm05 (mobilenetv2_dm05_coco_voc_trainval.tflite): Good resolution (530x530) and less noise.
- dm05-f16 (mobilenetv2_dm05_f16_coco_voc_trainval.tflite): Similar to the previous model.
- dm05-int8 (mobilenetv2_dm05_int8_coco_voc_trainval.tflite): Similar to the previous model.
- float16 (mobilenetv2_f16_coco_voc_trainval.tflite): Doesn't work (asks for more metadata).
- int8 (mobilenetv2_int8_coco_voc_trainval.tflite): Doesn't work (asks for more metadata).

https://www.kaggle.com/models/spsayakpaul/deeplabv3-mobilenetv3
- cityscapes: Not good for our images.
"""


def get_height(mask_image):
    """
    Returns height in pixels and foot height given a segmented image
    """
    height_mask   = np.max(mask_image, axis = 1)
    foot_height   = np.nonzero(height_mask)[0][-1]
    pixels_height = np.sum(height_mask)
    return pixels_height, foot_height

def get_horizontal_measure(mask_image, y_position):
    """
    Returns horizontal measure of the person in y = y_position given a segmented image
    """
    measure = np.sum(mask_image[y_position, :])
    return measure

def pixels2cm(measure, height_cm, height_pixels):
    """
    Returns measure in centimeters by computing the proportion between height_cm (real height)
    and height_pixels (height measured in the image)
    """
    return (height_cm/height_pixels) * measure

def get_hip_height(joints):
    """
    Compute the hip's height by averaging the y coordinates of the two hip joints
    """
    return (joints[0][1]+joints[1][1])/2

if __name__ == "__main__":
    
    models = sorted(["./models/good_models/"+i for i in os.listdir("./models/good_models/")])
    image_path = "./images/woman_beach.jpg"

    for filepath in models:
        # For models trained on ADE20K, the class "person" corresponds to index 13
        # For models trained on COCO, the class "person" corresponds to index 15
        person = 13 if "ade" in filepath else 15 
        print(filepath)

        # Copied and pasted from the aforementioned tutorial
        segmenter = vision.ImageSegmenter.create_from_file(filepath)
        image_file = vision.TensorImage.create_from_file(image_path)
        segmentation_result = segmenter.segment(image_file)
        segmented_image = segmentation_result.segmentations[0].category_mask

        # Make binary image. Person = 1 and Background = 0.
        mask_image = np.where(segmented_image==person, 1, 0)

        # Testing
        h, foot_h = get_height(mask_image)
        y_pixel   = foot_h-int(0.5*h)
        measure_h = get_horizontal_measure(mask_image, y_pixel)
        print(f"Height: {h} pixels")
        print(f"Measure in y = {y_pixel} : {measure_h} pixels")
        measure_cm = round(pixels2cm(measure_h, 160, h), 2)
        print(f"Measure in y = {y_pixel} (for height = 165cm) : {measure_cm} cm")

        # Plotting segmented image
        plt.text(10, foot_h-h+20, f"Height: {h} pixels", fontsize = 10, color = 'r')
        plt.text(10, y_pixel+20, f"Measure in y = {y_pixel} : {measure_h} pixels", fontsize = 10, color = 'b')
        plt.axhline(y = foot_h, color = 'r', linestyle = '-') 
        plt.axhline(y = foot_h-h+1, color = 'r', linestyle = '-') 
        plt.axhline(y = y_pixel, color = 'b', linestyle = '-') 
        model_name = filepath.split("/")[-1][:-7]
        image_name = image_path.split("/")[-1][:-4]
        #plt.imsave(f"./results/results_{image_name}/{model_name}_out.png", mask_image)
        plt.imshow(mask_image, cmap='gray')
        plt.show()