import os
import sys
import json
import datetime
import numpy as np
import skimage.draw

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils
from PIL import Image
import cv2
from barcode import BarcodeConfig
import matplotlib.pyplot as plt


import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
import numpy as np
import cv2
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point
import json
import glob
import math
from tqdm import tqdm
import pandas as pd
import pickle

class InferenceConfig(BarcodeConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
config = InferenceConfig()
# config.display()

DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=DEFAULT_LOGS_DIR)
weights_path = model.find_last()
model.load_weights(weights_path, by_name=True)


def evaluate_write_to_disk(model):
    base_dir = "/home/ec2-user/SageMaker/benchmarks/dataset/"
    for i in tqdm(range(97549, 100000)):
        test_image_path = "X/train_{}.png".format(i)
        prediction_path = "prediction/predict_{}.png".format(i)
        test_image = skimage.io.imread(base_dir + test_image_path)
        test_image = test_image.reshape(test_image.shape[0], test_image.shape[1], 1)
        
        r = model.detect([test_image])[0]
        if r['masks'].shape[-1] > 0 and r['masks'].shape[0] == r['masks'].shape[1]:
            prediction = r['masks'][:, :, 0]
            
        else:
            continue
        skimage.io.imsave(base_dir + prediction_path, prediction.squeeze().astype(np.uint8))
        
evaluate_write_to_disk(model)