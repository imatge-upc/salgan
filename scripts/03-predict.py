import os
import numpy as np
from tqdm import tqdm
import cv2
import glob
from utils import *
from constants import *
from models.model_bce import ModelBCE
import pdb

def test(path_to_images, path_output_maps, model_to_test=None):
    list_img_files = [k.split('/')[-1].split('.')[0] for k in glob.glob(os.path.join(path_to_images, '*'))]
    # Load Data
    for curr_file in tqdm(list_img_files):
        print os.path.join(path_to_images, curr_file + '.png')
        img = cv2.cvtColor(cv2.imread(os.path.join(path_to_images, curr_file + '.png'), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        predict(model=model_to_test, image_stimuli=img, name=curr_file, path_output_maps=path_output_maps)


def main():
    # Create network
    model = ModelBCE(INPUT_SIZE[0], INPUT_SIZE[1])
    # Here need to specify the epoch of model sanpshot
    load_weights(model.net['output'], path="test/gen_", epochtoload=30)
    # Here need to specify the path to images and output path
    test(path_to_images='/home/titan/Saeed/saliency-salgan-2017/images', path_output_maps='/home/titan/Saeed/saliency-salgan-2017/segmentation/', model_to_test=model)

if __name__ == "__main__":
    main()
