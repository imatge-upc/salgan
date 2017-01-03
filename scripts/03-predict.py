import os
import numpy as np
from tqdm import tqdm
import cv2
from utils import *
from constant import *
from models.model_bce import Model_BCE


def test(path_to_images, path_output_maps, model_to_test=None):
    list_img_files = [k.split('/')[-1].split('.')[0] for k in glob.glob(os.path.join(path_to_images, '*'))]
    # Load Data
    list_img_files.sort()
    for currFile in tqdm(list_img_files, ncols=20):
        print os.path.join(pathToImages, currFile + '.jpg')
        img = cv2.cvtColor(cv2.imread(os.path.join(pathToImages, currFile + '.jpg'), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        predict(model=model_to_test, image=img, name=currFile, pathOutputMaps=path_output_maps)


def main():
    # Create network
    model = Model_BCE(INPUT_SIZE[0], INPUT_SIZE[1], batch_size=8)
    load_weights(model, path='', epochtoload=80)
    test(path_to_images='/path/to/images', path_output_maps='/path/to/save/salmaps', model_to_test=model)

if __name__ == "__main__":
    main()