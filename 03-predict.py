import os
import glob
import numpy as np
from tqdm import tqdm
import cv2
from utils import *
from constants import *
from models.model_bce import ModelBCE

PATH_TO_IMAGES = '/home/users/jpang/scratch-local/lsun2016/saliency-2016-lsun/extraresults/image'
PATH_TO_SALMAPS = '/home/users/jpang/scratch-local/lsun2016/saliency-2016-lsun/extraresults/'


def test(path_to_images, path_output_maps, model_to_test=None):
    list_img_files = [k.split('/')[-1].split('.')[0] for k in glob.glob(os.path.join(path_to_images, '*'))]
    # Load Data
    list_img_files.sort()
    for curr_file in tqdm(list_img_files, ncols=20):
        print os.path.join(path_to_images, curr_file + '.jpg')
        img = cv2.cvtColor(cv2.imread(os.path.join(path_to_images, curr_file + '.jpg'), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        predict(model=model_to_test, image_stimuli=img, name=curr_file, path_output_maps=path_output_maps)


def main():
    # Create network
    model_name = 'resnet50_gan'
    epochtoload = 81
    model = ModelBCE(INPUT_SIZE[0], INPUT_SIZE[1], batch_size=8)
    print model_name + ' ' + str(epochtoload)
    load_weights(model.net['output'], path=model_name + '/gen_', epochtoload=epochtoload)
    test(path_to_images=PATH_TO_IMAGES, path_output_maps=PATH_TO_SALMAPS + model_name + '/', model_to_test=model)

if __name__ == "__main__":
    main()
