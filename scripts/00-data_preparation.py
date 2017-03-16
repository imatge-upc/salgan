import os
import numpy as np
from PIL import Image
from PIL import ImageOps
from scipy import misc
import scipy.io
from skimage import io
import cv2
import sys
import cPickle as pickle
import glob
import random
from tqdm import tqdm
from eliaLib import dataRepresentation
from constants import *
from PIL import Image
from PIL import ImageOps
import pdb


def augment_data():
    listImgFiles = [k.split('/')[-1].split('.')[0] for k in glob.glob(os.path.join(pathToImages, '*'))]
    listFilesTrain = [k for k in listImgFiles if 'train' in k]
    listFilesVal = [k for k in listImgFiles if 'train' not in k]
    for filenames in tqdm(listFilesTrain):
	for angle in [90, 180, 270]:
            src_im = Image.open(os.path.join(pathToImages,filenames+'.png'))
            gt_im = Image.open(os.path.join(pathToMaps,filenames+'mask.png'))
	    rot_im = src_im.rotate(angle,expand=True)
	    rot_gt = gt_im.rotate(angle,expand=True)
	    rot_im.save(os.path.join(pathToImages,filenames+'_'+str(angle)+'.png'))
	    rot_gt.save(os.path.join(pathToMaps,filenames+'_'+str(angle)+'mask.png'))
        vert_im = ImageOps.flip(src_im)
        vert_gt = ImageOps.flip(gt_im)
        horz_im = ImageOps.mirror(src_im)
        horz_gt = ImageOps.mirror(gt_im)
	vert_im.save(os.path.join(pathToImages,filenames+'_vert.png'))
	vert_gt.save(os.path.join(pathToMaps,filenames+'_vertmask.png'))
	horz_im.save(os.path.join(pathToImages,filenames+'_horz.png'))
	horz_gt.save(os.path.join(pathToMaps,filenames+'_horzmask.png'))
       
def split_data(fraction):
    listImgFiles = [k.split('/')[-1].split('.')[0] for k in glob.glob(os.path.join(pathToImages, '*'))]
    numSamples = len(listImgFiles)
    train_ind = np.random.choice(np.arange(0,numSamples),(1,np.ceil(fraction*numSamples)),replace=False).squeeze()
    val_ind = np.array(np.setdiff1d(np.arange(0,numSamples),train_ind)).squeeze()
    
    for k in train_ind:
        os.rename(os.path.join(pathToImages,listImgFiles[k] + '.png'),os.path.join(pathToImages,'train_'+listImgFiles[k]+'.png'))
        os.rename(os.path.join(pathToMaps,listImgFiles[k] + 'mask.png'),os.path.join(pathToMaps,'train_'+listImgFiles[k]+'mask.png'))
    for k in val_ind:
        os.rename(os.path.join(pathToImages,listImgFiles[k] + '.png'),os.path.join(pathToImages,'val_'+listImgFiles[k]+'.png'))
        os.rename(os.path.join(pathToMaps,listImgFiles[k] + 'mask.png'),os.path.join(pathToMaps,'val_'+listImgFiles[k]+'mask.png'))
def main():
#    split_data(0.8)
    augment_data()
if __name__== "__main__":
    main() 
