# Process raw data and save them into pickle file.
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

img_size = INPUT_SIZE
salmap_size = INPUT_SIZE

# Resize train/validation files

listImgFiles = [k.split('/')[-1].split('.')[0] for k in glob.glob(os.path.join(pathToMaps, '*'))]
listTestImages = [k.split('/')[-1].split('.')[0] for k in glob.glob(os.path.join(pathToImages, '*test*'))]

for currFile in tqdm(listImgFiles):
    tt = dataRepresentation.Target(os.path.join(pathToImages, currFile + '.jpg'),
                                   os.path.join(pathToMaps, currFile + '.mat'),
                                   os.path.join(pathToFixationMaps, currFile + '.mat'),
                                   dataRepresentation.LoadState.loaded, dataRepresentation.InputType.image,
                                   dataRepresentation.LoadState.loaded, dataRepresentation.InputType.saliencyMapMatlab,
                                   dataRepresentation.LoadState.unloaded, dataRepresentation.InputType.empty)

    # if tt.image.getImage().shape[:2] != (480, 640):
    #    print 'Error:', currFile

    imageResized = cv2.cvtColor(cv2.resize(tt.image.getImage(), img_size, interpolation=cv2.INTER_AREA),
                                cv2.COLOR_RGB2BGR)
    saliencyResized = cv2.resize(tt.saliency.getImage(), salmap_size, interpolation=cv2.INTER_AREA)

    cv2.imwrite(os.path.join(pathOutputImages, currFile + '.png'), imageResized)
    cv2.imwrite(os.path.join(pathOutputMaps, currFile + '.png'), saliencyResized)

# Resize test files

for currFile in tqdm(listTestImages):
    tt = dataRepresentation.Target(os.path.join(pathToImages, currFile + '.jpg'),
                                   os.path.join(pathToMaps, currFile + '.mat'),
                                   os.path.join(pathToFixationMaps, currFile + '.mat'),
                                   dataRepresentation.LoadState.loaded,dataRepresentation.InputType.image,
                                   dataRepresentation.LoadState.unloaded, dataRepresentation.InputType.empty,
                                   dataRepresentation.LoadState.unloaded, dataRepresentation.InputType.empty)

    imageResized = cv2.cvtColor(cv2.resize(tt.image.getImage(), img_size, interpolation=cv2.INTER_AREA),
                                cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(pathOutputImages, currFile + '.png'), imageResized)


# LOAD DATA

# Train

listFilesTrain = [k for k in listImgFiles if 'train' in k]
trainData = []
for currFile in tqdm(listFilesTrain):
    trainData.append(dataRepresentation.Target(os.path.join(pathOutputImages, currFile + '.png'),
                                               os.path.join(pathOutputMaps, currFile + '.png'),
                                               os.path.join(pathToFixationMaps, currFile + '.mat'),
                                               dataRepresentation.LoadState.loaded, dataRepresentation.InputType.image,
                                               dataRepresentation.LoadState.loaded, dataRepresentation.InputType.imageGrayscale,
                                               dataRepresentation.LoadState.loaded, dataRepresentation.InputType.fixationMapMatlab))

with open(os.path.join(pathToPickle, 'trainData.pickle'), 'wb') as f:
    pickle.dump(trainData, f)

# Validation

listFilesValidation = [k for k in listImgFiles if 'val' in k]
validationData = []
for currFile in tqdm(listFilesValidation):
    validationData.append(dataRepresentation.Target(os.path.join(pathOutputImages, currFile + '.png'),
                                                    os.path.join(pathOutputMaps, currFile + '.png'),
                                                    os.path.join(pathToFixationMaps, currFile + '.mat'),
                                                    dataRepresentation.LoadState.loaded, dataRepresentation.InputType.image,
                                                    dataRepresentation.LoadState.loaded, dataRepresentation.InputType.imageGrayscale,
                                                    dataRepresentation.LoadState.loaded, dataRepresentation.InputType.fixationMapMatlab))

with open(os.path.join(pathToPickle, 'validationData.pickle'), 'wb') as f:
    pickle.dump(validationData, f)

# Test

testData = []

for currFile in tqdm(listTestImages):
    testData.append(dataRepresentation.Target(os.path.join(pathOutputImages, currFile + '.png'),
                                              os.path.join(pathOutputMaps, currFile + '.png'),
                                              dataRepresentation.LoadState.loaded, dataRepresentation.InputType.image,
                                              dataRepresentation.LoadState.unloaded,
                                              dataRepresentation.InputType.empty))

with open(os.path.join(pathToPickle, 'testData.pickle'), 'wb') as f:
    pickle.dump(testData, f)
