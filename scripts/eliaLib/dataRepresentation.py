import cv2
import numpy as np
from enum import Enum
import scipy.io


class InputType(Enum):
    image = 0
    imageGrayscale = 1
    saliencyMapMatlab = 2
    fixationMapMatlab = 3
    empty = 100


class LoadState(Enum):
    unloaded = 0
    loaded = 1
    loadedCompressed = 2
    error = 100


###############################################################################################

class ImageContainer:
    def __init__(self, filePath, imageType, state=LoadState.unloaded):

        self.filePath = filePath
        self.state = state
        self.imageType = imageType

        if self.state == LoadState.unloaded:
            self.data = None
        elif self.state == LoadState.loaded:
            self.load()
        elif self.state == LoadState.loadedCompressed:
            self.loadCompressed()
        else:
            raise Exception('Unknown state when loading image')

    def load(self):

        if self.imageType == InputType.image:
            self.data = cv2.cvtColor(cv2.imread(self.filePath, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            self.state = LoadState.loaded
        if self.imageType == InputType.imageGrayscale:
            self.data = cv2.cvtColor(cv2.imread(self.filePath, cv2.IMREAD_COLOR), cv2.COLOR_BGR2GRAY)
            self.state = LoadState.loaded
        elif self.imageType == InputType.saliencyMapMatlab:
            self.data = (scipy.io.loadmat(self.filePath)['I'] * 255).astype(np.uint8)
            self.state = LoadState.loaded
        elif self.imageType == InputType.fixationMapMatlab:
            self.data = (scipy.io.loadmat(self.filePath)['I']).nonzero()
            self.state = LoadState.loaded
        elif self.imageType == InputType.empty:
            self.data = None

    def loadCompressed(self):

        if self.imageType == InputType.image:
            with open(self.filePath, 'rb') as f:
                data = f.read()
            self.data = np.fromstring(data, np.uint8)
            self.state = LoadState.loadedCompressed
        elif self.imageType == InputType.saliencyMapMatlab:
            self.state = LoadState.error
            raise Exception('Saliency maps do no have compressed handlind method enabled')
        elif self.imageType == InputType.empty:
            self.state = LoadState.error
            raise Exception('Empty images do no have compressed handlind method enabled')

    def getImage(self):

        if self.imageType == InputType.image:
            if self.state == LoadState.unloaded:
                return cv2.cvtColor(cv2.imread(self.filePath, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            elif self.state == LoadState.loaded:
                return self.data
            elif self.state == LoadState.loadedCompressed:
                return cv2.cvtColor(cv2.imdecode(self.data, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        elif self.imageType == InputType.imageGrayscale:
            if self.state == LoadState.unloaded:
                return cv2.cvtColor(cv2.imread(self.filePath, cv2.IMREAD_COLOR), cv2.COLOR_BGR2GRAY)
            elif self.state == LoadState.loaded:
                return self.data
            elif self.state == LoadState.loadedCompressed:
                return cv2.cvtColor(cv2.imdecode(self.data, cv2.IMREAD_COLOR), cv2.COLOR_BGR2GRAY)
        elif self.imageType == InputType.saliencyMapMatlab:
            if self.state == LoadState.unloaded:
                return (scipy.io.loadmat(self.filePath)['I'] * 255).astype(np.uint8)
            elif self.state == LoadState.loaded:
                return self.data
            elif self.state == LoadState.loadedCompressed:
                raise Exception('Saliency maps do no have compressed handlind method enabled')
                return None
        elif self.imageType == InputType.fixationMapMatlab:
            if self.state == LoadState.unloaded:
                return (scipy.io.loadmat(self.filePath)['I']).astype(np.uint8)
            elif self.state == LoadState.loaded:
                return self.data
            elif self.state == LoadState.loadedCompressed:
                raise Exception('Fixation maps do no have compressed handlind method enabled')
                return None
        elif self.imageType == InputType.empty:
            return None


###############################################################################################

# class Target():
#     def __init__(self, imagePath, saliencyPath,
#                  imageState=LoadState.unloaded, imageType=InputType.image,
#                  saliencyState=LoadState.unloaded, saliencyType=InputType.saliencyMapMatlab):
#         self.image = ImageContainer(imagePath, imageType, imageState)
#         self.saliency = ImageContainer(saliencyPath, saliencyType, saliencyState)

class Target():
    def __init__(self, imagePath, saliencyPath,fixationPath,
                 imageState=LoadState.unloaded, imageType=InputType.image,
                 saliencyState=LoadState.unloaded, saliencyType=InputType.saliencyMapMatlab,
                 fixationState=LoadState.unloaded, fixationType=InputType.fixationMapMatlab):
        self.image = ImageContainer(imagePath, imageType, imageState)
        self.saliency = ImageContainer(saliencyPath, saliencyType, saliencyState)
        self.fixation = ImageContainer(fixationPath, fixationType, fixationState)
