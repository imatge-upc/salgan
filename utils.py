import numpy as np
import cv2
import theano
from constants import HOME_DIR


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in xrange(0, len(l), n):
        yield l[i:i + n]


def load_weights(net, path, epochtoload):
    """
    Load a pretrained model
    :param epochtoload: epoch to load
    :param net: model object
    :param path: path of the weights to be set
    """
    with np.load(HOME_DIR + path + "modelWeights{:04d}.npz".format(epochtoload)) as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(net, param_values)


def feature_extraction(model, validationData, numEpoch, dir='test'):

    blob_img = np.zeros((1, 3, model.inputHeight, model.inputWidth), theano.config.floatX)
    blob_salmap = np.zeros((1, 1, model.height, model.width), theano.config.floatX)

    blob_img[0, ...] = validationData.image.data.astype(theano.config.floatX).transpose(2, 0, 1)
    blob_salmap[0, ...] = (validationData.saliency.data.astype(theano.config.floatX)) / 255.

    result = np.squeeze(model.featureFunction(blob_img, blob_salmap))

    featureMap = (result * 255.).astype(np.uint8)

    cv2.imwrite('./' + dir + '/validationRandomSaliencyPred_{:04d}.png'.format(numEpoch),
                cv2.cvtColor(featureMap.transpose(1, 2, 0), cv2.COLOR_RGB2BGR))

    # cv2.imwrite('./results/validationRandomImage_'+str(numEpoch)+'.png',
    #            cv2.cvtColor(validationData.image.data, cv2.COLOR_RGB2BGR))
    # cv2.imwrite('./results/validationRandomSaliencyGT_'+str(numEpoch)+'.png', validationData.saliency.data)


def predict(model, image_stimuli, numEpoch=None, name=None, pathOutputMaps=None):

    size = (image_stimuli.shape[1], image_stimuli.shape[0])

    if image_stimuli.shape[:2] != (model.inputHeight, model.inputWidth):
        image_stimuli = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

    blob = np.zeros((1, 3, model.inputHeight, model.inputWidth), theano.config.floatX)

    blob[0, ...] = (image_stimuli.astype(theano.config.floatX).transpose(2, 0, 1))

    result = np.squeeze(model.predictFunction(blob))
    saliencyMap = (result * 255).astype(np.uint8)

    # resize back to original size
    saliencyMap = cv2.resize(saliencyMap, size, interpolation=cv2.INTER_CUBIC)
    # blur
    # saliencyMap = cv2.GaussianBlur(saliencyMap, (61, 61), 0)
    # clip again
    saliencyMap = np.clip(saliencyMap, 0, 255)
    if name is None:
        # When we use for testing, there is no file name provided.
        cv2.imwrite('./' + pathOutputMaps + '/validationRandomSaliencyPred_{:04d}.png'.format(numEpoch), saliencyMap)
    else:
        cv2.imwrite(os.path.join(pathOutputMaps, name + '.jpg'), saliencyMap)


