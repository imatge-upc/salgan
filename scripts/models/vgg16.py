# VGG-16, 16-layer model from the paper:
# "Very Deep Convolutional Networks for Large-Scale Image Recognition"
# Original source: https://gist.github.com/ksimonyan/211839e770f7b538e2d8
# License: see http://www.robots.ox.ac.uk/~vgg/research/very_deep/

from lasagne.layers import Conv2DLayer as ConvLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.layers import InputLayer
from layers import RGBtoBGRLayer
from lasagne.layers import batch_norm
from lasagne.nonlinearities import sigmoid


def build(inputHeight, inputWidth, input_var):
    """
    Bulid only Convolutional part of the VGG-16 Layer model, all fully connected layers are removed.
    First 3 group of ConvLayers are fixed (not trainable).

    :param input_layer: Input layer of the network.
    :return: Dictionary that contains all layers.
    """

    net = {'input': InputLayer((None, 3, inputHeight, inputWidth), input_var=input_var)}
    print "Input: {}".format(net['input'].output_shape[1:])

    net['bgr'] = RGBtoBGRLayer(net['input'])

    net['conv1_1'] = batch_norm(ConvLayer(net['bgr'], 64, 3, pad=1, flip_filters=False))
    print "conv1_1: {}".format(net['conv1_1'].output_shape[1:])

    net['conv1_2'] = batch_norm(ConvLayer(net['conv1_1'], 64, 3, pad=1, flip_filters=False))
    print "conv1_2: {}".format(net['conv1_2'].output_shape[1:])

    net['pool1'] = PoolLayer(net['conv1_2'], 2)
    print "pool1: {}".format(net['pool1'].output_shape[1:])

    net['conv2_1'] = batch_norm(ConvLayer(net['pool1'], 128, 3, pad=1, flip_filters=False))
    print "conv2_1: {}".format(net['conv2_1'].output_shape[1:])

    net['conv2_2'] = batch_norm(ConvLayer(net['conv2_1'], 128, 3, pad=1, flip_filters=False))
    print "conv2_2: {}".format(net['conv2_2'].output_shape[1:])

    net['pool2'] = PoolLayer(net['conv2_2'], 2)
    print "pool2: {}".format(net['pool2'].output_shape[1:])

    net['conv3_1'] = batch_norm(ConvLayer(net['pool2'], 256, 3, pad=1, flip_filters=False))
    print "conv3_1: {}".format(net['conv3_1'].output_shape[1:])

    net['conv3_2'] = batch_norm(ConvLayer(net['conv3_1'], 256, 3, pad=1, flip_filters=False))
    print "conv3_2: {}".format(net['conv3_2'].output_shape[1:])

    net['conv3_3'] = batch_norm(ConvLayer(net['conv3_2'], 256, 3, pad=1, flip_filters=False))
    print "conv3_3: {}".format(net['conv3_3'].output_shape[1:])

    net['pool3'] = PoolLayer(net['conv3_3'], 2)
    print "pool3: {}".format(net['pool3'].output_shape[1:])

    net['conv4_1'] = batch_norm(ConvLayer(net['pool3'], 512, 3, pad=1, flip_filters=False))
    print "conv4_1: {}".format(net['conv4_1'].output_shape[1:])

    net['conv4_2'] =  batch_norm(ConvLayer(net['conv4_1'], 512, 3, pad=1, flip_filters=False))
    print "conv4_2: {}".format(net['conv4_2'].output_shape[1:])

    net['conv4_3'] =  batch_norm(ConvLayer(net['conv4_2'], 512, 3, pad=1, flip_filters=False))
    print "conv4_3: {}".format(net['conv4_3'].output_shape[1:])

    net['pool4'] = PoolLayer(net['conv4_3'], 2)
    print "pool4: {}".format(net['pool4'].output_shape[1:])

    net['conv5_1'] =  batch_norm(ConvLayer(net['pool4'], 512, 3, pad=1, flip_filters=False))
    print "conv5_1: {}".format(net['conv5_1'].output_shape[1:])

    net['conv5_2'] =  batch_norm(ConvLayer(net['conv5_1'], 512, 3, pad=1, flip_filters=False))
    print "conv5_2: {}".format(net['conv5_2'].output_shape[1:])

    net['conv5_3'] =  batch_norm(ConvLayer(net['conv5_2'], 512, 3, pad=1, flip_filters=False))
    print "conv5_3: {}".format(net['conv5_3'].output_shape[1:])
#    net['pool5'] = PoolLayer(net['conv5_3'], 2)
#    print "pool5: {}".format(net['pool5'].output_shape[1:])

    net['output'] =  batch_norm(ConvLayer(net['conv5_3'], 1, 1, pad=0,nonlinearity=sigmoid))
    print "output: {}".format(net['output'].output_shape[1:])

    return net

