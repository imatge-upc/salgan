from lasagne.layers import Conv2DLayer as ConvLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.layers import DenseLayer, InputLayer
from lasagne.nonlinearities import tanh, sigmoid
from lasagne.layers import batch_norm
from lasagne.init import GlorotUniform
import nn


def build(input_height, input_width, concat_var):
    """
    Build the discriminator, all weights initialized from scratch
    :param input_width:
    :param input_height: 
    :param concat_var: Theano symbolic tensor variable
    :return: Dictionary that contains the discriminator
    """

    net = {'input': InputLayer((None, 4, input_height, input_width), input_var=concat_var)}
    print "Input: {}".format(net['input'].output_shape[1:])

    net['merge'] = batch_norm(ConvLayer(net['input'], 3, 1, pad=0, W=GlorotUniform(gain="relu"),flip_filters=False))
    print "merge: {}".format(net['merge'].output_shape[1:])

    net['conv1'] = batch_norm(ConvLayer(net['merge'], 32, 3, pad=1,W=GlorotUniform(gain="relu")))
    print "conv1: {}".format(net['conv1'].output_shape[1:])

    net['pool1'] = PoolLayer(net['conv1'], 4)
    print "pool1: {}".format(net['pool1'].output_shape[1:])

    net['conv2_1'] = batch_norm(ConvLayer(net['pool1'], 64, 3, pad=1,W=GlorotUniform(gain="relu")))
    print "conv2_1: {}".format(net['conv2_1'].output_shape[1:])

    net['conv2_2'] = batch_norm(ConvLayer(net['conv2_1'], 64, 3, pad=1,W=GlorotUniform(gain="relu")))
    print "conv2_2: {}".format(net['conv2_2'].output_shape[1:])

    net['pool2'] = PoolLayer(net['conv2_2'], 2)
    print "pool2: {}".format(net['pool2'].output_shape[1:])

    net['conv3_1'] = batch_norm(ConvLayer(net['pool2'], 64, 3, pad=1,W=GlorotUniform(gain="relu")))
    print "conv3_1: {}".format(net['conv3_1'].output_shape[1:])

    net['conv3_2'] = batch_norm(ConvLayer(net['conv3_1'], 64, 3, pad=1,W=GlorotUniform(gain="relu")))
    print "conv3_2: {}".format(net['conv3_2'].output_shape[1:])

    net['pool3'] = PoolLayer(net['conv3_2'], 2)
    print "pool3: {}".format(net['pool3'].output_shape[1:])

    net['fc4'] = batch_norm(DenseLayer(net['pool3'], num_units=100,W=GlorotUniform(gain="relu")))
    print "fc4: {}".format(net['fc4'].output_shape[1:])

    net['fc5'] = batch_norm(DenseLayer(net['fc4'], num_units=2,W=GlorotUniform(gain="relu")))
    print "fc5: {}".format(net['fc5'].output_shape[1:])

    net['prob'] = batch_norm(DenseLayer(net['fc5'], num_units=1, W=GlorotUniform(gain=1.0),nonlinearity=sigmoid))
    print "prob: {}".format(net['prob'].output_shape[1:])

    return net
