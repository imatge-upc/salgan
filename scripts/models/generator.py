from lasagne.layers import Conv2DLayer as ConvLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.layers import Upscale2DLayer
from lasagne.nonlinearities import sigmoid
import lasagne
import cPickle
import vgg16
import resnet50
import unet
from constants import PATH_TO_VGG16_WEIGHTS
from constants import PATH_TO_RESNET50_WEIGHTS
from constants import PATH_TO_FCN_WEIGHTS
import numpy as np


def set_vgg16_pretrained_weights(net, path_to_model_weights=PATH_TO_VGG16_WEIGHTS):
    # Set out weights
    # VGG-16
    vgg16 = cPickle.load(open(path_to_model_weights))
    num_elements_to_set = 26  # Number of W and b elements for the first convolutional layers
    # lasagne.layers.set_all_param_values(net['conv5_3'], vgg16['param values'][:num_elements_to_set])
    lasagne.layers.set_all_param_values (net['conv5_3'], vgg16['params'][:num_elements_to_set])


def set_resnet50_pretrained_weights(net, path_to_model_weights=PATH_TO_RESNET50_WEIGHTS):
    # Set out weights
    # ResNet-50
    resnet50 = cPickle.load(open(path_to_model_weights))
    lasagne.layers.set_all_param_values(net['res5c_relu'], resnet50['values'][:-2])


def build_encoder_vgg16(input_height, input_width, input_var):
    # VGG-16
    encoder = vgg16.build(input_height, input_width, input_var, mean_img=np.array([103.007, 116.668, 122.68]))
    set_vgg16_pretrained_weights(encoder, path_to_model_weights=PATH_TO_FCN_WEIGHTS)
    return encoder


def build_encoder_resnet50(input_height, input_width, input_var):
    # ResNet-50
    encoder = resnet50.build(input_height, input_width, input_var)
    set_resnet50_pretrained_weights(encoder)
    return encoder


def build_encoder_unet(input_height, input_width, input_var):
    # Unet
    encoder = unet.build(input_height, input_width, input_var)
    return encoder


def build_decoder_x16(net):
    """
    This decoder scale the input x16
    :param net:
    :return:
    """
    net['uconv5_3']= ConvLayer(net['conv5_3'], 512, 3, pad=1)
    print "uconv5_3: {}".format(net['uconv5_3'].output_shape[1:])

    net['uconv5_2'] = ConvLayer(net['uconv5_3'], 512, 3, pad=1)
    print "uconv5_2: {}".format(net['uconv5_2'].output_shape[1:])

    net['uconv5_1'] = ConvLayer(net['uconv5_2'], 512, 3, pad=1)
    print "uconv5_1: {}".format(net['uconv5_1'].output_shape[1:])

    net['upool4'] = Upscale2DLayer(net['uconv5_1'], scale_factor=2)
    print "upool4: {}".format(net['upool4'].output_shape[1:])

    net['uconv4_3'] = ConvLayer(net['upool4'], 512, 3, pad=1)
    print "uconv4_3: {}".format(net['uconv4_3'].output_shape[1:])

    net['uconv4_2'] = ConvLayer(net['uconv4_3'], 512, 3, pad=1)
    print "uconv4_2: {}".format(net['uconv4_2'].output_shape[1:])

    net['uconv4_1'] = ConvLayer(net['uconv4_2'], 512, 3, pad=1)
    print "uconv4_1: {}".format(net['uconv4_1'].output_shape[1:])

    net['upool3'] = Upscale2DLayer(net['uconv4_1'], scale_factor=2)
    print "upool3: {}".format(net['upool3'].output_shape[1:])

    net['uconv3_3'] = ConvLayer(net['upool3'], 256, 3, pad=1)
    print "uconv3_3: {}".format(net['uconv3_3'].output_shape[1:])

    net['uconv3_2'] = ConvLayer(net['uconv3_3'], 256, 3, pad=1)
    print "uconv3_2: {}".format(net['uconv3_2'].output_shape[1:])

    net['uconv3_1'] = ConvLayer(net['uconv3_2'], 256, 3, pad=1)
    print "uconv3_1: {}".format(net['uconv3_1'].output_shape[1:])

    net['upool2'] = Upscale2DLayer(net['uconv3_1'], scale_factor=2)
    print "upool2: {}".format(net['upool2'].output_shape[1:])

    net['uconv2_2'] = ConvLayer(net['upool2'], 128, 3, pad=1)
    print "uconv2_2: {}".format(net['uconv2_2'].output_shape[1:])

    net['uconv2_1'] = ConvLayer(net['uconv2_2'], 128, 3, pad=1)
    print "uconv2_1: {}".format(net['uconv2_1'].output_shape[1:])

    net['upool1'] = Upscale2DLayer(net['uconv2_1'], scale_factor=2)
    print "upool1: {}".format(net['upool1'].output_shape[1:])

    net['uconv1_2'] = ConvLayer(net['upool1'], 64, 3, pad=1,)
    print "uconv1_2: {}".format(net['uconv1_2'].output_shape[1:])

    net['uconv1_1'] = ConvLayer(net['uconv1_2'], 64, 3, pad=1)
    print "uconv1_1: {}".format(net['uconv1_1'].output_shape[1:])

    net['output'] = ConvLayer(net['uconv1_1'], 1, 1, pad=0,nonlinearity=sigmoid)
    print "output: {}".format(net['output'].output_shape[1:])

    return net


def build_decoder(net):
    """
    This decoder scale the input x16
    :param net:
    :return:
    """
    net['uconv5_3']= ConvLayer(net['res5c_relu'], 512, 3, pad=1)
    print "uconv5_3: {}".format(net['uconv5_3'].output_shape[1:])

    net['uconv5_2'] = ConvLayer(net['uconv5_3'], 512, 3, pad=1)
    print "uconv5_2: {}".format(net['uconv5_2'].output_shape[1:])

    net['uconv5_1'] = ConvLayer(net['uconv5_2'], 512, 3, pad=1)
    print "uconv5_1: {}".format(net['uconv5_1'].output_shape[1:])

    net['output'] = ConvLayer(net['uconv5_1'], 1, 1, pad=0, nonlinearity=sigmoid)
    print "output: {}".format(net['output'].output_shape[1:])

    return net


def build(input_height, input_width, input_var):
    # Use VGG-16 Configuration
    # encoder = build_encoder_vgg16(input_height, input_width, input_var)
    # generator = build_decoder_x16(encoder)

    # Use ResNet-50 Configuration
    encoder = build_encoder_resnet50(input_height, input_width, input_var)
    generator = build_decoder(encoder)

    # Use Unet Configuration
    # encoder = build_encoder_unet(input_height, input_width, input_var)
    # generator = {'output': ConvLayer(encoder['expand_4_2'], 1, 1, pad=0, nonlinearity=sigmoid)}
    # print "output: {}".format(generator['output'].output_shape[1:])

    return generator
