from collections import OrderedDict
from lasagne.layers import (InputLayer, ConcatLayer, Pool2DLayer, ReshapeLayer, DimshuffleLayer, NonlinearityLayer,
                            DropoutLayer, Deconv2DLayer, batch_norm)
try:
    from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
except ImportError:
    from lasagne.layers import Conv2DLayer as ConvLayer
import lasagne
from lasagne.init import HeNormal
from lasagne.init import GlorotNormal
from layers import RGBtoBGRLayer
from lasagne.nonlinearities import sigmoid

def build(inputHeight, inputWidth, input_var,do_dropout=False):
    #net = OrderedDict()
    net = {'input': InputLayer((None, 3, inputHeight, inputWidth), input_var=input_var)}
    #net['input'] = InputLayer((None, 3, inputHeight, inputWidth), input_var=input_var)
    print "Input: {}".format(net['input'].output_shape[1:])

    net['bgr'] = RGBtoBGRLayer(net['input'])

    net['contr_1_1'] = batch_norm(ConvLayer(net['bgr'], 64, 3,pad='same',W=GlorotNormal(gain="relu")))
    print "convtr1_1: {}".format(net['contr_1_1'].output_shape[1:])
    net['contr_1_2'] = batch_norm(ConvLayer(net['contr_1_1'],64,3, pad='same',W=GlorotNormal(gain="relu")))
    print "convtr1_2: {}".format(net['contr_1_2'].output_shape[1:])
    net['pool1'] = Pool2DLayer(net['contr_1_2'], 2)
    print"pool1: {}".format(net['pool1'].output_shape[1:])
    

    net['contr_2_1'] = batch_norm(ConvLayer(net['pool1'], 128, 3, pad='same',W=GlorotNormal(gain="relu")))
    print "convtr2_1: {}".format(net['contr_2_1'].output_shape[1:])
    net['contr_2_2'] = batch_norm(ConvLayer(net['contr_2_1'], 128, 3, pad='same',W=GlorotNormal(gain="relu")))
    print "convtr2_2: {}".format(net['contr_2_2'].output_shape[1:])
    net['pool2'] = Pool2DLayer(net['contr_2_2'], 2)
    print "pool2: {}".format(net['pool2'].output_shape[1:])


    net['contr_3_1'] = batch_norm(ConvLayer(net['pool2'],256, 3, pad='same',W=GlorotNormal(gain="relu")))
    print "convtr3_1: {}".format(net['contr_3_1'].output_shape[1:])
    net['contr_3_2'] = batch_norm(ConvLayer(net['contr_3_1'], 256, 3, pad='same',W=GlorotNormal(gain="relu")))
    print "convtr3_2: {}".format(net['contr_3_2'].output_shape[1:])
    net['pool3'] = Pool2DLayer(net['contr_3_2'], 2)
    print "pool3: {}".format(net['pool3'].output_shape[1:])

    net['contr_4_1'] = batch_norm(ConvLayer(net['pool3'], 512, 3, pad='same',W=GlorotNormal(gain="relu")))
    print "convtr4_1: {}".format(net['contr_4_1'].output_shape[1:])
    net['contr_4_2'] = batch_norm(ConvLayer(net['contr_4_1'],512, 3,pad='same',W=GlorotNormal(gain="relu")))
    print "convtr4_2: {}".format(net['contr_4_2'].output_shape[1:])
    l = net['pool4'] = Pool2DLayer(net['contr_4_2'], 2)
    print "pool4: {}".format(net['pool4'].output_shape[1:])
    # the paper does not really describe where and how dropout is added. Feel free to try more options
    if do_dropout:
        l = DropoutLayer(l, p=0.4)

    net['encode_1'] = batch_norm(ConvLayer(l,1024, 3,pad='same', W=GlorotNormal(gain="relu")))
    print "encode_1: {}".format(net['encode_1'].output_shape[1:])
    net['encode_2'] = batch_norm(ConvLayer(net['encode_1'], 1024, 3,pad='same', W=GlorotNormal(gain="relu")))
    print "encode_2: {}".format(net['encode_2'].output_shape[1:])
    net['upscale1'] = batch_norm(Deconv2DLayer(net['encode_2'],1024, 2, 2, crop="valid", W=GlorotNormal(gain="relu")))
    print "upscale1: {}".format(net['upscale1'].output_shape[1:])

    net['concat1'] = ConcatLayer([net['upscale1'], net['contr_4_2']], cropping=(None, None, "center", "center"))
    print "concat1: {}".format(net['concat1'].output_shape[1:])
    net['expand_1_1'] = batch_norm(ConvLayer(net['concat1'], 512, 3,pad='same', W=GlorotNormal(gain="relu")))
    print "expand_1_1: {}".format(net['expand_1_1'].output_shape[1:])
    net['expand_1_2'] = batch_norm(ConvLayer(net['expand_1_1'],512, 3,pad='same',W=GlorotNormal(gain="relu")))
    print "expand_1_2: {}".format(net['expand_1_2'].output_shape[1:])
    net['upscale2'] = batch_norm(Deconv2DLayer(net['expand_1_2'], 512, 2, 2, crop="valid", W=GlorotNormal(gain="relu")))
    print "upscale2: {}".format(net['upscale2'].output_shape[1:])

    net['concat2'] = ConcatLayer([net['upscale2'], net['contr_3_2']], cropping=(None, None, "center", "center"))
    print "concat2: {}".format(net['concat2'].output_shape[1:])
    net['expand_2_1'] = batch_norm(ConvLayer(net['concat2'], 256, 3,pad='same',W=GlorotNormal(gain="relu")))
    print "expand_2_1: {}".format(net['expand_2_1'].output_shape[1:])
    net['expand_2_2'] = batch_norm(ConvLayer(net['expand_2_1'], 256, 3,pad='same',W=GlorotNormal(gain="relu")))
    print "expand_2_2: {}".format(net['expand_2_2'].output_shape[1:])
    net['upscale3'] = batch_norm(Deconv2DLayer(net['expand_2_2'],256, 2, 2, crop="valid",W=GlorotNormal(gain="relu")))
    print "upscale3: {}".format(net['upscale3'].output_shape[1:])

    net['concat3'] = ConcatLayer([net['upscale3'], net['contr_2_2']], cropping=(None, None, "center", "center"))
    print "concat3: {}".format(net['concat3'].output_shape[1:])
    net['expand_3_1'] = batch_norm(ConvLayer(net['concat3'], 128, 3,pad='same',W=GlorotNormal(gain="relu")))
    print "expand_3_1: {}".format(net['expand_3_1'].output_shape[1:])
    net['expand_3_2'] = batch_norm(ConvLayer(net['expand_3_1'],128, 3,pad='same', W=GlorotNormal(gain="relu")))
    print "expand_3_2: {}".format(net['expand_3_2'].output_shape[1:])
    net['upscale4'] = batch_norm(Deconv2DLayer(net['expand_3_2'], 128, 2, 2, crop="valid", W=GlorotNormal(gain="relu")))
    print "upscale4: {}".format(net['upscale4'].output_shape[1:])

    net['concat4'] = ConcatLayer([net['upscale4'], net['contr_1_2']], cropping=(None, None, "center", "center"))
    print "concat4: {}".format(net['concat4'].output_shape[1:])
    net['expand_4_1'] = batch_norm(ConvLayer(net['concat4'], 64, 3,pad='same', W=GlorotNormal(gain="relu")))
    print "expand_4_1: {}".format(net['expand_4_1'].output_shape[1:])
    net['expand_4_2'] = batch_norm(ConvLayer(net['expand_4_1'],64, 3,pad='same', W=GlorotNormal(gain="relu")))
    print "expand_4_2: {}".format(net['expand_4_2'].output_shape[1:])

    net['output'] = ConvLayer(net['expand_4_2'],1, 1, nonlinearity=sigmoid)
    print "output: {}".format(net['output'].output_shape[1:])
#    net['dimshuffle'] = DimshuffleLayer(net['output_segmentation'], (1, 0, 2, 3))
#    print "dimshuffle: {}".format(net['dimshuffle'].output_shape[1:])
#    net['reshapeSeg'] = ReshapeLayer(net['dimshuffle'], (2, -1))
#    print "reshapeSeg: {}".format(net['reshapeSeg'].output_shape[1:])
#    net['dimshuffle2'] = DimshuffleLayer(net['reshapeSeg'], (1, 0))
#    print "dimshuffle2: {}".format(net['dimshuffle2'].output_shape[1:])
#    net['output_flattened'] = NonlinearityLayer(net['dimshuffle2'], nonlinearity=lasagne.nonlinearities.softmax)
#    print "output_flattened: {}".format(net['output_flattened'].output_shape[1:])

    return net

