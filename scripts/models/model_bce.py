import lasagne
from lasagne.layers import InputLayer
import theano
import theano.tensor as T
import numpy as np

import generator
from model import Model


class ModelBCE(Model):
    def __init__(self, w, h, batch_size=32, lr=0.001):
        super(ModelBCE, self).__init__(w, h, batch_size)

        self.net = generator.build(self.inputHeight, self.inputWidth, self.input_var)

        output_layer_name = 'output'
        prediction = lasagne.layers.get_output(self.net[output_layer_name])

        test_prediction = lasagne.layers.get_output(self.net[output_layer_name], deterministic=True)
        self.predictFunction = theano.function([self.input_var], test_prediction)

        output_var_pooled = T.signal.pool.pool_2d(self.output_var, (4, 4), mode="average_exc_pad", ignore_border=True)
        prediction_pooled = T.signal.pool.pool_2d(prediction, (4, 4), mode="average_exc_pad", ignore_border=True)

        bce = lasagne.objectives.binary_crossentropy(prediction_pooled, output_var_pooled).mean()
        train_err = bce

        # parameters update and training
        G_params = lasagne.layers.get_all_params(self.net[output_layer_name], trainable=True)
        self.G_lr = theano.shared(np.array(lr, dtype=theano.config.floatX))
        G_updates = lasagne.updates.nesterov_momentum(train_err, G_params, learning_rate=self.G_lr, momentum=0.5)

        self.G_trainFunction = theano.function(inputs=[self.input_var, self.output_var], outputs=train_err, updates=G_updates,
                                               allow_input_downcast=True)
