import lasagne
from lasagne.layers import InputLayer
import theano
import theano.tensor as T
import numpy as np

import generator
from model import Model


class ModelBCE(Model):
    def __init__(self, w, h, batch_size=10, lr=0.01,regterm=1e-4,momentum=0.99):
        super(ModelBCE, self).__init__(w, h, batch_size)

        self.net = generator.build(self.inputHeight, self.inputWidth, self.input_var)

        output_layer_name = 'output'

        prediction = lasagne.layers.get_output(self.net[output_layer_name],deterministic=False)
        bce = lasagne.objectives.binary_crossentropy(prediction, self.output_var).mean()
        #+ 5e-2 * lasagne.regularization.regularize_network_params(self.net[output_layer_name], lasagne.regularization.l2)
        train_err = bce
        # parameters update and training
        G_params = lasagne.layers.get_all_params(self.net[output_layer_name], trainable=True)
        self.G_lr = theano.shared(np.array(lr, dtype=theano.config.floatX))
        G_updates = lasagne.updates.momentum(train_err, G_params, learning_rate=self.G_lr,momentum=momentum)
        self.G_trainFunction = theano.function(inputs=[self.input_var, self.output_var], outputs=train_err, updates=G_updates)

        test_prediction = lasagne.layers.get_output(self.net[output_layer_name],deterministic=True)
    	test_loss = lasagne.objectives.binary_crossentropy(test_prediction,self.output_var).mean()
	test_acc = lasagne.objectives.binary_accuracy(test_prediction,self.output_var).mean()
	self.G_valFunction = theano.function(inputs=[self.input_var, self.output_var],outputs=[test_loss,test_acc])
        self.predictFunction = theano.function([self.input_var], test_prediction)
