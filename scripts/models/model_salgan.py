import lasagne
from lasagne.layers import InputLayer
import theano
import theano.tensor as T
import numpy as np

import generator
import discriminator
from model import Model


class ModelSALGAN(Model):
    def __init__(self, w, h, batch_size=32, G_lr=3e-4, D_lr=3e-4, alpha=1/20.):
        super(ModelSALGAN, self).__init__(w, h, batch_size)

        # Build Generator
        self.net = generator.build(self.inputHeight, self.inputWidth, self.input_var)

        # Build Discriminator
        self.discriminator = discriminator.build(self.inputHeight, self.inputWidth,
                                                 T.concatenate([self.output_var, self.input_var], axis=1))

        # Set prediction function
        output_layer_name = 'output'

        prediction = lasagne.layers.get_output(self.net[output_layer_name])
        test_prediction = lasagne.layers.get_output(self.net[output_layer_name], deterministic=True)
        self.predictFunction = theano.function([self.input_var], test_prediction)

        disc_lab = lasagne.layers.get_output(self.discriminator['prob'],
                                             T.concatenate([self.output_var, self.input_var], axis=1))
        disc_gen = lasagne.layers.get_output(self.discriminator['prob'],
                                             T.concatenate([prediction, self.input_var], axis=1))

        # Downscale the saliency maps
        output_var_pooled = T.signal.pool.pool_2d(self.output_var, (4, 4), mode="average_exc_pad", ignore_border=True)
        prediction_pooled = T.signal.pool.pool_2d(prediction, (4, 4), mode="average_exc_pad", ignore_border=True)
        train_err = lasagne.objectives.binary_crossentropy(prediction_pooled, output_var_pooled).mean()
        + 1e-4 * lasagne.regularization.regularize_network_params(self.net[output_layer_name], lasagne.regularization.l2)

        # Define loss function and input data
        ones = T.ones(disc_lab.shape)
        zeros = T.zeros(disc_lab.shape)
        D_obj = lasagne.objectives.binary_crossentropy(T.concatenate([disc_lab, disc_gen], axis=0),
                                                       T.concatenate([ones, zeros], axis=0)).mean()
        + 1e-4 * lasagne.regularization.regularize_network_params(self.discriminator['prob'], lasagne.regularization.l2)

        G_obj_d = lasagne.objectives.binary_crossentropy(disc_gen, T.ones(disc_lab.shape)).mean()
        + 1e-4 * lasagne.regularization.regularize_network_params(self.net[output_layer_name], lasagne.regularization.l2)

        G_obj = G_obj_d + train_err * alpha
        cost = [G_obj, D_obj, train_err]

        # parameters update and training of Generator
        G_params = lasagne.layers.get_all_params(self.net[output_layer_name], trainable=True)
        self.G_lr = theano.shared(np.array(G_lr, dtype=theano.config.floatX))
        G_updates = lasagne.updates.adagrad(G_obj, G_params, learning_rate=self.G_lr)
        self.G_trainFunction = theano.function(inputs=[self.input_var, self.output_var], outputs=cost,
                                               updates=G_updates, allow_input_downcast=True)

        # parameters update and training of Discriminator
        D_params = lasagne.layers.get_all_params(self.discriminator['prob'], trainable=True)
        self.D_lr = theano.shared(np.array(D_lr, dtype=theano.config.floatX))
        D_updates = lasagne.updates.adagrad(D_obj, D_params, learning_rate=self.D_lr)
        self.D_trainFunction = theano.function([self.input_var, self.output_var], cost, updates=D_updates,
                                               allow_input_downcast=True)

