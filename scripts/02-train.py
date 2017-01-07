#   Two mode of training available:
#       - BCE: CNN training, NOT Adversarial Training here. Only learns the generator network.
#       - SALGAN: Adversarial Training. Updates weights for both Generator and Discriminator.
#   The training used data previously  processed using "01-data_preocessing.py"
import os
import numpy as np
import sys
import cPickle as pickle
import random
import cv2
import theano
import theano.tensor as T
import lasagne

from tqdm import tqdm
from constants import *
from models.model_salgan import ModelSALGAN
from models.model_bce import ModelBCE
from utils import *

flag = str(sys.argv[1])


def bce_batch_iterator(model, train_data, validation_sample):
    num_epochs = 301
    n_updates = 1
    nr_batches_train = int(len(train_data) / model.batch_size)
    for current_epoch in tqdm(range(num_epochs), ncols=20):
        e_cost = 0.

        random.shuffle(train_data)

        for currChunk in chunks(train_data, model.batch_size):

            if len(currChunk) != model.batch_size:
                continue

            batch_input = np.asarray([x.image.data.astype(theano.config.floatX).transpose(2, 0, 1) for x in currChunk],
                                     dtype=theano.config.floatX)

            batch_output = np.asarray([y.saliency.data.astype(theano.config.floatX) / 255. for y in currChunk],
                                      dtype=theano.config.floatX)
            batch_output = np.expand_dims(batch_output, axis=1)

            # train generator with one batch and discriminator with next batch
            G_cost = model.G_trainFunction(batch_input, batch_output)
            e_cost += G_cost
            n_updates += 1

        e_cost /= nr_batches_train

        print 'Epoch:', current_epoch, ' train_loss->', e_cost

        if current_epoch % 5 == 0:
            np.savez('./' + DIR_TO_SAVE + '/gen_modelWeights{:04d}.npz'.format(current_epoch),
                     *lasagne.layers.get_all_param_values(model.net['output']))
            predict(model=model, image_stimuli=validation_sample, num_epoch=current_epoch, path_output_maps=DIR_TO_SAVE)


def salgan_batch_iterator(model, train_data, validation_sample):
    num_epochs = 301
    nr_batches_train = int(len(train_data) / model.batch_size)
    n_updates = 1
    for current_epoch in tqdm(range(num_epochs), ncols=20):

        g_cost = 0.
        d_cost = 0.
        e_cost = 0.

        random.shuffle(train_data)

        for currChunk in chunks(train_data, model.batch_size):

            if len(currChunk) != model.batch_size:
                continue

            batch_input = np.asarray([x.image.data.astype(theano.config.floatX).transpose(2, 0, 1) for x in currChunk],
                                     dtype=theano.config.floatX)
            batch_output = np.asarray([y.saliency.data.astype(theano.config.floatX) / 255. for y in currChunk],
                                      dtype=theano.config.floatX)
            batch_output = np.expand_dims(batch_output, axis=1)

            # train generator with one batch and discriminator with next batch
            if n_updates % 2 == 0:
                G_obj, D_obj, G_cost = model.G_trainFunction(batch_input, batch_output)
                d_cost += D_obj
                g_cost += G_obj
                e_cost += G_cost
            else:
                G_obj, D_obj, G_cost = model.D_trainFunction(batch_input, batch_output)
                d_cost += D_obj
                g_cost += G_obj
                e_cost += G_cost

            n_updates += 1

        g_cost /= nr_batches_train
        d_cost /= nr_batches_train
        e_cost /= nr_batches_train

        # Save weights every 3 epoch
        if current_epoch % 3 == 0:
            np.savez('./' + DIR_TO_SAVE + '/gen_modelWeights{:04d}.npz'.format(current_epoch),
                     *lasagne.layers.get_all_param_values(model.net['output']))
            np.savez('./' + DIR_TO_SAVE + '/disrim_modelWeights{:04d}.npz'.format(current_epoch),
                     *lasagne.layers.get_all_param_values(model.discriminator['fc5']))
            predict(model=model, image_stimuli=validation_sample, numEpoch=current_epoch, pathOutputMaps=DIR_TO_SAVE)
        print 'Epoch:', current_epoch, ' train_loss->', (g_cost, d_cost, e_cost)


def train():
    """
    Train both generator and discriminator
    :return:
    """
    # Load data
    print 'Loading training data...'
    with open('../saliency-2016-lsun/validationSample240x320.pkl', 'rb') as f:
    # with open(TRAIN_DATA_DIR, 'rb') as f:
        train_data = pickle.load(f)
    print '-->done!'

    print 'Loading validation data...'
    with open('../saliency-2016-lsun/validationSample240x320.pkl', 'rb') as f:
    # with open(VALIDATION_DATA_DIR, 'rb') as f:
        validation_data = pickle.load(f)
    print '-->done!'

    # Choose a random sample to monitor the training
    num_random = random.choice(range(len(validation_data)))
    validation_sample = validation_data[num_random]
    cv2.imwrite('./' + DIR_TO_SAVE + '/validationRandomSaliencyGT.png', validation_sample.saliency.data)
    cv2.imwrite('./' + DIR_TO_SAVE + '/validationRandomImage.png', cv2.cvtColor(validation_sample.image.data,
                                                                                cv2.COLOR_RGB2BGR))

    # Create network

    if flag == 'salgan':
        model = ModelSALGAN(INPUT_SIZE[0], INPUT_SIZE[1])
        # Load a pre-trained model
        # load_weights(net=model.net['output'], path="nss/gen_", epochtoload=15)
        # load_weights(net=model.discriminator['fc5'], path="test_dialted/disrim_", epochtoload=54)
        salgan_batch_iterator(model, train_data, validation_sample.image.data)

    elif flag == 'bce':
        model = ModelBCE(INPUT_SIZE[0], INPUT_SIZE[1])
        # Load a pre-trained model
        # load_weights(net=model.net['output'], path='test/gen_', epochtoload=15)
        bce_batch_iterator(model, train_data, validation_sample.image.data)
    else:
        print "Invalid input argument."
if __name__ == "__main__":
    train()
