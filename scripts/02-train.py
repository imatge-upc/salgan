import os
import numpy as np
import sys
import cPickle as pickle
import random
import cv2
import theano
import theano.tensor as T
import lasagne
import matplotlib.pyplot as plt
from sympy.utilities.iterables import cartes
from tqdm import tqdm
from constants import *
from models.model_salgan import ModelSALGAN
from models.model_bce import ModelBCE
from utils import *
import pdb

flag = str(sys.argv[1])


def bce_batch_iterator(model, train_data, validation_data,validation_sample,epochs = 100, fig=False):
    num_epochs = epochs
    n_updates = 1
    nr_batches_train = int(len(train_data) / model.batch_size)
    nr_batches_val = int(len(validation_data) / model.batch_size)
    train_loss_plt, val_loss_plt, val_acc_plt = [[] for i in range(3)]
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
            G_cost = model.G_trainFunction(batch_input, batch_output)
            e_cost += G_cost
            n_updates += 1
        e_cost /= nr_batches_train
	train_loss_plt.append(e_cost)
        print '\n  train_loss->', e_cost

	v_cost = 0.
	v_acc = 0.
        for currChunk in chunks(validation_data, model.batch_size):
            if len(currChunk) != model.batch_size:
                continue
            batch_input = np.asarray([x.image.data.astype(theano.config.floatX).transpose(2, 0, 1) for x in currChunk],
                                        dtype=theano.config.floatX)

            batch_output = np.asarray([y.saliency.data.astype(theano.config.floatX) / 255. for y in currChunk],
                                         dtype=theano.config.floatX)
            batch_output = np.expand_dims(batch_output, axis=1)
            val_loss, val_accuracy = model.G_valFunction(batch_input,batch_output)
	    v_cost += val_loss
	    v_acc += val_accuracy
	v_cost /= nr_batches_val
	v_acc  /= nr_batches_val
	val_loss_plt.append(v_cost)
	val_acc_plt.append(v_acc)

        print "  validation_loss->", v_cost
        print "  validation_acc->", v_acc
	print("-----------------------------------------------")
        if current_epoch % 5 == 0:
	    if fig is True:
	        fig1 = plt.figure(1)
	        plt.title("Train and Val loss")
	        plt.xlabel("Epochs")
	        plt.ylabel("Loss")
    	        plt.plot(range(current_epoch+1),train_loss_plt,color='red',linestyle='-',label='Train Loss')
    	        plt.plot(range(current_epoch+1),val_loss_plt,color='blue',linestyle='-',label='Validation Loss')
	        plt.legend()
	        plt.savefig('./'+DIR_TO_SAVE+'/figure1_{:04d}.png'.format(current_epoch))
	        plt.close(fig1)
            np.savez('./' + DIR_TO_SAVE + '/gen_modelWeights{:04d}.npz'.format(current_epoch),
                     *lasagne.layers.get_all_param_values(model.net['output']))
            predict(model=model, image_stimuli=validation_sample, num_epoch=current_epoch, path_output_maps=DIR_TO_SAVE)
    return v_acc

def salgan_batch_iterator(model, train_data, validation_sample):
    num_epochs = 100
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
        if current_epoch % 5  == 0:
            np.savez('./' + DIR_TO_SAVE + '/gen_modelWeights{:04d}.npz'.format(current_epoch),
                     *lasagne.layers.get_all_param_values(model.net['output']))
            np.savez('./' + DIR_TO_SAVE + '/disrim_modelWeights{:04d}.npz'.format(current_epoch),
                     *lasagne.layers.get_all_param_values(model.discriminator['fc5']))
            predict(model=model, image_stimuli=validation_sample, num_epoch=current_epoch, path_output_maps=DIR_TO_SAVE)
        print 'Epoch:', current_epoch, ' train_loss->', (g_cost, d_cost, e_cost)


def train():
    """
    Train both generator and discriminator
    :return:
    """
    # Load data
    print 'Loading training data...'
    with open('/home/titan/Saeed/saliency-salgan-2017/data/pickle320x240/validationData.pickle', 'rb') as f:
    # with open(TRAIN_DATA_DIR, 'rb') as f:
        train_data = pickle.load(f)
    print '-->done!'

    print 'Loading validation data...'
    with open('/home/titan/Saeed/saliency-salgan-2017/data/pickle320x240/validationData.pickle', 'rb') as f:
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
        load_weights(net=model.net['output'], path="test_gen_only/gen_", epochtoload=10)
        # load_weights(net=model.discriminator['fc5'], path="test_dialted/disrim_", epochtoload=54)
        salgan_batch_iterator(model, train_data, validation_data, validation_sample.image.data)

    elif flag == 'bce':
        model = ModelBCE(INPUT_SIZE[0], INPUT_SIZE[1])
        # Load a pre-trained model
        # load_weights(net=model.net['output'], path='test/gen_', epochtoload=15)
        bce_batch_iterator(model, train_data, validation_data,validation_sample.image.data)
    else:
        print "Invalid input argument."
def cross_val(): 
    # Load data
    print 'Loading training data...'
    with open(TRAIN_DATA_DIR_CROSS, 'rb') as f:
        train_data = pickle.load(f)
    print '-->done!'

    print 'Loading validation data...'
    with open(VAL_DATA_DIR, 'rb') as f:
        validation_data = pickle.load(f)
    print '-->done!'

    lr_list = [0.1,0.01,0.001,0.0001]
    regterm_list = [1e-1,1e-2,1e-3,1e-4,1e-5]
    momentum_list = [0.5,0.7,0.9,0,99]
    lr, regterm,mom,acc = [[] for i in range(4)]
    for config_list in list(cartes(lr_list,regterm_list,momentum_list)):
	if flag == 'bce':
            model = ModelBCE(INPUT_SIZE[0], INPUT_SIZE[1],10,config_list[0],config_list[1],config_list[2])
            val_accuracy = bce_batch_iterator(model, train_data, validation_data,validation_sample.image.data)
  	    lr.append(config_list[0])
  	    regterm.append(config_list[1])
  	    mom.append(config_list[2])
  	    acc.append(val_accuracy)
    for l,r,m,a in zip(lr,regterm,mom,acc):
   	print ("lr: {}, lambda: {}, momentum: {}, accuracy: {}").format(l,r,m,a)
	print('------------------------------------------------------------------') 	    
    
if __name__ == "__main__":
    cross_val()
