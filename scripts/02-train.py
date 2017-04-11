import os
import numpy as np
import sys
import cPickle as pickle
import random
import cv2
import theano
import theano.tensor as T
import lasagne
from sympy.utilities.iterables import cartes
from tqdm import tqdm
from constants import *
from models.model_salgan import ModelSALGAN
from models.model_bce import ModelBCE
from utils import *
import pdb
import matplotlib

#####################################
#To bypass X11 for matplotlib in tmux
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#####################################
flag = str(sys.argv[1])

def bce_batch_iterator(model, train_data, validation_data, validation_sample, epochs = 10, fig=False):
    num_epochs = epochs+1
    n_updates = 1
    nr_batches_train = int(len(train_data) / model.batch_size)
    train_loss_plt, train_acc_plt, val_loss_plt, val_acc_plt = [[] for i in range(4)]
    for current_epoch in tqdm(range(num_epochs), ncols=20):
        counter = 0
        e_cost = 0.;tr_acc = 0.; tr_loss = 0.
        random.shuffle(train_data)
        for currChunk in chunks(train_data, model.batch_size):
            if len(currChunk) != model.batch_size:
                continue
            batch_input = np.asarray([x.image.data.astype(theano.config.floatX).transpose(2, 0, 1) for x in currChunk],dtype=theano.config.floatX)
            batch_output = np.asarray([y.saliency.data.astype(theano.config.floatX) / 255. for y in currChunk],dtype=theano.config.floatX)
            batch_output = np.expand_dims(batch_output, axis=1)
            G_cost = model.G_trainFunction(batch_input, batch_output)
            if counter < 20:
	        tr_l, tr_jac = model.G_valFunction(batch_input,batch_output)
                tr_loss += tr_l; tr_acc += tr_jac
		counter += 1
            e_cost += G_cost;
            n_updates += 1
        e_cost /= nr_batches_train; tr_acc /= counter; tr_loss /= counter
	train_loss_plt.append(tr_loss);train_acc_plt.append(tr_acc)
        print '\n  train_loss->', e_cost
        print '  train_accuracy(subset)->', tr_acc
        
	v_cost, v_acc = bce_feedforward(model,validation_data,True)
	val_loss_plt.append(v_cost);val_acc_plt.append(v_acc)
        if current_epoch % 5 == 0:
	    if fig is True:
		draw_figs(train_loss_plt, val_loss_plt, 'Train Loss', 'Val Loss')
		draw_figs(train_acc_plt, val_acc_plt, 'Train Acc', 'Val Acc')
            np.savez('./' + DIR_TO_SAVE + '/gen_modelWeights{:04d}.npz'.format(current_epoch),
                     *lasagne.layers.get_all_param_values(model.net['output']))
            predict(model=model, image_stimuli=validation_sample, num_epoch=current_epoch, path_output_maps=FIG_SAVE_DIR)
    return v_acc

def salgan_batch_iterator(model, train_data, validation_data,validation_sample,epochs = 20, fig=False):
    num_epochs = epochs+1
    nr_batches_train = int(len(train_data) / model.batch_size)
    train_loss_plt, train_acc_plt, val_loss_plt, val_acc_plt = [[] for i in range(4)]
    n_updates = 1
    for current_epoch in tqdm(range(num_epochs), ncols=20):
	g_cost = 0.; d_cost = 0.; e_cost = 0.
        random.shuffle(train_data)
        for currChunk in chunks(train_data, model.batch_size):
            if len(currChunk) != model.batch_size:
                continue
            batch_input = np.asarray([x.image.data.astype(theano.config.floatX).transpose(2, 0, 1) for x in currChunk],dtype=theano.config.floatX)
            batch_output = np.asarray([y.saliency.data.astype(theano.config.floatX) / 255. for y in currChunk],dtype=theano.config.floatX)
            batch_output = np.expand_dims(batch_output, axis=1)
            if n_updates % 2 == 0:
                G_obj, D_obj, G_cost = model.G_trainFunction(batch_input, batch_output)
                d_cost += D_obj; g_cost += G_obj; e_cost += G_cost
            else:
                G_obj, D_obj, G_cost = model.D_trainFunction(batch_input, batch_output)
                d_cost += D_obj; g_cost += G_obj; e_cost += G_cost
            n_updates += 1
        g_cost /= nr_batches_train
	d_cost /= nr_batches_train
	e_cost /= nr_batches_train	
	#Compute the Jaccard Index on the Validation
	v_cost, v_acc = bce_feedforward(model,validation_data,True)

	if current_epoch % 5  == 0:
            np.savez('./' + DIR_TO_SAVE + '/gen_modelWeights{:04d}.npz'.format(current_epoch),
                     *lasagne.layers.get_all_param_values(model.net['output']))
            np.savez('./' + DIR_TO_SAVE + '/disrim_modelWeights{:04d}.npz'.format(current_epoch),
                     *lasagne.layers.get_all_param_values(model.discriminator['fc5']))
            predict(model=model, image_stimuli=validation_sample, num_epoch=current_epoch, path_output_maps=FIG_SAVE_DIR)
    return v_acc

def bce_feedforward(model, validation_data, bPrint=False):
    nr_batches_val = int(len(validation_data) / model.batch_size)
    v_cost = 0.
    v_acc = 0.
    for currChunk in chunks(validation_data, model.batch_size):
        if len(currChunk) != model.batch_size:
            continue
        batch_input = np.asarray([x.image.data.astype(theano.config.floatX).transpose(2, 0, 1) for x in currChunk],dtype=theano.config.floatX)
        batch_output = np.asarray([y.saliency.data.astype(theano.config.floatX) / 255. for y in currChunk],dtype=theano.config.floatX)
        batch_output = np.expand_dims(batch_output, axis=1)
        val_loss, val_accuracy = model.G_valFunction(batch_input,batch_output)
        v_cost += val_loss
        v_acc += val_accuracy
    v_cost /= nr_batches_val
    v_acc /= nr_batches_val
    if bPrint is True:
        print "  validation_accuracy -->", v_acc
	print "  validation_loss -->", v_cost
	print "-----------------------------------------"
    return v_cost, v_acc

def draw_figs(x,y,current_epoch,label1,label2):
    fig1 = plt.figure(1)
    plt.plot(range(current_epoch+1),x,color='red',linestyle='-',label=label1)
    plt.plot(range(current_epoch+1),val_loss_plt,color='blue',linestyle='-',label=label2)
    if label == 'Train Loss': 
        plt.title("Train and Val loss");plt.xlabel("Epochs");plt.ylabel("Loss")
        plt.legend()
        plt.savefig('./'+FIG_SAVE_DIR+'/train_val_loss_{:04d}.png'.format(current_epoch))
        plt.close(fig1)
    else:
        plt.title("Train and Val Accuracy");plt.xlabel("Epochs");plt.ylabel("Acc")
        plt.legend()
        plt.savefig('./'+FIG_SAVE_DIR+'/train_val_acc_{:04d}.png'.format(current_epoch))
        plt.close(fig1)

def test():
    """
    Tests generator on the test set
    :return:
    """
    # Load data
    print 'Loading test data...'
    with open(TEST_DATA_DIR, 'rb') as f:
        test_data = pickle.load(f)
    print '-->done!'    
    model = ModelSALGAN(INPUT_SIZE[0], INPUT_SIZE[1],9,0.01,1e-05,0.01,0.2)
    load_weights(net=model.net['output'], path='weights/gen_', epochtoload=15) 
    bce_feedforward(model,test_data,bPrint=True)
def train():
    """
    Train both generator and discriminator
    :return:
    """
    # Load data
    print 'Loading training data...'
    with open(TRAIN_DATA_DIR, 'rb') as f:
        train_data = pickle.load(f)
    print '-->done!'

    print 'Loading test data...'
    with open(TEST_DATA_DIR, 'rb') as f:
        validation_data = pickle.load(f)
    print '-->done!'

    # Choose a random sample to monitor the training
    num_random = random.choice(range(len(validation_data)))
    validation_sample = validation_data[num_random]
    cv2.imwrite('./' + FIG_SAVE_DIR + '/validationRandomSaliencyGT.png', validation_sample.saliency.data)
    cv2.imwrite('./' + FIG_SAVE_DIR + '/validationRandomImage.png', cv2.cvtColor(validation_sample.image.data,
                                                                                cv2.COLOR_RGB2BGR))
    # Create network
    if flag == 'salgan':
        model = ModelSALGAN(INPUT_SIZE[0], INPUT_SIZE[1],9,0.01,1e-05,0.01,0.2)
        # Load a pre-trained model
        load_weights(net=model.net['output'], path="weights/gen_", epochtoload=15)
        load_weights(net=model.discriminator['fc5'], path="weights/disrim_", epochtoload=15)
        salgan_batch_iterator(model, train_data, validation_data,validation_sample.image.data,epochs=5)

    elif flag == 'bce':
        model = ModelBCE(INPUT_SIZE[0], INPUT_SIZE[1],10,0.05,1e-5,0.99)
        # Load a pre-trained model
        # load_weights(net=model.net['output'], path='test/gen_', epochtoload=15)
        bce_batch_iterator(model, train_data, validation_data,validation_sample.image.data,epochs=10)
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
    num_random = random.choice(range(len(validation_data)))
    validation_sample = validation_data[num_random]
    if flag == 'bce':
        lr_list = [0.1,0.01,0.001,0.05]
        regterm_list = [1e-1,1e-2,1e-3,1e-4,1e-5]
        momentum_list = [0.9,0.99]
        lr,regterm,mom,acc = [[] for i in range(4)]
        for config_list in list(cartes(lr_list,regterm_list,momentum_list)):
            model = ModelBCE(INPUT_SIZE[0], INPUT_SIZE[1],16,config_list[0],config_list[1],config_list[2])
            val_accuracy = bce_batch_iterator(model, train_data, validation_data,validation_sample.image.data,epochs=10)
      	    lr.append(config_list[0])
      	    regterm.append(config_list[1])
      	    mom.append(config_list[2])
      	    acc.append(val_accuracy)
        for l,r,m,a in zip(lr,regterm,mom,acc):
       	    print ("lr: {}, lambda: {}, momentum: {}, accuracy: {}").format(l,r,m,a)
       	    print('------------------------------------------------------------------') 	    
    
            print('--------------------------------The Best--------------------------') 	   
            best_idx = np.argmax(acc)
            print ("lr: {}, lambda: {}, momentum: {}, accuracy: {}").format(lr[best_idx],regterm[best_idx],mom[best_idx],acc[best_idx])
    elif flag == 'salgan':
        G_lr_list = [0.1,0.01,0.05]
        regterm_list = [1e-1,1e-2,1e-3,1e-4,1e-5]
        D_lr_list = [0.1,0.01,0.05]
        alpha_list = [1/5., 1/10., 1/20.]
        G_lr,regterm,D_lr,alpha,acc = [[] for i in range(5)]
        for config_list in list(cartes(G_lr_list,regterm_list,D_lr_list,alpha_list)):
            model = ModelSALGAN(INPUT_SIZE[0], INPUT_SIZE[1],9,config_list[0],config_list[1],config_list[2],config_list[3])
            val_accuracy = salgan_batch_iterator(model, train_data, validation_data,validation_sample.image.data,epochs=10)
      	    G_lr.append(config_list[0])
      	    regterm.append(config_list[1])
      	    D_lr.append(config_list[2])
            alpha.append(config_list[3])
      	    acc.append(val_accuracy)
        for g_l,r,d_l,al,a in zip(G_lr,regterm,D_lr,alpha,acc):
       	    print ("G_lr: {}, lambda: {}, D_lr: {}, alpha: {}, accuracy: {}").format(g_l,r,d_l,al,a)
    	    print('------------------------------------------------------------------') 	    
    
        print('--------------------------------The Best--------------------------') 	   
        best_idx = np.argmax(acc)
        print ("G_lr: {}, lambda: {}, D_lr: {}, alpha: {}, accuracy: {}").format(G_lr[best_idx],regterm[best_idx],D_lr[best_idx],alpha[best_idx],acc[best_idx])
    else:
        print("Please provide a correct argument")
if __name__ == "__main__":
    train()
