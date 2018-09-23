from __future__ import print_function

import numpy as np
#np.random.seed(43)
import tensorflow as tf
tf.set_random_seed(43)

import json

import sys

import gputil as gp

from mipego.mipego import Solution
from mipego.Bi_Objective import *

import keras
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
import os
import sys
import pandas as pd
import keras.backend as K
import math
from keras.callbacks import LearningRateScheduler
from keras.regularizers import l2

import time #CHRIS added to measure runtime of training

def CNN_conf(cfg,hist_save):
    verbose = 0
    batch_size = 100
    num_classes = 10
    epochs = 1 #CHRIS increased from 1 to 5 to make results less random and noisy
    data_augmentation = False
    num_predictions = 20
    logfile = 'mnist-cnn.log'
    savemodel = False

    # The data, shuffled and split between train and test sets:
    (x_train, y_train), (x_test, y_test) = mnist.load_data()    
    x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2],1)
    x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],1)
    
    cfg_df = pd.DataFrame(cfg, index=[0])

    # Convert class vectors to binary class matrices.
    y_train = keras.utils.to_categorical(y_train.flatten(), num_classes)
    y_test = keras.utils.to_categorical(y_test.flatten(), num_classes)

    model = Sequential()
    
    model.add(Dropout(cfg['dropout_0'],input_shape=x_train.shape[1:]))
    model.add(Conv2D(cfg['filters_0'], (cfg['k_0'], cfg['k_0']), padding='same', 
                     kernel_regularizer=l2(cfg['l2']), bias_regularizer=l2(cfg['l2'])))
    model.add(Activation(cfg['activation']))#kernel_initializer='random_uniform',
    
    #stack 0
    for i in range(cfg['stack_0']):
        model.add(Conv2D(cfg['filters_1'], (cfg['k_1'], cfg['k_1']), padding='same', 
                     kernel_regularizer=l2(cfg['l2']), bias_regularizer=l2(cfg['l2'])))
        model.add(Activation(cfg['activation']))
    #maxpooling as cnn
    model.add(Conv2D(cfg['filters_2'], (cfg['k_2'], cfg['k_2']), strides=(cfg['s_0'], cfg['s_0']), padding='same', 
                     kernel_regularizer=l2(cfg['l2']), bias_regularizer=l2(cfg['l2'])))
    model.add(Activation(cfg['activation']))
    model.add(Dropout(cfg['dropout_1']))
    
    #stack 1
    for i in range(cfg['stack_1']):
        model.add(Conv2D(cfg['filters_3'], (cfg['k_3'], cfg['k_3']), padding='same', 
                     kernel_regularizer=l2(cfg['l2']), bias_regularizer=l2(cfg['l2'])))
        model.add(Activation(cfg['activation']))
    model.add(Conv2D(cfg['filters_4'], (cfg['k_4'], cfg['k_4']), strides=(cfg['s_1'], cfg['s_1']), padding='same', 
                     kernel_regularizer=l2(cfg['l2']), bias_regularizer=l2(cfg['l2'])))
    model.add(Activation(cfg['activation']))
    model.add(Dropout(cfg['dropout_2']))

    #stack 2
    for i in range(cfg['stack_2']):
        model.add(Conv2D(cfg['filters_5'], (cfg['k_5'], cfg['k_5']), padding='same', 
                     kernel_regularizer=l2(cfg['l2']), bias_regularizer=l2(cfg['l2'])))
        model.add(Activation(cfg['activation']))
    if (cfg['stack_2']>0):
        model.add(Conv2D(cfg['filters_6'], (cfg['k_6'], cfg['k_6']), strides=(cfg['s_2'], cfg['s_2']), padding='same', 
                     kernel_regularizer=l2(cfg['l2']), bias_regularizer=l2(cfg['l2'])))
        model.add(Activation(cfg['activation']))
        model.add(Dropout(cfg['dropout_3']))
    
    #global averaging
    if (cfg['global_pooling']):
        model.add(GlobalAveragePooling2D())
    else:
        model.add(Flatten())
    
    
    
    #head
    model.add(Dense(num_classes, kernel_regularizer=l2(cfg['l2']), bias_regularizer=l2(cfg['l2'])))
    model.add(Activation(cfg['activ_dense']))
    
    cfg['decay'] = cfg['lr'] / float(epochs)
    def step_decay(epoch):
        initial_lrate = cfg['lr']
        drop = 0.1
        epochs_drop = 20.0
        lrate = initial_lrate * math.pow(drop,  
                                         math.floor((1+epoch)/epochs_drop))
        return lrate
    callbacks = []
    if (cfg['step'] == True):
        callbacks = [LearningRateScheduler(step_decay)]
        cfg['decay'] = 0.

    # initiate RMSprop optimizer
    #opt = keras.optimizers.rmsprop(lr= cfg['lr'], decay=cfg['decay'])
    opt = keras.optimizers.SGD(lr=cfg['lr'], momentum=0.9, decay=cfg['decay'], nesterov=False)

    # Let's train the model using RMSprop
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255.
    x_test /= 255.

    if not data_augmentation:
        print('Not using data augmentation.')
        start = time.time()
        hist = model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_data=(x_test, y_test),
                         callbacks=callbacks,
                         verbose=verbose,
                  shuffle=True)
        stop = time.time()
    else:
        print('Using real-time data augmentation.')
        # This will do preprocessing and realtime data augmentation:
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images
        datagen.fit(x_train)

        # Fit the model on the batches generated by datagen.flow().
        start = time.time()
        hist = model.fit_generator(datagen.flow(x_train, y_train,
                                         batch_size=batch_size), verbose=verbose,
                                   callbacks=callbacks,
                            epochs=epochs, steps_per_epoch = len(x_train)/batch_size,
                            validation_data=(x_test, y_test))
        stop = time.time()

    timer = stop-start
    print('run-time:')
    print(timer)
    hist_save.append(hist.history)
    if savemodel:
        model.save('best_model_mnist.h5')
    maxval = max(hist.history['val_acc'])
    #loss = -1 * math.log( 1.0 - max(hist.history['val_acc']) ) #np.amin(hist.history['val_loss'])
    loss = -1 * math.log(max(hist.history['val_acc']) ) #CHRIS minimizing this will maximize accuracy
    print('max val_acc:')
    print(max(hist.history['val_acc']))
    print('loss:')
    print(loss)
    #perf5 = max(hist.history['val_top_5_categorical_accuracy'])

    if logfile is not None:
        log_file = logfile #os.path.join(data_des, logfile)
        cfg_df['perf'] = maxval

        # save the configurations to log file
        if os.path.isfile(log_file): 
            cfg_df.to_csv(log_file, mode='a', header=False, index=False)
        else:
            cfg_df.to_csv(log_file, mode='w', header=True, index=False)
    return timer,loss


if len(sys.argv) < 2:
    print("usage: python3 load_data.py 'data_file_name.json' (optional: to be removed gpu's)")
    exit(0)
file_name = str(sys.argv[1])
with open(file_name) as f:
    for line in f:
        data = json.loads(line)

ref_time = None
ref_loss = None

conf_array = data[0]
fit_array = data[1]
time_array = data[2]
loss_array = data[3]
n_eval_array = data[4]
index_array = data[5]
name_array = data[6]

all_time_r2 = None
all_loss_r2 = None
if len(data) > 7:
    all_time_r2 = data[7]
    all_loss_r2 = data[8]


#print(data)
solutions = []
for i in range(len(conf_array)):
    solutions.append(Solution(x=conf_array[i],fitness=fit_array[i],n_eval=n_eval_array[i],index=index_array[i],var_name=name_array[i],loss=loss_array[i],time=time_array[i]))

print("len(solutions): " + str(len(solutions)))

pauser = 0.008

_time = [x.time for x in solutions]
_loss = [x.loss for x in solutions]

#print('time:')
#print(time)
#print('loss:')
#print(loss)
x_bound = min(0.0,min(_time)),max(_time)
y_bound = min(0.0,min(_loss)),max(_loss)

par = pareto(solutions)
quicksort_par(par,0,len(par)-1)
par_time = [x.time for x in par]
par_loss = [x.loss for x in par]
HV = hyper_vol(par, solutions, ref_time, ref_loss)
print("Hyper Volume:")
print(HV)
print("len pareto front:")
print(len(par))
print("paretofront:")
if all_time_r2 is not None and all_loss_r2 is not None:
    print("all_time_r2 average:")
    print(np.average(np.array(all_time_r2)))
    print("all_loss_r2 average:")
    print(np.average(np.array(all_loss_r2)))
for i in range(len(par)):
    print("time: " + str(par[i].time) + ", loss: " + str(par[i].loss) + ", acc: " + str(np.exp(-par[i].loss)))
hist_save = []
for x in par:
    available_gpus = gp.getAvailable(limit=5)

    if len(sys.argv) > 2:
        for i in range(3,int(len(sys.argv))):
            print(int(sys.argv[i]))
            try:
                available_gpus.remove(int(sys.argv[i]))
            except:
                pass
    print(available_gpus)
    #gpu = available_gpus[0]
    CNN_conf(x.tolist(),hist_save)
    with open('train_par_out.json', 'w') as outfile:
            json.dump(hist_save,outfile)
#for each in par build network and train
#save accuracy, time, iterations
