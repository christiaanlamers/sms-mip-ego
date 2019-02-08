from __future__ import print_function

import numpy as np
np.random.seed(44)
import tensorflow as tf
tf.set_random_seed(44)

import json

import sys

import gputil as gp

from mipego.mipego import Solution
from mipego.Bi_Objective import *

import keras
#from keras.datasets import mnist
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D,UpSampling2D,ZeroPadding2D,Concatenate
from keras.layers import Layer#CHRIS to define a layer yourself
import os
import sys
import pandas as pd
import keras.backend as K
import math
from keras.callbacks import LearningRateScheduler
from keras.regularizers import l2

import time #CHRIS added to measure runtime of training
from pynvml import * #CHRIS needed to test gpu memory capacity
#from fractions import gcd #CHRIS needed for proper upscaling

class TimedAccHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.accuracy_log = []
        self.timed = []
        self.start_time = time.time()
    
    def on_epoch_end(self, batch, logs={}):
        self.accuracy_log.append(logs.get('val_acc'))
        self.timed.append(time.time() - self.start_time)

def inv_gray(num):#TODO only for testing
    n = 0
    while num != 0:
        n = num ^ n
        num = num >> 1
    return n

class Skip_manager(object):
    def __init__(self,skip_ints,skip_ints_count):
        self.skip_ints= skip_ints
        self.skip_ints_count = skip_ints_count
        self.skip_connections = []
        self.layer_num = 0 #layer number of currently build layer
    
    def identity(self,num):
        return num
    
    def gray(self,num):
        return num ^ (num >> 1)
    
    def startpoint(self,func,num):
        return (func(num) >> self.layer_num) & 1

    def connect_skip(self,layer):
        #start skip connections
        for j in range(len(self.skip_ints)):
            if self.startpoint(self.gray,self.skip_ints[j]):
                self.skip_connections.append([layer,self.skip_ints_count[j],self.layer_num])#save layer output, skip counter, layer this skip connection starts (to remove duplicates)
    
        #end skip connections
        j = 0
        prev_skip = -1
        while j < len(self.skip_connections):
            if self.skip_connections[j][1] <= 0:
                #print(prev_skip,self.skip_connections[j][2])
                if prev_skip != self.skip_connections[j][2]:#this removes skip connection duplicates (works because same skip connections are next to eachother) TODO maybe better to make more robust
                    if K.int_shape(self.skip_connections[j][0])[1] != K.int_shape(layer)[1] or K.int_shape(self.skip_connections[j][0])[2] != K.int_shape(layer)[2]:
                        #CHRIS TODO add pooling, because this becomes too complex to train
                        pad_tpl1 = (int(np.floor(np.abs(K.int_shape(self.skip_connections[j][0])[1]-K.int_shape(layer)[1])/2)),int(np.ceil(np.abs(K.int_shape(self.skip_connections[j][0])[1]-K.int_shape(layer)[1])/2)))
                        pad_tpl2 = (int(np.floor(np.abs(K.int_shape(self.skip_connections[j][0])[2]-K.int_shape(layer)[2])/2)),int(np.ceil(np.abs(K.int_shape(self.skip_connections[j][0])[2]-K.int_shape(layer)[2])/2)))
                        #print(pad_tpl)
                        if K.int_shape(self.skip_connections[j][0])[1] < K.int_shape(layer)[1] and K.int_shape(self.skip_connections[j][0])[2] < K.int_shape(layer)[2]:
                            padded = ZeroPadding2D(padding=(pad_tpl1, pad_tpl2))(self.skip_connections[j][0])
                            layer = Concatenate()([layer, padded])
                        elif K.int_shape(self.skip_connections[j][0])[1] < K.int_shape(layer)[1] and K.int_shape(self.skip_connections[j][0])[2] >= K.int_shape(layer)[2]:
                            padded1 = ZeroPadding2D(padding=(pad_tpl1, 0))(self.skip_connections[j][0])
                            padded2 = ZeroPadding2D(padding=(0, pad_tpl2))(layer)
                            layer = Concatenate()([padded1, padded2])
                        elif K.int_shape(self.skip_connections[j][0])[1] >= K.int_shape(layer)[1] and K.int_shape(self.skip_connections[j][0])[2] < K.int_shape(layer)[2]:
                            padded1 = ZeroPadding2D(padding=(0, pad_tpl2))(self.skip_connections[j][0])
                            padded2 = ZeroPadding2D(padding=(pad_tpl1, 0))(layer)
                            layer = Concatenate()([padded1, padded2])
                        else:
                            #print(layer.shape)
                            padded = ZeroPadding2D(padding=(pad_tpl1, pad_tpl2))(layer)
                            #print(padded.shape)
                            #print(self.skip_connections[j][0].shape)
                            layer = Concatenate()([padded, self.skip_connections[j][0]])
                        #if upscaling is desired: (can result in enormous tensors though)
                        #shape1 = K.int_shape(layer)
                        #shape2 = K.int_shape(self.skip_connections[j][0])
                        #gcd_x = gcd(shape1[1], shape2[1])
                        #gcd_y = gcd(shape1[2], shape2[2])
                        #scale1 =shape2[1] // gcd_x, shape2[2] // gcd_y
                        #scale2 =shape1[1] // gcd_x, shape1[2] // gcd_y
                        #upscaled1 = UpSampling2D(size=scale1, interpolation='nearest')(layer)
                        #upscaled2 = UpSampling2D(size=scale2, interpolation='nearest')(self.skip_connections[j][0])
                        #layer = keras.layers.Concatenate()([upscaled1, upscaled2])
                    else:
                        layer = Concatenate()([layer, self.skip_connections[j][0]])
                prev_skip = self.skip_connections[j][2]
                del self.skip_connections[j]
            else:
                self.skip_connections[j][1] -= 1 #decrease skip connection counters
                j += 1
        self.layer_num +=1 #increase layer number where currently building takes place
        return layer


def CNN_conf(cfg,hist_save,epochs=1,test=False,gpu_no=0):
    verbose = 0 #CHRIS TODO set this to 0
    batch_size = 100
    num_classes = 10
    epochs = 2000 #CHRIS increased from 1 to 5 to make results less random and noisy
    data_augmentation = False
    num_predictions = 20
    logfile = 'mnist-cnn.log'
    savemodel = False

    # The data, shuffled and split between train and test sets:
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()#mnist.load_data()
    
    #CHRIS reshape only needed for mnist
    #x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2],1)
    #x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],1)
    
    cfg_df = pd.DataFrame(cfg, index=[0])

    # Convert class vectors to binary class matrices.
    y_train = keras.utils.to_categorical(y_train.flatten(), num_classes)
    y_test = keras.utils.to_categorical(y_test.flatten(), num_classes)
    
    #print('skip steps:')
    #print([cfg['skint_0'],cfg['skint_1'],cfg['skint_2']],[cfg['skst_0'],cfg['skst_1'],cfg['skst_2']])
    #(skip_ints,skip_ints_count) passed to Skip_manager constructor TODO get from cfg vector
    skip_manager = Skip_manager([cfg['skint_0'],cfg['skint_1'],cfg['skint_2']],[cfg['skst_0'],cfg['skst_1'],cfg['skst_2']])
    
    input1 = keras.layers.Input(shape=(x_train.shape[1],x_train.shape[2],x_train.shape[3]))
    
    layer = Dropout(cfg['dropout_0'],input_shape=x_train.shape[1:])(input1)
    layer = skip_manager.connect_skip(layer)
    #CHRIS removed following:
    #layer = Conv2D(cfg['filters_0'], (cfg['k_0'], cfg['k_0']), padding='same',kernel_regularizer=l2(cfg['l2']), bias_regularizer=l2(cfg['l2']))(layer)
    #layer = Activation(cfg['activation'])(layer)#kernel_initializer='random_uniform',
    #layer = skip_manager.connect_skip(layer)
    
    #stack 0
    for i in range(cfg['stack_0']):
        layer = Conv2D(cfg['filters_0'], (cfg['k_0'], cfg['k_0']), padding='same', kernel_regularizer=l2(cfg['l2']), bias_regularizer=l2(cfg['l2']))(layer)
        layer = Activation(cfg['activation'])(layer)
        layer = skip_manager.connect_skip(layer)
    if (cfg['stack_0']>0):
        #maxpooling as cnn
        if (cfg['no_pooling']):
            layer = Conv2D(cfg['filters_1'], (cfg['k_1'], cfg['k_1']), strides=(cfg['s_0'], cfg['s_0']), padding='same', kernel_regularizer=l2(cfg['l2']), bias_regularizer=l2(cfg['l2']))(layer)
        else:
            layer = MaxPooling2D(pool_size=(cfg['k_1'], cfg['k_1']), strides=(cfg['s_0'], cfg['s_0']), padding='same')(layer)
        layer = Activation(cfg['activation'])(layer)
        layer = Dropout(cfg['dropout_1'])(layer)
        layer = skip_manager.connect_skip(layer)
    
    #stack 1
    for i in range(cfg['stack_1']):
        layer = Conv2D(cfg['filters_2'], (cfg['k_2'], cfg['k_2']), padding='same', kernel_regularizer=l2(cfg['l2']), bias_regularizer=l2(cfg['l2']))(layer)
        layer = Activation(cfg['activation'])(layer)
        layer = skip_manager.connect_skip(layer)
    if (cfg['stack_1']>0):
        if (cfg['no_pooling']):
            layer = Conv2D(cfg['filters_3'], (cfg['k_3'], cfg['k_3']), strides=(cfg['s_1'], cfg['s_1']), padding='same', kernel_regularizer=l2(cfg['l2']), bias_regularizer=l2(cfg['l2']))(layer)
        else:
            layer = MaxPooling2D(pool_size=(cfg['k_3'], cfg['k_3']), strides=(cfg['s_1'], cfg['s_1']), padding='same')(layer)
        layer = Activation(cfg['activation'])(layer)
        layer = Dropout(cfg['dropout_2'])(layer)
        layer = skip_manager.connect_skip(layer)

    #stack 2
    for i in range(cfg['stack_2']):
        layer = Conv2D(cfg['filters_4'], (cfg['k_4'], cfg['k_4']), padding='same', kernel_regularizer=l2(cfg['l2']), bias_regularizer=l2(cfg['l2']))(layer)
        layer = Activation(cfg['activation'])(layer)
        layer = skip_manager.connect_skip(layer)
    if (cfg['stack_2']>0):
        if (cfg['no_pooling']):
            layer = Conv2D(cfg['filters_5'], (cfg['k_5'], cfg['k_5']), strides=(cfg['s_2'], cfg['s_2']), padding='same', kernel_regularizer=l2(cfg['l2']), bias_regularizer=l2(cfg['l2']))(layer)
        else:
            layer = MaxPooling2D(pool_size=(cfg['k_5'], cfg['k_5']), strides=(cfg['s_2'], cfg['s_2']), padding='same')(layer)
        layer = Activation(cfg['activation'])(layer)
        layer = Dropout(cfg['dropout_3'])(layer)
        layer = skip_manager.connect_skip(layer)

    #stack 3
    for i in range(cfg['stack_3']):
        layer = Conv2D(cfg['filters_6'], (cfg['k_6'], cfg['k_6']), padding='same', kernel_regularizer=l2(cfg['l2']), bias_regularizer=l2(cfg['l2']))(layer)
        layer = Activation(cfg['activation'])(layer)
        layer = skip_manager.connect_skip(layer)
    if (cfg['stack_3']>0):
        if (cfg['no_pooling']):
            layer = Conv2D(cfg['filters_7'], (cfg['k_7'], cfg['k_7']), strides=(cfg['s_3'], cfg['s_3']), padding='same', kernel_regularizer=l2(cfg['l2']), bias_regularizer=l2(cfg['l2']))(layer)
        else:
            layer = MaxPooling2D(pool_size=(cfg['k_7'], cfg['k_7']), strides=(cfg['s_3'], cfg['s_3']), padding='same')(layer)
        layer = Activation(cfg['activation'])(layer)
        layer = Dropout(cfg['dropout_4'])(layer)
        layer = skip_manager.connect_skip(layer)

    #stack 4
    for i in range(cfg['stack_4']):
        layer = Conv2D(cfg['filters_8'], (cfg['k_8'], cfg['k_8']), padding='same', kernel_regularizer=l2(cfg['l2']), bias_regularizer=l2(cfg['l2']))(layer)
        layer = Activation(cfg['activation'])(layer)
        layer = skip_manager.connect_skip(layer)
    if (cfg['stack_4']>0):
        if (cfg['no_pooling']):
            layer = Conv2D(cfg['filters_9'], (cfg['k_9'], cfg['k_9']), strides=(cfg['s_4'], cfg['s_4']), padding='same', kernel_regularizer=l2(cfg['l2']), bias_regularizer=l2(cfg['l2']))(layer)
        else:
            layer = MaxPooling2D(pool_size=(cfg['k_9'], cfg['k_9']), strides=(cfg['s_4'], cfg['s_4']), padding='same')(layer)
        layer = Activation(cfg['activation'])(layer)
        layer = Dropout(cfg['dropout_5'])(layer)
        layer = skip_manager.connect_skip(layer)

    #stack 5
    for i in range(cfg['stack_5']):
        layer = Conv2D(cfg['filters_10'], (cfg['k_10'], cfg['k_10']), padding='same', kernel_regularizer=l2(cfg['l2']), bias_regularizer=l2(cfg['l2']))(layer)
        layer = Activation(cfg['activation'])(layer)
        layer = skip_manager.connect_skip(layer)
    if (cfg['stack_5']>0):
        if (cfg['no_pooling']):
            layer = Conv2D(cfg['filters_11'], (cfg['k_11'], cfg['k_11']), strides=(cfg['s_5'], cfg['s_5']), padding='same', kernel_regularizer=l2(cfg['l2']), bias_regularizer=l2(cfg['l2']))(layer)
        else:
            layer = MaxPooling2D(pool_size=(cfg['k_11'], cfg['k_11']), strides=(cfg['s_5'], cfg['s_5']), padding='same')(layer)
        layer = Activation(cfg['activation'])(layer)
        layer = Dropout(cfg['dropout_6'])(layer)
        layer = skip_manager.connect_skip(layer)

    #stack 6
    for i in range(cfg['stack_6']):
        layer = Conv2D(cfg['filters_12'], (cfg['k_12'], cfg['k_12']), padding='same', kernel_regularizer=l2(cfg['l2']), bias_regularizer=l2(cfg['l2']))(layer)
        layer = Activation(cfg['activation'])(layer)
        layer = skip_manager.connect_skip(layer)
    if (cfg['stack_6']>0):
        if (cfg['no_pooling']):
            layer = Conv2D(cfg['filters_13'], (cfg['k_13'], cfg['k_13']), strides=(cfg['s_6'], cfg['s_6']), padding='same', kernel_regularizer=l2(cfg['l2']), bias_regularizer=l2(cfg['l2']))(layer)
        else:
            layer = MaxPooling2D(pool_size=(cfg['k_13'], cfg['k_13']), strides=(cfg['s_6'], cfg['s_6']), padding='same')(layer)
        layer = Activation(cfg['activation'])(layer)
        layer = Dropout(cfg['dropout_7'])(layer)
        layer = skip_manager.connect_skip(layer)

    #global averaging
    if (cfg['global_pooling']):
        layer = GlobalAveragePooling2D()(layer)
    else:
        layer = Flatten()(layer)
    
    
    
    #head
    if cfg['dense_size_0'] > 0:
        layer = Dense(cfg['dense_size_0'], kernel_regularizer=l2(cfg['l2']), bias_regularizer=l2(cfg['l2']))(layer)
        layer = Activation(cfg['activ_dense'])(layer)
    if cfg['dense_size_1'] > 0:
        layer = Dense(cfg['dense_size_1'], kernel_regularizer=l2(cfg['l2']), bias_regularizer=l2(cfg['l2']))(layer)
        layer = Activation(cfg['activ_dense'])(layer)
    layer = Dense(num_classes, kernel_regularizer=l2(cfg['l2']), bias_regularizer=l2(cfg['l2']))(layer)
    layer = Activation(cfg['activ_dense'])(layer)
    
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

    model = keras.models.Model(inputs=input1, outputs=layer)

    # Let's train the model using RMSprop
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    if test:
        return model #TODO remove this, just for testing
    
    #print("amount of parameters:")
    #print(model.count_params())
    #CHRIS test if gpu has enough memory
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(int(gpu_no))
    meminfo = nvmlDeviceGetMemoryInfo(handle)
    #max_size = meminfo.total #6689341440
    if meminfo.free/1024.**2 < 1.0:
        print('gpu is allready in use')
    nvmlShutdown()
    #if model.count_params()*4*2 >= max_size:#CHRIS *4*2: 4 byte per parameter times 2 for backpropagation
        #print('network too large for memory')
        #return 1000000000.0*(model.count_params()*4*2/max_size), 5.0*(model.count_params()*4*2/max_size)

    #max_size = 32828802 * 2 #CHRIS twice as large as RESnet-34-like implementation
    #max_size = 129200130 #CHRIS twice as wide as RESnet-34-like implementation with batchsize=10, one network of this size was able to be ran on tritanium gpu
    max_size = 130374394 #CHRIS twice as wide as RESnet-34-like implementation with batchsize=100, one network of this size was able to be ran on tritanium gpu
    #if model.count_params() > max_size:
        #print('network too large for implementation')
        #return 1000000000.0*(model.count_params()/max_size), 5.0*(model.count_params()/max_size)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255.
    x_test /= 255.

    hist_func = TimedAccHistory()
    
    if not data_augmentation:
        print('Not using data augmentation.')
        start = time.time()
        hist = model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_data=(x_test, y_test),
                         callbacks=[hist_func],
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
    #print('run-time:')
    #print(timer)
    hist_save.append([hist.history['val_acc'], hist_func.timed])

    if savemodel:
        model.save('best_model_mnist.h5')
    maxval = max(hist.history['val_acc'])
    #loss = -1 * math.log( 1.0 - max(hist.history['val_acc']) ) #np.amin(hist.history['val_loss'])
    loss = -1 * math.log(max(hist.history['val_acc']) ) #CHRIS minimizing this will maximize accuracy
    #print('max val_acc:')
    #print(max(hist.history['val_acc']))
    #print('loss:')
    #print(loss)
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
    conf_x = [conf_array[i][j] for j in name_array[i]]
    solutions.append(Solution(x=conf_x,fitness=fit_array[i],n_eval=n_eval_array[i],index=index_array[i],var_name=name_array[i],loss=loss_array[i],time=time_array[i]))

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
    available_gpus = []
    while True:
        available_gpus = gp.getAvailable(limit=5)
    
        if len(sys.argv) > 2:
            for i in range(2,int(len(sys.argv))):
                print(int(sys.argv[i]))
                try:
                    available_gpus.remove(int(sys.argv[i]))
                except:
                    pass
        if len(available_gpus) <= 0:
            print('no gpus available')
        else:
            break
    print('available gpus:')
    print(available_gpus)
    gpu = available_gpus[0]
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu)
    CNN_conf(x.to_dict(),hist_save,gpu_no=gpu)
    with open('train_par_skippy_fastest.json', 'w') as outfile:
            json.dump(hist_save,outfile)
    break
#for each in par build network and train
#save accuracy, time, iterations
