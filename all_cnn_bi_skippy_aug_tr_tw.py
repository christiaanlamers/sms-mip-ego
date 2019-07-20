from __future__ import print_function

import numpy as np
#np.random.seed(43)
import tensorflow as tf
tf.set_random_seed(43)

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

import setproctitle
import json

import sklearn
import sklearn.model_selection

#setproctitle.setproctitle('lamers c, do not use GPU 3-7 please')

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
    
    def set_dropout(self,dropout_val):
        for i in range(len(self.skip_connections)):
            self.skip_connections[i][3] = dropout_val
        return
    
    def pad_and_connect(self, layer, incoming_layer):
        if K.int_shape(incoming_layer)[1] != K.int_shape(layer)[1] or K.int_shape(incoming_layer)[2] != K.int_shape(layer)[2]:
            pad_tpl1 = (int(np.floor(np.abs(K.int_shape(incoming_layer)[1]-K.int_shape(layer)[1])/2)),int(np.ceil(np.abs(K.int_shape(incoming_layer)[1]-K.int_shape(layer)[1])/2)))
            pad_tpl2 = (int(np.floor(np.abs(K.int_shape(incoming_layer)[2]-K.int_shape(layer)[2])/2)),int(np.ceil(np.abs(K.int_shape(incoming_layer)[2]-K.int_shape(layer)[2])/2)))
            #print(pad_tpl)
            if K.int_shape(incoming_layer)[1] < K.int_shape(layer)[1] and K.int_shape(incoming_layer)[2] < K.int_shape(layer)[2]:
                padded = ZeroPadding2D(padding=(pad_tpl1, pad_tpl2))(incoming_layer)
                layer = Concatenate()([layer, padded])
            elif K.int_shape(incoming_layer)[1] < K.int_shape(layer)[1] and K.int_shape(incoming_layer)[2] >= K.int_shape(layer)[2]:
                padded1 = ZeroPadding2D(padding=(pad_tpl1, 0))(incoming_layer)
                padded2 = ZeroPadding2D(padding=(0, pad_tpl2))(layer)
                layer = Concatenate()([padded1, padded2])
            elif K.int_shape(incoming_layer)[1] >= K.int_shape(layer)[1] and K.int_shape(incoming_layer)[2] < K.int_shape(layer)[2]:
                padded1 = ZeroPadding2D(padding=(0, pad_tpl2))(incoming_layer)
                padded2 = ZeroPadding2D(padding=(pad_tpl1, 0))(layer)
                layer= Concatenate()([padded1, padded2])
            else:
                #print(layer.shape)
                padded = ZeroPadding2D(padding=(pad_tpl1, pad_tpl2))(layer)
                #print(padded.shape)
                #print(incoming_layer.shape)
                layer= Concatenate()([padded, incoming_layer])
        else:
            layer= Concatenate()([layer, incoming_layer])
        return layer

    def pool_pad_connect(self, layer, incoming_layer,dropout_val):
        if K.int_shape(incoming_layer)[1] != K.int_shape(layer)[1] or K.int_shape(incoming_layer)[2] != K.int_shape(layer)[2]:
            #print('layer dimensions:')
            #print(K.int_shape(layer)[1], K.int_shape(layer)[2])
            #print('incoming_layer dimensions:')
            #print(K.int_shape(incoming_layer)[1],  K.int_shape(incoming_layer)[2])
            if K.int_shape(incoming_layer)[1] < K.int_shape(layer)[1] and K.int_shape(incoming_layer)[2] < K.int_shape(layer)[2]:
                pass
            elif K.int_shape(incoming_layer)[1] < K.int_shape(layer)[1] and K.int_shape(incoming_layer)[2] >= K.int_shape(layer)[2]:
                scalar = int(np.ceil(K.int_shape(incoming_layer)[2] / K.int_shape(layer)[2]))
                incoming_layer = MaxPooling2D(pool_size=(1, scalar), strides=(1, scalar), padding='same')(incoming_layer)
                print('warning: code used that is not tested, see: all_cnn_bi_skippy.py --> pool_pad_connect()')
            elif K.int_shape(incoming_layer)[1] >= K.int_shape(layer)[1] and K.int_shape(incoming_layer)[2] < K.int_shape(layer)[2]:
                scalar = int(np.ceil(K.int_shape(incoming_layer)[1] / K.int_shape(layer)[1]))
                incoming_layer = MaxPooling2D(pool_size=(scalar, 1), strides=(scalar, 1), padding='same')(incoming_layer)
                print('warning: code used that is not tested, see: all_cnn_bi_skippy.py --> pool_pad_connect()')
            else: #K.int_shape(incoming_layer)[1] > K.int_shape(layer)[1] and K.int_shape(incoming_layer)[2] > K.int_shape(layer)[2]
                scalar_1 =  int(np.ceil(K.int_shape(incoming_layer)[1] / K.int_shape(layer)[1]))
                scalar_2 =  int(np.ceil(K.int_shape(incoming_layer)[2] / K.int_shape(layer)[2]))
                incoming_layer = MaxPooling2D(pool_size=(scalar_1, scalar_2), strides=(scalar_1, scalar_2), padding='same')(incoming_layer)
                #print('Did a max pool')
        if dropout_val is not None:
            incoming_layer = Dropout(dropout_val)(incoming_layer)
        return self.pad_and_connect(layer, incoming_layer)

    def start_skip(self,layer):
        for j in range(len(self.skip_ints)):
            if self.skip_ints_count[j] > 1 and self.startpoint(self.identity,self.skip_ints[j]):#CHRIS skip connections smaller than 2 are not made, thus mean no skip connection.
                self.skip_connections.append([layer,self.skip_ints_count[j],self.layer_num,None])#save layer output, skip counter, layer this skip connection starts (to remove duplicates)
        return layer
    
    def end_skip(self,layer,filters,kernel,regulizer,act):
        for j in range(len(self.skip_connections)):
            self.skip_connections[j][1] -= 1 #decrease skip connection counters
        j = 0
        prev_skip = -1
        connected = False #CHRIS check if an end skip connection is made
        while j < len(self.skip_connections):
            if self.skip_connections[j][1] <= 0:
                #print(prev_skip,self.skip_connections[j][2])
                if prev_skip != self.skip_connections[j][2]:#this removes skip connection duplicates (works because same skip connections are next to eachother) TODO maybe better to make more robust
                    #CHRIS TODO add pooling, because this becomes too complex to train
                    #layer = self.pad_and_connect(layer, self.skip_connections[j][0])#CHRIS warning! pad_and_connect does not do dropout!
                    layer = self.pool_pad_connect(layer, self.skip_connections[j][0],self.skip_connections[j][3])
                    connected = True#CHRIS an end skip connection is made
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
                prev_skip = self.skip_connections[j][2]
                del self.skip_connections[j]
            else:
                j += 1
        if connected and K.int_shape(layer)[3] > filters:#CHRIS we only want projection if an end skip connection is made, hence: ''connected''
            #CHRIS convolution to bound amount of features
            #CHRIS can funcion as addition, or projection followed by addition
            layer = Conv2D(filters, (1,1), padding='same', kernel_regularizer=l2(regulizer), bias_regularizer=l2(regulizer))(layer)#CHRIS kernel value set to (1,1) in order to simply act as projection
            #layer = Activation(act)(layer)
        for j in range(len(self.skip_connections)):#CHRIS TODO this is a bit hacky
            self.skip_connections[j][1] += 1 #decrease skip connection counters
        return layer

    def connect_skip(self,layer,filters,kernel,regulizer,act):
        
        #end skip connections
        layer = self.end_skip(layer,filters,kernel,regulizer,act)
        for j in range(len(self.skip_connections)):#CHRIS TODO this is a bit hacky
            self.skip_connections[j][1] -= 1 #decrease skip connection counters
        
        #start skip connections
        layer = self.start_skip(layer)
        
        self.layer_num +=1 #increase layer number where currently building takes place
        return layer


def CNN_conf(cfg,epochs=1,test=False,gpu_no=0,verbose=0,save_name='skippy_test_train_hist_aug_tr_tw',data_augmentation=True, use_validation=True):
    #batch_size = 100
    num_classes = 10
    num_predictions = 20
    logfile = 'mnist-cnn-aug-tr-tw.log'
    savemodel = False
    
    data_augmentation=True#TODO remove this
    use_validation=True#TODO remove this
    
    batch_size = cfg['batch_size_sp']
    #epochs = cfg['epoch_sp']

    # The data, shuffled and split between train and test sets:
    #(x_train, y_train), (x_test, y_test) = mnist.load_data()
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()#mnist.load_data()
    
    #CHRIS reshape only needed for mnist
    #x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2],1)
    #x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],1)
    
    
    if use_validation:
        x_train,x_val,y_train, y_val = sklearn.model_selection.train_test_split(x_train,y_train, test_size=2000, train_size=None, random_state=42,shuffle=True,stratify=y_train)
    
    cfg_df = pd.DataFrame(cfg, index=[0])

    # Convert class vectors to binary class matrices.
    y_train = keras.utils.to_categorical(y_train.flatten(), num_classes)
    y_test = keras.utils.to_categorical(y_test.flatten(), num_classes)
    
    #print('skip steps:')
    #print([cfg['skint_0'],cfg['skint_1'],cfg['skint_2']],[cfg['skst_0'],cfg['skst_1'],cfg['skst_2']])
    #(skip_ints,skip_ints_count) passed to Skip_manager constructor TODO get from cfg vector
    skint_0 = 0
    skint_1 = 0
    skint_2 = 0
    skint_3 = 0
    skint_4 = 0
    
    network_depth = cfg['stack_0'] + cfg['stack_1'] + cfg['stack_2'] + cfg['stack_3'] + cfg['stack_4'] + cfg['stack_5'] + cfg['stack_6']+7
    if cfg['skstep_0'] > 1:
        cnt = 0
        skint_0 = 1
        while cnt <= network_depth:
            skint_0 = skint_0 << cfg['skstep_0']
            skint_0 += 1
            cnt += cfg['skstep_0']
        skint_0 = skint_0 << cfg['skstart_0']

    if cfg['skstep_1'] > 1:
        cnt = 0
        skint_1 = 1
        while cnt <= network_depth:
            skint_1 = skint_1 << cfg['skstep_1']
            skint_1 += 1
            cnt += cfg['skstep_1']
        skint_1 = skint_1 << cfg['skstart_1']

    if cfg['skstep_2'] > 1:
        cnt = 0
        skint_2 = 1
        while cnt <= network_depth:
            skint_2 = skint_2 << cfg['skstep_2']
            skint_2 += 1
            cnt += cfg['skstep_2']
        skint_2 = skint_2 << cfg['skstart_2']

    if cfg['skstep_3'] > 1:
        cnt = 0
        skint_3 = 1
        while cnt <= network_depth:
            skint_3 = skint_3 << cfg['skstep_3']
            skint_3 += 1
            cnt += cfg['skstep_3']
        skint_3 = skint_3 << cfg['skstart_3']

    if cfg['skstep_4'] > 1:
        cnt = 0
        skint_4 = 1
        while cnt <= network_depth:
            skint_4 = skint_4 << cfg['skstep_4']
            skint_4 += 1
            cnt += cfg['skstep_4']
        skint_4 = skint_4 << cfg['skstart_4']
    
    skip_manager = Skip_manager([skint_0,skint_1,skint_2,skint_3,skint_4],[cfg['skstep_0'],cfg['skstep_1'],cfg['skstep_2'],cfg['skstep_3'],cfg['skstep_4']])
    
    #skip_manager = Skip_manager([0,0,0,0,0],[cfg['skstep_0'],cfg['skstep_1'],cfg['skstep_2'],cfg['skstep_3'],cfg['skstep_4']])
    
    input1 = keras.layers.Input(shape=(x_train.shape[1],x_train.shape[2],x_train.shape[3]))
    layer=input1

    filter_amount = x_train.shape[3]#CHRIS the filter amount for the sake of skip connections lags a bit behind the stack it is in, so it must be assigned separately
    layer = skip_manager.connect_skip(layer,filter_amount,cfg['k_0'],cfg['l2'],cfg['activation'])
    
    layer = Dropout(cfg['dropout_0'],input_shape=x_train.shape[1:])(layer)#CHRIS TODO reengage this line!
    skip_manager.set_dropout(cfg['dropout_0'])
    #CHRIS removed following:
    #layer = Conv2D(cfg['filters_0'], (cfg['k_0'], cfg['k_0']), padding='same',kernel_regularizer=l2(cfg['l2']), bias_regularizer=l2(cfg['l2']))(layer)
    #layer = Activation(cfg['activation'])(layer)#kernel_initializer='random_uniform',
    #layer = skip_manager.connect_skip(layer)

    
    #stack 0
    for i in range(cfg['stack_0']):
        filter_amount = cfg['filters_0']
        layer = Conv2D(cfg['filters_0'], (cfg['k_0'], cfg['k_0']), padding='same', kernel_regularizer=l2(cfg['l2']), bias_regularizer=l2(cfg['l2']))(layer)
        layer = skip_manager.connect_skip(layer,filter_amount,cfg['k_0'],cfg['l2'],cfg['activation'])
        layer = Activation(cfg['activation'])(layer)
    if (cfg['stack_0']>0):
        #maxpooling as cnn
        if not (cfg['max_pooling']):
            filter_amount = cfg['filters_1']
            layer = Conv2D(cfg['filters_1'], (cfg['k_1'], cfg['k_1']), strides=(cfg['s_0'], cfg['s_0']), padding='same', kernel_regularizer=l2(cfg['l2']), bias_regularizer=l2(cfg['l2']))(layer)
            layer = skip_manager.connect_skip(layer,filter_amount,cfg['k_0'],cfg['l2'],cfg['activation'])
            layer = Activation(cfg['activation'])(layer)
        else:
            #layer = skip_manager.end_skip(layer,filter_amount,cfg['k_0'],cfg['l2'],cfg['activation'])
            layer = MaxPooling2D(pool_size=(cfg['k_1'], cfg['k_1']), strides=(cfg['s_0'], cfg['s_0']), padding='same')(layer)
        layer = Dropout(cfg['dropout_1'])(layer)
        skip_manager.set_dropout(cfg['dropout_1'])
    
    #stack 1
    for i in range(cfg['stack_1']):
        filter_amount = cfg['filters_2']
        layer = Conv2D(cfg['filters_2'], (cfg['k_2'], cfg['k_2']), padding='same', kernel_regularizer=l2(cfg['l2']), bias_regularizer=l2(cfg['l2']))(layer)
        layer = skip_manager.connect_skip(layer,filter_amount,cfg['k_2'],cfg['l2'],cfg['activation'])
        layer = Activation(cfg['activation'])(layer)
    if (cfg['stack_1']>0):
        if not (cfg['max_pooling']):
            filter_amount = cfg['filters_3']
            layer = Conv2D(cfg['filters_3'], (cfg['k_3'], cfg['k_3']), strides=(cfg['s_1'], cfg['s_1']), padding='same', kernel_regularizer=l2(cfg['l2']), bias_regularizer=l2(cfg['l2']))(layer)
            layer = skip_manager.connect_skip(layer,filter_amount,cfg['k_2'],cfg['l2'],cfg['activation'])
            layer = Activation(cfg['activation'])(layer)
        else:
            #layer = skip_manager.end_skip(layer,filter_amount,cfg['k_2'],cfg['l2'],cfg['activation'])
            layer = MaxPooling2D(pool_size=(cfg['k_3'], cfg['k_3']), strides=(cfg['s_1'], cfg['s_1']), padding='same')(layer)
        layer = Dropout(cfg['dropout_2'])(layer)
        skip_manager.set_dropout(cfg['dropout_2'])

    #stack 2
    for i in range(cfg['stack_2']):
        filter_amount = cfg['filters_4']
        layer = Conv2D(cfg['filters_4'], (cfg['k_4'], cfg['k_4']), padding='same', kernel_regularizer=l2(cfg['l2']), bias_regularizer=l2(cfg['l2']))(layer)
        layer = skip_manager.connect_skip(layer,filter_amount,cfg['k_4'],cfg['l2'],cfg['activation'])
        layer = Activation(cfg['activation'])(layer)
    if (cfg['stack_2']>0):
        if not (cfg['max_pooling']):
            filter_amount = cfg['filters_5']
            layer = Conv2D(cfg['filters_5'], (cfg['k_5'], cfg['k_5']), strides=(cfg['s_2'], cfg['s_2']), padding='same', kernel_regularizer=l2(cfg['l2']), bias_regularizer=l2(cfg['l2']))(layer)
            layer = skip_manager.connect_skip(layer,filter_amount,cfg['k_4'],cfg['l2'],cfg['activation'])
            layer = Activation(cfg['activation'])(layer)
        else:
            #layer = skip_manager.end_skip(layer,filter_amount,cfg['k_4'],cfg['l2'],cfg['activation'])
            layer = MaxPooling2D(pool_size=(cfg['k_5'], cfg['k_5']), strides=(cfg['s_2'], cfg['s_2']), padding='same')(layer)
        layer = Dropout(cfg['dropout_3'])(layer)
        skip_manager.set_dropout(cfg['dropout_3'])

    #stack 3
    for i in range(cfg['stack_3']):
        filter_amount = cfg['filters_6']
        layer = Conv2D(cfg['filters_6'], (cfg['k_6'], cfg['k_6']), padding='same', kernel_regularizer=l2(cfg['l2']), bias_regularizer=l2(cfg['l2']))(layer)
        layer = skip_manager.connect_skip(layer,filter_amount,cfg['k_6'],cfg['l2'],cfg['activation'])
        layer = Activation(cfg['activation'])(layer)
    if (cfg['stack_3']>0):
        if not (cfg['max_pooling']):
            filter_amount = cfg['filters_7']
            layer = Conv2D(cfg['filters_7'], (cfg['k_7'], cfg['k_7']), strides=(cfg['s_3'], cfg['s_3']), padding='same', kernel_regularizer=l2(cfg['l2']), bias_regularizer=l2(cfg['l2']))(layer)
            layer = skip_manager.connect_skip(layer,filter_amount,cfg['k_6'],cfg['l2'],cfg['activation'])
            layer = Activation(cfg['activation'])(layer)
        else:
            #layer = skip_manager.end_skip(layer,filter_amount,cfg['k_6'],cfg['l2'],cfg['activation'])
            layer = MaxPooling2D(pool_size=(cfg['k_7'], cfg['k_7']), strides=(cfg['s_3'], cfg['s_3']), padding='same')(layer)
        layer = Dropout(cfg['dropout_4'])(layer)
        skip_manager.set_dropout(cfg['dropout_4'])

    #stack 4
    for i in range(cfg['stack_4']):
        filter_amount = cfg['filters_8']
        layer = Conv2D(cfg['filters_8'], (cfg['k_8'], cfg['k_8']), padding='same', kernel_regularizer=l2(cfg['l2']), bias_regularizer=l2(cfg['l2']))(layer)
        layer = skip_manager.connect_skip(layer,filter_amount,cfg['k_8'],cfg['l2'],cfg['activation'])
        layer = Activation(cfg['activation'])(layer)
    if (cfg['stack_4']>0):
        if not (cfg['max_pooling']):
            filter_amount = cfg['filters_9']
            layer = Conv2D(cfg['filters_9'], (cfg['k_9'], cfg['k_9']), strides=(cfg['s_4'], cfg['s_4']), padding='same', kernel_regularizer=l2(cfg['l2']), bias_regularizer=l2(cfg['l2']))(layer)
            layer = skip_manager.connect_skip(layer,filter_amount,cfg['k_8'],cfg['l2'],cfg['activation'])
            layer = Activation(cfg['activation'])(layer)
        else:
            #layer = skip_manager.end_skip(layer,filter_amount,cfg['k_8'],cfg['l2'],cfg['activation'])
            layer = MaxPooling2D(pool_size=(cfg['k_9'], cfg['k_9']), strides=(cfg['s_4'], cfg['s_4']), padding='same')(layer)
        layer = Dropout(cfg['dropout_5'])(layer)
        skip_manager.set_dropout(cfg['dropout_5'])

    #stack 5
    for i in range(cfg['stack_5']):
        filter_amount = cfg['filters_10']
        layer = Conv2D(cfg['filters_10'], (cfg['k_10'], cfg['k_10']), padding='same', kernel_regularizer=l2(cfg['l2']), bias_regularizer=l2(cfg['l2']))(layer)
        layer = skip_manager.connect_skip(layer,filter_amount,cfg['k_10'],cfg['l2'],cfg['activation'])
        layer = Activation(cfg['activation'])(layer)
    if (cfg['stack_5']>0):
        if not (cfg['max_pooling']):
            filter_amount = cfg['filters_11']
            layer = Conv2D(cfg['filters_11'], (cfg['k_11'], cfg['k_11']), strides=(cfg['s_5'], cfg['s_5']), padding='same', kernel_regularizer=l2(cfg['l2']), bias_regularizer=l2(cfg['l2']))(layer)
            layer = skip_manager.connect_skip(layer,filter_amount,cfg['k_10'],cfg['l2'],cfg['activation'])
            layer = Activation(cfg['activation'])(layer)
        else:
            #layer = skip_manager.end_skip(layer,filter_amount,cfg['k_10'],cfg['l2'],cfg['activation'])
            layer = MaxPooling2D(pool_size=(cfg['k_11'], cfg['k_11']), strides=(cfg['s_5'], cfg['s_5']), padding='same')(layer)
        layer = Dropout(cfg['dropout_6'])(layer)
        skip_manager.set_dropout(cfg['dropout_6'])

    #stack 6
    for i in range(cfg['stack_6']):
        filter_amount = cfg['filters_12']
        layer = Conv2D(cfg['filters_12'], (cfg['k_12'], cfg['k_12']), padding='same', kernel_regularizer=l2(cfg['l2']), bias_regularizer=l2(cfg['l2']))(layer)
        layer = skip_manager.connect_skip(layer,filter_amount,cfg['k_12'],cfg['l2'],cfg['activation'])
        layer = Activation(cfg['activation'])(layer)
    if (cfg['stack_6']>0):
        if not (cfg['max_pooling']):
            filter_amount = cfg['filters_13']
            layer = Conv2D(cfg['filters_13'], (cfg['k_13'], cfg['k_13']), strides=(cfg['s_6'], cfg['s_6']), padding='same', kernel_regularizer=l2(cfg['l2']), bias_regularizer=l2(cfg['l2']))(layer)
            layer = skip_manager.connect_skip(layer,filter_amount,cfg['k_12'],cfg['l2'],cfg['activation'])
            layer = Activation(cfg['activation'])(layer)
        else:
            #layer = skip_manager.end_skip(layer,filter_amount,cfg['k_12'],cfg['l2'],cfg['activation'])
            layer = MaxPooling2D(pool_size=(cfg['k_13'], cfg['k_13']), strides=(cfg['s_6'], cfg['s_6']), padding='same')(layer)
        layer = Dropout(cfg['dropout_7'])(layer)
        skip_manager.set_dropout(cfg['dropout_7'])

    #layer = input1#TODO remove this
    #global averaging
    if (cfg['global_pooling']):
        layer = GlobalAveragePooling2D()(layer)
        #layer = Dropout(cfg['dropout_7'])(layer) #TODO note that this line was removed between the "tweaked" experiment and the "better data augmentation" experiment
    else:
        layer = Flatten()(layer)
    
    
    #head
    if cfg['dense_size_0'] > 0:
        layer = Dense(cfg['dense_size_0'], kernel_regularizer=l2(cfg['l2']), bias_regularizer=l2(cfg['l2']))(layer)
        layer = Activation(cfg['activation'])(layer)
        layer = Dropout(cfg['dropout_8'])(layer)
    if cfg['dense_size_1'] > 0:
        layer = Dense(cfg['dense_size_1'], kernel_regularizer=l2(cfg['l2']), bias_regularizer=l2(cfg['l2']))(layer)
        layer = Activation(cfg['activation'])(layer)
        layer = Dropout(cfg['dropout_9'])(layer)
    layer = Dense(num_classes, kernel_regularizer=l2(cfg['l2']), bias_regularizer=l2(cfg['l2']))(layer)
    out = Activation(cfg['activ_dense'])(layer)
    
    cfg['decay'] = cfg['lr'] / float(epochs)
    def step_decay(epoch):
        initial_lrate = cfg['lr']
        drop = cfg['drop']
        epochs_drop = cfg['epochs_drop']
        lrate = initial_lrate * math.pow(drop,  math.floor((1+epoch)/epochs_drop))
        return lrate

    hist_func = TimedAccHistory()
    callbacks = [hist_func]
    #if (cfg['step'] == True):
    callbacks = [LearningRateScheduler(step_decay),hist_func]
    cfg['decay'] = 0.

    if cfg['optimizer'] == "SGD":
        opt = keras.optimizers.SGD(lr=cfg['lr'], momentum=cfg['momentum'], decay=0.0, nesterov=False)
    elif cfg['optimizer'] == "RMSprop":
        opt = keras.optimizers.RMSprop(lr=cfg['lr'], rho=cfg['rho'], epsilon=None, decay=0.0)
    elif cfg['optimizer'] == "Adagrad":
        opt = keras.optimizers.Adagrad(lr=cfg['lr'], epsilon=None, decay=0.0)
    elif cfg['optimizer'] == "Adadelta":
        opt = keras.optimizers.Adadelta(lr=cfg['lr'], rho=cfg['rho'], epsilon=None, decay=0.0)
    elif cfg['optimizer'] == "Adam":
        opt = keras.optimizers.Adam(lr=cfg['lr'], beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    elif cfg['optimizer'] == "Adamax":
        opt = keras.optimizers.Adamax(lr=cfg['lr'], beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
    elif cfg['optimizer'] == "Nadam":
        opt = keras.optimizers.Nadam(lr=cfg['lr'], beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
    else:
        print("error, unknown optimizer")
        return

    model = keras.models.Model(inputs=input1, outputs=out)

    # Let's train the model using RMSprop
    model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])#TODO 'adam' moet zijn: opt
    #model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

    if test:
        return model #TODO remove this, just for testing
    
    #print("amount of parameters:")
    #print(model.count_params())
    #CHRIS test if gpu has enough memory
    #nvmlInit()
    #handle = nvmlDeviceGetHandleByIndex(int(gpu_no))
    #meminfo = nvmlDeviceGetMemoryInfo(handle)
    #max_size = meminfo.total #6689341440
    #if meminfo.free/1024.**2 < 1.0:
    #    print('gpu is allready in use')
    #nvmlShutdown()
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
    x_val = x_val.astype('float32')
    
    if not data_augmentation or data_augmentation:# CHRIS data augmentation handles normalizationTODO remove or data_augmentation
        x_train /= 255.
        x_test /= 255.
        x_val /= 255.
    
    if True:#In case of training for validation
        x_test = x_val
        y_test = y_val

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
        #datagen = ImageDataGenerator(
        #    featurewise_center=False,  # set input mean to 0 over the dataset
        #    samplewise_center=False,  # set each sample mean to 0
        #    featurewise_std_normalization=False,  # divide inputs by std of the dataset
        #    samplewise_std_normalization=False,  # divide each input by its std
        #    zca_epsilon=1*10**-6,
        #    zca_whitening=False,  # apply ZCA whitening
        #    rotation_range=180,  # randomly rotate images in the range (degrees, 0 to 180)
        #    width_shift_range=0.25,  # randomly shift images horizontally (fraction of total width)
        #    height_shift_range=0.25,  # randomly shift images vertically (fraction of total height)
        #    shear_range=20.0,
        #    zoom_range=2.0,
        #    channel_shift_range=0.0,
        #    fill_mode='wrap',#('constant','nearest',reflect','wrap')
        #    cval=0.0,
        #    horizontal_flip=True,  # randomly flip images
        #    vertical_flip=True,  # randomly flip images
        #    rescale=1.0)
            
        datagen = ImageDataGenerator(
             featurewise_center=cfg['featurewise_center'],  # set input mean to 0 over the dataset
             samplewise_center=cfg['samplewise_center'],  # set each sample mean to 0
             featurewise_std_normalization=cfg['featurewise_std_normalization'],  # divide inputs by std of the dataset
             samplewise_std_normalization=cfg['samplewise_std_normalization'],  # divide each input by its std
             zca_epsilon=cfg['zca_epsilon'],
             zca_whitening=cfg['zca_whitening'],  # apply ZCA whitening
             rotation_range=cfg['rotation_range'],  # randomly rotate images in the range (degrees, 0 to 180)
             width_shift_range=cfg['width_shift_range'],  # randomly shift images horizontally (fraction of total width)
             height_shift_range=cfg['height_shift_range'],  # randomly shift images vertically (fraction of total height)
             shear_range=cfg['shear_range'],
             zoom_range=cfg['zoom_range'],
             channel_shift_range=cfg['channel_shift_range'],
             fill_mode=cfg['fill_mode'],#('constant','nearest',reflect','wrap')
             cval=cfg['cval'],
             horizontal_flip=cfg['horizontal_flip'],  # randomly flip images
             vertical_flip=cfg['vertical_flip'],  # randomly flip images
             rescale=1.0)
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

    #CHRIS append network training history to file
    eval_training_hist = [time.time(),hist.history['acc'],hist.history['val_acc'],hist_func.timed]
    with open(save_name + '_eval_train_hist.json', 'a') as outfile:
        json.dump(eval_training_hist,outfile)
        outfile.write('\n')

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

#CHRIS testcode
def test_skippy():
    from mipego.mipego import Solution #TODO remove this, only for testing
    from mipego.SearchSpace import ContinuousSpace, NominalSpace, OrdinalSpace
    from keras.utils import plot_model
    
    with open('skippy_test_train_hist' + '_eval_train_hist.json', 'w') as f:
        f.write('')
    #define the search space.
    #objective = obj_func('./all-cnn_bi.py')
    activation_fun = ["softmax"]
    activation_fun_conv = ["elu","relu","tanh","sigmoid","selu"]

    filters = OrdinalSpace([10, 100], 'filters') * 14 #TODO [0,100] should be [0,600]
    kernel_size = OrdinalSpace([1, 8], 'k') * 14
    strides = OrdinalSpace([1, 5], 's') * 7
    stack_sizes = OrdinalSpace([0, 4], 'stack') * 7 #TODO [0,4] should be [0,7]

    activation = NominalSpace(activation_fun_conv, "activation")  # activation function
    activation_dense = NominalSpace(activation_fun, "activ_dense") # activation function for dense layer
    step = NominalSpace([True, False], "step")  # step
    global_pooling = NominalSpace([True, False], "global_pooling")  # global_pooling
    
    #skippy parameters
    skstart = OrdinalSpace([0, 50], 'skstart') * 5
    skstep = OrdinalSpace([1, 50], 'skstep') * 5
    max_pooling = NominalSpace([True, False], "max_pooling")
    dense_size = OrdinalSpace([0,2000],'dense_size')*2
    #skippy parameters

    drop_out = ContinuousSpace([1e-5, .9], 'dropout') * 10        # drop_out rate
    lr_rate = ContinuousSpace([1e-4, 1.0e-0], 'lr')        # learning rate
    l2_regularizer = ContinuousSpace([1e-5, 1e-2], 'l2')# l2_regularizer

    search_space =  stack_sizes * strides * filters *  kernel_size * activation * activation_dense * drop_out * lr_rate * l2_regularizer * step * global_pooling * skstart * skstep * max_pooling * dense_size
    
    n_init_sample = 1
    samples = search_space.sampling(n_init_sample)
    print(samples)
    var_names = search_space.var_name.tolist()
    print(var_names)
    
    #a sample
    #samples = [[1, 1, 1, 1, 2, 3, 10, 10, 5, 10, 10, 10, 10, 3, 4, 2, 1, 3, 1, 3, 'relu', 'softmax', 0.7105013348601977, 0.24225495530708516, 0.5278997344637044, 0.7264822991098491, 0.0072338759099408985, 0.00010867041652507452, False, True]]

    #test parameters
    #original parameters
    #RESnet-34-like
    stack_0 = 1
    stack_1 = 6
    stack_2 = 4
    stack_3 = 4
    stack_4 = 6
    stack_5 = 6
    stack_6 = 6
    s_0=2#1#2
    s_1=2
    s_2=1#1
    s_3=2
    s_4=1
    s_5=2
    s_6=1
    filters_0=64
    filters_1=64
    filters_2=64
    filters_3=64
    filters_4=128
    filters_5=128
    filters_6=128
    filters_7=128
    filters_8=256
    filters_9=256
    filters_10=256
    filters_11=256
    filters_12=512
    filters_13=512
    k_0=7
    k_1=1
    k_2=3
    k_3=1
    k_4=3
    k_5=1
    k_6=3
    k_7=1
    k_8=3
    k_9=1
    k_10=3
    k_11=1
    k_12=3
    k_13=1
    activation='relu'
    activ_dense='softmax'
    dropout_0=0.001
    dropout_1=0.001
    dropout_2=0.001
    dropout_3=0.001
    dropout_4=0.001
    dropout_5=0.001
    dropout_6=0.001
    dropout_7=0.001
    dropout_8=0.001
    dropout_9=0.001
    lr=0.01
    l2=0.0001
    step=False#True
    global_pooling=True

    #skippy parameters
    om_en_om = 1
    ranges = [stack_6,stack_5,stack_4,stack_3,stack_2,stack_1,stack_0]
    for w in range(len(ranges)):#TODO testcode: remove
        om_en_om = om_en_om << 1
        for z in range(ranges[w]//2):
            om_en_om = om_en_om << 2
            om_en_om += 1
    om_en_om = om_en_om << 1
    skstart_0 = 1#inv_gray(om_en_om)#3826103921638#2**30-1
    skstart_1 = 1#19283461627361826#2**30-1
    skstart_2 = 1#473829102637452916#2**30-1
    skstart_3 = 1#473829102637452916#2**30-1
    skstart_4 = 1#473829102637452916#2**30-1
    skstep_0 = 2
    skstep_1 = 1
    skstep_2 = 1
    skstep_3 = 1
    skstep_4 = 1
    max_pooling = True
    dense_size_0 = 1000
    dense_size_1 = 0
    #skippy parameters

    #assembling parameters
    samples = [[stack_0, stack_1, stack_2, stack_3, stack_4, stack_5, stack_6, s_0, s_1, s_2, s_3, s_4, s_5, s_6, filters_0, filters_1, filters_2, filters_3, filters_4, filters_5, filters_6, filters_7, filters_8, filters_9, filters_10, filters_11, filters_12, filters_13,k_0, k_1, k_2, k_3, k_4, k_5, k_6, k_7, k_8, k_9, k_10, k_11, k_12, k_13, activation, activ_dense, dropout_0, dropout_1, dropout_2, dropout_3, dropout_4, dropout_5, dropout_6, dropout_7, dropout_8, dropout_9, lr, l2, step, global_pooling, skstart_0, skstart_1, skstart_2, skstart_3, skstart_4, skstep_0, skstep_1, skstep_2, skstep_3, skstep_4, max_pooling, dense_size_0, dense_size_1]]
    
    #var_names
    #['stack_0', 'stack_1', 'stack_2', 's_0', 's_1', 's_2', 'filters_0', 'filters_1', 'filters_2', 'filters_3', 'filters_4', 'filters_5', 'filters_6', 'k_0', 'k_1', 'k_2', 'k_3', 'k_4', 'k_5', 'k_6', 'activation', 'activ_dense', 'dropout_0', 'dropout_1', 'dropout_2', 'dropout_3', 'lr', 'l2', 'step', 'global_pooling']

    dropout_mult = 1.0
    lr_mult = 1.0
    X = [Solution(s, index=k, var_name=var_names) for k, s in enumerate(samples)]
    vla = {'s_5': 3, 'filters_9': 339, 'dropout_2': 0.3496803660826153, 's_6': 3, 'stack_0': 3, 'dropout_6': 0.19656398582242512, 'rotation_range': 31, 'filters_7': 202, 'dropout_0': 0.005004155479145995, 's_3': 3, 's_1': 1, 'filters_1': 120, 'k_8': 9, 's_0': 2, 'stack_3': 1, 'samplewise_std_normalization': False, 'dropout_1': 0.16597601970646955, 'cval': 0.24779638415638786, 'filters_6': 535, 'stack_6': 0, 'filters_3': 473, 'zoom_range': 0.02446592218470434, 'dense_size_0': 1344, 'zca_whitening': False, 'k_1': 11, 'dropout_3': 0.11567654065541401, 'k_0': 2, 'k_3': 4, 'filters_5': 109, 'zca_epsilon': 1.2393513955305375e-06, 'activation': 'selu', 'max_pooling': True, 'filters_4': 191, 'filters_11': 507, 'filters_8': 353, 'filters_13': 350, 's_2': 3, 'dropout_7': 0.2567508735733037, 'k_12': 4, 'channel_shift_range': 0.002134671459292783, 'filters_2': 397, 'skstep_1': 8, 'skstep_0': 2, 'k_6': 8, 'skstart_1': 0, 'skstep_2': 2, 'vertical_flip': False, 'dropout_9': 0.282536761776477, 'width_shift_range': 0.11326574574565945, 'k_2': 4, 'skstart_0': 6, 'dense_size_1': 1216, 's_4': 3, 'shear_range': 4.413108635288765, 'stack_1': 1, 'filters_10': 305, 'horizontal_flip': True, 'k_11': 9, 'filters_12': 474, 'k_7': 9, 'featurewise_std_normalization': False, 'skstep_3': 2, 'stack_2': 2, 'global_pooling': True, 'step': False, 'batch_size_sp': 75, 'l2': 0.00043366714416766863, 'stack_5': 0, 'fill_mode': 'nearest', 'skstep_4': 1, 'skstart_4': 1, 'samplewise_center': False, 'k_13': 4, 'filters_0': 246, 'skstart_2': 6, 'k_4': 14, 'k_9': 5, 'lr': 0.003521543292982737, 'dropout_4': 0.025329970168830974, 'featurewise_center': False, 'k_10': 10, 'dropout_5': 0.09773198911653828, 'skstart_3': 3, 'height_shift_range': 0.5512549395731117, 'stack_4': 5, 'dropout_8': 0.10220859898802954, 'activ_dense': 'softmax', 'k_5': 10}
    cfg = {'lr': 0.0016057778383265723, 'drop': 0.8237039131904371, 'optimizer': 'SGD', 'epochs_drop': 13, 'rho': 0.923183503914636, 'momentum': 0.9584685019441632}
    for a in cfg:
        vla[a] = cfg[a]
    print(X)
    print(X[0].to_dict())
    #cfg = [Solution(x, index=len(self.data) + i, var_name=self.var_names) for i, x in enumerate(X)]
    test = False
    if test:
        #model = CNN_conf(X[0].to_dict(),test=test)
        model = CNN_conf(vla,test=test)
        plot_model(model, to_file='model_skippy_test.png',show_shapes=True,show_layer_names=True)
        model.summary()
        print(model.count_params())
        print(str(model.count_params() * 4 * 2 / 1024/1024/1024) + ' Gb')
    else:
        #timer, loss = CNN_conf(X[0].to_dict(),test=test,epochs= 2000,verbose=1)
        #timer, loss = CNN_conf(vla,test=test,epochs= 200,verbose=1)
        timer, loss = CNN_conf(vla,test=test,epochs= 200,verbose=1,data_augmentation=True,use_validation=True) #TODO use this for data augmentation and make sure the val set is used for val accuracy, not the test set
        print('timer, loss:')
        print(timer, loss)

if __name__ == '__main__':
    #system arguments (configuration)
    if len(sys.argv) > 2 and sys.argv[1] == '--cfg':
        cfg = {'s_5': 3, 'filters_9': 339, 'dropout_2': 0.3496803660826153, 's_6': 3, 'stack_0': 3, 'dropout_6': 0.19656398582242512, 'rotation_range': 31, 'filters_7': 202, 'dropout_0': 0.005004155479145995, 's_3': 3, 's_1': 1, 'filters_1': 120, 'k_8': 9, 's_0': 2, 'stack_3': 1, 'samplewise_std_normalization': False, 'dropout_1': 0.16597601970646955, 'cval': 0.24779638415638786, 'filters_6': 535, 'stack_6': 0, 'filters_3': 473, 'zoom_range': 0.02446592218470434, 'dense_size_0': 1344, 'zca_whitening': False, 'k_1': 11, 'dropout_3': 0.11567654065541401, 'k_0': 2, 'k_3': 4, 'filters_5': 109, 'zca_epsilon': 1.2393513955305375e-06, 'activation': 'selu', 'max_pooling': True, 'filters_4': 191, 'filters_11': 507, 'filters_8': 353, 'filters_13': 350, 's_2': 3, 'dropout_7': 0.2567508735733037, 'k_12': 4, 'channel_shift_range': 0.002134671459292783, 'filters_2': 397, 'skstep_1': 8, 'skstep_0': 2, 'k_6': 8, 'skstart_1': 0, 'skstep_2': 2, 'vertical_flip': False, 'dropout_9': 0.282536761776477, 'width_shift_range': 0.11326574574565945, 'k_2': 4, 'skstart_0': 6, 'dense_size_1': 1216, 's_4': 3, 'shear_range': 4.413108635288765, 'stack_1': 1, 'filters_10': 305, 'horizontal_flip': True, 'k_11': 9, 'filters_12': 474, 'k_7': 9, 'featurewise_std_normalization': False, 'skstep_3': 2, 'stack_2': 2, 'global_pooling': True, 'step': False, 'batch_size_sp': 75, 'l2': 0.00043366714416766863, 'stack_5': 0, 'fill_mode': 'nearest', 'skstep_4': 1, 'skstart_4': 1, 'samplewise_center': False, 'k_13': 4, 'filters_0': 246, 'skstart_2': 6, 'k_4': 14, 'k_9': 5, 'lr': 0.003521543292982737, 'dropout_4': 0.025329970168830974, 'featurewise_center': False, 'k_10': 10, 'dropout_5': 0.09773198911653828, 'skstart_3': 3, 'height_shift_range': 0.5512549395731117, 'stack_4': 5, 'dropout_8': 0.10220859898802954, 'activ_dense': 'softmax', 'k_5': 10}
        cfg_xtra = eval(sys.argv[2])
        for a in cfg_xtra:
            cfg[a] = cfg_xtra[a]
        if len(sys.argv) > 3:
            gpu = sys.argv[3]
            epochs = int(sys.argv[4])
            save_name = str(sys.argv[5])
            data_augmentation = str(sys.argv[6]) == 'True'
            use_validation = str(sys.argv[7]) == 'True'
            
            os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
            os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu)
        print(CNN_conf(cfg,gpu_no=gpu,epochs=epochs,save_name=save_name,data_augmentation=data_augmentation,use_validation=use_validation))
        K.clear_session()
    else:
        print('switching to test mode')
        test_skippy()
