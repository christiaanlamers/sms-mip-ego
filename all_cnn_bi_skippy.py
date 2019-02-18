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
    
    def pad_and_connect(self, layer, incoming_layer):
        if K.int_shape(incoming_layer)[1] != K.int_shape(layer)[1] or K.int_shape(incoming_layer)[2] != K.int_shape(layer)[2]:
            pad_tpl1 = (int(np.floor(np.abs(K.int_shape(incoming_layer)[1]-K.int_shape(layer)[1])/2)),int(np.ceil(np.abs(K.int_shape(incoming_layer)[1]-K.int_shape(layer)[1])/2)))
            pad_tpl2 = (int(np.floor(np.abs(K.int_shape(incoming_layer)[2]-K.int_shape(layer)[2])/2)),int(np.ceil(np.abs(K.int_shape(incoming_layer)[2]-K.int_shape(layer)[2])/2)))
            #print(pad_tpl)
            if K.int_shape(incoming_layer)[1] < K.int_shape(layer)[1] and K.int_shape(incoming_layer)[2] < K.int_shape(layer)[2]:
                padded = ZeroPadding2D(padding=(pad_tpl1, pad_tpl2))(incoming_layer)
                return Concatenate()([layer, padded])
            elif K.int_shape(incoming_layer)[1] < K.int_shape(layer)[1] and K.int_shape(incoming_layer)[2] >= K.int_shape(layer)[2]:
                padded1 = ZeroPadding2D(padding=(pad_tpl1, 0))(incoming_layer)
                padded2 = ZeroPadding2D(padding=(0, pad_tpl2))(layer)
                return Concatenate()([padded1, padded2])
            elif K.int_shape(incoming_layer)[1] >= K.int_shape(layer)[1] and K.int_shape(incoming_layer)[2] < K.int_shape(layer)[2]:
                padded1 = ZeroPadding2D(padding=(0, pad_tpl2))(incoming_layer)
                padded2 = ZeroPadding2D(padding=(pad_tpl1, 0))(layer)
                return Concatenate()([padded1, padded2])
            else:
                #print(layer.shape)
                padded = ZeroPadding2D(padding=(pad_tpl1, pad_tpl2))(layer)
                #print(padded.shape)
                #print(incoming_layer.shape)
                return Concatenate()([padded, incoming_layer])
        else:
            return Concatenate()([layer, incoming_layer])

    def pool_pad_connect(self, layer, incoming_layer):
        if K.int_shape(incoming_layer)[1] < K.int_shape(layer)[1] and K.int_shape(incoming_layer)[2] < K.int_shape(layer)[2]:
            pass
        elif K.int_shape(incoming_layer)[1] < K.int_shape(layer)[1] and K.int_shape(incoming_layer)[2] >= K.int_shape(layer)[2]:
            print('error, not implemented yet')
            pass
        elif K.int_shape(incoming_layer)[1] >= K.int_shape(layer)[1] and K.int_shape(incoming_layer)[2] < K.int_shape(layer)[2]:
            print('error, not implemented yet')
            pass
        else: #K.int_shape(incoming_layer)[1] > K.int_shape(layer)[1] and K.int_shape(incoming_layer)[2] > K.int_shape(layer)[2]
            scalar_1 =  K.int_shape(incoming_layer)[1] // K.int_shape(layer)[1]
            scalar_2 =  K.int_shape(incoming_layer)[2] // K.int_shape(layer)[2]
            incoming_layer = MaxPooling2D(pool_size=(scalar_1, scalar_2), strides=(scalar_1, scalar_2), padding='same')(incoming_layer)
        return self.pad_and_connect(layer, incoming_layer)

    def connect_skip(self,layer):
        #start skip connections
        for j in range(len(self.skip_ints)):
            if self.skip_ints_count[j] > 1 and self.startpoint(self.gray,self.skip_ints[j]):#CHRIS skip connections smaller than 2 are not made, thus mean no skip connection.
                self.skip_connections.append([layer,self.skip_ints_count[j],self.layer_num])#save layer output, skip counter, layer this skip connection starts (to remove duplicates)
    
        #end skip connections
        j = 0
        prev_skip = -1
        while j < len(self.skip_connections):
            if self.skip_connections[j][1] <= 0:
                #print(prev_skip,self.skip_connections[j][2])
                if prev_skip != self.skip_connections[j][2]:#this removes skip connection duplicates (works because same skip connections are next to eachother) TODO maybe better to make more robust
                    #CHRIS TODO add pooling, because this becomes too complex to train
                    #layer = self.pad_and_connect(layer, self.skip_connections[j][0])
                    layer = self.pool_pad_connect(layer, self.skip_connections[j][0])
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
                self.skip_connections[j][1] -= 1 #decrease skip connection counters
                j += 1
        self.layer_num +=1 #increase layer number where currently building takes place
        return layer


def CNN_conf(cfg,epochs=1,test=False,gpu_no=0,verbose=0):
    batch_size = 100
    num_classes = 10
    data_augmentation = False
    num_predictions = 20
    logfile = 'mnist-cnn.log'
    savemodel = False

    # The data, shuffled and split between train and test sets:
    #(x_train, y_train), (x_test, y_test) = mnist.load_data()
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
    
    layer = Dropout(cfg['dropout_0'],input_shape=x_train.shape[1:])(input1)#CHRIS TODO reengage this line!
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
        if not (cfg['max_pooling']):
            layer = Conv2D(cfg['filters_1'], (cfg['k_1'], cfg['k_1']), strides=(cfg['s_0'], cfg['s_0']), padding='same', kernel_regularizer=l2(cfg['l2']), bias_regularizer=l2(cfg['l2']))(layer)
            layer = Activation(cfg['activation'])(layer)
        else:
            layer = MaxPooling2D(pool_size=(cfg['k_1'], cfg['k_1']), strides=(cfg['s_0'], cfg['s_0']), padding='same')(layer)
        layer = Dropout(cfg['dropout_1'])(layer)
        layer = skip_manager.connect_skip(layer)
    
    #stack 1
    for i in range(cfg['stack_1']):
        layer = Conv2D(cfg['filters_2'], (cfg['k_2'], cfg['k_2']), padding='same', kernel_regularizer=l2(cfg['l2']), bias_regularizer=l2(cfg['l2']))(layer)
        layer = Activation(cfg['activation'])(layer)
        layer = skip_manager.connect_skip(layer)
    if (cfg['stack_1']>0):
        if not (cfg['max_pooling']):
            layer = Conv2D(cfg['filters_3'], (cfg['k_3'], cfg['k_3']), strides=(cfg['s_1'], cfg['s_1']), padding='same', kernel_regularizer=l2(cfg['l2']), bias_regularizer=l2(cfg['l2']))(layer)
            layer = Activation(cfg['activation'])(layer)
        else:
            layer = MaxPooling2D(pool_size=(cfg['k_3'], cfg['k_3']), strides=(cfg['s_1'], cfg['s_1']), padding='same')(layer)
        layer = Dropout(cfg['dropout_2'])(layer)
        layer = skip_manager.connect_skip(layer)

    #stack 2
    for i in range(cfg['stack_2']):
        layer = Conv2D(cfg['filters_4'], (cfg['k_4'], cfg['k_4']), padding='same', kernel_regularizer=l2(cfg['l2']), bias_regularizer=l2(cfg['l2']))(layer)
        layer = Activation(cfg['activation'])(layer)
        layer = skip_manager.connect_skip(layer)
    if (cfg['stack_2']>0):
        if not (cfg['max_pooling']):
            layer = Conv2D(cfg['filters_5'], (cfg['k_5'], cfg['k_5']), strides=(cfg['s_2'], cfg['s_2']), padding='same', kernel_regularizer=l2(cfg['l2']), bias_regularizer=l2(cfg['l2']))(layer)
            layer = Activation(cfg['activation'])(layer)
        else:
            layer = MaxPooling2D(pool_size=(cfg['k_5'], cfg['k_5']), strides=(cfg['s_2'], cfg['s_2']), padding='same')(layer)
        layer = Dropout(cfg['dropout_3'])(layer)
        layer = skip_manager.connect_skip(layer)

    #stack 3
    for i in range(cfg['stack_3']):
        layer = Conv2D(cfg['filters_6'], (cfg['k_6'], cfg['k_6']), padding='same', kernel_regularizer=l2(cfg['l2']), bias_regularizer=l2(cfg['l2']))(layer)
        layer = Activation(cfg['activation'])(layer)
        layer = skip_manager.connect_skip(layer)
    if (cfg['stack_3']>0):
        if not (cfg['max_pooling']):
            layer = Conv2D(cfg['filters_7'], (cfg['k_7'], cfg['k_7']), strides=(cfg['s_3'], cfg['s_3']), padding='same', kernel_regularizer=l2(cfg['l2']), bias_regularizer=l2(cfg['l2']))(layer)
            layer = Activation(cfg['activation'])(layer)
        else:
            layer = MaxPooling2D(pool_size=(cfg['k_7'], cfg['k_7']), strides=(cfg['s_3'], cfg['s_3']), padding='same')(layer)
        layer = Dropout(cfg['dropout_4'])(layer)
        layer = skip_manager.connect_skip(layer)

    #stack 4
    for i in range(cfg['stack_4']):
        layer = Conv2D(cfg['filters_8'], (cfg['k_8'], cfg['k_8']), padding='same', kernel_regularizer=l2(cfg['l2']), bias_regularizer=l2(cfg['l2']))(layer)
        layer = Activation(cfg['activation'])(layer)
        layer = skip_manager.connect_skip(layer)
    if (cfg['stack_4']>0):
        if not (cfg['max_pooling']):
            layer = Conv2D(cfg['filters_9'], (cfg['k_9'], cfg['k_9']), strides=(cfg['s_4'], cfg['s_4']), padding='same', kernel_regularizer=l2(cfg['l2']), bias_regularizer=l2(cfg['l2']))(layer)
            layer = Activation(cfg['activation'])(layer)
        else:
            layer = MaxPooling2D(pool_size=(cfg['k_9'], cfg['k_9']), strides=(cfg['s_4'], cfg['s_4']), padding='same')(layer)
        layer = Dropout(cfg['dropout_5'])(layer)
        layer = skip_manager.connect_skip(layer)

    #stack 5
    for i in range(cfg['stack_5']):
        layer = Conv2D(cfg['filters_10'], (cfg['k_10'], cfg['k_10']), padding='same', kernel_regularizer=l2(cfg['l2']), bias_regularizer=l2(cfg['l2']))(layer)
        layer = Activation(cfg['activation'])(layer)
        layer = skip_manager.connect_skip(layer)
    if (cfg['stack_5']>0):
        if not (cfg['max_pooling']):
            layer = Conv2D(cfg['filters_11'], (cfg['k_11'], cfg['k_11']), strides=(cfg['s_5'], cfg['s_5']), padding='same', kernel_regularizer=l2(cfg['l2']), bias_regularizer=l2(cfg['l2']))(layer)
            layer = Activation(cfg['activation'])(layer)
        else:
            layer = MaxPooling2D(pool_size=(cfg['k_11'], cfg['k_11']), strides=(cfg['s_5'], cfg['s_5']), padding='same')(layer)
        layer = Dropout(cfg['dropout_6'])(layer)
        layer = skip_manager.connect_skip(layer)

    #stack 6
    for i in range(cfg['stack_6']):
        layer = Conv2D(cfg['filters_12'], (cfg['k_12'], cfg['k_12']), padding='same', kernel_regularizer=l2(cfg['l2']), bias_regularizer=l2(cfg['l2']))(layer)
        layer = Activation(cfg['activation'])(layer)
        layer = skip_manager.connect_skip(layer)
    if (cfg['stack_6']>0):
        if not (cfg['max_pooling']):
            layer = Conv2D(cfg['filters_13'], (cfg['k_13'], cfg['k_13']), strides=(cfg['s_6'], cfg['s_6']), padding='same', kernel_regularizer=l2(cfg['l2']), bias_regularizer=l2(cfg['l2']))(layer)
            layer = Activation(cfg['activation'])(layer)
        else:
            layer = MaxPooling2D(pool_size=(cfg['k_13'], cfg['k_13']), strides=(cfg['s_6'], cfg['s_6']), padding='same')(layer)
        layer = Dropout(cfg['dropout_7'])(layer)
        layer = skip_manager.connect_skip(layer)

    #layer = input1#TODO remove this
    #global averaging
    if (cfg['global_pooling']):
        layer = GlobalAveragePooling2D()(layer)
    else:
        layer = Flatten()(layer)
    
    
    #head
    layer = Dense(num_classes, kernel_regularizer=l2(cfg['l2']), bias_regularizer=l2(cfg['l2']))(layer)
    out = Activation(cfg['activ_dense'])(layer)
    
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
    #print('run-time:')
    #print(timer)

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
    skints = OrdinalSpace([0, 2**50-1], 'skint') * 3#CHRIS TODO tweak this
    skst = OrdinalSpace([2, 10], 'skst') * 3
    max_pooling = NominalSpace([True, False], "max_pooling")
    #skippy parameters

    drop_out = ContinuousSpace([1e-5, .9], 'dropout') * 8        # drop_out rate
    lr_rate = ContinuousSpace([1e-4, 1.0e-0], 'lr')        # learning rate
    l2_regularizer = ContinuousSpace([1e-5, 1e-2], 'l2')# l2_regularizer

    search_space =  stack_sizes * strides * filters *  kernel_size * activation * activation_dense * drop_out * lr_rate * l2_regularizer * step * global_pooling * skints * skst * max_pooling
    
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
    stack_0 = 1#1
    stack_1 = 1#6
    stack_2 = 1#4
    stack_3 = 1#4
    stack_4 = 1#6
    stack_5 = 1#6
    stack_6 = 1#6
    s_0=1#2
    s_1=1#2
    s_2=1
    s_3=2#2
    s_4=1
    s_5=1#2
    s_6=1
    filters_0=32#64*2
    filters_1=32#64*2
    filters_2=32#64*2
    filters_3=32#64*2
    filters_4=32#128*2
    filters_5=32#128*2
    filters_6=32#128*2
    filters_7=32#128*2
    filters_8=64#256*2
    filters_9=64#256*2
    filters_10=64#256*2
    filters_11=64#256*2
    filters_12=64#512*2
    filters_13=64#512*2
    k_0=3#7
    k_1=3
    k_2=3
    k_3=3
    k_4=3
    k_5=3
    k_6=3
    k_7=3
    k_8=3
    k_9=3
    k_10=3
    k_11=3
    k_12=3
    k_13=3
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
    lr=0.01
    l2=0.0001
    step=False#True
    global_pooling=False#True

    #skippy parameters
    om_en_om = 1
    ranges = [stack_6,stack_5,stack_4,stack_3,stack_2,stack_1,stack_0]
    for w in range(len(ranges)):#TODO testcode: remove
        om_en_om = om_en_om << 1
        for z in range(ranges[w]//2):
            om_en_om = om_en_om << 2
            om_en_om += 1
    om_en_om = om_en_om << 1
    skint_0 = inv_gray(2**30-1)#inv_gray(om_en_om)#3826103921638#2**30-1
    skint_1 = 0#19283461627361826#2**30-1
    skint_2 = 0#473829102637452916#2**30-1
    skst_0 = 2
    skst_1 = 0
    skst_2 = 0
    max_pooling = True
    #skippy parameters

    #assembling parameters
    samples = [[stack_0, stack_1, stack_2, stack_3, stack_4, stack_5, stack_6, s_0, s_1, s_2, s_3, s_4, s_5, s_6, filters_0, filters_1, filters_2, filters_3, filters_4, filters_5, filters_6, filters_7, filters_8, filters_9, filters_10, filters_11, filters_12, filters_13,k_0, k_1, k_2, k_3, k_4, k_5, k_6, k_7, k_8, k_9, k_10, k_11, k_12, k_13, activation, activ_dense, dropout_0, dropout_1, dropout_2, dropout_3, dropout_4, dropout_5, dropout_6, dropout_7, lr, l2, step, global_pooling, skint_0, skint_1, skint_2, skst_0, skst_1, skst_2, max_pooling]]
    
    #var_names
    #['stack_0', 'stack_1', 'stack_2', 's_0', 's_1', 's_2', 'filters_0', 'filters_1', 'filters_2', 'filters_3', 'filters_4', 'filters_5', 'filters_6', 'k_0', 'k_1', 'k_2', 'k_3', 'k_4', 'k_5', 'k_6', 'activation', 'activ_dense', 'dropout_0', 'dropout_1', 'dropout_2', 'dropout_3', 'lr', 'l2', 'step', 'global_pooling']

    
    X = [Solution(s, index=k, var_name=var_names) for k, s in enumerate(samples)]
    vla = {'s_0': 3, 'l2': 4.4274387289657325e-05, 'filters_7': 423, 'dense_size_1': 992, 'filters_12': 295, 'stack_0': 0, 'filters_2': 53, 'global_pooling': True, 'dropout_6': 0.5606577615096975, 'filters_13': 115, 'filters_4': 396, 'stack_4': 0, 'k_9': 6, 'activation': 'tanh', 'dropout_1': 0.07267176147234225, 'filters_5': 405, 'filters_1': 250, 'k_7': 7, 'filters_3': 408, 'stack_2': 0, 'max_pooling': True, 'dropout_7': 0.12689965102483852, 's_2': 4, 'filters_8': 455, 'dropout_4': 0.8991002969243431, 'k_11': 5, 'skst_0': 7, 'k_4': 3, 'dropout_3': 0.5966691482903116, 'step': False, 'dense_size_0': 583, 'stack_1': 1, 'k_0': 3, 'skint_1': 505527202345094, 'k_1': 5, 'k_8': 1, 'stack_6': 0, 'lr': 0.6919959357016345, 'activ_dense': 'softmax', 'filters_6': 305, 's_1': 3, 'filters_9': 226, 's_4': 2, 'stack_3': 1, 'skst_1': 5, 'skst_2': 6, 'dropout_2': 0.039087410518674766, 'k_12': 4, 'k_3': 6, 'dropout_5': 0.46057411289276423, 'skint_0': 957176709324259, 'k_5': 4, 'k_2': 3, 's_3': 1, 'filters_0': 195, 'k_6': 1, 'k_13': 2, 'skint_2': 1098454353499063, 'filters_11': 107, 'filters_10': 257, 'k_10': 4, 'stack_5': 0, 's_6': 1, 's_5': 2, 'dropout_0': 0.6293343140822664}
    print(X)
    print(X[0].to_dict())
    #cfg = [Solution(x, index=len(self.data) + i, var_name=self.var_names) for i, x in enumerate(X)]
    test = True
    if test:
        model = CNN_conf(X[0].to_dict(),test=test)
        #model = CNN_conf(vla,test=test)
        plot_model(model, to_file='model_skippy_test.png',show_shapes=True,show_layer_names=True)
        model.summary()
        print(model.count_params())
        print(str(model.count_params() * 4 * 2 / 1024/1024/1024) + ' Gb')
    else:
        timer, loss = CNN_conf(X[0].to_dict(),test=test,epochs= 2000,verbose=1)
        print('timer, loss:')
        print(timer, loss)

if __name__ == '__main__':#CHRIS TODO will this wreck the entire method?
    #system arguments (configuration)
    if len(sys.argv) > 2 and sys.argv[1] == '--cfg':
        cfg = eval(sys.argv[2])
        if len(sys.argv) > 3:
            gpu = sys.argv[3]
        
            os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
            os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu)
        print(CNN_conf(cfg,gpu_no=gpu,epochs=1))
        K.clear_session()
    else:
        print('switching to test mode')
        test_skippy()
