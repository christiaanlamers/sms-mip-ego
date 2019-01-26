from __future__ import print_function

import numpy as np
#np.random.seed(43)
import tensorflow as tf
tf.set_random_seed(43)

import keras
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D,UpSampling2D,ZeroPadding2D
import os
import sys
import pandas as pd
import keras.backend as K
import math
from keras.callbacks import LearningRateScheduler
from keras.regularizers import l2

import time #CHRIS added to measure runtime of training
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
                        pad_tpl1 = (int(np.floor(np.abs(K.int_shape(self.skip_connections[j][0])[1]-K.int_shape(layer)[1])/2)),int(np.ceil(np.abs(K.int_shape(self.skip_connections[j][0])[1]-K.int_shape(layer)[1])/2)))
                        pad_tpl2 = (int(np.floor(np.abs(K.int_shape(self.skip_connections[j][0])[2]-K.int_shape(layer)[2])/2)),int(np.ceil(np.abs(K.int_shape(self.skip_connections[j][0])[2]-K.int_shape(layer)[2])/2)))
                        #print(pad_tpl)
                        if K.int_shape(self.skip_connections[j][0])[1] < K.int_shape(layer)[1] and K.int_shape(self.skip_connections[j][0])[2] < K.int_shape(layer)[2]:
                            padded = ZeroPadding2D(padding=(pad_tpl1, pad_tpl2))(self.skip_connections[j][0])
                            layer = keras.layers.Concatenate()([layer, padded])
                        elif K.int_shape(self.skip_connections[j][0])[1] < K.int_shape(layer)[1] and K.int_shape(self.skip_connections[j][0])[2] >= K.int_shape(layer)[2]:
                            padded1 = ZeroPadding2D(padding=(pad_tpl1, 0))(self.skip_connections[j][0])
                            padded2 = ZeroPadding2D(padding=(0, pad_tpl2))(layer)
                            layer = keras.layers.Concatenate()([padded1, padded2])
                        elif K.int_shape(self.skip_connections[j][0])[1] >= K.int_shape(layer)[1] and K.int_shape(self.skip_connections[j][0])[2] < K.int_shape(layer)[2]:
                            padded1 = ZeroPadding2D(padding=(0, pad_tpl2))(self.skip_connections[j][0])
                            padded2 = ZeroPadding2D(padding=(pad_tpl1, 0))(layer)
                            layer = keras.layers.Concatenate()([padded1, padded2])
                        else:
                            #print(layer.shape)
                            padded = ZeroPadding2D(padding=(pad_tpl1, pad_tpl2))(layer)
                            #print(padded.shape)
                            #print(self.skip_connections[j][0].shape)
                            layer = keras.layers.Concatenate()([padded, self.skip_connections[j][0]])
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
                        layer = keras.layers.Concatenate()([layer, self.skip_connections[j][0]])
                prev_skip = self.skip_connections[j][2]
                del self.skip_connections[j]
            else:
                self.skip_connections[j][1] -= 1 #decrease skip connection counters
                j += 1
        self.layer_num +=1 #increase layer number where currently building takes place
        return layer


def CNN_conf(cfg,epochs=1,test=False):
    verbose = 0
    batch_size = 100
    num_classes = 10
    epochs = 100 #CHRIS increased from 1 to 5 to make results less random and noisy
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
        layer = Conv2D(cfg['filters_1'], (cfg['k_1'], cfg['k_1']), strides=(cfg['s_0'], cfg['s_0']), padding='same', kernel_regularizer=l2(cfg['l2']), bias_regularizer=l2(cfg['l2']))(layer)
        layer = Activation(cfg['activation'])(layer)
        layer = skip_manager.connect_skip(layer)
        layer = Dropout(cfg['dropout_1'])(layer)
    
    #stack 1
    for i in range(cfg['stack_1']):
        layer = Conv2D(cfg['filters_2'], (cfg['k_2'], cfg['k_2']), padding='same', kernel_regularizer=l2(cfg['l2']), bias_regularizer=l2(cfg['l2']))(layer)
        layer = Activation(cfg['activation'])(layer)
        layer = skip_manager.connect_skip(layer)
    if (cfg['stack_1']>0):
        layer = Conv2D(cfg['filters_3'], (cfg['k_3'], cfg['k_3']), strides=(cfg['s_1'], cfg['s_1']), padding='same', kernel_regularizer=l2(cfg['l2']), bias_regularizer=l2(cfg['l2']))(layer)
        layer = Activation(cfg['activation'])(layer)
        layer = skip_manager.connect_skip(layer)
        layer = Dropout(cfg['dropout_2'])(layer)

    #stack 2
    for i in range(cfg['stack_2']):
        layer = Conv2D(cfg['filters_4'], (cfg['k_4'], cfg['k_4']), padding='same', kernel_regularizer=l2(cfg['l2']), bias_regularizer=l2(cfg['l2']))(layer)
        layer = Activation(cfg['activation'])(layer)
        layer = skip_manager.connect_skip(layer)
    if (cfg['stack_2']>0):
        layer = Conv2D(cfg['filters_5'], (cfg['k_5'], cfg['k_5']), strides=(cfg['s_2'], cfg['s_2']), padding='same', kernel_regularizer=l2(cfg['l2']), bias_regularizer=l2(cfg['l2']))(layer)
        layer = Activation(cfg['activation'])(layer)
        layer = skip_manager.connect_skip(layer)
        layer = Dropout(cfg['dropout_3'])(layer)

    #stack 3
    for i in range(cfg['stack_3']):
        layer = Conv2D(cfg['filters_6'], (cfg['k_6'], cfg['k_6']), padding='same', kernel_regularizer=l2(cfg['l2']), bias_regularizer=l2(cfg['l2']))(layer)
        layer = Activation(cfg['activation'])(layer)
        layer = skip_manager.connect_skip(layer)
    if (cfg['stack_3']>0):
        layer = Conv2D(cfg['filters_7'], (cfg['k_7'], cfg['k_7']), strides=(cfg['s_3'], cfg['s_3']), padding='same', kernel_regularizer=l2(cfg['l2']), bias_regularizer=l2(cfg['l2']))(layer)
        layer = Activation(cfg['activation'])(layer)
        layer = skip_manager.connect_skip(layer)
        layer = Dropout(cfg['dropout_4'])(layer)

    #stack 4
    for i in range(cfg['stack_4']):
        layer = Conv2D(cfg['filters_8'], (cfg['k_8'], cfg['k_8']), padding='same', kernel_regularizer=l2(cfg['l2']), bias_regularizer=l2(cfg['l2']))(layer)
        layer = Activation(cfg['activation'])(layer)
        layer = skip_manager.connect_skip(layer)
    if (cfg['stack_4']>0):
        layer = Conv2D(cfg['filters_9'], (cfg['k_9'], cfg['k_9']), strides=(cfg['s_4'], cfg['s_4']), padding='same', kernel_regularizer=l2(cfg['l2']), bias_regularizer=l2(cfg['l2']))(layer)
        layer = Activation(cfg['activation'])(layer)
        layer = skip_manager.connect_skip(layer)
        layer = Dropout(cfg['dropout_5'])(layer)

    #global averaging
    if (cfg['global_pooling']):
        layer = GlobalAveragePooling2D()(layer)
    else:
        layer = Flatten()(layer)
    
    
    
    #head
    layer = Dense(cfg['dense_size'], kernel_regularizer=l2(cfg['l2']), bias_regularizer=l2(cfg['l2']))(layer)#TODO add more dense layers, or add option in cfg vector
    layer = Activation(cfg['activ_dense'])(layer)
    layer = Dense(num_classes, kernel_regularizer=l2(cfg['l2']), bias_regularizer=l2(cfg['l2']))(layer)#TODO add more dense layers, or add option in cfg vector
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

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255.
    x_test /= 255.

    if not data_augmentation:
        print('Not using data augmentation.')
        verbose = True #TODO remove this
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
        verbose = True #TODO remove this
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



#system arguments (configuration)
if len(sys.argv) > 2 and sys.argv[1] == '--cfg':
    cfg = eval(sys.argv[2])
    if len(sys.argv) > 3:
        gpu = sys.argv[3]
        
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu)
    print(CNN_conf(cfg))
    K.clear_session()

#CHRIS testcode
def test_skippy():
    from mipego.mipego import Solution #TODO remove this, only for testing
    from mipego.SearchSpace import ContinuousSpace, NominalSpace, OrdinalSpace
    from keras.utils import plot_model
    #define the search space.
    #objective = obj_func('./all-cnn_bi.py')
    activation_fun = ["softmax"]
    activation_fun_conv = ["elu","relu","tanh","sigmoid","selu"]

    filters = OrdinalSpace([10, 600], 'filters') * 10
    kernel_size = OrdinalSpace([1, 6], 'k') * 10
    strides = OrdinalSpace([1, 5], 's') * 5
    stack_sizes = OrdinalSpace([0, 12], 'stack') * 5
    #TODO_CHRIS these changes are just for cigar test function
    #filters = OrdinalSpace([0, 5], 'filters') * 7
    #kernel_size = OrdinalSpace([0, 5], 'k') * 7
    #strides = OrdinalSpace([0, 5], 's') * 3
    #stack_sizes = OrdinalSpace([0, 5], 'stack') * 3
    #TODO_CHRIS these changes are just for cigar test function

    activation = NominalSpace(activation_fun_conv, "activation")  # activation function
    activation_dense = NominalSpace(activation_fun, "activ_dense") # activation function for dense layer
    step = NominalSpace([True, False], "step")  # step
    global_pooling = NominalSpace([True, False], "global_pooling")  # global_pooling
    
    #skippy parameters
    skints = OrdinalSpace([0, 2**30], 'skint') * 3
    skst = OrdinalSpace([2, 10], 'skst') * 3
    dense_size = OrdinalSpace([1, 2000], 'dense_size')
    #skippy parameters

    drop_out = ContinuousSpace([1e-5, .9], 'dropout') * 6        # drop_out rate
    lr_rate = ContinuousSpace([1e-4, 1.0e-0], 'lr')        # learning rate
    l2_regularizer = ContinuousSpace([1e-5, 1e-2], 'l2')# l2_regularizer
    #TODO_CHRIS these changes are just for cigar test function
    #drop_out = ContinuousSpace([0.0, .9], 'dropout') * 4        # drop_out rate
    #lr_rate = ContinuousSpace([0.0, 1.0e-0], 'lr')        # learning rate
    #l2_regularizer = ContinuousSpace([0.0, 1e-2], 'l2')# l2_regularizer
    #TODO_CHRIS these changes are just for cigar test function

    search_space =  stack_sizes * strides * filters *  kernel_size * activation * activation_dense * drop_out * lr_rate * l2_regularizer * step * global_pooling * skints * skst * dense_size
    
    n_init_sample = 1
    samples = search_space.sampling(n_init_sample)
    print(samples)
    var_names = search_space.var_name.tolist()
    print(var_names)
    
    #a sample
    #samples = [[1, 1, 1, 1, 2, 3, 10, 10, 5, 10, 10, 10, 10, 3, 4, 2, 1, 3, 1, 3, 'relu', 'softmax', 0.7105013348601977, 0.24225495530708516, 0.5278997344637044, 0.7264822991098491, 0.0072338759099408985, 0.00010867041652507452, False, True]]

    #test parameters
    #original parameters
    stack_0 = 1
    stack_1 = 0
    stack_2 = 0
    stack_3 = 0
    stack_4 = 0
    s_0=2
    s_1=1
    s_2=1
    s_3=1
    s_4=1
    filters_0=20
    filters_1=15
    filters_2=128
    filters_3=128
    filters_4=256
    filters_5=256
    filters_6=512
    filters_7=128
    filters_8=256
    filters_9=256
    k_0=3
    k_1=3
    k_2=3
    k_3=3
    k_4=3
    k_5=3
    k_6=3
    k_7=3
    k_8=3
    k_9=3
    activation='relu'
    activ_dense='softmax'
    dropout_0=0.0
    dropout_1=0.0
    dropout_2=0.0
    dropout_3=0.0
    dropout_4=0.0
    dropout_5=0.0
    lr=0.01
    l2=0.01
    step=False
    global_pooling=True

    #skippy parameters
    om_en_om = 1
    for w in range((stack_0+stack_1+stack_2+stack_3+stack_4)//2):#TODO testcode: remove
        om_en_om = om_en_om << 2
        om_en_om += 1
    om_en_om = om_en_om << 3
    skint_0 = 0#3826103921638#2**30-1
    skint_1 = 0#19283461627361826#2**30-1
    skint_2 = 0#473829102637452916#2**30-1
    skst_0 = 2
    skst_1 = 3
    skst_2 = 5
    dense_size = 128
    #skippy parameters

    #assembling parameters
    samples = [[stack_0, stack_1, stack_2, stack_3, stack_4, s_0, s_1, s_2, s_3, s_4, filters_0, filters_1, filters_2, filters_3, filters_4, filters_5, filters_6, filters_7, filters_8, filters_9, k_0, k_1, k_2, k_3, k_4, k_5, k_6, k_7, k_8, k_9, activation, activ_dense, dropout_0, dropout_1, dropout_2, dropout_3, dropout_4, dropout_5, lr, l2, step, global_pooling, skint_0, skint_1, skint_2, skst_0, skst_1, skst_2, dense_size]]
    
    #var_names
    #['stack_0', 'stack_1', 'stack_2', 's_0', 's_1', 's_2', 'filters_0', 'filters_1', 'filters_2', 'filters_3', 'filters_4', 'filters_5', 'filters_6', 'k_0', 'k_1', 'k_2', 'k_3', 'k_4', 'k_5', 'k_6', 'activation', 'activ_dense', 'dropout_0', 'dropout_1', 'dropout_2', 'dropout_3', 'lr', 'l2', 'step', 'global_pooling']

    
    X = [Solution(s, index=k, var_name=var_names) for k, s in enumerate(samples)]
    print(X)
    #cfg = [Solution(x, index=len(self.data) + i, var_name=self.var_names) for i, x in enumerate(X)]
    test = False
    if test:
        model = CNN_conf(X[0].to_dict(),test=test)
        plot_model(model, to_file='model_skippy_test.png',show_shapes=True,show_layer_names=True)
        model.summary()
    else:
        timer, loss = CNN_conf(X[0].to_dict(),test=test)
        print('timer, loss:')
        print(timer, loss)
#CHRIS uncomment following to test the code
test_skippy()
