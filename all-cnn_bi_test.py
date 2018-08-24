from __future__ import print_function

import numpy as np
np.random.seed(43)
import tensorflow as tf
tf.set_random_seed(43)

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
import deap.benchmarks
import numbers


def CNN_conf(cfg):
    vector = []
    for x in cfg:
        if isinstance(cfg[x], numbers.Number) and cfg[x] is not None:
            vector.append(float(cfg[x]))
    print('vector in testfunction:')
    print(vector)
    #tuple = deap.benchmarks.schaffer_mo(vector)
    ans = deap.benchmarks.zdt1(vector)
    tuple = (ans[0],ans[1])
    return tuple



#system arguments (configuration)
if len(sys.argv) > 2 and sys.argv[1] == '--cfg':
    cfg = eval(sys.argv[2])
    if len(sys.argv) > 3:
        gpu = sys.argv[3]
        
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu)
    print(CNN_conf(cfg))
    K.clear_session()
