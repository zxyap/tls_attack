import os
import json
import fnmatch
import argparse
import logging
from datetime import datetime
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, LSTM, Dense, Lambda
from tensorflow.keras.models import Model
import numpy as np
import config_cross_ref as config_cross_ref
#import utils_datagen as utilsDatagen

# Fking hack to use the BatchGenerator from rnn-model-many2one
# TODO: Combine rnn-model and rnn-model-many2one module since they share alot of functions
import sys
sys.path.append(os.path.join('..','rnn-model-many2one'))
#import utils as utilsMany2one

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # Suppress Tensorflow debugging information for INFO level
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.backend import set_session

parser = argparse.ArgumentParser()
# parser.add_argument('-s', '--savedir', help='Input the directory path to save the prediction results', required=True)
parser.add_argument('-g', '--gpu', help='Flag for using GPU in model training', action='store_true')
args = parser.parse_args()


#####################################################
# Custom activation function
#####################################################
def custom_activation(x):
    return K.tanh(x) * 1.5


#####################################################
# PRE-CONFIGURATION
#####################################################

# Setting of CPU/GPU configuration for TF
if args.gpu:
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    tf_config.log_device_placement = False
sess = tf.Session(config=tf_config)
set_session(sess)  # set this TensorFlow session as the default session for Keras

number_of_dimensions = 128
number_of_examples = 12

input_ = Input(shape=(3, 1))
lstm, hidden, cell = LSTM(units=number_of_dimensions, return_state=True)(input_)
dense = Dense(10, activation='softmax')(lstm)
model = Model(inputs=input_, outputs=dense)

data = np.array([0.1, 0.2, 0.3]).reshape((1,3,1))
# make and show prediction
print(model.predict(data))
"""
with K.get_session() as sess:
    x = np.zeros((number_of_examples, 3, 1))
    cell_state = sess.run(cell, feed_dict={input_: x})
    print(cell_state)
    """
