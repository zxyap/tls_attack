import os
import fnmatch
import json
import numpy as np
import copy as cp
import math
import logging
import argparse
from datetime import datetime
import config_cross_ref as config_cross_ref
import utils_datagen as utilsDatagen

from tensorflow.python import debug as tf_debug
import tensorflow as tf
import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.backend import set_session

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

parser = argparse.ArgumentParser()
parser.add_argument('-g', '--gpu', help='Flag for using GPU in model training', action='store_true')
parser.add_argument('-b', '--debug', help='run in debug mode', action='store_true')
args = parser.parse_args()
##############################################
# LOGGING
##############################################
now = datetime.now()
current_time = now.strftime("%Y-%m-%d_%H-%M-%S")

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# create a file handler
handler = logging.FileHandler('debug_logs/evaluation_{}.log'.format(current_time))
handler.setLevel(logging.INFO)

# create a logging format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# add the file handler to the logger
logger.addHandler(handler)

#####################################################
# HELPER FUNCTIONS
#####################################################
def getNoised(base_traffic, noise_traffic, percentage, seq_len):
    noised_array = []
    original_array = []
    for normal, attack in zip(base_traffic, noise_traffic):
        normal = normal[:seq_len]
        print("normal has length of {}".format(str(len(normal))))
        appending_len = math.floor(len(normal)/100 * percentage)
        if appending_len < len(attack):
            base = cp.deepcopy(normal)
            original = cp.deepcopy(normal)
            original_array.append(original)
            noise = cp.deepcopy(attack[:appending_len])
            base.extend(noise)
            noised_array.append(base)
        else:
            print('The attack length required is shorter than base traffic, no noised traffic generated for this base traffic')
    return original_array, noised_array


##set session##
tf_config = None
if args.gpu:
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    tf_config.log_device_placement = False
sess = tf.Session(config=tf_config)
if args.debug:
    print('Entering debug mode..')
    #sess = keras.backend.set_session(tf_debug.LocalCLIDebugWrapperSession(tf.Session()))
    sess = keras.backend.set_session(tf_debug.TensorBoardDebugWrapperSession(tf.Session(), "DESKTOP-DBVD5I7:6064"))
else:
    set_session(sess)

#log_dir = "tf_logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
#tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

##load model##

many2one_model = None

if config_cross_ref.many2one_modeldir and os.path.exists(config_cross_ref.many2one_modeldir):
    #many2one_model = keras.models.load_model(config_cross_ref.many2one_modeldir)
    many2one_model = load_model(config_cross_ref.many2one_modeldir)
    print("loading many2one_model..")

## load single feature file ##
feature_filename_template = 'features_tls_*.csv'
rootdir =  config_cross_ref.mixed_dos_featuredir
feature_filepath = os.path.join(rootdir, fnmatch.filter(os.listdir(rootdir), feature_filename_template)[0])

##load labels ##
labels_filename_template = "labels_type_*.csv"
rootdir =  config_cross_ref.mixed_dos_labels
labels_filepath = os.path.join(rootdir, fnmatch.filter(os.listdir(rootdir), labels_filename_template)[0])

# Load the mmap data and byte offset for each dataset
feature_mmap_byteoffsets = None
print('Loading features into memory..')
feature_mmap_byteoffset = utilsDatagen.get_mmapdata_and_byteoffset(feature_filepath)

#Load labels mmap data object and byte offset for corresponding datasets
labels_mmap_byteoffset = None
print('Loading labels into memory..')
labels_mmap_byteoffset = utilsDatagen.get_mmapdata_and_byteoffset(labels_filepath)

# Load min-max features from file
if not os.path.exists(config_cross_ref.minmax_dir):
    print('Error: Min-max feature file does not exist')
    exit()
with open(config_cross_ref.minmax_dir) as f:
    min_max_feature_list = json.load(f)
    min_max_feature = (np.array(min_max_feature_list[0]),np.array(min_max_feature_list[1]))

# Initialize the normalization function
norm_fn = utilsDatagen.normalize(3, min_max_feature)
denorm_fn = utilsDatagen.denormalize(3, min_max_feature)


# Initialize data generator for prediction

mmap,byteoffset = feature_mmap_byteoffset
data_generator = utilsDatagen.BatchGenerator_DSW(mmap, byteoffset, list(range(len(byteoffset))),
                                                     config_cross_ref.BATCH_SIZE,config_cross_ref.SEQUENCE_LEN,
                                                     norm_fn, return_batch_info=True)

mmap, byteoffset = labels_mmap_byteoffset
label_generator = utilsDatagen.BatchLabelGenerator(mmap, byteoffset, list(range(len(byteoffset))),
                                                         config_cross_ref.BATCH_SIZE, config_cross_ref.SEQUENCE_LEN, None, None, False)

label2id = {'normal':0, 'breach':1, 'poodle':2, 'rc4':3, 'dos':4}
id2label = {v:k for k,v in label2id.items()}
metrics = ['mean_acc'] 
for batch_data, label_data in zip(data_generator, label_generator):
    normal_list, attack_list = utilsDatagen.separate_traffic(batch_data, label_data, metrics, norm_fn, 1000, True, False)
    original_list, noised_list = getNoised(normal_list,attack_list, 5, 50)
    simple_list = noised_list[0]
    np_data = np.array(simple_list)
    np_data = np_data.reshape(1, np_data.shape[0], np_data.shape[1])
    window_input = utilsDatagen.preprocess_data(np_data, 1000, norm_fn, False)
    #result = many2one_model.predict(window_input, callbacks=[tensorboard_callback])
    result = many2one_model.predict(window_input)
    print('result from the model is ' + str(result))
    #base_acc_list, noise_acc_list = utilsDatagen.getNoiseMetrics(original_list, noised_list, many2one_model, norm_fn, id2label, 1000, logger, False) 