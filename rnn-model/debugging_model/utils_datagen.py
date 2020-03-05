import json
import math
import mmap
import logging
import argparse
import numpy as np
import copy as cp
from random import shuffle
from random import randint
from functools import partial
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences

import utils_metric as utilsMetric

import os
import sys
sys.path.append(os.path.join('..', 'rnn-model-many2one'))
#import utils_many2one as utilsMany2one

def find_lines(data):
    for i, char in enumerate(data):
        if char == b'\n':
            yield i

def get_mmapdata_and_byteoffset(feature_file):
    ########################################################################
    # Biggest saviour: shuffling a large file w/o loading in memory
    # >>> https://stackoverflow.com/questions/24492331/shuffle-a-large-list-of-items-without-loading-in-memory
    ########################################################################

    # Creating a list of byte offset for each line
    with open(feature_file, 'r') as f:
        mmap_data = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        start = 0
        byte_offset = []
        for end in find_lines(mmap_data):
            byte_offset.append((start, end))
            start = end + 1
    return mmap_data, byte_offset

def get_min_max(mmap_data, byte_offset):
    min_feature = None
    max_feature = None
    for start,end in byte_offset:
        dataline = mmap_data[start:end+1].decode('ascii').strip().rstrip(',')
        dataline = np.array(json.loads('['+dataline+']'))
        if min_feature is not None:
            dataline = np.vstack((min_feature, dataline))
        if max_feature is not None:
            dataline = np.vstack((max_feature, dataline))
        min_feature = np.min(dataline, axis=0)
        max_feature = np.max(dataline, axis=0)

    return (min_feature, max_feature)

def split_train_test(dataset_size, split_ratio, seed):
    # Shuffling the indices to give a random train test split
    indices = np.random.RandomState(seed=seed).permutation(dataset_size)
    split_idx = math.ceil((1-split_ratio)*dataset_size)
    train_idx, test_idx = indices[:split_idx], indices[split_idx:]
    # Avoid an empty list in test set
    if len(test_idx) == 0:
        test_idx = train_idx[-1:]
        train_idx = train_idx[:-1]
    return train_idx, test_idx

def normalize(option, min_max_feature=None):
    def l2_norm(batch_data):
        l2_norm = np.linalg.norm(batch_data, axis=2, keepdims=True)
        batch_data = np.divide(batch_data, l2_norm, out=np.zeros_like(batch_data), where=l2_norm!=0.0)
        return batch_data

    def min_max_norm(batch_data):
        min_feature, max_feature = min_max_feature
        # Dimension 20~62 of ciphersuite are frequency values and should not be normalized like other features
        min_feature[20:63] = 0
        max_feature[20:63] = 1
        num = batch_data-min_feature
        den = max_feature-min_feature
        batch_data = np.divide(num, den, out=np.zeros_like(num), where=den!=0.0)
        batch_data[batch_data<0] = 0  # if < min, set to 0
        batch_data[batch_data>1] = 1  # if > max, set to 1
        assert (batch_data <= 1).all() and (batch_data >= 0).all()
        return batch_data

    def reciprocal_norm(batch_data):
        batch_data[batch_data < 0] = 0  # Set -ve values to 0 because -ve values are mapped weirdly for this func
        batch_data = batch_data/(1+batch_data)
        return batch_data

    if option == 1:
        return l2_norm
    elif option == 2:
        return min_max_norm
    elif option == 3:
        return reciprocal_norm
    else:
        print('Error: Option is not valid')
        return

def denormalize(option, min_max_feature=None):
    def min_max_denorm(batch_norm_data):
        min_feature, max_feature = min_max_feature
        batch_data = (batch_norm_data * (max_feature - min_feature)) + min_feature
        return batch_data

    def reciprocal_denorm(batch_norm_data):
        batch_data = batch_norm_data/(1-batch_norm_data)
        return batch_data

    if option == 2:
        return min_max_denorm
    elif option == 3:
        return reciprocal_denorm
    else:
        print('Error: Option is not valid')
        return

def get_feature_vector(selected_idx, mmap_data, byte_offset, sequence_len, norm_fn):
    selected_byte_offset = [byte_offset[i] for i in selected_idx]
    selected_data = []
    for start,end in selected_byte_offset:
        dataline = mmap_data[start:end+1].decode('ascii').strip().rstrip(',')
        selected_data.append(json.loads('['+dataline+']'))
    selected_seq_len = [len(data) for data in selected_data]
    selected_inputs,selected_targets = preprocess_data(selected_data, pad_len=sequence_len, norm_fn=norm_fn)
    return (selected_inputs, selected_targets, selected_seq_len)

def preprocess_data(batch_data, pad_len, norm_fn, to_slice):
    #print("batch_data size is : \n")
    #print(len(batch_data))   
    #print(batch_data)
    # Step 1: Pad sequences
    batch_data = pad_sequences(batch_data, maxlen=pad_len, dtype='float32', padding='post', truncating='post', value=0.0)

    #print("batch_data shape after padding is : \n")
    #print(batch_data.shape)
    #print(batch_data.shape[0], batch_data.shape[1], batch_data.shape[2])   
    
    # Step 2: Scale features with a normalization function
    batch_data = norm_fn(batch_data)
    # Step 3: Append zero to start of the sequence
    #packet_zero = np.zeros((batch_data.shape[0], 1, batch_data.shape[2]))
    #batch_data = np.concatenate((packet_zero, batch_data), axis=1)
    # Step 4: Split the data into inputs and targets
    #print("batch_data shape after extra 0 is : \n")
    #print(batch_data.shape[0], batch_data.shape[1], batch_data.shape[2])
    if to_slice :
        # Step 3: Append zero to start of the sequence
        packet_zero = np.zeros((batch_data.shape[0], 1, batch_data.shape[2]))
        batch_data = np.concatenate((packet_zero, batch_data), axis=1)
        # Step 4: Split the data into inputs and targets
        batch_inputs = batch_data[:,:-1,:]  #:-1 everything until the last item in the list
        batch_targets = batch_data[:,1:,:]
        #print("batch_input shape is : \n")
        #print(batch_inputs.shape[0], batch_inputs.shape[1], batch_inputs.shape[2])
     
        return batch_inputs, batch_targets
    else:
        return batch_data


def slidingWindow(a, window_size, stepsize):
    #input must be 2D array
    return np.hstack( a[i:1+i-window_size or None:stepsize] for i in range(0,window_size) )

def preprocess_data_sw(batch_data, window_size, step_size, pad_len, norm_fn, to_pad=True):
    batch_data = np.array(batch_data) 
    feature_num = 147 
    total = []
    #loop through all pcap files to obtain (num_windows, window_size/number of packets in window, num_features) representation of the pcap file
    for a in batch_data:
        #print(a.shape)
        #print("\n")
        #returns an array of windows of the particular pcap file
        a_new = slidingWindow(a, window_size=window_size, stepsize=step_size)
        #print(a_new.shape)
        #transforms (a.size/(window_size*feature_num) , window_size*feature_num) into (num_windows, window_size/number of packets in window, num_features)
        window_num = (int)(a_new.size/(window_size*feature_num))
        a_new = a_new.reshape(window_num,window_size,feature_num)
        #print("the new shape of a_new is \n")
        #print(a_new.shape)
        #print("\n")
        if to_pad == True : ##do not pad for appending window/dynamic SW
                a_new = pad_sequences(a_new, maxlen=pad_len, dtype='float32', padding='post', truncating='post', value=0.0) #pad along the window_size axis.
                # Step 2: Scale features with a normalization function
                a_new = norm_fn(a_new)
                #print("a_new shape after padding  \n")
                #print(a_new.shape)
                #print("\n")
                packet_zero = np.zeros((a_new.shape[0], 1, a_new.shape[2]))
                a_new = np.concatenate((packet_zero, a_new), axis=1) #concentenate along the window_size axis
                #print("after concat and padding \n")
                #print(a_new.shape)
                #print("\n")

        a_new.tolist()
        total.append(a_new)

    total_np = np.array(total)
    return total_np

def preprocess_pure(batch_data, norm_fn):
    # Step 1: Scale features with a normalization function
    batch_data = norm_fn(batch_data)
    # Step 2: Append zero to start of the sequence
    packet_zero = np.zeros((batch_data.shape[0], 1, batch_data.shape[2]))
    batch_data = np.concatenate((packet_zero, batch_data), axis=1)
    return batch_data


def preprocess_labels(batch_labels, window_size, step_size):
    batch_labels = np.array(batch_labels) 
    #print("shape of labels before sliding window \n")
    #print(batch_labels.shape)
    total = []
    for a in batch_labels:
        a = np.array(a)
        a = a.reshape(a.shape[0], 1) ##dont have to do extra processing as [[1], [1], [1]] -> [[1,1,1]], after slidingWindow, should be (num_window, window_size)
        label = slidingWindow(a, window_size=window_size, stepsize=step_size)
        #print("shape of label's after sliding window is \n")
        #print(label.shape)
        label.tolist()
        total.append(label)

    return np.array(total)

class BatchGenerator(Sequence):
    def __init__(self, mmap_data, byte_offset, selected_idx, batch_size, sequence_len, norm_fn, return_batch_info=False):
        self.mmap_data = mmap_data
        self.byte_offset = byte_offset
        self.selected_idx = selected_idx
        self.batch_size = batch_size
        self.sequence_len = sequence_len
        self.norm_fn = norm_fn
        self.return_batch_info = return_batch_info

    def __len__(self):
        return int(np.ceil(len(self.selected_idx)/float(self.batch_size)))

    def __getitem__(self, idx):
        batch_idx = self.selected_idx[idx*self.batch_size:(idx+1)*self.batch_size]
        batch_byte_offset = [self.byte_offset[i] for i in batch_idx]


        print("batch_idx is : \n")
        print(batch_idx)
        print("batch_byte_offset is : \n")
        print(batch_byte_offset)
        
        batch_data = []
        for start,end in batch_byte_offset:
            dataline = self.mmap_data[start:end+1].decode('ascii').strip().rstrip(',')
            batch_data.append(json.loads('['+dataline+']'))
        batch_inputs, batch_targets = preprocess_data(batch_data, pad_len=self.sequence_len, norm_fn=self.norm_fn, to_slice=True)

        if not self.return_batch_info:
            return (batch_inputs, batch_targets)

        batch_info = {}
        batch_seq_len = [len(data) for data in batch_data]
        batch_info['seq_len'] = np.array(batch_seq_len)  # Standardize output into numpy array
        batch_info['batch_idx'] = batch_idx  # batch_idx is already a numpy array

        return (batch_inputs, batch_targets, batch_info)

    def on_epoch_end(self):
        shuffle(self.selected_idx)

class BatchGenerator_small(Sequence):
    def __init__(self, mmap_data, byte_offset, selected_idx, batch_size, sequence_len, norm_fn, return_batch_info=False):
        self.mmap_data = mmap_data
        self.byte_offset = byte_offset
        self.selected_idx = selected_idx
        self.batch_size = batch_size
        self.sequence_len = 100
        self.norm_fn = norm_fn
        self.return_batch_info = return_batch_info

    def __len__(self):
        return int(np.ceil(len(self.selected_idx)/float(self.batch_size)))

    def __getitem__(self, idx):
        batch_idx = self.selected_idx[idx*self.batch_size:(idx+1)*self.batch_size]
        batch_byte_offset = [self.byte_offset[i] for i in batch_idx]


        print("batch_idx is : \n")
        print(batch_idx)
        print("batch_byte_offset is : \n")
        print(batch_byte_offset)
        
        batch_data = []
        for start,end in batch_byte_offset:
            dataline = self.mmap_data[start:end+1].decode('ascii').strip().rstrip(',')
            batch_data.append(json.loads('['+dataline+']'))
        batch_inputs, batch_targets = preprocess_data_small(batch_data, pad_len=self.sequence_len, norm_fn=self.norm_fn)

        if not self.return_batch_info:
            return (batch_inputs, batch_targets)

        batch_info = {}
        batch_seq_len = [len(data) for data in batch_data]
        batch_info['seq_len'] = np.array(batch_seq_len)  # Standardize output into numpy array
        batch_info['batch_idx'] = batch_idx  # batch_idx is already a numpy array

        return (batch_inputs, batch_targets, batch_info)

    def on_epoch_end(self):
        shuffle(self.selected_idx)

class BatchGenerator_DSW(Sequence):
    def __init__(self, mmap_data, byte_offset, selected_idx, batch_size, sequence_len, norm_fn, return_batch_info=False):
        self.mmap_data = mmap_data
        self.byte_offset = byte_offset
        self.selected_idx = selected_idx
        self.batch_size = batch_size
        self.sequence_len = sequence_len
        self.norm_fn = norm_fn
        self.return_batch_info = return_batch_info

    def __len__(self):
        return int(np.ceil(len(self.selected_idx)/float(self.batch_size)))

    def __getitem__(self, idx):
        batch_idx = self.selected_idx[idx*self.batch_size:(idx+1)*self.batch_size]
        batch_byte_offset = [self.byte_offset[i] for i in batch_idx]


        print("batch_idx is : \n")
        print(batch_idx)
        print("batch_byte_offset is : \n")
        print(batch_byte_offset)
        
        batch_data = []
        for start,end in batch_byte_offset:
            dataline = self.mmap_data[start:end+1].decode('ascii').strip().rstrip(',')
            batch_data.append(json.loads('['+dataline+']'))
        #batch_inputs, batch_targets = preprocess_data(batch_data, pad_len=self.sequence_len, norm_fn=self.norm_fn)

        if not self.return_batch_info:
            return batch_data

        batch_info = {}
        batch_seq_len = [len(data) for data in batch_data]
        batch_info['seq_len'] = np.array(batch_seq_len)  # Standardize output into numpy array
        batch_info['batch_idx'] = batch_idx  # batch_idx is already a numpy array

        return (batch_data, batch_info)

class BatchGenerator_pure(Sequence):
    def __init__(self, mmap_data, byte_offset, selected_idx, batch_size, sequence_len, norm_fn, return_batch_info=False):
        self.mmap_data = mmap_data
        self.byte_offset = byte_offset
        self.selected_idx = selected_idx
        self.batch_size = batch_size
        self.sequence_len = sequence_len
        self.norm_fn = norm_fn
        self.return_batch_info = return_batch_info

    def __len__(self):
        return int(np.ceil(len(self.selected_idx)/float(self.batch_size)))

    def __getitem__(self, idx):
        batch_idx = self.selected_idx[idx*self.batch_size:(idx+1)*self.batch_size]
        batch_byte_offset = [self.byte_offset[i] for i in batch_idx]


        print("batch_idx is : \n")
        print(batch_idx)
        print("batch_byte_offset is : \n")
        print(batch_byte_offset)
        
        batch_data = []
        for start,end in batch_byte_offset:
            dataline = self.mmap_data[start:end+1].decode('ascii').strip().rstrip(',')
            batch_data.append(json.loads('['+dataline+']'))

        if not self.return_batch_info:
            return batch_data

        batch_info = {}
        batch_seq_len = [len(data) for data in batch_data]
        batch_info['seq_len'] = np.array(batch_seq_len)  # Standardize output into numpy array
        batch_info['batch_idx'] = batch_idx  # batch_idx is already a numpy array

        return batch_data, batch_info


class BatchGenerator_ordered(Sequence):
    def __init__(self, mmap_data, byte_offset, selected_idx, batch_size, sequence_len, window_size, step_size, norm_fn, to_pad=True, return_batch_info=False):
        self.mmap_data = mmap_data
        self.byte_offset = byte_offset
        self.selected_idx = selected_idx
        self.batch_size = batch_size
        self.sequence_len = sequence_len
        self.window_size = window_size
        self.step_size = step_size
        self.norm_fn = norm_fn
        self.to_pad = to_pad
        self.return_batch_info = return_batch_info

    def __len__(self):
        return int(np.ceil(len(self.selected_idx)/float(self.batch_size)))

    def __getitem__(self, idx):
        batch_idx = self.selected_idx[idx*self.batch_size:(idx+1)*self.batch_size]
        batch_byte_offset = [self.byte_offset[i] for i in batch_idx]


        #print("batch_idx is : \n")
        #print(batch_idx)
        #print("batch_byte_offset is : \n")
        #print(batch_byte_offset)
        
        batch_data = []
        for start,end in batch_byte_offset:
            dataline = self.mmap_data[start:end+1].decode('ascii').strip().rstrip(',')
            batch_data.append(json.loads('['+dataline+']'))
        batch_inputs = preprocess_data_sw(batch_data, self.window_size, self.step_size, pad_len=self.sequence_len, norm_fn=self.norm_fn, to_pad=self.to_pad)
        #print("shape of batch inputs is \n")
        #print(batch_inputs.shape)

        if not self.return_batch_info:
            return batch_inputs

        batch_info = {}
        batch_seq_len = [len(data) for data in batch_data]
        batch_info['seq_len'] = np.array(batch_seq_len)  # Standardize output into numpy array
        batch_info['batch_idx'] = batch_idx  # batch_idx is already a numpy array

        return (batch_inputs, batch_info)



class BatchLabelGenerator(Sequence):
    def __init__(self, mmap_data, byte_offset, selected_label_idx, batch_size, sequence_len, window_size, step_size, isSlide):
        self.mmap_data = mmap_data
        self.byte_offset = byte_offset
        self.selected_idx = selected_label_idx
        self.batch_size = batch_size    
        self.sequence_len = sequence_len    
        self.window_size = window_size
        self.step_size = step_size
        self.isSlide = isSlide

    def __len__(self):
        return int(np.ceil(len(self.selected_idx)/float(self.batch_size)))

    def __getitem__(self, idx):
        batch_idx = self.selected_idx[idx*self.batch_size:(idx+1)*self.batch_size]
        batch_byte_offset = [self.byte_offset[i] for i in batch_idx]

        #print("batch_idx is : \n")
        #print(batch_idx)
        #print("batch_byte_offset is : \n")
        #print(batch_byte_offset)
        
        batch_labels = []
        for start,end in batch_byte_offset:
            dataline = self.mmap_data[start:end+1].decode('ascii').strip().rstrip(',')
            dataline = json.loads(dataline)
            #print("data typeis \n")
            #print(dataline)
            #dataline = list(dataline)
            #print(dataline)
            batch_labels.append(dataline)
        #print("shape of the full batch is \n")
        #print(batch_labels)
        #print('\n')
        if self.isSlide :
            batch_labels_windowed = preprocess_labels(batch_labels, self.window_size, self.step_size)
            #print("shape of labels window is \n")
            #print(batch_labels_windowed.shape)
            #print("the full batch is \n")
            #print(batch_labels)
            return batch_labels_windowed
        else:
            return batch_labels




PKT_LEN_THRESHOLD = 100 # <<< CHANGE THIS VALUE
                        # for computation of mean over big packets. If -1, computation over all packets

def compute_metrics_for_batch(model, batch_data, metrics, denorm_fn):
    # Metric supported: 'seq_len', 'idx', 'acc', 'mean_acc'
    #                   'squared_error', 'mean_squared_error', 'true', 'predict', <an int value for the dim number>

    batch_inputs, batch_true, batch_info = batch_data
    output = {}
    for metric in metrics:
        batch_seq_len = batch_info['seq_len']
        if metric == 'idx':
            batch_idx = batch_info['batch_idx']
            output[metric] = batch_idx

        elif metric == 'seq_len':
            output[metric] = batch_seq_len

        else:
            batch_predict = model.predict_on_batch(batch_inputs)
            if metric == 'acc' or metric == 'mean_acc':
                padded_batch_acc = utilsMetric.calculate_acc_of_traffic(batch_predict, batch_true)
                masked_batch_acc = np.ma.array(padded_batch_acc)
                # Mask based on true seq len for every row
                for i in range(len(batch_seq_len)):
                    masked_batch_acc[i, batch_seq_len[i]:] = np.ma.masked
                if metric == 'acc':
                    output[metric] = masked_batch_acc
                elif metric == 'mean_acc':
                    if PKT_LEN_THRESHOLD > 0 and denorm_fn:
                        denorm_batch_true = denorm_fn(batch_true) #denorm the 'y' values
                        mask = generate_mask_from_pkt_len(denorm_batch_true) #generate a array of '0's and '1's to determine which pckts to mask based on the len of the packet.
                        masked2_batch_acc = np.ma.array(masked_batch_acc) #
                        masked2_batch_acc.mask = mask #decide which to mask.
                        batch_mean_acc_over_big_pkts = np.mean(masked2_batch_acc, axis=-1)
                        output[metric] = batch_mean_acc_over_big_pkts 

                    elif PKT_LEN_THRESHOLD == -1:
                        batch_mean_acc = np.mean(masked_batch_acc, axis=-1)
                        output[metric] = batch_mean_acc

            elif metric == 'squared_error' or metric == 'mean_squared_error':
                padded_batch_squared_error = utilsMetric.calculate_squared_error_of_traffic(batch_predict, batch_true)
                masked_batch_squared_error = np.ma.array(padded_batch_squared_error)
                # Mask based on true seq len for every row
                for i in range(len(batch_seq_len)):
                    masked_batch_squared_error[i, batch_seq_len[i]:, :] = np.ma.masked
                if metric == 'squared_error':
                    output[metric] = masked_batch_squared_error
                elif metric == 'mean_squared_error':
                    batch_mean_squared_error = np.mean(masked_batch_squared_error, axis=1)
                    output[metric] = batch_mean_squared_error

            elif type(metric) == int:  # dim number
                output[metric] = (batch_true[:, :, metric:metric + 1], batch_predict[:, :, metric:metric + 1])

            elif metric == 'true':
                output[metric] = batch_true

            elif metric == 'predict':
                output[metric] = batch_predict
    return output

def compute_accuracy(model, batch_data, metrics, true_seq_len, denorm_fn, packet_acc, appending_len, logger):
    # Metric supported: 'seq_len', 'idx', 'acc', 'mean_acc'
    #                   'squared_error', 'mean_squared_error', 'true', 'predict', <an int value for the dim number>

    batch_inputs, batch_true = batch_data
    output = {}
    for metric in metrics:
        if metric == 'idx':
            batch_idx = batch_info['batch_idx']
            output[metric] = batch_idx

        elif metric == 'seq_len':
            output[metric] = batch_seq_len

        else:
            batch_predict = model.predict_on_batch(batch_inputs)
            #print("batch_predict shape is : \n")
            #print(batch_predict.shape[0], batch_predict.shape[1], batch_predict.shape[2])
            if metric == 'acc' or metric == 'mean_acc':
                padded_batch_acc = utilsMetric.calculate_acc_of_traffic(batch_predict, batch_true)
                #print("padded_batch_acc is ")
                #print(padded_batch_acc.shape)
                masked_batch_acc = np.ma.array(padded_batch_acc)
                #print("shape of the masked_batch_acc")
                #print(masked_batch_acc.shape)
                #print("\n")
                #print("batch_seq_len is \n")
                #print(batch_seq_len[0])
                #print("\n")
                # Mask based on true seq len for every row
                #masked_batch_acc[:, true_seq_len:] = np.ma.masked
                if packet_acc:
                    logger.info("Only taking into consideration the packets of appending len {}".format(str(appending_len)))
                    #print(masked_batch_acc[0][49], masked_batch_acc[0][50], masked_batch_acc[0][51], masked_batch_acc[0][52])
                    masked_batch_acc[:, true_seq_len:] = np.ma.masked
                    masked_batch_acc[:, :true_seq_len-appending_len] = np.ma.masked
                    #print(masked_batch_acc[0][49], masked_batch_acc[0][50], masked_batch_acc[0][51], masked_batch_acc[0][52])
                    #print(np.mean(masked_batch_acc, axis=-1))
                    #masked_batch_acc = cp.deepcopy(padded_batch_acc[:, true_seq_len:appending_len])
                else :
                    logger.info("Taking true sequence length")
                    masked_batch_acc[:, true_seq_len:] = np.ma.masked
                    #print(np.mean(masked_batch_acc, axis=-1))
                #print("value of the masked_batch_acc \n")
                #print(masked_batch_acc)
                #print("\n")
                if metric == 'acc':
                    output[metric] = masked_batch_acc
                elif metric == 'mean_acc':
                    if PKT_LEN_THRESHOLD > 0 and denorm_fn:
                        #print("shape of batch_true is \n")
                        #print(batch_true.shape)
                        #print("\n")
                        denorm_batch_true = denorm_fn(batch_true) #denorm the 'y' values
                        mask = generate_mask_from_pkt_len(denorm_batch_true) #generate a array of '0's and '1's to determine which pckts to mask based on the len of the packet.
                        #print("value of mask \n")
                        #print(mask)
                        #print("\n")
                        masked2_batch_acc = np.ma.array(masked_batch_acc) #
                        masked2_batch_acc.mask = mask #decide which to mask.
                        print("After packet len mask")
                        print(masked_batch_acc[0][50], masked_batch_acc[0][51])
                        batch_mean_acc_over_big_pkts = np.mean(masked2_batch_acc, axis=-1)
                        output[metric] = batch_mean_acc_over_big_pkts 

                    elif PKT_LEN_THRESHOLD == -1:
                        batch_mean_acc = np.mean(masked_batch_acc, axis=-1)
                        output[metric] = batch_mean_acc

            elif metric == 'squared_error' or metric == 'mean_squared_error':
                padded_batch_squared_error = utilsMetric.calculate_squared_error_of_traffic(batch_predict, batch_true)
                masked_batch_squared_error = np.ma.array(padded_batch_squared_error)
                # Mask based on true seq len for every row
                masked_batch_squared_error[:, true_seq_len:, :] = np.ma.masked
                if metric == 'squared_error':
                    output[metric] = masked_batch_squared_error
                elif metric == 'mean_squared_error':
                    batch_mean_squared_error = np.mean(masked_batch_squared_error, axis=1)
                    output[metric] = batch_mean_squared_error

            elif type(metric) == int:  # dim number
                output[metric] = (batch_true[:, :, metric:metric + 1], batch_predict[:, :, metric:metric + 1])

            elif metric == 'true':
                output[metric] = batch_true

            elif metric == 'predict':
                output[metric] = batch_predict
    return output


def compute_metrics_for_pure(models, batch_data, metrics, denorm_fn, label):
    # Metric supported: 'seq_len', 'idx', 'acc', 'mean_acc'
    #                   'squared_error', 'mean_squared_error', 'true', 'predict', <an int value for the dim number>
    model_counter = {"dos" : 0, "normal" : 0 , "breach": 0, "poodle" : 0, "rc4" : 0 }
    batch_inputs, batch_true, batch_info = batch_data
    output = {}
    correct_predictions = 0;
    for metric in metrics:
        batch_seq_len = batch_info['seq_len']
        if metric == 'idx':
            batch_idx = batch_info['batch_idx']
            output[metric] = batch_idx

        elif metric == 'seq_len':
            output[metric] = batch_seq_len

        else:
            for batch_input, batch_input_len, batch_input_true in zip(batch_inputs, batch_seq_len, batch_true) :
                labels = []
                model_results = []
                batch_input = batch_input.reshape(1,1000,147)
                batch_input_true = batch_input_true.reshape(1, 1000, 147)
                for model_label in models.keys():
                    #print("using model " + str(model_label))
                    labels.append(model_label)
                    model = models[model_label]
                    batch_predict = model.predict_on_batch(batch_input)
                    #print("batch_predict shape is : \n")
                    #print(batch_predict.shape[0], batch_predict.shape[1], batch_predict.shape[2])
                    if metric == 'acc' or metric == 'mean_acc':
                        padded_batch_acc = utilsMetric.calculate_acc_of_traffic(batch_predict, batch_input_true)
                        masked_batch_acc = np.ma.array(padded_batch_acc)
                        #print("shape of the masked_batch_acc")
                        #print(masked_batch_acc.shape)
                        #print("\n")
                        #print("batch_seq_len is \n")
                        #print(batch_seq_len[0])
                        #print("\n")
                        # Mask based on true seq len for every row
                        masked_batch_acc[0, batch_input_len:] = np.ma.masked
                        #print("value of the masked_batch_acc \n")
                        #print(masked_batch_acc.shape)
                        #print("\n")
                        if metric == 'acc':
                            output[metric] = masked_batch_acc
                        elif metric == 'mean_acc':
                            if PKT_LEN_THRESHOLD > 0 and denorm_fn:
                                #print("shape of batch_true is \n")
                                #print(batch_true.shape)
                                #print("\n")
                                denorm_batch_true = denorm_fn(batch_true) #denorm the 'y' values
                                mask = generate_mask_from_pkt_len(denorm_batch_true) #generate a array of '0's and '1's to determine which pckts to mask?
                                #print("value of mask \n")
                                #print(mask)
                                #print("\n")
                                masked2_batch_acc = np.ma.array(masked_batch_acc) #
                                masked2_batch_acc.mask = mask #decide which to mask.
                         #       print("masked2_batch_acc is \n")
                         #       print(masked2_batch_acc.shape)
                         #       print("\n")
                                batch_mean_acc_over_big_pkts = np.mean(masked2_batch_acc, axis=-1)
                                model_results.append(batch_mean_acc_over_big_pkts[0])
                                #print("value of batch_mean_acc_over_big_pkts \n")
                                #print(batch_mean_acc_over_big_pkts)
                                #print("\n")
                                output[metric] = batch_mean_acc_over_big_pkts #batch_mean_acc is only 1 value

                            elif PKT_LEN_THRESHOLD == -1:
                                batch_mean_acc = np.mean(masked_batch_acc, axis=-1)
                                output[metric] = batch_mean_acc

                    elif metric == 'squared_error' or metric == 'mean_squared_error':
                        padded_batch_squared_error = utilsMetric.calculate_squared_error_of_traffic(batch_predict, batch_true)
                        masked_batch_squared_error = np.ma.array(padded_batch_squared_error)
                        # Mask based on true seq len for every row
                        for i in range(len(batch_seq_len)):
                            masked_batch_squared_error[i, batch_seq_len[i]:, :] = np.ma.masked
                        if metric == 'squared_error':
                            output[metric] = masked_batch_squared_error
                        elif metric == 'mean_squared_error':
                            batch_mean_squared_error = np.mean(masked_batch_squared_error, axis=1)
                            output[metric] = batch_mean_squared_error

                    elif type(metric) == int:  # dim number
                        output[metric] = (batch_true[:, :, metric:metric + 1], batch_predict[:, :, metric:metric + 1])

                    elif metric == 'true':
                        output[metric] = batch_true

                    elif metric == 'predict':
                        output[metric] = batch_predict

                temp_array = np.array(model_results)
                max_indice = np.argmax(temp_array)
                prediction_label = labels[max_indice] #gets the prediction label for each traffic flow
                #print("max model:")
                #print(prediction_label)
                model_counter[prediction_label] = model_counter[prediction_label] + 1

    return model_counter, len(batch_inputs)

def compute_metrics_for_pure2(model, batch_data, metrics, denorm_fn, id2label, label):
    # Metric supported: 'seq_len', 'idx', 'acc', 'mean_acc'
    #                   'squared_error', 'mean_squared_error', 'true', 'predict', <an int value for the dim number>
    model_counter = {"dos" : 0, "normal" : 0 , "breach": 0, "poodle" : 0, "rc4" : 0 }
    batch_inputs, batch_true, batch_info = batch_data
    output = {}
    correct_predictions = 0;
    for metric in metrics:
        batch_seq_len = batch_info['seq_len']
        if metric == 'idx':
            batch_idx = batch_info['batch_idx']
            output[metric] = batch_idx

        elif metric == 'seq_len':
            output[metric] = batch_seq_len

        else:
            for batch_input, batch_input_len, batch_input_true in zip(batch_inputs, batch_seq_len, batch_true) :
                labels = []
                model_results = []
                batch_input = batch_input.reshape(1,1000,147)
                batch_input_true = batch_input_true.reshape(1, 1000, 147)
                
                traffic_predictions = model.predict_on_batch(batch_input)
                max_indice = np.argmax(traffic_predictions[0])
                prediction_label = id2label[max_indice]
                #print("max model:")
                #print(prediction_label)
                model_counter[prediction_label] = model_counter[prediction_label] + 1

    return model_counter, len(batch_inputs)

def compute_metrics_for_pure3(model, batch_data, metrics, seq_len, denorm_fn, id2label, label, is4D):
    # Metric supported: 'seq_len', 'idx', 'acc', 'mean_acc'
    #                   'squared_error', 'mean_squared_error', 'true', 'predict', <an int value for the dim number>
    model_counter = {"dos" : 0, "normal" : 0 , "breach": 0, "poodle" : 0, "rc4" : 0 }
    batch_inputs, batch_info = batch_data
    output = {}
    correct_predictions = 0;
    for metric in metrics:
        batch_seq_len = batch_info['seq_len']
        if metric == 'idx':
            batch_idx = batch_info['batch_idx']
            output[metric] = batch_idx

        elif metric == 'seq_len':
            output[metric] = batch_seq_len

        else:
            for batch_input, batch_input_len in zip(batch_inputs, batch_seq_len) :
                labels = []
                model_results = []
                print(batch_input.shape)
                if is4D:
                    batch_input = batch_input.reshape(1,seq_len,147,1)
                else :
                    batch_input = batch_input.reshape(1,seq_len,147)
                traffic_predictions = model.predict_on_batch(batch_input)
                max_indice = np.argmax(traffic_predictions[0])
                prediction_label = id2label[max_indice]
                #print("max model:")
                #print(prediction_label)
                model_counter[prediction_label] = model_counter[prediction_label] + 1

    return model_counter, len(batch_inputs)


def getMaxPrediction(model_results):
    max_model = None
    max = 0
    for traffic_type in model_results.keys() :
        if model_results[traffic_type] > max : 
            max = model_results[traffic_type]
            max_model = traffic_type
    return max_model

def get_window_label(window, type):
    sum = np.sum(window, axis=0)
    #print("sum is :\n")
    #print(sum)
    #print("\n")
    majority = sum/((window.shape[0]) * type)
    if majority >= 0.5:
        return "dos"
    else:
        return "normal"
def get_mixed_metrics(traffic):
    max_length = 0
    max_length_sec = 0
    max_traffic = None
    max_traffic_sec = None
    for traffic_type in traffic.keys():
        traffic_len = len(traffic[traffic_type])
        if traffic_len > max_length :
            max_length = traffic_len
            max_traffic = traffic_type
        elif traffic_len > max_length_sec :
            max_length_sec = traffic_len
            max_traffic_sec = traffic_type
        else :
            continue
    return (max_traffic, max_traffic_sec)

def mixed_prediction(models, batch_data,  window_size, metrics, norm_fn, denorm_fn, threshold):
    # Metric supported: 'seq_len', 'idx', 'acc', 'mean_acc'
    #                   'squared_error', 'mean_squared_error', 'true', 'predict', <an int value for the dim number>
    #batch_labels expected : (batch_size, num_windows, window_size)
    #batch_inputs expected: (batch_size, num_windows, window_size, num_features)
    batch_inputs, batch_info = batch_data
    #print("batch_inputs shape is \n")
    #print(batch_inputs.shape)
    #print("\n")
    for metric in metrics:
        batch_seq_len = batch_info['seq_len']
        if metric == 'idx':
            batch_idx = batch_info['batch_idx']
            output[metric] = batch_idx

        elif metric == 'seq_len':
            output[metric] = batch_seq_len
        else:
            #loop through all the batches
            results = []
            for batch in batch_inputs :
                #traffic = {'normal': [], 'rc4' : [], 'breach' : [], 'dos' : [], 'poodle' : []}
                traffic = {'normal': [], 'breach' : [], 'dos' : [], 'poodle' : []}

                for window in batch :
                    print('length of normal : ' + str(len(traffic['normal'])))
                    #print('length of rc4 : ' + str(len(traffic['rc4'])))
                    print('length of breach : ' + str(len(traffic['breach'])))
                    print('length of dos : ' + str(len(traffic['dos'])))
                    print('length of poodle : ' + str(len(traffic['poodle'])))
                    model_results = {'normal': 0, 'rc4' : 0, 'breach' : 0, 'dos' : 0, 'poodle': 0}
                    for traffic_type in traffic.keys():
                        #print('length of '+ traffic_type + ' traffic')
                        #print(len(traffic[traffic_type]))
                        #print('\n')
                        #print('shape of window')
                        #print(window.shape)
                        #print('\n')
                        curr_seq = traffic[traffic_type][:]
                        #print('length of ' + traffic_type + ' before appending')
                        #print(len(curr_seq))
                        #print('\n')
                        curr_seq.extend(window)
                        #print('length of ' + traffic_type + ' after appending')
                        #print(len(curr_seq))
                        #print('\n')
                        np_data = np.array(curr_seq)
                        #print('shape of np_data before reshaping is')
                        #print(np_data.shape)
                        #print('\n')
                        np_data = np_data.reshape(1, np_data.shape[0], np_data.shape[1])
                        #print('shape of np_data after reshaping is')
                        #print(np_data.shape)
                        #print('\n')
                        np_data = pad_sequences(np_data, maxlen=1000, dtype='float32', padding='post', truncating='post', value=0.0)
                        np_data = norm_fn(np_data)
                        packet_zero = np.zeros((np_data.shape[0], 1, np_data.shape[2]))
                        batch_data = np.concatenate((packet_zero, np_data), axis=1)
                        window_input = batch_data[:, :-1, :]
                        window_true = batch_data[:, 1:, :]
                        window_predict = models[traffic_type].predict_on_batch(window_input)
                        if metric == 'acc' or metric == 'mean_acc':
                            padded_batch_acc = utilsMetric.calculate_acc_of_traffic(window_predict, window_true) #obtain cosine similarity of each packet => (1,1000)
                            masked_batch_acc = np.ma.array(padded_batch_acc) #just make it a into a masked array without masking specifications first

                            #print("shape of the masked_batch_acc")
                            #print(masked_batch_acc.shape)
                            #print("\n")
                            # Mask all the results for > window size. 
                            
                            masked_batch_acc[0,  len(curr_seq):] = np.ma.masked #0 is needed because its a 2D array with only 1 1D array because our input shape is 1,1000, 147
                            #print("value of the masked_batch_acc \n")
                            #print(masked_batch_acc)
                            #print("\n")
                            if metric == 'acc':
                                output[metric] = masked_batch_acc
                            elif metric == 'mean_acc':
                                if PKT_LEN_THRESHOLD > 0 and denorm_fn:
                                    #print("shape of window_true \n")
                                    #print(window_true.shape)
                                    #print("\n")
                                    denorm_batch_true = denorm_fn(window_true)
                                    mask = generate_mask_from_pkt_len(denorm_batch_true)
                                    #print("shape of mask \n")
                                    #print(mask.shape)
                                    #print("\n")
                                    masked2_batch_acc = np.ma.array(masked_batch_acc)
                                    masked2_batch_acc.mask = mask #mask again based on length of packet, if len of packet < length threshold, mask it.
                                    batch_mean_acc_over_big_pkts = np.mean(masked2_batch_acc, axis=-1) #get the mean cosine similarity of all relevant packets. batch_mean_acc_over_big_pkts is a 1D array with only one value.
                                    #print('big packets mean')
                                    #print(batch_mean_acc_over_big_pkts)
                                    model_results[traffic_type] = batch_mean_acc_over_big_pkts[0] #appends each model's results to perform np.argmax...
                                elif PKT_LEN_THRESHOLD == -1:
                                    batch_mean_acc = np.mean(masked_batch_acc, axis=-1)
                                    model_results[traffic_type] = batch_mean_acc[0]
                    print(model_results)
                    max_traffic_type = getMaxPrediction(model_results)
                    print('the max traffic type is ')
                    print(max_traffic_type)
                    print('\n')
                    if model_results[max_traffic_type] > threshold : #if the prediction is less than threshold, we unappend it from the max_traffic_type
                        traffic[max_traffic_type].extend(window)
                mixed_traffic = get_mixed_metrics(traffic)
                results.append(mixed_traffic)
    return results

def get_mixed_labels(models, batch_data,  window_size, metrics, norm_fn, denorm_fn, threshold):
    # Metric supported: 'seq_len', 'idx', 'acc', 'mean_acc'
    #                   'squared_error', 'mean_squared_error', 'true', 'predict', <an int value for the dim number>
    #batch_labels expected : (batch_size, num_windows, window_size)
    #batch_inputs expected: (batch_size, num_windows, window_size, num_features)
    batch_inputs, batch_info = batch_data
    #print("batch_inputs shape is \n")
    #print(batch_inputs.shape)
    #print("\n")
    for metric in metrics:
        batch_seq_len = batch_info['seq_len']
        if metric == 'idx':
            batch_idx = batch_info['batch_idx']
            output[metric] = batch_idx

        elif metric == 'seq_len':
            output[metric] = batch_seq_len
        else:
            #loop through all the batches
            results = []
            batch_traffic_labels = []
            for batch in batch_inputs :
                #traffic = {'normal': [], 'rc4' : [], 'breach' : [], 'dos' : [], 'poodle' : []}
                traffic = {'normal': [], 'breach' : [], 'dos' : [], 'poodle' : []}
                full_traffic_labels = []
                for window in batch :
                    window_packets_label = []
                    print('length of normal : ' + str(len(traffic['normal'])))
                    #print('length of rc4 : ' + str(len(traffic['rc4'])))
                    print('length of breach : ' + str(len(traffic['breach'])))
                    print('length of dos : ' + str(len(traffic['dos'])))
                    print('length of poodle : ' + str(len(traffic['poodle'])))
                    model_results = {'normal': 0, 'rc4' : 0, 'breach' : 0, 'dos' : 0, 'poodle': 0}
                    for traffic_type in traffic.keys():
                        #print('length of '+ traffic_type + ' traffic')
                        #print(len(traffic[traffic_type]))
                        #print('\n')
                        #print('shape of window')
                        #print(window.shape)
                        #print('\n')
                        curr_seq = traffic[traffic_type][:]
                        #print('length of ' + traffic_type + ' before appending')
                        #print(len(curr_seq))
                        #print('\n')
                        curr_seq.extend(window)
                        #print('length of ' + traffic_type + ' after appending')
                        #print(len(curr_seq))
                        #print('\n')
                        np_data = np.array(curr_seq)
                        #print('shape of np_data before reshaping is')
                        #print(np_data.shape)
                        #print('\n')
                        np_data = np_data.reshape(1, np_data.shape[0], np_data.shape[1])
                        #print('shape of np_data after reshaping is')
                        #print(np_data.shape)
                        #print('\n')
                        np_data = pad_sequences(np_data, maxlen=1000, dtype='float32', padding='post', truncating='post', value=0.0)
                        np_data = norm_fn(np_data)
                        packet_zero = np.zeros((np_data.shape[0], 1, np_data.shape[2]))
                        batch_data = np.concatenate((packet_zero, np_data), axis=1)
                        window_input = batch_data[:, :-1, :]
                        window_true = batch_data[:, 1:, :]
                        window_predict = models[traffic_type].predict_on_batch(window_input)
                        if metric == 'acc' or metric == 'mean_acc':
                            padded_batch_acc = utilsMetric.calculate_acc_of_traffic(window_predict, window_true) #obtain cosine similarity of each packet => (1,1000)
                            masked_batch_acc = np.ma.array(padded_batch_acc) #just make it a into a masked array without masking specifications first

                            #print("shape of the masked_batch_acc")
                            #print(masked_batch_acc.shape)
                            #print("\n")
                            # Mask all the results for > window size. 
                            
                            masked_batch_acc[0,  len(curr_seq):] = np.ma.masked #0 is needed because its a 2D array with only 1 1D array because our input shape is 1,1000, 147
                            #print("value of the masked_batch_acc \n")
                            #print(masked_batch_acc)
                            #print("\n")
                            if metric == 'acc':
                                output[metric] = masked_batch_acc
                            elif metric == 'mean_acc':
                                if PKT_LEN_THRESHOLD > 0 and denorm_fn:
                                    #print("shape of window_true \n")
                                    #print(window_true.shape)
                                    #print("\n")
                                    denorm_batch_true = denorm_fn(window_true)
                                    mask = generate_mask_from_pkt_len(denorm_batch_true)
                                    #print("shape of mask \n")
                                    #print(mask.shape)
                                    #print("\n")
                                    masked2_batch_acc = np.ma.array(masked_batch_acc)
                                    masked2_batch_acc.mask = mask #mask again based on length of packet, if len of packet < length threshold, mask it.
                                    batch_mean_acc_over_big_pkts = np.mean(masked2_batch_acc, axis=-1) #get the mean cosine similarity of all relevant packets. batch_mean_acc_over_big_pkts is a 1D array with only one value.
                                    #print('big packets mean')
                                    #print(batch_mean_acc_over_big_pkts)
                                    model_results[traffic_type] = batch_mean_acc_over_big_pkts[0] #appends each model's results to perform np.argmax...
                                elif PKT_LEN_THRESHOLD == -1:
                                    batch_mean_acc = np.mean(masked_batch_acc, axis=-1)
                                    model_results[traffic_type] = batch_mean_acc[0]
                    print(model_results)
                    max_traffic_type = getMaxPrediction(model_results)
                    print('the max traffic type is ')
                    print(max_traffic_type)
                    print('\n')
                    traffic[max_traffic_type].extend(window)
                    for i in range(0, window_size):
                        window_packets_label += [max_traffic_type] 
                    full_traffic_labels.append(window_packets_label)
                batch_traffic_labels.append(full_traffic_labels)
    return batch_traffic_labels

def dynamic_sw(models, batch_data,  step_size, window_size, metrics, norm_fn, denorm_fn, threshold, logger):
    # Metric supported: 'seq_len', 'idx', 'acc', 'mean_acc'
    #                   'squared_error', 'mean_squared_error', 'true', 'predict', <an int value for the dim number>
    #batch_labels expected : (batch_size, num_windows, window_size)
    #batch_inputs expected: (batch_size, num_windows, window_size, num_features)
    max_len = 1000
    batch_inputs, batch_info = batch_data
    curr_window_size = window_size
    curr_step_size = step_size
    for metric in metrics:
        batch_seq_len = batch_info['seq_len']
        if metric == 'idx':
            batch_idx = batch_info['batch_idx']
            output[metric] = batch_idx

        elif metric == 'seq_len':
            output[metric] = batch_seq_len
        else:
            #loop through all the batches
            batch_traffic_labels = []
            for batch in batch_inputs :
                #start and end indexes...
                start = 0
                end = 0
                prev_start = None
                prev_end = None
                ### declare all data structures ###
                logger.info("Length of the batch is {}".format(len(batch)))
                #windowed_traffic = {'normal': [], 'breach' : [], 'dos' : [], 'poodle' : []}
                windowed_traffic = {'normal': [], 'breach' : [], 'dos' : [], 'poodle' : [], 'rc4' : []}
                model_results = {'normal': 0, 'rc4' : 0, 'breach' : 0, 'dos' : 0, 'poodle': 0}
                packet_predictions = []
                while end < len(batch):
                    logger.info('start is {}'.format(str(start)))
                    logger.info('end is {}'.format(str(end)))
                    logger.info('len of packet prediction is {}'.format(len(packet_predictions)))
                    #1. append window packets to each appending window.
                    # search through the overlapping windows to find all potential conflicts.
                    # conflicts is key(traffic_type) and value(number of packets in the overlapping window)
                    # the conflicts dictionary tell us how many packets to remove from each windowed_traffic so that the full window can be appended without repeating packets
                    # seq_copy holds the sequence up until start index for each traffic_type => no packets in current window, including those that belong to itself, so that full current window can be appended to it without any repeating packets
                    # Then using the windowed_traffic(includes packet in current window for all conflicting types) and seq_copy(conflicting types), we can get the difference in pred value for S and S'
                    # window_primes holds the windows that contains the overlapping packets to append to max_traffic type without the conflicting packets to each conflicting traffic type. It should only contain the 4 other traffic types besides the current max traffic type
                    # window_primes also contain packets that are to the right of the overlapping region (fresh to the right of the window)
                    conflicts = {}
                    seq_copy = {'normal': [], 'breach' : [], 'dos' : [], 'poodle' : [], 'rc4' : []}
                    window_primes = {}
                    window_copy = cp.deepcopy(batch[start:start+curr_window_size]) #even if window_size > remaining batch data size, will take all that is remaining
                    if len(packet_predictions) != 0 :#this is not the first window
                        #TODO : need to account when window_size and step_size changes
                        #overlapping windows only include the overlapping packets
                        overlapping_window = packet_predictions[start:prev_end+1] #shallow copy so that we can make changes to it directly, packet_predictions is 2D to make referencing changes eg. [['dos'], ['normal']]
                        getConflicts(conflicts, overlapping_window)
                        logger.info('Conflicts dictionary state is : {}'.format(str(conflicts)))
                    else:
                        pass
                    for traffic_type in models.keys():
                        if len(packet_predictions) == 0:
                            logger.info("First window, no preappending to do")
                            #First window, no existing sequence for any traffic, append full window to all types and predict
                            full_seq = cp.deepcopy(window_copy[:])
                        else:
                            # Not the first window, remove all the previous duplicate packets that are present in this window.
                            # If the there are no packets that belong to this specific traffic type, then conflicts[traffic_type] = 0, no packets will be removed, the full window will be appended
                            seq_len = len(windowed_traffic[traffic_type])
                            logger.info("len of existing sequence {} is {}".format(traffic_type, seq_len))
                            logger.info('the number of conflict packets for {} is {} '.format(traffic_type, conflicts.get(traffic_type, 0)))
                            seq_copy[traffic_type] = cp.deepcopy(windowed_traffic[traffic_type][:seq_len - conflicts.get(traffic_type, 0)])
                            to_append = cp.deepcopy(window_copy[:])
                            full_seq = cp.deepcopy(seq_copy[traffic_type])
                            full_seq.extend(to_append[:])
                        true_seq_len = len(full_seq)
                        full_seq = np.array(full_seq)
                        model_results[traffic_type]= many2many_predict(models, full_seq, traffic_type, true_seq_len, norm_fn, denorm_fn, logger)
                    ###TODO : There could be None for traffic type as the prediction may not match for all types.
                    max_traffic_type = getMaxPrediction(model_results)
                    logger.info("models results are {}".format(str(model_results)))
                    max_traffic_pred = model_results[max_traffic_type]
                    logger.info("max result is {} with pred_value of {}".format(max_traffic_type, str(max_traffic_pred)))

                    ####Threshold checking####
                    if max_traffic_pred < lower_threshold :
                        logger.info("The prediction value of {} is less than lower_threshold, expecting that there is too much noise in the window, reducing window size...".format(max_traffic_type))
                        curr_window_size -= 5
                        change = start + curr_window_size - 1
                        if change < prev_end :
                            ### window end will not be less then prev_end windwow ###
                            end = prev_end
                        else : 
                            end = start + curr_window_size - 1
                        curr_step_size = math.ceil((end - start + 1) / 2)

                    elif max_traffic_pred < threshold :
                        #TODO readjustment of window size..

                        logger.info("The prediction value of {} is less than threshold, suspecting that there is not enough information, expanding window size...".format(max_traffic_type))
                        #only have to adjust the end index of the window
                        curr_window_size += 5
                        end = start + curr_window_size - 1
                        curr_step_size = math.ceil((end - start + 1) / 2)
                    else :
                        logger.info('prediction value of {} exceeds threshold of {}'.format(str(max_traffic_pred), str(threshold)))
                        # now we resolve the overlapping packets if the threshold is met.
                        if len(packet_predictions) == 0 :
                            #there are no overlapping packets as it is the first window.
                            for i in range(curr_window_size):
                                packet_predictions += [[max_traffic_type]]
                            logger.info("First window, no conflicting packets...")
                            #append the window to the max traffic type
                            windowed_traffic[max_traffic_type].extend(window_copy[:])
                        else :
                            curr_traffic_type = max_traffic_type
                            #construct the window primes for each conflicting traffic.
                            construct_windows(window_primes, conflicts, overlapping_window, curr_traffic_type, window_copy, logger)
                            #resolve fn will remove all the packets from the conflicting windowed_traffic if necessary, update the packet_predictions list and append any overlapping packets that belong to the curr max traffic type
                            resolve_m2m(models, norm_fn, denorm_fn, conflicts, window_copy, windowed_traffic, seq_copy, window_primes, max_traffic_type, max_traffic_pred, overlapping_window, logger)
                            #append the rest of the packet that are not in the overlapping portion
                            if curr_window_size > len(window_copy):
                                #reached the end of the batch data, the actual window is actually lesser than the determined window_size, every packet is in the overlapping region, nothing more to append
                                pass
                            else:
                                #append the rest of the packet that are not in the overlapping portion only if curr end index is > prev end index
                                if end > prev_end : 
                                    logger.info('prev end index {} is and current end index at {}'.format(str(prev_end), str(end)))
                                    appending_tail = cp.deepcopy(batch[prev_end+1:end+1]) # if end > len(batch),it will just extend nothing to the exising window
                                    logger.info('length of appending tail is {}'.format(len(appending_tail)))
                                    windowed_traffic[max_traffic_type].extend(appending_tail)
                                    ##add the new packets to the packet_predictions list
                                    for i in range(len(appending_tail)):
                                        packet_predictions += [[max_traffic_type]]
                                else:
                                    logger.info('prev end index {} is same at current end index at {}'.format(len(prev_end, end)))
                                    pass
                                #print('len of normal after appending ' + str(len(windowed_traffic['normal'])))
                        prev_start = start
                        prev_end = end
                        start = start + curr_step_size
                        end = start + curr_window_size - 1
                        logger.info('Sliding the window from prev start index {} to new start index {}'.format(str(prev_start), str(start)))             

                    #reset the results
                    model_results = {key: 0 for key in model_results}

                ##flatten the 2D pakcet predictions to 1D
                packet_predictions = np.array(packet_predictions)
                packet_predictions = packet_predictions.flatten()
                packet_predictions = packet_predictions.tolist()
                batch_traffic_labels.append(packet_predictions)
    

    return batch_traffic_labels



def dynamic_sw2(models, many2one, id2label, batch_data,  step_size, window_size, metrics, norm_fn, denorm_fn, threshold, logging):
    # Metric supported: 'seq_len', 'idx', 'acc', 'mean_acc'
    #                   'squared_error', 'mean_squared_error', 'true', 'predict', <an int value for the dim number>
    #batch_labels expected : (batch_size, num_windows, window_size)
    #batch_inputs expected: (batch_size, num_windows, window_size, num_features)
    batch_inputs, batch_info = batch_data
    curr_window_size = window_size
    curr_step_size = step_size
    #print("batch_inputs shape is \n")
    #print(batch_inputs.shape)
    #print("\n")
    for metric in metrics:
        batch_seq_len = batch_info['seq_len']
        if metric == 'idx':
            batch_idx = batch_info['batch_idx']
            output[metric] = batch_idx

        elif metric == 'seq_len':
            output[metric] = batch_seq_len
        else:
            #loop through all the batches
            batch_traffic_labels = []
            for batch in batch_inputs :
                start = 0
                #traffic = {'normal': [], 'rc4' : [], 'breach' : [], 'dos' : [], 'poodle' : []}
                windowed_traffic = {'normal': [], 'breach' : [], 'dos' : [], 'poodle' : []}
                appending_window = {'normal': [], 'breach' : [], 'dos' : [], 'poodle' : [], 'rc4' : []}
                model_results = {'normal': 0, 'rc4' : 0, 'breach' : 0, 'dos' : 0, 'poodle': 0}
                full_traffic_labels = []
                prev_max_traffic = None
                resizing_count = 0
                while len(batch) != 0:
                    logging.info("START OF LOGGING for window size of {} and overlap of {}".format(str(curr_window_size), str(curr_step_size)))
                    logging.info('shape of batch is:')
                    logging.info(str(np.array(batch).shape))
                    logging.info('length of normal : ' + str(len(windowed_traffic['normal'])))
                    #print('length of rc4 : ' + str(len(traffic['rc4'])))
                    logging.info('length of breach : ' + str(len(windowed_traffic['breach'])))
                    logging.info('length of dos : ' + str(len(windowed_traffic['dos'])))
                    logging.info('length of poodle : ' + str(len(windowed_traffic['poodle'])))
                    window_packets_label = []
                    window = batch[start : curr_window_size]
                    for traffic_type in windowed_traffic.keys():
                        partial_traffic = cp.deepcopy(windowed_traffic[traffic_type][:])
                        appending_len = curr_window_size - curr_step_size
                        logging.info("TRAFFIC TYPE is " + traffic_type)
                        if prev_max_traffic != traffic_type or appending_len > len(window):
                            appending_window[traffic_type] = cp.deepcopy(window[:])
                        else :
                            logging.info("window size is " + str(curr_window_size))
                            logging.info("step size is " + str(curr_step_size))
                            logging.info("overlapping window size is " + str(appending_len))
                            appending_window[traffic_type] = cp.deepcopy(window[curr_window_size - curr_step_size :])
                        logging.info("appending window of size {} to {} of length {} ".format(str(len(appending_window[traffic_type])), traffic_type, str(len(partial_traffic))))
                        appending_copy = cp.deepcopy(appending_window[traffic_type])
                        partial_traffic.extend(appending_copy)
                        np_data = np.array(partial_traffic)
                        logging.info('shape of {} window before preprocessing and padding is {}:'.format(traffic_type, str(np_data.shape)))
                        #logging.info(str(np_data.shape))
                        np_data = np_data.reshape(1, np_data.shape[0], np_data.shape[1])
                        #print('shape of np_data after reshaping is')
                        #print(np_data.shape)
                        #print('\n')
                        window_input, window_true = preprocess_data(np_data, pad_len=1000, norm_fn=norm_fn)
                        #np_data = pad_sequences(np_data, maxlen=1000, dtype='float32', padding='post', truncating='post', value=0.0)
                        #np_data = norm_fn(np_data)
                        #packet_zero = np.zeros((np_data.shape[0], 1, np_data.shape[2]))
                        #batch_data = np.concatenate((packet_zero, np_data), axis=1)
                        #window_input = batch_data[:, :-1, :]
                        #window_true = batch_data[:, 1:, :]
                        window_predict = models[traffic_type].predict(window_input)
                        if metric == 'acc' or metric == 'mean_acc':
                            padded_batch_acc = utilsMetric.calculate_acc_of_traffic(window_predict, window_true) #obtain cosine similarity of each packet => (1,1000)
                            masked_batch_acc = np.ma.array(padded_batch_acc) #just make it a into a masked array without masking specifications first

                            #print("shape of the masked_batch_acc")
                            #print(masked_batch_acc.shape)
                            #print("\n")
                            # Mask all the results for > window size. 
                            
                            masked_batch_acc[0,  len(partial_traffic):] = np.ma.masked #0 is needed because its a 2D array with only 1 1D array because our input shape is 1,1000, 147
                            masked_batch_acc[0, :len(partial_traffic)-len(appending_window[traffic_type])] = np.ma.masked
                            #print("value of the masked_batch_acc \n")
                            #print(masked_batch_acc)
                            #print("\n")
                            if metric == 'acc':
                                output[metric] = masked_batch_acc
                            elif metric == 'mean_acc':
                                if PKT_LEN_THRESHOLD > 0 and denorm_fn:
                                    #print("shape of window_true \n")
                                    #print(window_true.shape)
                                    #print("\n")
                                    denorm_batch_true = denorm_fn(window_true)
                                    mask = generate_mask_from_pkt_len(denorm_batch_true)
                                    #print("shape of mask \n")
                                    #print(mask.shape)
                                    #print("\n")
                                    masked2_batch_acc = np.ma.array(masked_batch_acc)
                                    masked2_batch_acc.mask = mask #mask again based on length of packet, if len of packet < length threshold, mask it.
                                    batch_mean_acc_over_big_pkts = np.mean(masked2_batch_acc, axis=-1) #get the mean cosine similarity of all relevant packets. batch_mean_acc_over_big_pkts is a 1D array with only one value.
                                    #print('big packets mean')
                                    #print(batch_mean_acc_over_big_pkts)
                                    model_results[traffic_type] = batch_mean_acc_over_big_pkts[0] #appends each model's results to perform np.argmax...
                                elif PKT_LEN_THRESHOLD == -1:
                                    batch_mean_acc = np.mean(masked_batch_acc, axis=-1)
                                    model_results[traffic_type] = batch_mean_acc[0]
                    logging.info("The threshold is " + str(threshold))
                    logging.info("Showing results for all model types : ")
                    logging.info(str(model_results))
                    max_traffic_type = getMaxPrediction(model_results)
                    logging.info('the max traffic type is ' + max_traffic_type)
                    
                    if model_results[max_traffic_type] > threshold:
                        windowed_traffic[max_traffic_type].extend(appending_window[max_traffic_type])
                        del batch[start:curr_step_size]
                        for i in range(0, len(appending_window[max_traffic_type])):
                            window_packets_label += [max_traffic_type]
                        full_traffic_labels.extend(window_packets_label)
                        ##### if the window_size or step_size changes based on threshold, change it back to default after threshold has been reached 
                        curr_window_size = window_size
                        curr_step_size = step_size
                        prev_max_traffic = max_traffic_type
                        resizing_count = 0
                    else :
                        resizing_count += 1
                        if resizing_count == 110:
                            """
                            logging.info('Using the many2one model due to inability to cross threshold.')
                            traffic_predictions = many2one.predict_on_batch(window_input)
                            max_indice = np.argmax(traffic_predictions[0])
                            max_traffic_type = id2label[max_indice]
                            if max_traffic_type == 'rc4' :
                                logging.info('rc4 was detected, skipping it....')
                            else :
                                windowed_traffic[max_traffic_type].extend(appending_window[max_traffic_type])
                            del batch[start:curr_step_size]
                            for i in range(0, len(appending_window[max_traffic_type])):
                                window_packets_label += [max_traffic_type]
                                logging.info("using the many2one model, {} was predicted and appending window of length {} to existing {} windowed traffic of length {}".format(max_traffic_type, max_traffic_type, str(len(appending_window[max_traffic_type])), str(len(windowed_traffic[max_traffic_type]))))
                            full_traffic_labels.extend(window_packets_label)
                            curr_window_size = window_size
                            curr_step_size = step_size
                            prev_max_traffic = max_traffic_type
                            resizing_count = 0
                            #logging.info("using the many2one model, {} was predicted and appending window of length {} to existing {} windowed traffic of length {}".format(max_traffic_type, max_traffic_type, str(len(appending_window[max_traffic_type])), str(len(windowed_traffic[max_traffic_type]))))
                            """
                            logging.info("The many2many model is unable to produce a max result that exceeds the threshold {} despite all possible window resizings, terminating the program....".format(str(threshold)))
                            exit()
                            
                        else:
                            curr_window_size = (curr_window_size + 10) % 1000 ##subjected to change
                            if curr_window_size == 0 :
                                curr_window_size = 10
                            logging.info("increasing window size to " + str(curr_window_size))
                            #curr_step_size = curr_window_size ###subjected to change
                    logging.info("END OF LOGGING for window size of {} and overlap of {}".format(str(curr_window_size), str(curr_step_size)))
                    """
                    windowed_traffic[max_traffic_type].extend(appending_window[max_traffic_type])
                    del batch[start:curr_step_size]
                    for i in range(curr_window_size - curr_step_size, curr_window_size):
                        window_packets_label += [max_traffic_type]
                    """
                    full_traffic_labels.extend(window_packets_label)
                batch_traffic_labels.append(full_traffic_labels)
    return batch_traffic_labels

def dynamic_sw3(models, many2one, id2label, batch_data,  step_size, window_size, metrics, norm_fn, denorm_fn, threshold, logging):
    # Metric supported: 'seq_len', 'idx', 'acc', 'mean_acc'
    #                   'squared_error', 'mean_squared_error', 'true', 'predict', <an int value for the dim number>
    #batch_labels expected : (batch_size, num_windows, window_size)
    #batch_inputs expected: (batch_size, num_windows, window_size, num_features)
    batch_inputs, batch_info = batch_data
    curr_window_size = window_size
    curr_step_size = step_size
    for metric in metrics:
        batch_seq_len = batch_info['seq_len']
        if metric == 'idx':
            batch_idx = batch_info['batch_idx']
            output[metric] = batch_idx

        elif metric == 'seq_len':
            output[metric] = batch_seq_len
        else:
            #loop through all the batches
            batch_traffic_labels = []
            for batch in batch_inputs :
                start = 0
                windowed_traffic = {'normal': [], 'breach' : [], 'dos' : [], 'poodle' : []}
                appending_window = {'normal': [], 'breach' : [], 'dos' : [], 'poodle' : [], 'rc4' : []}
                model_results = {'normal': 0, 'rc4' : 0, 'breach' : 0, 'dos' : 0, 'poodle': 0}
                full_traffic_labels = []
                prev_max_traffic = None
                resizing_count = 0
                prev_window = None
                while len(batch) != 0:
                    logging.info("START OF LOGGING for window size of {} and overlap of {}".format(str(curr_window_size), str(curr_step_size)))
                    logging.info('shape of batch is:')
                    logging.info(str(np.array(batch).shape))
                    logging.info('length of normal : ' + str(len(windowed_traffic['normal'])))
                    #print('length of rc4 : ' + str(len(traffic['rc4'])))
                    logging.info('length of breach : ' + str(len(windowed_traffic['breach'])))
                    logging.info('length of dos : ' + str(len(windowed_traffic['dos'])))
                    logging.info('length of poodle : ' + str(len(windowed_traffic['poodle'])))
                    window_packets_label = []
                    curr_window = cp.deepcopy(batch[start : curr_window_size])
                    for traffic_type in windowed_traffic.keys():
                        logging.info("TRAFFIC TYPE is " + traffic_type)
                        partial_traffic = cp.deepcopy(windowed_traffic[traffic_type][:])
                        appending_window[traffic_type] = cp.deepcopy(curr_window[:])
                        logging.info("appending window of size {} to {} of length {} ".format(str(len(appending_window[traffic_type])), traffic_type, str(len(partial_traffic))))
                        appending_copy = cp.deepcopy(appending_window[traffic_type])
                        partial_traffic.extend(appending_copy)
                        np_data = np.array(partial_traffic)
                        logging.info('shape of {} window before preprocessing and padding is {}:'.format(traffic_type, str(np_data.shape)))
                        #logging.info(str(np_data.shape))
                        np_data = np_data.reshape(1, np_data.shape[0], np_data.shape[1])
                        #print('shape of np_data after reshaping is')
                        #print(np_data.shape)
                        #print('\n')
                        window_input, window_true = preprocess_data(np_data, 1000, norm_fn, True)
                        window_predict = models[traffic_type].predict(window_input)
                        if metric == 'acc' or metric == 'mean_acc':
                            padded_batch_acc = utilsMetric.calculate_acc_of_traffic(window_predict, window_true) #obtain cosine similarity of each packet => (1,1000)
                            masked_batch_acc = np.ma.array(padded_batch_acc) #just make it a into a masked array without masking specifications first

                            #print("shape of the masked_batch_acc")
                            #print(masked_batch_acc.shape)
                            #print("\n")
                            # Mask all the results for > window size. 
                            
                            masked_batch_acc[0,  len(partial_traffic):] = np.ma.masked #0 is needed because its a 2D array with only 1 1D array because our input shape is 1,1000, 147
                            #masked_batch_acc[0, :len(partial_traffic)-len(appending_window[traffic_type])] = np.ma.masked
                            #print("value of the masked_batch_acc \n")
                            #print(masked_batch_acc)
                            #print("\n")
                            if metric == 'acc':
                                output[metric] = masked_batch_acc
                            elif metric == 'mean_acc':
                                if PKT_LEN_THRESHOLD > 0 and denorm_fn:
                                    #print("shape of window_true \n")
                                    #print(window_true.shape)
                                    #print("\n")
                                    denorm_batch_true = denorm_fn(window_true)
                                    mask = generate_mask_from_pkt_len(denorm_batch_true)
                                    #print("shape of mask \n")
                                    #print(mask.shape)
                                    #print("\n")
                                    masked2_batch_acc = np.ma.array(masked_batch_acc)
                                    masked2_batch_acc.mask = mask #mask again based on length of packet, if len of packet < length threshold, mask it.
                                    batch_mean_acc_over_big_pkts = np.mean(masked2_batch_acc, axis=-1) #get the mean cosine similarity of all relevant packets. batch_mean_acc_over_big_pkts is a 1D array with only one value.
                                    #print('big packets mean')
                                    #print(batch_mean_acc_over_big_pkts)
                                    model_results[traffic_type] = batch_mean_acc_over_big_pkts[0] #appends each model's results to perform np.argmax...
                                elif PKT_LEN_THRESHOLD == -1:
                                    batch_mean_acc = np.mean(masked_batch_acc, axis=-1)
                                    model_results[traffic_type] = batch_mean_acc[0]
                    logging.info("The threshold is " + str(threshold))
                    logging.info("Showing results for all model types : ")
                    logging.info(str(model_results))
                    max_traffic_type = getMaxPrediction(model_results)
                    logging.info('the max traffic type is ' + max_traffic_type)
                    curr_pred_value = model_results[max_traffic_type]
                    
                    if curr_pred_value > threshold:
                        overlapping_size = curr_window_size-curr_step_size
                        if max_traffic_type != prev_max_traffic and prev_max_traffic != None and curr_step_size != curr_window_size:
                            overlap_pred = get_overlap_pred(curr_window, curr_pred_value, windowed_traffic[max_traffic_type], max_traffic_type, prev_window_pred_value, windowed_traffic[prev_max_traffic], prev_max_traffic, models, overlapping_size, norm_fn, denorm_fn, logging)
                            if overlap_pred == max_traffic_type : ## overlapping packets belong to max_traffic
                                del windowed_traffic[prev_max_traffic][-overlapping_size:] #delete the last x packets from the prev_traffic seq
                                windowed_traffic[max_traffic_type].extend(appending_window[max_traffic_type])
                                for i in range(0, len(appending_window[max_traffic_type])): #append all packets in curr_traffic_type
                                     window_packets_label += [max_traffic_type]
                                full_traffic_labels.extend(window_packets_label)
                            else :
                                del appending_window[max_traffic_type][:overlapping_size] #overlapping packets belong to prev_traffic, delete first x packets from curr appending window
                                windowed_traffic[max_traffic_type].extend(appending_window[max_traffic_type])
                                for i in range(0, overlapping_size):
                                    window_packets_label += [prev_max_traffic] #append the packet label for prev_max_traffic
                                for i in range(0, len(appending_window[max_traffic_type])):
                                    window_packets_label += [max_traffic_type] #append the packet label for curr_traffic_type
                                full_traffic_labels.extend(window_packets_label)

                        elif prev_max_traffic != None and max_traffic_type == prev_max_traffic and curr_step_size != curr_window_size: ##
                            windowed_traffic[max_traffic_type].extend(appending_window[max_traffic_type][overlapping_size:])
                            for i in range(0, len(appending_window[max_traffic_type])): # if curr_step_size != curr_window_size, overlapping_size = 0
                                window_packets_label += [max_traffic_type]
                        else:### first window
                            windowed_traffic[max_traffic_type].extend(appending_window[max_traffic_type])
                            for i in range(0, len(appending_window[max_traffic_type])-overlapping_size): # if curr_step_size != curr_window_size, overlapping_size = 0
                                window_packets_label += [max_traffic_type]
                            full_traffic_labels.extend(window_packets_label)
                        del batch[start:curr_step_size]
                        ##### if the window_size or step_size changes based on threshold, change it back to default after threshold has been reached 
                        curr_window_size = window_size
                        curr_step_size = step_size
                        prev_max_traffic = max_traffic_type
                        prev_window = cp.deepcopy(curr_window)
                        prev_window_pred_value = curr_pred_value  
                        resizing_count = 0
                    else :
                        logging.info('prediction did not cross threshold...')
                        resizing_count += 1
                        if resizing_count == 110:
                            logging.info("The many2many model is unable to produce a max result that exceeds the threshold {} despite all possible window resizings, terminating the program....".format(str(threshold)))
                            exit()
                            
                        else:
                            curr_window_size = (curr_window_size + 10) % 1000 ##subjected to change
                            if curr_window_size == 0 :
                                curr_window_size = 10
                            logging.info("increasing window size to " + str(curr_window_size))
                            #curr_step_size = curr_window_size ###subjected to change
                    logging.info("END OF LOGGING for window size of {} and overlap of {}".format(str(curr_window_size), str(curr_step_size)))
                    full_traffic_labels.extend(window_packets_label)
                batch_traffic_labels.append(full_traffic_labels)
    return batch_traffic_labels

def many2many_predict(models, np_data, traffic_type, seq_len, norm_fn, denorm_fn, logger):
    np_data = np_data.reshape(1, np_data.shape[0], np_data.shape[1])
    window_input, window_true = preprocess_data(np_data, 1000, norm_fn, True)
    window_predict = models[traffic_type].predict(window_input)
    padded_batch_acc = utilsMetric.calculate_acc_of_traffic(window_predict, window_true) #obtain cosine similarity of each packet => (1,1000)
    masked_batch_acc = np.ma.array(padded_batch_acc) #just make it a into a masked array without masking specifications first
    masked_batch_acc[0,  seq_len:] = np.ma.masked #0 is needed because its a 2D array with only 1 1D array because our input shape is 1,1000, 147
    if PKT_LEN_THRESHOLD > 0 and denorm_fn:
        denorm_batch_true = denorm_fn(window_true)
        mask = generate_mask_from_pkt_len(denorm_batch_true)
        masked2_batch_acc = np.ma.array(masked_batch_acc)
        masked2_batch_acc.mask = mask #mask again based on length of packet, if len of packet < length threshold, mask it.
        batch_mean_acc_over_big_pkts = np.mean(masked2_batch_acc, axis=-1) #get the mean cosine similarity of all relevant packets. batch_mean_acc_over_big_pkts is a 1D array with only one value.
        return batch_mean_acc_over_big_pkts[0] #appends each model's results to perform np.argmax...
    elif PKT_LEN_THRESHOLD == -1:
        batch_mean_acc = np.mean(masked_batch_acc, axis=-1)
        return batch_mean_acc[0]

def resolve_m2m(models, norm_fn, denorm_fn, conflicts, window_copy, windowed_traffic, seq_copy, window_primes, max_traffic_type, max_traffic_pred, overlapping_window, logger):
    for traffic_type in window_primes.keys():
        logger.info('Resolving for traffic type {}'.format(traffic_type))
        #1.obtain the conflicting traffics metrics: S and S'
        # seq_copy of a certain traffic type may be empty, meaning that all the packets in the windowed_traffic must be in the overlapping region. 
        # If that is the case, feed the overlapping packets straight into the conflicting traffic type model and the max model to decide which it belongs to. Then alter the data structures accordingly. windowed_traffic ->
        full_seq = cp.deepcopy(windowed_traffic[traffic_type])
        np_full_seq = np.array(full_seq)
        true_seq_len = len(np_full_seq)
        full_seq_pred = many2many_predict(models, np_full_seq, traffic_type, true_seq_len, norm_fn, denorm_fn, logger)

        full_seq_prime = cp.deepcopy(seq_copy[traffic_type])
        np_full_seq_prime = np.array(full_seq_prime)
        true_seq_len = len(np_full_seq_prime)
        if true_seq_len == 0:
            full_seq = cp.deepcopy(windowed_traffic[traffic_type])
            np_full_seq = np.array(full_seq)
            simplifed_comparison(models, conflicts, windowed_traffic, seq_copy, overlapping_window, window_copy, traffic_type, max_traffic_type, np_full_seq, norm_fn, denorm_fn, logger)
            continue
            
        full_seq_prime_pred = many2many_predict(models, np_full_seq_prime, traffic_type, true_seq_len, norm_fn, denorm_fn, logger)
        
        #2. obtain the current max traffic metrics
        # If the max_curr_seq is empty -> means that end of seq has been reached and the every packet is in the overlapping region
        max_curr_seq = cp.deepcopy(seq_copy[max_traffic_type])
        logger.info('len of window prime for {} is {}'.format(traffic_type, str(len(window_primes[traffic_type]))))
        max_curr_seq.extend(window_primes[traffic_type])
        np_max_curr_seq = np.array(max_curr_seq)
        true_seq_len = len(np_max_curr_seq)
        max_curr_seq_pred = many2many_predict(models, np_max_curr_seq, max_traffic_type, true_seq_len, norm_fn, denorm_fn, logger)

        #3 compare the 4 metrics:
        #case 1 : full_seq > full_seq_prime and max_traffic_pred < traffic_pred, means that overlapping packets belong more to the previous conflicting traffic type than in the current traffic type
        #case 2 : full_seq < full_seq_prime and max_traffic_pred > traffic_pred, means that overlapping packetes belong more to the current traffic type than previous conflicting traffic type
        #case 3 : full_seq < full_seq_prime and max_traffic_pred < traffic_pred, take the minimum difference between the two, least drop in accuracy means the packets belong more to the specific traffic type
        #case 4 : full_seq > full_seq_prime and max_traffic_pred > traffic_pred, take the max difference between the two, max rise in accuracy means the packets belong more to the specific traffic type
        #case 5 : equal change in both or no change in both, take any one of the two traffic types.
        # if one traffic type stays stagnant and 
        #                                       1. the other traffic type rises, traffic belongs to it
        #                                       2. the other traffic drops, traffic belongs to the stagnant traffic.
        if full_seq_pred > full_seq_prime_pred and max_traffic_pred < max_curr_seq_pred:
            #belongs to conflicting traffic
            #no change in packet_predictions
            pass
        elif full_seq_pred < full_seq_prime_pred and max_traffic_pred > max_curr_seq_pred:
            #belongs to current max traffic
            #overwrite the packet predictions
            for packet_pred in overlapping_window:
                if packet_pred[0] == traffic_type:
                    packet_pred[0] = max_traffic_type
                else:
                    pass
            #delete the last x packets from the conflicting windowed traffic
            del windowed_traffic[traffic_type][-conflicts[traffic_type]:]
        elif full_seq_pred < full_seq_prime_pred and max_traffic_pred < max_curr_seq_pred:
            #get the minimum fall in value
            if full_seq_prime_pred - full_seq_pred < max_curr_seq_pred - max_traffic_pred :
                #fall is larger for current max traffic type, therefore packets belong to conflicting traffic_type
                # no need to overwrite packet predictions
                pass
            else:
                #fall is larger for conflicting traffic type, therefore packets belong to current max traffic type traffic_type
                #overwrite packet predictions
                for packet_pred in overlapping_window:
                    if packet_pred[0] == traffic_type:
                        packet_pred[0] = max_traffic_type
                    else:
                        pass
                #delete the last x packets from the conflicting windowed traffic
                del windowed_traffic[traffic_type][-conflicts[traffic_type]:]
        elif full_seq_pred > full_seq_prime_pred and max_traffic_pred > max_curr_seq_pred:
            #get the maximum rise in value
            if full_seq_pred - full_seq_prime_pred > max_traffic_pred - max_curr_seq_pred:
                #rise is larger for conflicting traffic type, therefore packets belong to conflicting traffic_type
                # no need to overwrite packet predictions
                pass
            else:
                #rise is larger for current max traffic type, therefore packets belong to current max traffic type traffic_type
                #overwrite packet predictions
                for packet_pred in overlapping_window:
                    if packet_pred[0] == traffic_type:
                        packet_pred[0] = max_traffic_type
                    else:
                        pass
                #delete the last x packets from the conflicting windowed traffic
                del windowed_traffic[traffic_type][-conflicts[traffic_type]:]
        else:
            #both have no change
            #generate a random number to assign it
            guess = randint(0,1)
            if guess == 1:
                #take it as current max traffic type
                for packet_pred in overlapping_window:
                    if packet_pred[0] == traffic_type:
                        packet_pred[0] = max_traffic_type
                    else:
                        pass
                #delete the last x packets from the conflicting windowed traffic
                del windowed_traffic[traffic_type][-conflicts[traffic_type]:]
            else:
                #take it as conflicting traffic type
                pass

        #4. from the overwritten packet_pred, form a list of overlapping actual packets to append to curr_traffic type 
        curr_traffic_app = []
        copy_window_copy = cp.deepcopy(window_copy[:])
        for i in range(len(overlapping_window)):
            if overlapping_window[i][0] == max_traffic_type :
                curr_traffic_app += [copy_window_copy[i]]
            else:
                pass
        # append to the max traffic type for overlapping portion
        # get the sequence for max traffic that has no overlapping packets.
        # append the overlapping packet that belongs to the max traffic type
        windowed_traffic[max_traffic_type] = cp.deepcopy(seq_copy[max_traffic_type])
        windowed_traffic[max_traffic_type].extend(curr_traffic_app)

def simplifed_comparison(models, conflicts, windowed_traffic, seq_copy, overlapping_window, window_copy, traffic_type, max_traffic_type, np_full_seq, norm_fn, denorm_fn, logger):
    #compare the max traffic type and conflicting traffic type
    #overwrite the packet predictions list
    # if == conflicting traffic, no change.
    # if == max traffic.. 
    seq_len = len(np_full_seq)

    ## for conflicting traffic
    np_data_copy = cp.deepcopy(np_full_seq)
    conflicting_pred = many2many_predict(models, np_data_copy, traffic_type, seq_len, norm_fn, denorm_fn, logger)

    ## for current traffic
    np_data_copy = cp.deepcopy(np_full_seq)
    curr_pred = many2many_predict(models, np_data_copy, max_traffic_type, seq_len, norm_fn, denorm_fn, logger)

    if conflicting_pred > curr_pred:
        logger.info("packets belong in the conflicting traffic")
        pass
    else :
        logger.info("packets belong in the current max traffic..")
        #delete the packets from the windowed_traffic of conflicting traffic type
        del windowed_traffic[traffic_type][-conflicts[traffic_type]:]
        #overwrite for packet prediction
        for packet_pred in overlapping_window:
            if packet_pred[0] == traffic_type:
                packet_pred[0] = max_traffic_type
            else:
                pass
        #from the overwritten packet_pred, form a list of overlapping actual packets to append to curr_traffic type 
        curr_traffic_app = []
        copy_window_copy = cp.deepcopy(window_copy[:])
        for i in range(len(overlapping_window)):
            if overlapping_window[i][0] == max_traffic_type :
                curr_traffic_app += [copy_window_copy[i]]
            else:
                pass
        # append to the max traffic type for overlapping portion
        # get the sequence for max traffic that has no overlapping packets.
        # append the overlapping packet that belongs to the max traffic type
        windowed_traffic[max_traffic_type] = cp.deepcopy(seq_copy[max_traffic_type])
        windowed_traffic[max_traffic_type].extend(curr_traffic_app)



def get_overlap_pred(curr_window, curr_seq_pred_value, curr_seq, curr_traffic_type, prev_seq_pred_value, prev_seq, prev_traffic_type, models, overlapping_size, norm_fn, denorm_fn, logger):
    ### get the pred values for when without overlapping size
    """ 
    curr_seq_pred_value : prediction of the curr window appended with the exising sequence
    curr_seq : the existing seq of the traffic type of the curr_window without appending the curr_window
    curr_window : the curr window sliced from the original seq

    prev_seq_pred_value : prediction of the prev_window appended to its respective existing seq
    prev_seq : the seq that the prev_window is appended to
    """
    prev_seq_m_pred_value = None
    curr_seq_m_pred_value = None
    ### remove the overlapping packets from the curr_window and append it to its respective traffic sequence
    curr_window_m = cp.deepcopy(curr_window[overlapping_size:])
    logger.info("len of curr_window_m is {}".format(str(len(curr_window_m))))
    curr_seq_copy = cp.deepcopy(curr_seq[:])
    logger.info("len of curr_seq_copy is {}".format(str(len(curr_seq_copy))))
    curr_seq_copy.extend(curr_window_m)
    curr_seq_m = curr_seq_copy[:]
    logger.info("len of curr_seq_m is {}".format(str(len(curr_seq_m))))
    ### remove the overlapping pakcets - at the end - of the prev seq
    prev_seq_m = cp.deepcopy(prev_seq[:-overlapping_size])
    logger.info("len of the prev_seq is {}".format(str(len(prev_seq))))
    logger.info("len of the prev_seq_m is {}".format(str(len(prev_seq_m))))
    
    ###get the prediction values for prev_seq
    prev_seq_m = np.array(prev_seq_m)
    print("The shape of prev_seq_m")
    print(prev_seq_m.shape)
    np_prev_seq_m = prev_seq_m.reshape(1, prev_seq_m.shape[0], prev_seq_m.shape[1])
    window_input, window_true = preprocess_data(np_prev_seq_m, 1000, norm_fn, True)
    window_predict = models[prev_traffic_type].predict(window_input)
    padded_batch_acc = utilsMetric.calculate_acc_of_traffic(window_predict, window_true) #obtain cosine similarity of each packet => (1,1000)
    masked_batch_acc = np.ma.array(padded_batch_acc) #just make it a into a masked array without masking specifications first
    # Mask all the results for > window size.               
    masked_batch_acc[0,  len(prev_seq_m):] = np.ma.masked #0 is needed because its a 2D array with only 1 1D array because our input shape is 1,1000, 147
    denorm_batch_true = denorm_fn(window_true)
    mask = generate_mask_from_pkt_len(denorm_batch_true)
    masked2_batch_acc = np.ma.array(masked_batch_acc)
    masked2_batch_acc.mask = mask #mask again based on length of packet, if len of packet < length threshold, mask it.
    batch_mean_acc_over_big_pkts = np.mean(masked2_batch_acc, axis=-1) #get the mean cosine similarity of all relevant packets. batch_mean_acc_over_big_pkts is a 1D array with only one value.
    prev_seq_m_pred_value = batch_mean_acc_over_big_pkts[0] #appends each model's results to perform np.argmax...

    ###repeat for curr_seq
    curr_seq_m = np.array(curr_seq_m)
    logger.info("shape of curr_seq_m is {}".format(str(curr_seq_m.shape)))
    np_curr_seq_m = curr_seq_m.reshape(1, curr_seq_m.shape[0], curr_seq_m.shape[1])
    window_input, window_true = preprocess_data(np_curr_seq_m, 1000, norm_fn, True)
    window_predict = models[curr_traffic_type].predict(window_input)
    padded_batch_acc = utilsMetric.calculate_acc_of_traffic(window_predict, window_true) #obtain cosine similarity of each packet => (1,1000)
    masked_batch_acc = np.ma.array(padded_batch_acc) #just make it a into a masked array without masking specifications first
    masked_batch_acc[0,  len(curr_seq_m):] = np.ma.masked #0 is needed because its a 2D array with only 1 1D array because our input shape is 1,1000, 147
    denorm_batch_true = denorm_fn(window_true)
    mask = generate_mask_from_pkt_len(denorm_batch_true)
    masked2_batch_acc = np.ma.array(masked_batch_acc)
    masked2_batch_acc.mask = mask #mask again based on length of packet, if len of packet < length threshold, mask it.
    batch_mean_acc_over_big_pkts = np.mean(masked2_batch_acc, axis=-1) #get the mean cosine similarity of all relevant packets. batch_mean_acc_over_big_pkts is a 1D array with only one value.
    curr_seq_m_pred_value = batch_mean_acc_over_big_pkts[0] #appends each model's results to perform np.argmax...

    overlapped_traffic_type = evaluate_overlapped_packets(curr_seq_m_pred_value, curr_seq_pred_value, prev_seq_m_pred_value, prev_seq_pred_value, prev_traffic_type, curr_traffic_type)
    return overlapped_traffic_type

def evaluate_overlapped_packets(curr_seq_m_pred_value, curr_seq_pred_value, prev_seq_m_pred_value, prev_seq_pred_value, prev_traffic_type, curr_traffic_type):
    if curr_seq_m_pred_value < curr_seq_pred_value and prev_seq_m_pred_value > prev_seq_pred_value : #means that packets belong to curr_seq traffic type
        return curr_traffic_type
    elif curr_seq_m_pred_value > curr_seq_pred_value and prev_seq_m_pred_value < prev_seq_pred_value : #means that packets belong to prev_seq traffic type
        return prev_traffic_type
    elif curr_seq_m_pred_value < curr_seq_pred_value and prev_seq_m_pred_value < prev_seq_pred_value: #both increase, can belong to both, take the higher increment
        if curr_seq_pred_value < prev_seq_pred_value:
            return prev_traffic_type
        else :
            return curr_traffic_type
    else : #both decrease, take the lower decrease
        if curr_seq_m_pred_value < prev_seq_m_pred_value:
            return prev_traffic_type
        else :
            return curr_traffic_type

def get_cs(window_len, appending_len, window_true, window_predict):
    padded_batch_acc = utilsMetric.calculate_acc_of_traffic(window_predict, window_true) #obtain cosine similarity of each packet => (1,1000)
    masked_batch_acc = np.ma.array(padded_batch_acc) #just make it a into a masked array without masking specifications first
    #print("shape of the masked_batch_acc")
    #print(masked_batch_acc.shape)
    #print("\n")
    # Mask all the results for > window size. 
    masked_batch_acc[0,  0: window_len] = np.ma.masked #0 is needed because its a 2D array with only 1 1D array because our input shape is 1,1000, 147
    masked_batch_acc[0, window_len + appending_len :]
    batch_mean_acc_over_big_pkts = np.mean(masked_batch_acc, axis=-1)

def dynamic_sw_many2one(model, batch_data,  step_size, window_size, metrics, norm_fn, denorm_fn, threshold, id2labels, logger):
     # Metric supported: 'seq_len', 'idx', 'acc', 'mean_acc'
    #                   'squared_error', 'mean_squared_error', 'true', 'predict', <an int value for the dim number>
    #batch_labels expected : (batch_size, num_windows, window_size)
    #batch_inputs expected: (batch_size, num_windows, window_size, num_features)
    max_len = 1000
    batch_inputs, batch_info = batch_data
    curr_window_size = window_size
    curr_step_size = step_size
    #print("batch_inputs shape is \n")
    #print(batch_inputs.shape)
    #print("\n")
    for metric in metrics:
        batch_seq_len = batch_info['seq_len']
        if metric == 'idx':
            batch_idx = batch_info['batch_idx']
            output[metric] = batch_idx

        elif metric == 'seq_len':
            output[metric] = batch_seq_len
        else:
            #loop through all the batches
            batch_traffic_labels = []
            for batch in batch_inputs :
                start = 0
                logger.info("Start of the batch is")
                logger.info(str(np.array(batch).shape))
                #traffic = {'normal': [], 'rc4' : [], 'breach' : [], 'dos' : [], 'poodle' : []}
                windowed_traffic = {'normal': [], 'breach' : [], 'dos' : [], 'poodle' : [], 'rc4' : []}
                appending_window = {'normal': [], 'breach' : [], 'dos' : [], 'poodle' : [], 'rc4' : []}
                model_results = {'normal': 0, 'rc4' : 0, 'breach' : 0, 'dos' : 0, 'poodle': 0}
                full_traffic_labels = []
                prev_max_traffic = None
                resizing_count = 0
                while len(batch) != 0:
                    logger.info('shape of batch is:')
                    logger.info(str(np.array(batch).shape))
                    logger.info('length of normal : ' + str(len(windowed_traffic['normal'])))
                    logger.info('length of rc4 : ' + str(len(windowed_traffic['rc4'])))
                    logger.info('length of breach : ' + str(len(windowed_traffic['breach'])))
                    logger.info('length of dos : ' + str(len(windowed_traffic['dos'])))
                    logger.info('length of poodle : ' + str(len(windowed_traffic['poodle'])))
                    window_packets_label = []
                    window = batch[start : curr_window_size]
                    np_data = np.array(window)
                    logger.info("Shape of the window before preprocessing and padding is {}".format(np_data.shape))
                    max_traffic_type, max_pred_value = many2one_predict(np_data, id2labels, model, norm_fn)
                    #np_data = np_data.reshape(1, np_data.shape[0], np_data.shape[1])
                    #window_input, window_true = preprocess_data(np_data, pad_len=1000, norm_fn=norm_fn)
                    #traffic_predictions = model.predict_on_batch(window_input)
                    #max_indice = np.argmax(traffic_predictions[0])
                    #max_traffic_type = id2labels[max_indice]
                    model_results[max_traffic_type] = max_pred_value
                    logger.info('The prediction value for the max traffic type {} is {}'.format(max_traffic_type, model_results[max_traffic_type]))
                    if model_results[max_traffic_type] < threshold :
                        resizing_count += 1
                        if resizing_count == 110:
                            logger.info("Looping through all previous traffic windows for preappending")
                            for traffic_type in windowed_traffic.keys():
                                model_results[traffic_type] = 0
                                curr_window_size = window_size
                                curr_step_size = step_size
                                logger.info("The current window_size is {} and the current step_size is {}".format(str(curr_window_size), str(curr_step_size)))
                                preappending_data = windowed_traffic[traffic_type][:]
                                logger.info("The length of the window to append is {}".format(str(len(window[curr_window_size - curr_step_size :]))))
                                if curr_window_size - curr_step_size > len(window):
                                    preappending_data.extend(window[:])
                                else :
                                    preappending_data.extend(window[curr_window_size - curr_step_size:])
                                preappending_data = np.array(preappending_data)
                                logger.info("Appending to the previous windows of {}. Shape of the newly constructed window before preprocessing and padding is {}".format(traffic_type, str(len(preappending_data))))
                                traffic_type, pred_value = many2one_predict(preappending_data, id2labels, model, norm_fn)
                                model_results[traffic_type] = pred_value
                            max_traffic_type = getMaxPrediction(model_results)
                            max_pred_value = model_results[max_traffic_type]
                            logger.info('With previous appended windows, the prediction value for the max traffic type {} is {}'.format(max_traffic_type, max_pred_value))
                            logger.info("The results of the model after appending all traffic types are : " + str(model_results))
                            if max_pred_value < threshold:
                                logger.info("The many2many model is unable to produce a max result that exceeds the threshold {} despite all possible window resizings, terminating the program....".format(str(threshold)))
                                exit()
                            else:
                                resizing_count = 0
                                logger.info('the max traffic type is {}, appending window of length {} to existing {} of length {}'.format(max_traffic_type, str(len(window)), max_traffic_type, str(len(windowed_traffic[max_traffic_type]))))
                                windowed_traffic[max_traffic_type].extend(window[curr_window_size - curr_step_size:])
                                del batch[start:curr_step_size]
                                ###Generate the labels dos, rc4, normal etc...
                                for i in range(0, window_size):
                                    window_packets_label += [max_traffic_type]
                                full_traffic_labels.extend(window_packets_label)
                        elif curr_window_size == max_len:
                            curr_window_size = 10
                            curr_step_size = curr_window_size
                        else : 
                            curr_window_size += 10
                            curr_step_size = curr_window_size
                    else :
                        resizing_count = 0
                        logger.info('the max traffic type is {}, appending window of length {} to existing {} of length {}'.format(max_traffic_type, str(len(window)), max_traffic_type, str(len(windowed_traffic[max_traffic_type]))))
                        windowed_traffic[max_traffic_type].extend(window[curr_window_size - curr_step_size:])
                        del batch[start:curr_step_size]
                        ###Generate the labels dos, rc4, normal etc...
                        for i in range(curr_window_size - curr_step_size, curr_window_size):
                            window_packets_label += [max_traffic_type]
                        full_traffic_labels.extend(window_packets_label)
                        curr_window_size = window_size
                        curr_step_size = step_size
                batch_traffic_labels.append(full_traffic_labels)
    return batch_traffic_labels

def dynamic_sw_many2one2(model, batch_data,  step_size, window_size, metrics, norm_fn, denorm_fn, threshold, id2labels, logger, is4D):
     # Metric supported: 'seq_len', 'idx', 'acc', 'mean_acc'
    #                   'squared_error', 'mean_squared_error', 'true', 'predict', <an int value for the dim number>
    #batch_labels expected : (batch_size, num_windows, window_size)
    #batch_inputs expected: (batch_size, num_windows, window_size, num_features)
    max_len = 1000
    batch_inputs, batch_info = batch_data
    curr_window_size = window_size
    curr_step_size = step_size
    #print("batch_inputs shape is \n")
    #print(batch_inputs.shape)
    #print("\n")
    for metric in metrics:
        batch_seq_len = batch_info['seq_len']
        if metric == 'idx':
            batch_idx = batch_info['batch_idx']
            output[metric] = batch_idx

        elif metric == 'seq_len':
            output[metric] = batch_seq_len
        else:
            #loop through all the batches
            batch_traffic_labels = []
            for batch in batch_inputs :
                start = 0
                logger.info("Start of the batch is")
                logger.info(str(np.array(batch).shape))
                #traffic = {'normal': [], 'rc4' : [], 'breach' : [], 'dos' : [], 'poodle' : []}
                windowed_traffic = {'normal': [], 'breach' : [], 'dos' : [], 'poodle' : [], 'rc4' : []}
                appending_window = {'normal': [], 'breach' : [], 'dos' : [], 'poodle' : [], 'rc4' : []}
                model_results = {'normal': 0, 'rc4' : 0, 'breach' : 0, 'dos' : 0, 'poodle': 0}
                full_traffic_labels = []
                prev_max_traffic = None
                resizing_count = 0
                while len(batch) != 0:
                    logger.info('shape of batch is:')
                    logger.info(str(np.array(batch).shape))
                    logger.info('length of normal : ' + str(len(windowed_traffic['normal'])))
                    logger.info('length of rc4 : ' + str(len(windowed_traffic['rc4'])))
                    logger.info('length of breach : ' + str(len(windowed_traffic['breach'])))
                    logger.info('length of dos : ' + str(len(windowed_traffic['dos'])))
                    logger.info('length of poodle : ' + str(len(windowed_traffic['poodle'])))
                    window_packets_label = []
                    window = cp.deepcopy(batch[start : curr_window_size])
                    to_append = cp.deepcopy(window[curr_window_size - curr_step_size:])
                    ####problem : did not deepcopy the appending traffic for all different traffic types
                    for traffic_type in windowed_traffic.keys():
                        ## reset the results from before
                        model_results[traffic_type] = 0 
                        appending_traffic = cp.deepcopy(windowed_traffic[traffic_type][:])
                        if traffic_type == prev_max_traffic:
                            appending_traffic.extend(window[curr_window_size - curr_step_size :])
                        else:
                            appending_traffic.extend(window[:])
                        appending_traffic = np.array(appending_traffic)
                        logger.info("Shape of the traffic for {} before preprocessing and padding is {}".format(traffic_type, str(appending_traffic.shape)))
                        traffic_type, pred_value = many2one_predict(appending_traffic, id2labels, model, norm_fn, max_len, logger, is4D)
                        if model_results[traffic_type] < pred_value:
                            model_results[traffic_type] = pred_value
                        logger.info('The resultant traffic type {} has prediction value of {}'.format(traffic_type, str(pred_value)))
                    max_traffic_type = getMaxPrediction(model_results)
                    logger.info("The result for each traffic type are : {}".format(str(model_results)))
                    logger.info('The prediction value for the max traffic type {} is {}'.format(max_traffic_type, model_results[max_traffic_type]))
                    
                    if model_results[max_traffic_type] < threshold :
                        resizing_count += 1
                        if resizing_count == 110:
                            logger.info("Looping through all previous traffic windows for preappending")
                            for traffic_type in windowed_traffic.keys():
                                curr_window_size = window_size
                                curr_step_size = step_size
                                logger.info("The current window_size is {} and the current step_size is {}".format(str(curr_window_size), str(curr_step_size)))
                                preappending_data = windowed_traffic[traffic_type][:]
                                logger.info("The length of the window to append is {}".format(str(len(window[curr_window_size - curr_step_size :]))))
                                if curr_window_size - curr_step_size > len(window):
                                    preappending_data.extend(window[:])
                                else :
                                    preappending_data.extend(window[curr_window_size - curr_step_size:])
                                preappending_data = np.array(preappending_data)
                                logger.info("Appending to the previous windows of {}. Shape of the newly constructed window before preprocessing and padding is {}".format(traffic_type, str(len(preappending_data))))
                                traffic_type, pred_value = many2one_predict(preappending_data, id2labels, model, norm_fn, max_len, logger, is4D)
                                model_results[traffic_type] = pred_value
                            max_traffic_type = getMaxPrediction(model_results)
                            max_pred_value = model_results[max_traffic_type]
                            logger.info('With previous appended windows, the prediction value for the max traffic type {} is {}'.format(max_traffic_type, max_pred_value))
                            logger.info("The results of the model after appending all traffic types are : " + str(model_results))
                            if max_pred_value < threshold:
                                logger.info("The many2many model is unable to produce a max result that exceeds the threshold {} despite all possible window resizings, terminating the program....".format(str(threshold)))
                                exit()
                            else:
                                resizing_count = 0
                                logger.info('the max traffic type is {}, appending window of length {} to existing {} of length {}'.format(max_traffic_type, str(len(window)), max_traffic_type, str(len(windowed_traffic[max_traffic_type]))))
                                windowed_traffic[max_traffic_type].extend(window[curr_window_size - curr_step_size:])
                                del batch[start:curr_step_size]
                                ###Generate the labels dos, rc4, normal etc...
                                for i in range(0, window_size):
                                    window_packets_label += [max_traffic_type]
                                full_traffic_labels.extend(window_packets_label)
                        elif curr_window_size == max_len:
                            curr_window_size = 10
                            curr_step_size = curr_window_size
                        else : 
                            curr_window_size += 10
                            curr_step_size = curr_window_size
                    else :
                        prev_max_traffic = max_traffic_type
                        resizing_count = 0
                        logger.info('the max traffic type is {}, appending window of length {} to existing {} of length {}'.format(max_traffic_type, str(len(window)), max_traffic_type, str(len(windowed_traffic[max_traffic_type]))))
                        windowed_traffic[max_traffic_type].extend(to_append)
                        del batch[start:curr_step_size]
                        ###Generate the labels dos, rc4, normal etc...
                        for i in range(0, window_size):
                            window_packets_label += [max_traffic_type]
                        full_traffic_labels.extend(window_packets_label)
                    """
                    logger.info('the max traffic type is {}, appending window of length {} to existing {} of length {}'.format(max_traffic_type, str(len(window)), max_traffic_type, str(len(windowed_traffic[max_traffic_type]))))
                    if max_traffic_type == prev_max_traffic: 
                        windowed_traffic[max_traffic_type].extend(window[curr_window_size - curr_step_size:])
                    else :
                        windowed_traffic[max_traffic_type].extend(window[:])
                    prev_max_traffic = max_traffic_type
                    del batch[start:curr_step_size]
                    for i in range(curr_window_size - curr_step_size, curr_window_size):
                        window_packets_label += [max_traffic_type]
                    full_traffic_labels.extend(window_packets_label)
                    """
                batch_traffic_labels.append(full_traffic_labels)
    return batch_traffic_labels

def dynamic_sw_many2one3(model, batch_data,  step_size, window_size, metrics, norm_fn, denorm_fn, threshold, id2labels, logger, is4D):
     # Metric supported: 'seq_len', 'idx', 'acc', 'mean_acc'
    #                   'squared_error', 'mean_squared_error', 'true', 'predict', <an int value for the dim number>
    #batch_inputs expected: (batch_size, window_size, num_features)
    max_len = 1000
    batch_inputs, batch_info = batch_data
    curr_window_size = window_size
    curr_step_size = step_size
    for metric in metrics:
        batch_seq_len = batch_info['seq_len']
        if metric == 'idx':
            batch_idx = batch_info['batch_idx']
            output[metric] = batch_idx

        elif metric == 'seq_len':
            output[metric] = batch_seq_len
        else:
            #loop through all the batches
            batch_traffic_labels = []
            for batch in batch_inputs :
                #start and end indexes...
                start = 0
                end = None
                prev_start = None
                prev_end = None
                ### declare all data structures ###
                logger.info("Length of the batch is {}".format(len(batch)))
                windowed_traffic = {'normal': [], 'breach' : [], 'dos' : [], 'poodle' : [], 'rc4' : []}
                model_results = {'normal': 0, 'rc4' : 0, 'breach' : 0, 'dos' : 0, 'poodle': 0}
                packet_predictions = []
                while start < len(batch):
                    end = start + curr_window_size - 1
                    logger.info('start is {}'.format(str(start)))
                    logger.info('end is {}'.format(str(end)))
                    logger.info('len of packet prediction is {}'.format(len(packet_predictions)))
                    #1. append window packets to each appending window.
                    # search through the overlapping windows to find all potential conflicts.
                    # conflicts is key(traffic_type) and value(number of packets in the overlapping window)
                    # the conflicts dictionary tell us how many packets to remove from each windowed_traffic so that the full window can be appended without repeating packets
                    # seq_copy holds the sequence up until start index for each traffic_type => no packets in current window, including those that belong to itself, so that full current window can be appended to it without any repeating packets
                    # Then using the windowed_traffic(includes packet in current window for all conflicting types) and seq_copy(conflicting types), we can get the difference in pred value for S and S'
                    # window_primes holds the windows that contains the overlapping packets to append to max_traffic type without the conflicting packets to each conflicting traffic type. It should only contain the 4 other traffic types besides the current max traffic type
                    # window_primes also contain packets that are to the right of the overlapping region (fresh to the right of the window)
                    conflicts = {}
                    seq_copy = {'normal': [], 'breach' : [], 'dos' : [], 'poodle' : [], 'rc4' : []}
                    window_primes = {}
                    window_copy = cp.deepcopy(batch[start:start+curr_window_size]) #even if window_size > remaining batch data size, will take all that is remaining
                    if len(packet_predictions) != 0 :#this is not the first window
                        #TODO : need to account when window_size and step_size changes
                        #overlapping windows only include the overlapping packets
                        overlapping_window = packet_predictions[start:prev_end+1] #shallow copy so that we can make changes to it directly, packet_predictions is 2D to make referencing changes eg. [['dos'], ['normal']]
                        getConflicts(conflicts, overlapping_window)
                        logger.info('Conflicts dictionary state is : {}'.format(str(conflicts)))
                    else:
                        pass
                    for traffic_type in windowed_traffic.keys():
                        if len(packet_predictions) == 0:
                            logger.info("First window, no preappending to do")
                            #First window, no existing sequence for any traffic, append full window to all types and predict
                            full_seq = cp.deepcopy(window_copy[:])
                        else:
                            # Not the first window, remove all the previous duplicate packets that are present in this window.
                            # If the there are no packets that belong to this specific traffic type, then conflicts[traffic_type] = 0, no packets will be removed, the full window will be appended
                            seq_len = len(windowed_traffic[traffic_type])
                            logger.info("len of existing sequence {} is {}".format(traffic_type, seq_len))
                            logger.info('the number of conflict packets for {} is {} '.format(traffic_type, conflicts.get(traffic_type, 0)))
                            seq_copy[traffic_type] = cp.deepcopy(windowed_traffic[traffic_type][:seq_len - conflicts.get(traffic_type, 0)])
                            to_append = cp.deepcopy(window_copy[:])
                            full_seq = cp.deepcopy(seq_copy[traffic_type])
                            full_seq.extend(to_append[:])

                        full_seq = np.array(full_seq)
                        traffic_pred, value_pred = many2one_predict(full_seq, id2labels, model, norm_fn, max_len, logger, is4D)
                        if traffic_type == traffic_pred :
                            #only consider when the traffic prediction is equal to the traffic_type it is appended to. Else the packets strictly does not belong to the traffic_type as it produces another traffic type which we denote by '0'
                            model_results[traffic_type] = value_pred
                        else :
                            logger.info("The model took in seq of {} but output prediction of {}".format(traffic_type, traffic_pred))
                    ###TODO : There could be None for traffic type as the prediction may not match for all types.
                    logger.info("All traffic types have been processed. Showing models results: \n")
                    max_traffic_type = getMaxPrediction(model_results)
                    logger.info("models results are {}".format(str(model_results)))
                    max_traffic_pred = model_results[max_traffic_type]
                    logger.info("max result is {} with pred_value of {}".format(max_traffic_type, str(max_traffic_pred)))
                    if max_traffic_pred > threshold:
                        logger.info('prediction value of {} exceeds threshold of {}'.format(str(max_traffic_pred), str(threshold)))
                        # now we resolve the overlapping packets if the threshold is met.
                        if len(packet_predictions) == 0 :
                            #there are no overlapping packets as it is the first window.
                            for i in range(curr_window_size):
                                packet_predictions += [[max_traffic_type]]
                            logger.info("First window, no conflicting packets...")
                            #append the window to the max traffic type
                            windowed_traffic[max_traffic_type].extend(window_copy[:])
                        else :
                            curr_traffic_type = max_traffic_type
                            #construct the window primes for each conflicting traffic.
                            construct_windows(window_primes, conflicts, overlapping_window, curr_traffic_type, window_copy, logger)
                            #resolve fn will remove all the packets from the conflicting windowed_traffic if necessary, update the packet_predictions list and append any overlapping packets that belong to the curr max traffic type 
                            resolve(model, id2labels, norm_fn, max_len, conflicts, window_copy, windowed_traffic, seq_copy, window_primes, max_traffic_type, max_traffic_pred, overlapping_window, logger, is4D)
                            #append the rest of the packet that are not in the overlapping portion
                            if curr_window_size > len(window_copy):
                                #reached the end of the batch data, the actual window is actually lesser than the determined window_size, every packet is in the overlapping region, nothing more to append
                                pass
                            else:
                                #append the rest of the packet that are not in the overlapping portion only if curr end index is > prev end index
                                if end > prev_end : 
                                    logger.info('prev end index {} is and current end index at {}'.format(str(prev_end), str(end)))
                                    appending_tail = cp.deepcopy(batch[prev_end+1:end+1]) # if end > len(batch),it will just extend nothing to the exising window
                                    logger.info('length of appending tail is {}'.format(len(appending_tail)))
                                    windowed_traffic[max_traffic_type].extend(appending_tail)
                                    ##add the new packets to the packet_predictions list
                                    for i in range(len(appending_tail)):
                                        packet_predictions += [[max_traffic_type]]
                                else:
                                    logger.info('prev end index {} is same at current end index at {}'.format(len(prev_end, end)))
                                    pass
                                #print('len of normal after appending ' + str(len(windowed_traffic['normal'])))
                        prev_start = start
                        prev_end = end
                        start = start + curr_step_size
                        logger.info('\n')
                        logger.info('Staring the next iteration of sliding window...')
                        logger.info('Sliding the window from prev start index {} to new start index {}'.format(str(prev_start), str(start)))             
                    else :
                        #TODO readjustment of window size..
                        logger.info("The prediction value of {} does not less than threshold, readjusting window size...".format(max_traffic_type))

                    #reset the results
                    model_results = {key: 0 for key in model_results}

                ##flatten the 2D pakcet predictions to 1D
                packet_predictions = np.array(packet_predictions)
                packet_predictions = packet_predictions.flatten()
                packet_predictions = packet_predictions.tolist()
                batch_traffic_labels.append(packet_predictions)
    

    return batch_traffic_labels


def getConflicts(conflicts, overlapping_window):
    for packet_pred in overlapping_window:
        if packet_pred[0] in conflicts:
            conflicts[packet_pred[0]] += 1
        else:
            conflicts[packet_pred[0]] = 1

def construct_windows(window_primes, conflicts, overlapping_window, curr_traffic_type, window_copy, logger):
    ##window copy shape is (num_packet, dims)

    for traffic_type in conflicts.keys():
        if traffic_type == curr_traffic_type:
            pass
        else:
            window_primes[traffic_type] = []
            copy_window_copy = cp.deepcopy(window_copy[:])
            #window_primes stores the corresponding windows to append to max traffic type for each conflicting traffic type in the overlapping region + the tail portion of the window where there are no conflicts. key(conflicting traffic type) value(window to append)
            tail_start = 0
            for i in range(len(overlapping_window)):
                if overlapping_window[i][0] == traffic_type:
                    #do not add the conflicting packet to the window primes as we want to construct the window_prime without the conflicting packet of traffic_type
                    pass
                else :
                    #add the overlapping packet as it either belongs to curr_traffic_type or != traffic type
                    window_primes[traffic_type] += [copy_window_copy[i]]
                tail_start += 1
            window_primes[traffic_type].extend(copy_window_copy[tail_start:])
    logger.info('The number of traffic type in window_primes is {}:'.format(str(window_primes.keys())))

def resolve(model, id2labels, norm_fn, seq_len, conflicts, window_copy, windowed_traffic, seq_copy, window_primes, max_traffic_type, max_traffic_pred, overlapping_window, logger, is4D):
    for traffic_type in window_primes.keys():
        logger.info('Resolving for traffic type {}'.format(traffic_type))
        #1.obtain the conflicting traffics metrics: S and S'
        full_seq = cp.deepcopy(windowed_traffic[traffic_type])
        np_full_seq = np.array(full_seq)
        full_traffic_pred, full_seq_pred = many2one_predict(np_full_seq, id2labels, model, norm_fn, seq_len, logger, is4D)

        ###TODO : there might be a chance where the existing sequence is only in the overlapping window...
        full_seq_prime = cp.deepcopy(seq_copy[traffic_type])
        np_full_seq_prime = np.array(full_seq_prime)
        prime_traffic_pred, full_seq_prime_pred = many2one_predict(np_full_seq_prime, id2labels, model, norm_fn, seq_len, logger, is4D)
        
        #2. obtain the current max traffic metrics
        max_curr_seq = cp.deepcopy(seq_copy[max_traffic_type])
        max_curr_seq.extend(window_primes[traffic_type])
        np_max_curr_seq = np.array(max_curr_seq)
        traffic_pred, max_curr_seq_pred = many2one_predict(np_max_curr_seq, id2labels, model, norm_fn, seq_len, logger, is4D)

        #3 compare the 4 metrics:
        #case 1 : full_seq > full_seq_prime and max_traffic_pred < traffic_pred, means that overlapping packets belong more to the previous conflicting traffic type than in the current traffic type
        #case 2 : full_seq < full_seq_prime and max_traffic_pred > traffic_pred, means that overlapping packetes belong more to the current traffic type than previous conflicting traffic type
        #case 3 : full_seq < full_seq_prime and max_traffic_pred < traffic_pred, take the minimum difference between the two, least drop in accuracy means the packets belong more to the specific traffic type
        #case 4 : full_seq > full_seq_prime and max_traffic_pred > traffic_pred, take the max difference between the two, max rise in accuracy means the packets belong more to the specific traffic type
        #case 5 : equal change in both or no change in both, take any one of the two traffic types.
        # if one traffic type stays stagnant and 
        #                                       1. the other traffic type rises, traffic belongs to it
        #                                       2. the other traffic drops, traffic belongs to the stagnant traffic.
        if full_seq_pred > full_seq_prime_pred and max_traffic_pred < max_curr_seq_pred:
            #belongs to conflicting traffic
            #no change in packet_predictions
            pass
        elif full_seq_pred < full_seq_prime_pred and max_traffic_pred > max_curr_seq_pred:
            #belongs to current max traffic
            #overwrite the packet predictions
            for packet_pred in overlapping_window:
                if packet_pred[0] == traffic_type:
                    packet_pred[0] = max_traffic_type
                else:
                    pass
            #delete the last x packets from the conflicting windowed traffic
            del windowed_traffic[traffic_type][-conflicts[traffic_type]:]
        elif full_seq_pred < full_seq_prime_pred and max_traffic_pred < max_curr_seq_pred:
            #get the minimum fall in value
            if full_seq_prime_pred - full_seq_pred < max_curr_seq_pred - max_traffic_pred :
                #fall is larger for current max traffic type, therefore packets belong to conflicting traffic_type
                # no need to overwrite packet predictions
                pass
            else:
                #fall is larger for conflicting traffic type, therefore packets belong to current max traffic type traffic_type
                #overwrite packet predictions
                for packet_pred in overlapping_window:
                    if packet_pred[0] == traffic_type:
                        packet_pred[0] = max_traffic_type
                    else:
                        pass
                #delete the last x packets from the conflicting windowed traffic
                del windowed_traffic[traffic_type][-conflicts[traffic_type]:]
        elif full_seq_pred > full_seq_prime_pred and max_traffic_pred > max_curr_seq_pred:
            #get the maximum rise in value
            if full_seq_pred - full_seq_prime_pred > max_traffic_pred - max_curr_seq_pred:
                #rise is larger for conflicting traffic type, therefore packets belong to conflicting traffic_type
                # no need to overwrite packet predictions
                pass
            else:
                #rise is larger for current max traffic type, therefore packets belong to current max traffic type traffic_type
                #overwrite packet predictions
                for packet_pred in overlapping_window:
                    if packet_pred[0] == traffic_type:
                        packet_pred[0] = max_traffic_type
                    else:
                        pass
                #delete the last x packets from the conflicting windowed traffic
                del windowed_traffic[traffic_type][-conflicts[traffic_type]:]
        else:
            #both have no change
            #generate a random number to assign it
            guess = randint(0,1)
            if guess == 1:
                #take it as current max traffic type
                for packet_pred in overlapping_window:
                    if packet_pred[0] == traffic_type:
                        packet_pred[0] = max_traffic_type
                    else:
                        pass
                #delete the last x packets from the conflicting windowed traffic
                del windowed_traffic[traffic_type][-conflicts[traffic_type]:]
            else:
                #take it as conflicting traffic type
                pass

        #4. from the overwritten packet_pred, form a list of overlapping actual packets to append to curr_traffic type 
        curr_traffic_app = []
        copy_window_copy = cp.deepcopy(window_copy[:])
        for i in range(len(overlapping_window)):
            if overlapping_window[i][0] == max_traffic_type :
                curr_traffic_app += [copy_window_copy[i]]
            else:
                pass
        # append to the max traffic type for overlapping portion
        windowed_traffic[max_traffic_type].extend(curr_traffic_app)        

def many2one_predict(np_data, id2labels, model, norm_fn, seq_len, logger, is4D):
    np_data = np_data.reshape(1, np_data.shape[0], np_data.shape[1])
    window_input = preprocess_data(np_data, seq_len, norm_fn, False)
    if is4D : 
        window_input = window_input.reshape(window_input.shape[0], window_input.shape[1], window_input.shape[2], 1)
    traffic_predictions = model.predict_on_batch(window_input)
    logger.info("many2one prediction - the traffic prediction array is : {}".format(str(traffic_predictions)))
    max_indice = np.argmax(traffic_predictions[0])
    return id2labels[max_indice], traffic_predictions[0][max_indice]



def many2one_mixed_eval(batch_data, id2labels, model, norm_fn, seq_len, logger, is4D):
    batch_metrics = []
    for data in batch_data:
        data = data[:seq_len]
        pred, acc = many2one_predict(np.array(data), id2labels, model, norm_fn, seq_len, logger, is4D)
        logger.info("many2one model has predicted traffic type of  {} with accuracy of {}".format(pred, str(acc)))
        batch_metrics.append((pred, acc))
    return batch_metrics

def get_lstm_states(batch_data, model, norm_fn, seq_len, logger):
    for data in batch_data:
        np_data = np.array(data)
        np_data = np_data.reshape(1, np_data.shape[0], np_data.shape[1])
        window_input, window_true = preprocess_data(np_data, pad_len=seq_len, norm_fn=norm_fn)
        traffic_predictions = model.predict_on_batch(window_input)
        print(traffic_predictions)

def getNoiseMetrics(normal_list, noised_list, many2one_model, norm_fn, id2labels, seq_len, logger, is4D):
    base_results = []
    noised_results = []
    for normal, noised in zip(normal_list, noised_list):
        logger.info("Base traffic has a length of {} while the noised traffic has length of {}".format(len(normal), len(noised)))
        base_pred, base_acc = many2one_predict(np.array(normal), id2labels, many2one_model, norm_fn, seq_len, logger, is4D)
        base_tuple = (base_pred, base_acc)
        base_results.append(base_tuple)
        noise_pred, noise_acc = many2one_predict(np.array(noised), id2labels, many2one_model, norm_fn, seq_len, logger, is4D)
        noised_tuple = (noise_pred , noise_acc)
        logger.info("After prediction in getNoiseMetrics, base_tuple has {} and noised_tuple has {}".format(str(base_tuple), str(noised_tuple)))
        noised_results.append(noised_tuple)
    return base_results, noised_results

def getNoiseMetrics_old(noised_list, normal_list, model, norm_fn, denorm_fn, metrics, seq_len, logger):
    base_results = []
    noised_results = []
    for normal, noised in zip(normal_list, noised_list):
        normal_len = len(normal)
        noised_len = len(noised)
        logger.info("Base traffic has a length of {} while the noised traffic has length of {}".format(str(normal_len), str(noised_len)))
        normal = np.array(normal)
        normal = normal.reshape(1, normal.shape[0], normal.shape[1])
        base_inputs, base_targets = preprocess_data(normal, seq_len, norm_fn, True)
        base_result = compute_accuracy(model, (base_inputs, base_targets), metrics, normal_len, denorm_fn, False, None, logger)
        base_tuple = ('normal', base_result['mean_acc'][0])
        base_results.append(base_tuple)
        noised = np.array(noised)
        noised = noised.reshape(1, noised.shape[0], noised.shape[1])
        noised_inputs, noised_targets = preprocess_data(noised, seq_len, norm_fn, True)
        noised_result = compute_accuracy(model, (noised_inputs, noised_targets), metrics, noised_len, denorm_fn, False, None, logger)
        noised_tuple = ('normal', noised_result['mean_acc'][0])
        noised_results.append(noised_tuple)
        logger.info("After prediction in getNoiseMetrics, base_tuple has {} and noised_tuple has {}".format(str(base_tuple), str(noised_tuple)))
    return base_results, noised_results


def get_app_packets_acc(original_list, noised_list, models, norm_fn, denorm_fn, metrics, seq_len, logger):
    base_pred = []
    appended_pred = []
    for normal, noised in zip(original_list, noised_list):
        base_results = {'normal': 0, 'rc4' : 0, 'breach' : 0, 'dos' : 0, 'poodle': 0}
        noised_results = {'normal': 0, 'rc4' : 0, 'breach' : 0, 'dos' : 0, 'poodle': 0}
        normal_len = len(normal)
        noised_len = len(noised)
        appending_len = noised_len - normal_len
        logger.info("Base traffic has a length of {} while the noised traffic has length of {}, appending length is {}".format(str(normal_len), str(noised_len), str(appending_len)))
        for traffic_type in models.keys():
            logger.info("using model of {} type".format(str(traffic_type)))
            model = models[traffic_type]
            normal_copy = cp.deepcopy(normal)
            normal_array = np.array(normal_copy)
            normal_array = normal_array.reshape(1, normal_array.shape[0], normal_array.shape[1])
            base_inputs, base_targets = preprocess_data(normal_array, seq_len, norm_fn, True)
            base_results[traffic_type] = compute_accuracy(model, (base_inputs, base_targets), metrics, normal_len, denorm_fn, False, None, logger)['mean_acc'][0]
            noised_copy = cp.deepcopy(noised)
            noised_array = np.array(noised_copy)
            noised_array = noised_array.reshape(1, noised_array.shape[0], noised_array.shape[1])
            noised_inputs, noised_targets = preprocess_data(noised_array, seq_len, norm_fn, True)
            noised_results[traffic_type] = compute_accuracy(model, (noised_inputs, noised_targets), metrics, noised_len, denorm_fn, True, appending_len, logger)['mean_acc'][0]
        normal_type = getMaxPrediction(base_results)
        normal_acc = noised_results[normal_type]
        normal_tuple = (normal_type, normal_acc)
        base_pred.append(normal_tuple)
        appended_type = getMaxPrediction(noised_results)
        appended_acc = noised_results[appended_type]
        appended_tuple = (appended_type, appended_acc)
        appended_pred.append(appended_tuple)
    return base_pred, appended_pred

def validate_windows(models, batch_data, labels_data, window_size, type, metrics, denorm_fn):
    # Metric supported: 'seq_len', 'idx', 'acc', 'mean_acc'
    #                   'squared_error', 'mean_squared_error', 'true', 'predict', <an int value for the dim number>
    #batch_labels expected : (batch_size, num_windows, window_size)
    #batch_inputs expected: (batch_size, num_windows, window_size, num_features)
    batch_inputs, batch_info = batch_data
    batch_labels = labels_data
    #print("batch_inputs shape is \n")
    #print(batch_inputs.shape)
    #print("\n")
    for metric in metrics:
        batch_seq_len = batch_info['seq_len']
        if metric == 'idx':
            batch_idx = batch_info['batch_idx']
            output[metric] = batch_idx

        elif metric == 'seq_len':
            output[metric] = batch_seq_len

        else:
            batch_percentage = []
            batch_label_window_acc = []
            batch_prediction_acc = []
            max_window = 0
            #loop through all the batches
            for batch, batch_label in zip(batch_inputs, batch_labels) :
                #print("num of windows for this batch is " + str(batch.shape[0]))
                if max_window < batch.shape[0] :
                    max_window = batch.shape[0]
                batch_inputs = batch[:,:-1,:]  #:-1 everything until the last item in the list
                batch_targets = batch[:,1:,:]
                correct_predictions = 0
                num_windows = batch_inputs.shape[0]
                window_label_acc = []
                prediction_acc = []
                #get the each window, 
                #batch_label is expected
                for window, window_true, window_packet_label in zip(batch_inputs, batch_targets, batch_label):
                    window = window.reshape(1, 1000, 147) #window is intitially (1000,147), model predict requires 3D, window size is always padded to 1000
                    window_true = window_true.reshape(1, 1000, 147)
                    model_results = []
                    labels = [] #holds the name of the models. ordering of the models is preserved
                    for model_label in models.keys():
                        labels.append(model_label)
                        model = models[model_label]
                        window_label = get_window_label(window_packet_label, type)
                        window_predict = model.predict_on_batch(window)
                        if metric == 'acc' or metric == 'mean_acc':
                            padded_batch_acc = utilsMetric.calculate_acc_of_traffic(window_predict, window_true) #obtain cosine similarity of each packet => (1,1000)
                            masked_batch_acc = np.ma.array(padded_batch_acc) #just make it a into a masked array without masking specifications first

                            #print("shape of the masked_batch_acc")
                            #print(masked_batch_acc.shape)
                            #print("\n")
                            # Mask all the results for > window size. 
                            
                            masked_batch_acc[0,  window_size:] = np.ma.masked #0 is needed because its a 2D array with only 1 1D array because our input shape is 1,1000, 147
                            #print("value of the masked_batch_acc \n")
                            #print(masked_batch_acc)
                            #print("\n")
                            if metric == 'acc':
                                output[metric] = masked_batch_acc
                            elif metric == 'mean_acc':
                                if PKT_LEN_THRESHOLD > 0 and denorm_fn:
                                    #print("shape of window_true \n")
                                    #print(window_true.shape)
                                    #print("\n")
                                    denorm_batch_true = denorm_fn(window_true)
                                    mask = generate_mask_from_pkt_len(denorm_batch_true)
                                    #print("shape of mask \n")
                                    #print(mask.shape)
                                    #print("\n")
                                    masked2_batch_acc = np.ma.array(masked_batch_acc)
                                    masked2_batch_acc.mask = mask #mask again based on length of packet, if len of packet < length threshold, mask it.
                                    batch_mean_acc_over_big_pkts = np.mean(masked2_batch_acc, axis=-1) #get the mean cosine similarity of all relevant packets. batch_mean_acc_over_big_pkts is a 1D array with only one value.
                                    model_results.append(batch_mean_acc_over_big_pkts[0]) #appends each model's results to perform np.argmax...
                                    if model_label == window_label:
                                        window_label_acc.append(batch_mean_acc_over_big_pkts[0]) #gets the accuracy of the window label model (the model that is supposed to get the highest cosine similarity)
                                elif PKT_LEN_THRESHOLD == -1:
                                    batch_mean_acc = np.mean(masked_batch_acc, axis=-1)
                                    model_results.append(batch_mean_acc[0])
                    print("the model results are \n")
                    print(model_results)
                    print("\n")
                    #print("the prediction_acc is \n")
                    #print(prediction_acc)
                    #print("\n")
                    temp_array = np.array(model_results)
                    max_indice = np.argmax(temp_array)
                    prediction_acc.append(temp_array[max_indice]) #appends the best prediction accuracy value for each window
                    prediction_label = labels[max_indice] #gets the prediction label for each window
                    if prediction_label == window_label: 
                        correct_predictions += 1

                ### Append to batch data structures after all windows are been processed ###
                #print("number of correct guesses is {} and number of windows is {}".format(correct_predictions, num_windows))
                #print("the length window_label_acc is \n")
                #print(str(len(prediction_acc)))
                #print("\n")
                #print("the length prediction_acc is \n")
                #print(str(len(window_label_acc)))
                #print("\n")
                batch_percentage.append((correct_predictions/num_windows)* 100 )
                batch_label_window_acc.append(window_label_acc)
                batch_prediction_acc.append(prediction_acc)
                #print("the batch window_label_acc is \n")
                #print(str(len(batch_label_window_acc)))
                #print("\n")
                #print("the batch prediction_acc is \n")
                #print(str(len(batch_prediction_acc)))
                #print("\n")    

            return batch_percentage, batch_label_window_acc, batch_prediction_acc, max_window


##############################
###PURE TRAFFIC SEPARATION ###
##############################
def separate(mixed_data, data_labels):
    ### takes one traffic flow and returns its normal and attack
    normal = []
    attack = []
    for data, label in zip(mixed_data, data_labels):
        if label == 0 : 
            normal.append(data)
        else :
            attack.append(data)
    return normal, attack

def separate_traffic(batch_data, labels_data, metrics, norm_fn, seq_len, isNoise, is_slice):
    # Metric supported: 'seq_len', 'idx', 'acc', 'mean_acc'
    #                   'squared_error', 'mean_squared_error', 'true', 'predict', <an int value for the dim number>
    batch_inputs, batch_info = batch_data
    batch_labels = labels_data
    pad_len = seq_len;
    batch_inputs = np.array(batch_inputs)
    #print("batch_inputs shape is \n")
    #print(batch_inputs.shape)
    #print("\n")
    batch_labels = np.array(batch_labels)
    #print("batch_labels shape is \n")
    #print(batch_labels.shape)
    #print("\n")
    #batch_labels = batch_labels.reshape(batch_labels.shape[1], batch_labels.shape[2])
    normal_list = []
    attack_list = []
    for data, label in zip(batch_inputs, batch_labels):
        normal, attack = separate(data, label)
        if len(normal) > 0:
            normal_list.append(normal)
        if len(attack) > 0:
            attack_list.append(attack)
        #print("normal shape is \n")
        #print(np.array(normal).shape)
        #print("\n")
        #print("attack shape is \n")
        #print(np.array(attack).shape)
        #print("\n")
    #print("normal shape is \n")
    #print(np.array(normal_list).shape)
    #print("\n")
    #print("attack shape is \n")
    #print(np.array(attack_list).shape)
    #print("\n")
    if isNoise :
        return normal_list, attack_list
    else :
        normal_batch_info = {}
        attack_batch_info = {}
        batch_seq_len = [len(data) for data in normal_list]
        #print("the length of normal batch_seq_len is \n")
        #print(batch_seq_len)
        #print("\n")
        normal_batch_info['seq_len'] = np.array(batch_seq_len)  # Standardize output into numpy array
        batch_seq_len = [len(data) for data in attack_list]
        attack_batch_info['seq_len'] = np.array(batch_seq_len)
        #print("the length of attack batch_seq_len is \n")
        #print(batch_seq_len)
        #print("\n")
        if len(normal_list) > 0 :
            if is_slice :
                normal_inputs, normal_targets = preprocess_data(normal_list, pad_len, norm_fn, True)
            else:
                normal_inputs = preprocess_data(normal_list, pad_len, norm_fn, False)
        else : 
            normal_inputs = []
            normal_targets = []
        if len(attack_list) > 0 :
            if is_slice:
                attack_inputs, attack_targets = preprocess_data(attack_list, pad_len, norm_fn, True)
            else :
                attack_inputs = preprocess_data(attack_list, pad_len, norm_fn, False)
        else :
            attack_inputs = []
            attack_targets = []

        #print("shape of attack inputs is \n")
        #print(normal_targets.shape)
        #print("\n")
        if is_slice : 
            return (normal_inputs, normal_targets, normal_batch_info), (attack_inputs, attack_targets, attack_batch_info)
        else :
            return (normal_inputs, normal_batch_info), (attack_inputs, attack_batch_info)





def generate_mask_from_pkt_len(batch_data):
    batch_pktlen = batch_data[:, :, 7] #7th feature is the length of the packet
    mask = batch_pktlen <= PKT_LEN_THRESHOLD  ##generate a boolean array of shape (1,1000)
    return mask

def compute_metrics_generator(model, data_generator, metrics, denorm_fn=None):
    for batch_data in data_generator:
        output = compute_metrics_for_batch(model, batch_data, metrics, denorm_fn)
        yield output

# For defining float values between 0 and 1 for argparse
def restricted_float(x):
    x = float(x)
    if x < 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError('{} not in range [0.0, 1.0]'.format(x))
    return x

