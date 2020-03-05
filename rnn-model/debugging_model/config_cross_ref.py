import os
from datetime import datetime

normal_featuredir = None
breach_featuredir = None
poodle_featuredir = None
rc4_featuredir = None
dos_featuredir = None

mixed_dos_labels = None

# Write your directory path to the feature files here
root_featuredir = os.path.join('../..','feature-extraction','extracted-features')
#normal_featuredir = os.path.join(root_featuredir, 'normal-36k')
#breach_featuredir = os.path.join(root_featuredir, 'breach-10k')
#poodle_featuredir = os.path.join(root_featuredir, 'poodle-10k')
#rc4_featuredir = os.path.join(root_featuredir, 'rc4-10k')
#dos_featuredir = os.path.join(root_featuredir, 'thc-tls-dos-10k')

##mix traffic###
#mixed_dos_featuredir = os.path.join(root_featuredir, 'mixed_thc-tls-dos')
mixed_dos_featuredir = os.path.join(root_featuredir, 'single_mixed_thc-tls-dos')
#mixed_rc4_featuredir = os.path.join(root_featuredir, 'mixed_rc4_normal')

#Write the labels directory path here.
root_labeldir = os.path.join('../..','feature-extraction','extracted-features')

##mix labels##
#mixed_dos_labels = os.path.join(root_labeldir, 'mixed_thc-tls-dos')
mixed_dos_labels = os.path.join(root_labeldir, 'single_mixed_thc-tls-dos')
#mixed_rc4_labels = os.path.join(root_labeldir, 'mixed_rc4_normal')

minmax_dir = os.path.join('../..', 'feature-extraction', 'features_minmax_ref.csv')

# Write your directory path to the models here
root_modeldir = '../trained-rnn'

##attention LSTM epochs = 5###
#many2one_modeldir = os.path.join(root_modeldir, 'attention', 'expt_2020-03-03_08-21-14', 'rnnmodel_2020-03-03_08-21-14.h5') #attention model with no activation function for sigmoid -> not sensitive at all
#many2one_modeldir = os.path.join(root_modeldir, 'attention', 'expt_2020-03-03_12-43-58', 'rnnmodel_2020-03-03_12-43-58.h5') #attention with sigmoid activation in attention layer -> not sensitive, normal = 1
#many2one_modeldir = os.path.join(root_modeldir, 'attention', 'expt_2020-03-03_15-45-55', 'rnnmodel_2020-03-03_15-45-55.h5') #attention with sigmoid activation in attention layer + attention_width = 10 -> not sensitive, normal = 1
#many2one_modeldir = os.path.join(root_modeldir, 'attention', 'expt_2020-03-03_19-07-54', 'rnnmodel_2020-03-03_19-07-54.h5') #attention with sigmoid activation in attention layer + attention_width = 10 , history=True -> not sensitive, normal = 1
#many2one_modeldir = os.path.join(root_modeldir, 'attention', 'expt_2020-03-04_10-27-46', 'rnnmodel_2020-03-04_10-27-46.h5') #attention with sigmoid activation in attention layer + attention_width = 5 with name -> not sensitive, normal = 1
#many2one_modeldir = os.path.join(root_modeldir, 'attention', 'expt_2020-03-04_13-11-13', 'rnnmodel_2020-03-04_13-11-13.h5') #attention with sigmoid activation in attention layer + attention_width = 5 with name, multiplicative attention ,  -> not sensitve with normal as base
#many2one_modeldir = os.path.join(root_modeldir, 'attention', 'expt_2020-03-04_15-39-09', 'rnnmodel_2020-03-04_15-39-09.h5') #attention with 100 seq_len, sigmoid activation in attention layer + attention_width = 5 with name, multiplicative attention
#many2one_modeldir = os.path.join(root_modeldir, 'attention', 'expt_2020-03-04_20-06-03', 'rnnmodel_2020-03-04_20-06-03.h5') #attention with 5k normal, sigmoid activation in attention layer + attention_width = 5 with name, multiplicative attention

#many2one_modeldir = os.path.join(root_modeldir, 'many2one', 'expt_2020-02-25_14-42-40', 'rnnmodel_2020-02-25_14-42-40.h5') #cnn_lstm, (2,4) kernel, stride 4
#many2one_modeldir = os.path.join(root_modeldir, 'many2one', 'expt_2020-02-25_22-04-32', 'rnnmodel_2020-02-25_22-04-32.h5') #cnn_lstm, (4,8) kernel, stride 4
#many2one_modeldir = os.path.join(root_modeldir, 'many2one', 'expt_2020-02-26_13-51-27', 'rnnmodel_2020-02-26_13-51-27.h5') #cnn_lstm, (4,8) kernel, stride 4, custom activation with sigmoid
#many2one_modeldir = os.path.join(root_modeldir, 'many2one', 'expt_2020-02-26_22-48-34', 'rnnmodel_2020-02-26_22-48-34.h5') #cnn_lstm, (4,8) kernel, stride 4, custom activation with sigmoid, dropout(0.2) at LSTM

many2one_modeldir = os.path.join(root_modeldir, 'many2one','expt_2019-12-25_19-16-29','rnnmodel_2019-12-25_19-16-29.h5')#original, small drop for insertion
#many2one_modeldir = os.path.join(root_modeldir, 'many2one','expt_2020-01-30_22-03-40','rnnmodel_2020-01-30_22-03-40.h5')#softmax to sigmoid
#many2one_modeldir = os.path.join(root_modeldir, 'many2one','expt_2020-01-31_19-42-54','rnnmodel_2020-01-31_19-42-54.h5')#sigmoid
#many2one_modeldir = os.path.join(root_modeldir, 'many2one','expt_2020-02-01_18-01-32','rnnmodel_2020-02-01_18-01-32.h5')#sigmoid with constraints -10<x<10
#many2one_modeldir = os.path.join(root_modeldir, 'many2one','expt_2020-02-02_14-56-24','rnnmodel_2020-02-02_14-56-24.h5')#sigmoid with constraints -5<x<5
#many2one_modeldir = os.path.join(root_modeldir, 'many2one','expt_2020-02-03_10-11-59','rnnmodel_2020-02-03_10-11-59.h5')#sigmoid with constraints -2.5<x<2.5
#many2one_modeldir = os.path.join(root_modeldir, 'many2one','expt_2020-02-04_11-48-30','rnnmodel_2020-02-04_11-48-30.h5')#sigmoid with constraints -2.5<x<2.5, 147
#many2one_modeldir = os.path.join(root_modeldir, 'many2one','expt_2020-02-05_16-08-14','rnnmodel_2020-02-05_16-08-14.h5')#sigmoid with constraints -2.5<x<2.5, 147, 10k-all
#many2one_modeldir = os.path.join(root_modeldir, 'many2one','expt_2020-02-06_15-18-30','rnnmodel_2020-02-06_15-18-30.h5')#sigmoid with constraints -1.5<x<1.5, 147
#many2one_modeldir = os.path.join(root_modeldir, 'many2one','expt_2020-02-06_08-39-19','rnnmodel_2020-02-06_08-39-19.h5')#sigmoid with constraints -1.5<x<1.5, 147, 10k-all, n.n
#many2one_modeldir = os.path.join(root_modeldir, 'many2one','expt_2020-02-07_17-12-03','rnnmodel_2020-02-07_17-12-03.h5')#100-seq length, sigmoid, constraints -1.5<x<1.5. n.n
#many2one_modeldir = os.path.join(root_modeldir, 'many2one','expt_2020-02-08_17-17-18','rnnmodel_2020-02-08_17-17-18.h5')#100-seq length, sigmoid without constraints . changes(up accuracy when inserted), changes(up accuracy when appended)
#many2one_modeldir = os.path.join(root_modeldir, 'many2one','expt_2020-02-09_09-25-01','rnnmodel_2020-02-09_09-25-01.h5') #original with softmax, 100 seq_len
#many2one_modeldir = os.path.join(root_modeldir, 'many2one','expt_2020-02-13_15-45-12','rnnmodel_2020-02-13_15-45-12.h5') #sigmoid with constraint -1.5<x<1.5, 1000 seq_len, two dense 5
#many2one_modeldir = os.path.join(root_modeldir, 'many2one','expt_2020-02-17_11-48-13','rnnmodel_2020-02-17_11-48-13.h5') #10k all, original with dropout(0.2), softmax
#many2one_modeldir = os.path.join(root_modeldir, 'many2one','expt_2020-02-17_18-01-55','rnnmodel_2020-02-17_18-01-55.h5') #36k normal, original with dropout(0.2), softmax. (too skew towards normal)
#many2one_modeldir = os.path.join(root_modeldir, 'many2one','expt_2020-02-19_13-48-34','rnnmodel_2020-02-19_13-48-34.h5') #epoch 8, batch_size 32, 36k normal, original with dropout(0.2), softmax. (too skew towards normal)


normal_modeldir = os.path.join(root_modeldir, 'normal-36k-gpu','expt_2019-09-16_15-56-03','rnnmodel_2019-09-16_15-56-03.h5')
breach_modeldir = os.path.join(root_modeldir, 'breach-10k-gpu','expt_2019-09-16_15-56-32','rnnmodel_2019-09-16_15-56-32.h5')
poodle_modeldir = os.path.join(root_modeldir, 'poodle-10k-gpu','expt_2019-09-17_13-11-15','rnnmodel_2019-09-17_13-11-15.h5')
rc4_modeldir = os.path.join(root_modeldir, 'rc4-10k-gpu','expt_2019-09-17_13-59-18','rnnmodel_2019-09-17_13-59-18.h5')
dos_modeldir = os.path.join(root_modeldir, 'thc-tls-dos-10k-gpu','expt_2019-09-16_15-56-56','rnnmodel_2019-09-16_15-56-56.h5')
	
# Write your model configuration here
DATETIME_NOW = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
SEQUENCE_LEN = 1000
SPLIT_RATIO = 0.05  # Validation dataset as a %
BATCH_SIZE = 4
SEED = 2019

# ID for traffic labels
label2id = {'normal':0, 'breach':1, 'poodle':2, 'rc4':3, 'dos':4}
id2label = {v:k for k,v in label2id.items()}