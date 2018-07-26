from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf
import numpy as np
import scipy.sparse
import sklearn.metrics

from gcn.utils import *
from gcn.models import GCN, MLP
from sklearn.decomposition import PCA
from sklearn import preprocessing
lb = preprocessing.LabelBinarizer()

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4 , 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_float('weight_decay_wb', 5e-4 , 'Weight for L2 loss on W and b matrix.')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')
flags.DEFINE_integer('early_stopping', 100, 'Tolerance for early stopping (# of epochs).')

# Load data
# Load data
adj1 = np.load('/home/yld8809/dep_mat_tep_all_train.npy')
adj1 = adj1.item()
adj2 = np.load('/home/yld8809/dep_mat_tep_all_test.npy')
adj2 = adj2.item()
adj = scipy.sparse.block_diag((adj1,adj2))
adj = scipy.sparse.coo_matrix(adj, dtype=np.float32)


features1 = np.load('/home/yld8809/word_embedding_tep_all_train.npy')
features2 = np.load('/home/yld8809/word_embedding_tep_all_test.npy')

features = np.vstack((features1,features2))
#pca = sklearn.decomposition.PCA(n_components=0.8)
features1 = np.float32(features1[:,7:])
#pca.fit(features1)


features1 = []
features2 = []

num_doc = int(max(np.float32(features[:,1])))+1

ind_dict = np.float32(features[:,1:7])
features = np.float32(features[:,7:])
#features = pca.transform(features)

label1 = np.load('/home/yld8809/all_rel/tep_all_train.npy')[:,[2,8]]
label2 = np.load('/home/yld8809/all_rel/tep_all_test.npy')[:,[2,8]]
label = np.vstack((label1,label2))

label_match = np.in1d(np.float32(label[:,0]),np.asarray(range(num_doc)))
label = label[label_match,:]
label = lb.fit_transform(label[:,1])

#==============================================================================
# adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data('cora')
#==============================================================================
word_index = ind_dict
unique_word_index=[]; 
[unique_word_index.append(tuple(r)) for r in word_index if tuple(r) not in unique_word_index]; 
unique_word_index = np.asarray(unique_word_index);

size_unique_word_index = np.shape(unique_word_index)[0]
all_size = np.zeros(shape=[np.shape(ind_dict)[0],1])
for sent_ind in range(0,size_unique_word_index):
    match_ind = (ind_dict[:,:] == unique_word_index[sent_ind,:])
    match_ind = np.sum(match_ind,axis = 1)
    match_ind = np.where(match_ind == 6)
    all_size[match_ind,0] = sent_ind

all_size = np.float32(all_size)

all_phrase = np.empty(shape=[0, 1])
for sent_ind in range(0,len(unique_word_index)):
    match_ind = np.where(all_size[:,0] == sent_ind)[0]
    temp_embedding_ind = ind_dict[match_ind,:][0]
    all_phrase_temp = np.zeros(shape=[np.shape(match_ind)[0],1])
    all_phrase_temp[int(temp_embedding_ind[2]):int(temp_embedding_ind[3]+1),0] = (sent_ind+1)*2-1
    all_phrase_temp[int(temp_embedding_ind[4]):int(temp_embedding_ind[5]+1),0] = (sent_ind+1)*2
    all_phrase = np.vstack((all_phrase, all_phrase_temp))

max_size = 0
for epoch in range(0,label.shape[0]):

    current_ind = np.asarray(epoch) 
    matched_word_ind = np.asarray([j for j, item in enumerate(all_size[:,0]) if item in current_ind])
    
    max_size = max(max_size,matched_word_ind.shape[0])
    
features_tuple = []
adj_tuple = []
for epoch in range(0,label.shape[0]):
    print(epoch,label.shape[0])
    
    current_ind = np.asarray(epoch) 
    matched_word_ind = [j for j, item in enumerate(all_size[:,0]) if item in current_ind]
    
    features_train_feed = features[matched_word_ind,:]
    
    features_temp = np.zeros(shape=[max_size,300])
    features_temp[0:np.asarray(matched_word_ind).shape[0],:] = features_train_feed[:,:]
    features_tuple = features_tuple + [scipy.sparse.coo_matrix(features_temp)]
    
    adj_temp = np.zeros(shape=[max_size,max_size])
    adj_ori = (adj.tocsr()[matched_word_ind, :].tocsc()[:, matched_word_ind]).todense()
    adj_temp[0:np.asarray(matched_word_ind).shape[0],0:np.asarray(matched_word_ind).shape[0]] = adj_ori[:,:]
    adj_tuple = adj_tuple + [scipy.sparse.coo_matrix(adj_temp)]

np.save("/home/yld8809/tep_features_padded", features_tuple)
np.save("/home/yld8809/tep_adj_padded", adj_tuple)