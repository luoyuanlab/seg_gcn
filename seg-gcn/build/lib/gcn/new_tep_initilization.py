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

flags.DEFINE_float('learning_rate', 0.003, 'Initial learning rate.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4 , 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_float('weight_decay_wb', 5e-4 , 'Weight for L2 loss on W and b matrix.')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')
flags.DEFINE_integer('early_stopping', 100, 'Tolerance for early stopping (# of epochs).')

# Load data
adj = np.load('/home/yld8809/tep_adj_tep_2_padded.npy')

features1 = np.load('/home/yld8809/word_embedding_tep_all_train.npy')
features2 = np.load('/home/yld8809/word_embedding_tep_all_test.npy')
features = np.vstack((features1,features2))
features1 = []
features2 = []
ind_dict = np.float32(features[:,1:7])

features = np.load('/home/yld8809/tep_features_padded.npy')


#pca = sklearn.decomposition.PCA(n_components=0.8)
#pca.fit(features1)


#features = pca.transform(features)

label1 = np.load('/home/yld8809/all_rel/tep_all_train.npy')[:,[2,8]]
label2 = np.load('/home/yld8809/all_rel/tep_all_test.npy')[:,[2,8]]
label = np.vstack((label1,label2))

label = lb.fit_transform(label[:,1])

word_index = ind_dict
unique_word_index=[]; 
[unique_word_index.append(tuple(r)) for r in word_index if tuple(r) not in unique_word_index]; 
unique_word_index = np.asarray(unique_word_index);

# Define placeholders
placeholders = {
    'eigvec': tf.placeholder(tf.float32, shape=tf.TensorShape([adj[0].shape[0], adj[0].shape[0]-1])),
    'all_phrase': tf.placeholder(tf.float32, shape=(None, 1)),                                      
    'features': tf.placeholder(tf.float32, shape=(None, features[0].shape[1])),
    'labels': tf.placeholder(tf.float32, shape=(None, label.shape[1])),
    'labels_mask': tf.placeholder(tf.int32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'theta': tf.placeholder(tf.float32,shape=[adj[0].shape[0]-1]) 
}

with tf.device('/gpu:0'):
    
    # Create model
    model = GCN(placeholders, input_dim=features[0].shape[1], logging=False)
    

    sess = tf.Session(config=tf.ConfigProto(
      allow_soft_placement=True, log_device_placement=False))
    
    # Init variables
    sess.run(tf.global_variables_initializer())

    cost_val = []
      
    rand_ind = np.asarray(range(0,label1.shape[0]))
    np.random.shuffle(rand_ind)

    for epoch in range(0,label1.shape[0]):
        t = time.time()
        print("epoch",epoch)
        current_ind = np.asarray(rand_ind[epoch])
        
        features_train_feed = features[current_ind].todense()
    
        adj_par = adj[current_ind]
        
        eigvec_par,eigval_par = SGC(adj_par, adj[0].shape[0]-1)
        
        all_phrase_par = np.zeros(shape=[adj[0].shape[0],1])
        temp_embedding_ind = unique_word_index[current_ind,:]
        all_phrase_par[int(temp_embedding_ind[2]):int(temp_embedding_ind[3]+1),0] = 1
        all_phrase_par[int(temp_embedding_ind[4]):int(temp_embedding_ind[5]+1),0] = 2

        y_train = np.float32(label[current_ind,:]).reshape(1,-1)
        
        train_mask = np.full((1,1), True, dtype=bool).reshape(-1)

        feed_dict = construct_feed_dict_sgc(features_train_feed, eigvec_par, y_train, train_mask, all_phrase_par, eigval_par, placeholders)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})

        # Training step
        outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)


        # Print results
        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
              "train_acc=", "{:.5f}".format(outs[2]), "time=", "{:.5f}".format(time.time() - t))


print("Optimization Finished!")

outs_test_out_sum = np.empty(shape=[0,label.shape[1]])
y_test_sum = np.empty(shape=[0,label.shape[1]])

for i in range(label1.shape[0],label.shape[0]):
    
    print("i",i)
    current_ind = np.asarray(i)   
    
    features_test_feed = features[current_ind].todense()
    adj_par = adj[current_ind]
    
    eigvec_par = SGC(adj_par, adj[0].shape[0]-1)
    
    all_phrase_par = np.zeros(shape=[adj[0].shape[0],1])
    temp_embedding_ind = unique_word_index[current_ind,:]
    all_phrase_par[int(temp_embedding_ind[2]):int(temp_embedding_ind[3]+1),0] = 1
    all_phrase_par[int(temp_embedding_ind[4]):int(temp_embedding_ind[5]+1),0] = 2

    y_test = np.float32(label[current_ind,:]).reshape(1,-1)
    
    test_mask = np.full((1,1), True, dtype=bool).reshape(-1)

            
    # Testing
    feed_dict_test_sing = construct_feed_dict_sgc(features_test_feed, eigvec_par, y_test, test_mask, all_phrase_par, eigval_par, placeholders)
    outs_test_out = sess.run([model.outputs], feed_dict=feed_dict_test_sing)[0]
    outs_test_out_sum = np.vstack((outs_test_out_sum,outs_test_out))
    y_test_sum = np.vstack((y_test_sum, y_test))
                            
y_test_arg = y_test_sum.argmax(axis=1)
outs_test_out_arg = outs_test_out_sum.argmax(axis=1)


print(sklearn.metrics.precision_score(y_test_arg, outs_test_out_arg, average='micro'))
print(sklearn.metrics.recall_score(y_test_arg, outs_test_out_arg, average='micro'))
print(sklearn.metrics.f1_score(y_test_arg, outs_test_out_arg, average='micro'))
cm_tep = sklearn.metrics.confusion_matrix(y_test_arg, outs_test_out_arg)
np.save("/home/yld8809/cm_tep", cm_tep)
