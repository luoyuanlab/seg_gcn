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

flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4 , 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_float('weight_decay_wb', 5e-4 , 'Weight for L2 loss on W and b matrix.')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')
flags.DEFINE_integer('early_stopping', 100, 'Tolerance for early stopping (# of epochs).')

# Load data
adj = np.load('/home/yld8809/tp_adj_1_1_padded.npy')

features1 = np.load('/home/yld8809/word_embedding_tp_all_train.npy')
features2 = np.load('/home/yld8809/word_embedding_tp_all_test.npy')
features = np.vstack((features1,features2))
features1 = []
features2 = []
ind_dict = np.float32(features[:,1:7])

features = np.load('/home/yld8809/tp_features_padded.npy')


#pca = sklearn.decomposition.PCA(n_components=0.8)
#pca.fit(features1)


#features = pca.transform(features)

label1 = np.load('/home/yld8809/all_rel/tp_all_train.npy')[:,[2,8]]
label2 = np.load('/home/yld8809/all_rel/tp_all_test.npy')[:,[2,8]]
label = np.vstack((label1,label2))

label = lb.fit_transform(label[:,1])
# make tenp the first label
label = np.hstack((label[:,np.asarray(np.where(lb.classes_ == 'TrnP')).reshape(-1)],label[:,np.asarray(np.where(lb.classes_ != 'TrnP')).reshape(-1)]))

word_index = ind_dict
unique_word_index=[]; 
[unique_word_index.append(tuple(r)) for r in word_index if tuple(r) not in unique_word_index]; 
unique_word_index = np.asarray(unique_word_index);

rand_ind = np.asarray(range(0,label1.shape[0]))
np.random.shuffle(rand_ind)
num_batch = 100


num_coe = adj[0].shape[0]
# Define placeholders
placeholders = {
    'support': tf.placeholder(tf.float32, shape=tf.TensorShape([None, adj[0].shape[0], adj[0].shape[0]])),
    'all_phrase': tf.placeholder(tf.float32, shape=(None, adj[0].shape[0], 1)),                                      
    'features': tf.placeholder(tf.float32, shape=(None, adj[0].shape[0], features[0].shape[1])),
    'labels': tf.placeholder(tf.float32, shape=(None, label.shape[1])),
    'labels_mask': tf.placeholder(tf.int32),
    'dropout': tf.placeholder_with_default(0., shape=())
}

epoch_val = num_batch-1
current_ind_list = rand_ind[range(round((label1.shape[0]/num_batch)*epoch_val),round((label1.shape[0]/num_batch)*(epoch_val+1)))]
features_val_feed = np.zeros(shape=[current_ind_list.shape[0], adj[0].shape[0], features[0].shape[1]],dtype=np.float32)
support_val = np.zeros(shape=[current_ind_list.shape[0], adj[0].shape[0], adj[0].shape[0]],dtype=np.float32)
all_phrase_val = np.zeros(shape=[current_ind_list.shape[0], adj[0].shape[0], 1],dtype=np.float32)
y_val = np.zeros(shape=[current_ind_list.shape[0], label.shape[1]],dtype=np.float32)
val_mask = np.full((current_ind_list.shape[0],1), True, dtype=bool).reshape(-1)

for epoch in range(0,current_ind_list.shape[0]):
    current_ind = current_ind_list[epoch]

    features_val_feed[epoch,:,:] = features[current_ind].todense()

    support_val[epoch,:,:] = preprocess_adj(adj[current_ind])

    temp_embedding_ind = unique_word_index[current_ind,:]
    all_phrase_val[epoch, int(temp_embedding_ind[2]):int(temp_embedding_ind[3]+1),0] = 1
    all_phrase_val[epoch, int(temp_embedding_ind[4]):int(temp_embedding_ind[5]+1),0] = 2

    y_val[epoch,:] = np.float32(label[current_ind,:]).reshape(1,-1)
    
feed_dict_val = construct_feed_dict_sgc(features_val_feed, support_val, y_val, val_mask, all_phrase_val, placeholders)

with tf.device('/cpu:0'):
    
    # Create model
    model = GCN(placeholders, input_dim=features[0].shape[1], logging=False)
    

    sess = tf.Session(config=tf.ConfigProto(
      allow_soft_placement=True, log_device_placement=False))
    
    # Init variables
    sess.run(tf.global_variables_initializer())

    cost_val = []
    for epoch_h in range(0,100):   

        shuffle_ind_per_epoch = np.asarray(range(0,num_batch-1))
        np.random.shuffle(shuffle_ind_per_epoch)
    
        for epoch_m in shuffle_ind_per_epoch:
            t = time.time()
            print("epoch",epoch_m)
            current_ind_list = rand_ind[range(int(round((label1.shape[0]/num_batch)*epoch_m)),int(round((label1.shape[0]/num_batch)*(epoch_m+1))))]
            
            features_train_feed = np.zeros(shape=[current_ind_list.shape[0], adj[0].shape[0], features[0].shape[1]],dtype=np.float32)
            support_val = np.zeros(shape=[current_ind_list.shape[0], adj[0].shape[0], adj[0].shape[0]],dtype=np.float32)
            all_phrase_par = np.zeros(shape=[current_ind_list.shape[0], adj[0].shape[0], 1],dtype=np.float32)
            y_train = np.zeros(shape=[current_ind_list.shape[0], label.shape[1]],dtype=np.float32)
            train_mask = np.full((current_ind_list.shape[0],1), True, dtype=bool).reshape(-1)
            
            for epoch in range(0,current_ind_list.shape[0]):
                current_ind = current_ind_list[epoch]

                features_train_feed[epoch,:,:] = features[current_ind].todense()

                
                support_par[epoch,:,:] = preprocess_adj(adj[current_ind])

                
                temp_embedding_ind = unique_word_index[current_ind,:]
                all_phrase_par[epoch, int(temp_embedding_ind[2]):int(temp_embedding_ind[3]+1),0] = 1
                all_phrase_par[epoch, int(temp_embedding_ind[4]):int(temp_embedding_ind[5]+1),0] = 2

                y_train[epoch,:] = np.float32(label[current_ind,:]).reshape(1,-1)

                

            feed_dict = construct_feed_dict_sgc(features_train_feed, support_par, y_train, train_mask, all_phrase_par, placeholders)
            feed_dict.update({placeholders['dropout']: FLAGS.dropout})

            # Training step
            outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)

            # val step
            outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
            cost_val.append(outs_val[0])
            
            # Print results
            print("Epoch_h:", '%04d' % (epoch_h), "Epoch_m:", '%04d' % (epoch_m), "train_loss=", "{:.5f}".format(outs[1]),
                  "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(outs_val[0]), "time=", "{:.5f}".format(time.time() - t))


print("Optimization Finished!")

current_ind_list = np.asarray(range(label1.shape[0],label.shape[0]))

features_test_feed = np.zeros(shape=[current_ind_list.shape[0], adj[0].shape[0], features[0].shape[1]],dtype=np.float32)
support_test = np.zeros(shape=[current_ind_list.shape[0], adj[0].shape[0], adj[0].shape[0]],dtype=np.float32)
all_phrase_test = np.zeros(shape=[current_ind_list.shape[0], adj[0].shape[0], 1],dtype=np.float32)
y_test = np.zeros(shape=[current_ind_list.shape[0], label.shape[1]],dtype=np.float32)
test_mask = np.full((current_ind_list.shape[0],1), True, dtype=bool).reshape(-1)

for epoch in range(0,current_ind_list.shape[0]):
    current_ind = current_ind_list[epoch]

    features_test_feed[epoch,:,:] = features[current_ind].todense()

    support_test[epoch,:,:] = preprocess_adj(adj[current_ind])


    temp_embedding_ind = unique_word_index[current_ind,:]
    all_phrase_test[epoch, int(temp_embedding_ind[2]):int(temp_embedding_ind[3]+1),0] = 1
    all_phrase_test[epoch, int(temp_embedding_ind[4]):int(temp_embedding_ind[5]+1),0] = 2

    y_test[epoch,:] = np.float32(label[current_ind,:]).reshape(1,-1)


            
# Testing
feed_dict_test_sing = construct_feed_dict_sgc(features_test_feed, support_test, y_test, test_mask, all_phrase_test,                                          placeholders)

outs_test_out = sess.run([model.outputs], feed_dict=feed_dict_test_sing)[0]
    
                            
y_test_arg = y_test.argmax(axis=1)
outs_test_out_arg = outs_test_out.argmax(axis=1)


print(sklearn.metrics.precision_score(y_test_arg, outs_test_out_arg, average='micro'))
print(sklearn.metrics.recall_score(y_test_arg, outs_test_out_arg, average='micro'))
print(sklearn.metrics.f1_score(y_test_arg, outs_test_out_arg, average='micro'))
cm = sklearn.metrics.confusion_matrix(y_test_arg, outs_test_out_arg)

np.save("/home/yld8809/cm_tp", cm)
np.save("/home/yld8809/cost_val_tp",cost_val)
