from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf
import numpy as np
import scipy.sparse

from gcn.utils import *
from gcn.models import GCN, MLP

from sklearn import preprocessing
lb = preprocessing.LabelBinarizer()

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
#==============================================================================
# flags.DEFINE_string('dataset', 'cora', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
#==============================================================================
flags.DEFINE_string('model', 'gcn', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
#==============================================================================
# flags.DEFINE_integer('hidden1', 16, 'Number of units in hidden layer 1.')
#==============================================================================
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')

# Load data
adj1 = np.load('/home/yld8809/dep_mat_pp_all_train.npy')
adj1 = adj1.item()
adj2 = np.load('/home/yld8809/dep_mat_pp_all_test.npy')
adj2 = adj2.item()
adj = scipy.sparse.block_diag((adj1,adj2))
adj = scipy.sparse.coo_matrix(adj, dtype=np.float32)

features1 = np.load('/home/yld8809/word_embedding_pp_all_train.npy')
features2 = np.load('/home/yld8809/word_embedding_pp_all_test.npy')
features = np.vstack((features1,features2))
features1 = []
features2 = []

num_doc = int(max(np.float32(features[:,1])))+1

ind_dict = np.float32(features[:,1:7])
features = np.float32(features[:,7:])

label1 = np.load('/home/yld8809/all_rel/pp_all_train.npy')[:,[2,8]]
label2 = np.load('/home/yld8809/all_rel/pp_all_test.npy')[:,[2,8]]
label = np.vstack((label1,label2))

label_match = np.in1d(np.float32(label[:,0]),np.asarray(range(num_doc)))
label = label[label_match,:]
label = lb.fit_transform(label[:,1])
label = np.hstack((label, 1 - label))

y_train = np.zeros(label.shape)
y_val = np.zeros(label.shape)
y_test = np.zeros(label.shape)
y_train[0:round(label1.shape[0]*9/10),:] = label[0:round(label1.shape[0]*9/10),:]
y_val[round(label1.shape[0]*9/10):round(label1.shape[0]),:] = label[round(label1.shape[0]*9/10):round(label1.shape[0]),:]
y_test[round(label1.shape[0]):round(label.shape[0]),:] = label[round(label1.shape[0]):round(label.shape[0]),:]
   
train_mask = np.full((label.shape[0],1), False, dtype=bool)
val_mask = np.full((label.shape[0],1), False, dtype=bool)
test_mask = np.full((label.shape[0],1), False, dtype=bool)

train_mask[0:round(label1.shape[0]*9/10),0] = True
val_mask[round(label1.shape[0]*9/10):round(label1.shape[0]),0] = True
test_mask[round(label1.shape[0]):round(label.shape[0]),0] = True
          
train_mask = train_mask.reshape(-1)
test_mask = test_mask.reshape(-1)
val_mask = val_mask.reshape(-1)

      
y_test = np.float32(y_test)
y_train = np.float32(y_train)
y_val = np.float32(y_val)
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
    

#==============================================================================
#     all_size = np.vstack((all_size, np.shape(match_ind)[1]))
#     max_all_size = max(all_size) 
#==============================================================================

#==============================================================================
# mat = np.asarray([[0,1],[1,0]])
# support = [preprocess_adj(mat)]
#        
#==============================================================================
## Vali
val_ind = np.asarray(range(round(label1.shape[0]*9/10),round(label1.shape[0])))
matched_word_val_ind = [i for i, item in enumerate(all_size[:,0]) if item in val_ind]
features_val = features[matched_word_val_ind,:]
features_val = scipy.sparse.csr_matrix(features_val)
features_val = preprocess_features(features_val)
    
adj_val = scipy.sparse.coo_matrix(adj.tocsr()[matched_word_val_ind, :].tocsc()[:, matched_word_val_ind])

if FLAGS.model == 'gcn':
    support_val = [preprocess_adj(adj_val)]
    num_supports = 1
    model_func = GCN
elif FLAGS.model == 'gcn_cheby':
    support_val = chebyshev_polynomials(adj_val, FLAGS.max_degree)
    num_supports = 1 + FLAGS.max_degree
    model_func = GCN
else:
    raise ValueError('Invalid argument for model: ' + str(FLAGS.model))
        
all_phrase_val = all_phrase[matched_word_val_ind,:]
y_val = y_val[val_ind,:]
val_mask = val_mask[val_ind]

## test
test_ind = np.asarray(range(round(label1.shape[0]),round(label.shape[0])))
matched_word_test_ind = [i for i, item in enumerate(all_size[:,0]) if item in test_ind]

features_test = features[matched_word_test_ind,:]
features_test = scipy.sparse.csr_matrix(features_test)
features_test = preprocess_features(features_test)
    
adj_test = scipy.sparse.coo_matrix(adj.tocsr()[matched_word_test_ind, :].tocsc()[:, matched_word_test_ind])

if FLAGS.model == 'gcn':
    support_test = [preprocess_adj(adj_test)]
    num_supports = 1
    model_func = GCN
elif FLAGS.model == 'gcn_cheby':
    support_test = chebyshev_polynomials(adj_test, FLAGS.max_degree)
    num_supports = 1 + FLAGS.max_degree
    model_func = GCN
else:
    raise ValueError('Invalid argument for model: ' + str(FLAGS.model))
        
all_phrase_test = all_phrase[matched_word_test_ind,:]
y_test = y_test[test_ind,:]
test_mask = test_mask[test_ind]
                      
# Some preprocessing
support_feed = []
features_feed = []
all_phrase_feed = []
y_train_feed = []
train_mask_feed = []

num_batch = 50

for i in range(0,num_batch):
    current_ind = np.asarray(range((i*round(label1.shape[0]*9/10/num_batch)),((i+1)*round(label1.shape[0]*9/10/num_batch))))    
    matched_word_ind = [j for j, item in enumerate(all_size[:,0]) if item in current_ind]

    features_par = features[matched_word_ind,:]
    features_par = scipy.sparse.csr_matrix(features_par)
    features_par = preprocess_features(features_par)
    
    adj_par = scipy.sparse.coo_matrix(adj.tocsr()[matched_word_ind, :].tocsc()[:, matched_word_ind])

    if FLAGS.model == 'gcn':
        support_par = [preprocess_adj(adj_par)]
        num_supports = 1
        model_func = GCN
    elif FLAGS.model == 'gcn_cheby':
        support_par = chebyshev_polynomials(adj_par, FLAGS.max_degree)
        num_supports = 1 + FLAGS.max_degree
        model_func = GCN
    else:
        raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

    support_feed = support_feed + support_par
    features_feed = features_feed + list(features_par)
    
    
# Define placeholders
placeholders = {
    'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'all_phrase': tf.placeholder(tf.float32, shape=(None, all_phrase.shape[1])),                                      
    'features': tf.sparse_placeholder(tf.float32, shape=(None, features_val[2][1])),
    'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
    'labels_mask': tf.placeholder(tf.int32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
}

with tf.device('/cpu:0'):
    
    # Create model
    model = model_func(placeholders, input_dim=features_val[2][1], logging=False)

    # Define model evaluation function
    def evaluate(features, support, labels, mask, all_phrase, placeholders):
        t_test = time.time()
        feed_dict_val = construct_feed_dict(features, support, labels, mask, all_phrase, placeholders)
        outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
        return outs_val[0], outs_val[1], (time.time() - t_test)

    # Initialize session
    # sess = tf.Session(config=tf.ConfigProto(
    #       allow_soft_placement=True, log_device_placement=False))
    sess = tf.Session()
    # Init variables
    sess.run(tf.global_variables_initializer())

    cost_val = []
    #

    for epoch in range(0,num_batch):

        t = time.time()
        # Construct feed dictionary
        current_ind = np.asarray(range((epoch*round(label1.shape[0]*9/10/num_batch)),((epoch+1)*round(label1.shape[0]*9/10/num_batch))))    
        matched_word_ind = [epoch for epoch, item in enumerate(all_size[:,0]) if item in current_ind]

        all_phrase_par = all_phrase[matched_word_ind,:]
        y_train_par = y_train[current_ind,:]
        train_mask_par = train_mask[current_ind]

        features_par = tuple(features_feed[epoch*3:(epoch+1)*3])
        support_par = [tuple(support_feed[epoch])]

        feed_dict = construct_feed_dict(features_par, support_par, y_train_par, train_mask_par, all_phrase_par, placeholders)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})

        # Training step
        outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)

        # Validation
        cost, acc, duration = evaluate(features_val, support_val, y_val, val_mask, all_phrase_val, placeholders)
        cost_val.append(cost)

        # Print results
        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
              "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost),
              "val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))

#        if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
#            print("Early stopping...")
#            break    

print("Optimization Finished!")

# Testing
test_cost, test_acc, test_duration = evaluate(features_test, support_test, y_test, test_mask, all_phrase_test, placeholders)
print("Test set results:", "cost=", "{:.5f}".format(test_cost),
      "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))

print(model.outputs)

