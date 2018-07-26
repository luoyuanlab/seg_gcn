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
flags.DEFINE_integer('hidden1', 16, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')

# Load data
adj = np.float32(np.load('/home/yld8809/dep_mat_all_20.npy'))
features = np.load('/home/yld8809/word_embedding_all_20.npy')
num_doc = int(max(features[:,1]))+1

ind_dict = np.float32(features[:,1:7])
features = np.float32(features[:,7:])

label = np.load('/home/yld8809/beth_rel_label.npy')
label_match = np.in1d(np.float32(label[:,0]),np.asarray(range(num_doc)))
label = label[label_match,:]
label = lb.fit_transform(label[:,2])

y_train = np.zeros(label.shape)
y_val = np.zeros(label.shape)
y_test = np.zeros(label.shape)
y_train[0:round(label.shape[0]*7/10),:] = label[0:round(label.shape[0]*7/10),:]
y_val[round(label.shape[0]*7/10)+1:round(label.shape[0]*9/10),:] = label[round(label.shape[0]*7/10)+1:round(label.shape[0]*9/10),:]
y_test[round(label.shape[0]*9/10)+1:,:] = label[round(label.shape[0]*9/10)+1:,:]
   
train_mask = np.full((label.shape[0],1), False, dtype=bool)
val_mask = np.full((label.shape[0],1), False, dtype=bool)
test_mask = np.full((label.shape[0],1), False, dtype=bool)
train_mask[0:round(label.shape[0]*7/10),0] = True
val_mask[round(label.shape[0]*7/10)+1:round(label.shape[0]*9/10),0] = True
test_mask[round(label.shape[0]*9/10)+1:,0] = True

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

       
# Some preprocessing
features = scipy.sparse.csr_matrix(features)
features = preprocess_features(features)
if FLAGS.model == 'gcn':
    support = [preprocess_adj(adj)]
    num_supports = 1
    model_func = GCN
elif FLAGS.model == 'gcn_cheby':
    support = chebyshev_polynomials(adj, FLAGS.max_degree)
    num_supports = 1 + FLAGS.max_degree
    model_func = GCN
elif FLAGS.model == 'dense':
    support = [preprocess_adj(adj)]  # Not used
    num_supports = 1
    model_func = MLP
else:
    raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

# Define placeholders
placeholders = {
    'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'all_phrase': tf.placeholder(tf.float32, shape=(all_phrase.shape[0], all_phrase.shape[1])),
    'size_unique_word_index': tf.placeholder(tf.int32),                                         
    'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
    'labels': tf.placeholder(tf.float32, shape=(y_train.shape[0], y_train.shape[1])),
    'labels_mask': tf.placeholder(tf.int32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
}

# Create model
model = model_func(placeholders, input_dim=features[2][1], logging=False)


# Initialize session
sess = tf.Session()


# Define model evaluation function
def evaluate(features, support, labels, mask, all_phrase, size_unique_word_index, placeholders):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(features, support, labels, mask, all_phrase, size_unique_word_index, placeholders)
    outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], (time.time() - t_test)


# Init variables
sess.run(tf.global_variables_initializer())

cost_val = []

with tf.device("/gpu:0"):
# Train model
    for epoch in range(FLAGS.epochs):

        t = time.time()
        # Construct feed dictionary
        feed_dict = construct_feed_dict(features, support, y_train, train_mask, all_phrase, size_unique_word_index, placeholders)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})

        # Training step
        outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)

        # Validation
        cost, acc, duration = evaluate(features, support, y_val, val_mask, all_phrase, size_unique_word_index, placeholders)
        cost_val.append(cost)

        # Print results
        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
              "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost),
              "val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))

        if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
            print("Early stopping...")
            break

print("Optimization Finished!")

# Testing
test_cost, test_acc, test_duration = evaluate(features, support, y_test, test_mask, all_phrase, size_unique_word_index, placeholders)
print("Test set results:", "cost=", "{:.5f}".format(test_cost),
      "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))

print(model.outputs)

#==============================================================================
# 
# import numpy as np
# x = np.float32([[1,2,3],[4,5,6]])
# sess.run(tf.shape(x)[0])
# x = tf.reshape(x, [-1, 1, 2, 3])
# x = tf.nn.max_pool(x, ksize=[1, 1, 2, 1], strides=[1,1,2,1],
#                           padding='SAME')
# sess = tf.Session()
# sess.run(tf.boolean_mask(ind_dict,match_ind))
# sess.run(match_ind)
# 
# x1 = np.float32([[1,2,3],[4,5,6]])
# x1 = tf.reshape(x1, [-1, 1, 2, 3])
# x1 = tf.nn.max_pool(x1, ksize=[1, 1, 2, 1], strides=[1,1,2,1],
#                           padding='SAME')
# sess = tf.Session()
# sess.run(x1)
# 
# sess.run(tf.concat([x,x1],3))
# 
# word_index = word_embedding_all[:,1:7]
# unique_word_index=[]; 
# [unique_word_index.append(tuple(r)) for r in word_index if tuple(r) not in unique_word_index]; 
# unique_word_index = np.asarray(unique_word_index);
# 
# word_embedding_all = np.float32(word_embedding_all[:,1:7] )
# unique_word_index = np.float32(unique_word_index)
# 
# all_phrase = tf.convert_to_tensor(all_phrase)
# ind_dict = tf.convert_to_tensor(ind_dict)
# word_embedding_all = np.load(r'C:\Users\liyifu\Desktop\Northwestern\word_embedding_all_20.npy')
# word_embedding_all = tf.convert_to_tensor(np.float32(word_embedding_all[:,7:]))
# sent_ind = 0
# output = np.empty(shape=[0, 600])
# for sent_ind in range(0,len(unique_word_index)):
#     match_ind1 = tf.equal(all_phrase[:,0], (sent_ind+1)*2-1)
#     match_ind2 = tf.equal(all_phrase[:,0], (sent_ind+1)*2)

#     x1 = tf.boolean_mask(word_embedding_all,match_ind1)
#     x2 = tf.boolean_mask(word_embedding_all,match_ind2)
#                         
#     x1 = tf.reduce_max(x1, reduction_indices=[0])
#     x2 = tf.reduce_max(x2, reduction_indices=[0])
#     
#     x_concat = tf.concat([x1,x2],0) 
#     x_concat = tf.reshape(x_concat, [1, 600])  
#     output = tf.concat([output,x_concat],0)
#     
#     
# unique_word_index = np.vstack(set(map(tuple, word_index)))
# 
# 
#==============================================================================
