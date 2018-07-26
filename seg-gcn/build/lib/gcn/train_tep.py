from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf
import numpy as np
import scipy.sparse
import sklearn.metrics

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

flags.DEFINE_string('model', 'gcn', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4 , 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_float('weight_decay_wb', 5e-4 , 'Weight for L2 loss on W and b matrix.')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')
flags.DEFINE_integer('early_stopping', 50, 'Tolerance for early stopping (# of epochs).')

# Load data
adj1 = np.load('/home/yld8809/dep_mat_tep_all_train.npy')
adj1 = adj1.item()
adj2 = np.load('/home/yld8809/dep_mat_tep_all_test.npy')
adj2 = adj2.item()
adj = scipy.sparse.block_diag((adj1,adj2))
adj = scipy.sparse.coo_matrix(adj, dtype=np.float32)
adj.setdiag(2,k=-1)
adj.setdiag(2,k=1)


    
features1 = np.load('/home/yld8809/word_embedding_tep_all_train.npy')
features2 = np.load('/home/yld8809/word_embedding_tep_all_test.npy')
features = np.vstack((features1,features2))
features1 = []
features2 = []

num_doc = int(max(np.float32(features[:,1])))+1

ind_dict = np.float32(features[:,1:7])
features = np.float32(features[:,7:])

label1 = np.load('/home/yld8809/all_rel/tep_all_train.npy')[:,[2,8]]
label2 = np.load('/home/yld8809/all_rel/tep_all_test.npy')[:,[2,8]]
label = np.vstack((label1,label2))

label_match = np.in1d(np.float32(label[:,0]),np.asarray(range(num_doc)))
label = label[label_match,:]
label = lb.fit_transform(label[:,1])

y_train = label[0:label1.shape[0],:]
y_test = label[label1.shape[0]:label.shape[0],:]
   
train_mask = np.full((label1.shape[0],1), True, dtype=bool)
test_mask = np.full((label2.shape[0],1), True, dtype=bool)
          
train_mask = train_mask.reshape(-1)
test_mask = test_mask.reshape(-1)
      
y_test = np.float32(y_test)
y_train = np.float32(y_train)
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
    
# adj added with immediate neighborhood
for current_ind in range(0,label.shape[0]):
    matched_word_ind = [j for j, item in enumerate(all_size[:,0]) if item in np.asarray(current_ind)]
    
    adj[matched_word_ind, :][:, matched_word_ind].setdiag(1,k=-1)
    adj[matched_word_ind, :][:, matched_word_ind].setdiag(1,k=1)
    
    
adj = scipy.sparse.coo_matrix(adj)

## test
test_ind = np.asarray(range(label1.shape[0],label.shape[0]))
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
                      
# Some preprocessing
support_feed = []
features_feed = []
all_phrase_feed = []
y_train_feed = []
train_mask_feed = []

# random sampling of training data
num_batch = 50
rand_ind = np.asarray(range(0,label1.shape[0]))
np.random.shuffle(rand_ind)

for i in range(0,num_batch):
    current_ind = np.sort(np.asarray(rand_ind[round(i*(rand_ind.shape[0]/num_batch)):round((i+1)*(rand_ind.shape[0]/num_batch))]))    
    matched_word_ind = [j for j, item in enumerate(all_size[:,0]) if item in current_ind]

    features_par = features[matched_word_ind,:]
    features_par = scipy.sparse.csr_matrix(features_par)
    features_par = preprocess_features(features_par)
    
    adj_par = scipy.sparse.coo_matrix(adj.tocsr()[matched_word_ind, :].tocsc()[:, matched_word_ind])

    if FLAGS.model == 'gcn':
        support_par = [preprocess_adj(adj_par)]
        num_supports = 1
        model_func = GCN
        support_feed = support_feed + support_par
    elif FLAGS.model == 'gcn_cheby':
        support_par = chebyshev_polynomials(adj_par, FLAGS.max_degree)
        num_supports = 1 + FLAGS.max_degree
        model_func = GCN
        support_par = [support_par]
        support_feed = support_feed + support_par
    else:
        raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

    features_feed = features_feed + list(features_par)
    
    
# Define placeholders
placeholders = {
    'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'all_phrase': tf.placeholder(tf.float32, shape=(None, all_phrase.shape[1])),                                      
    'features': tf.sparse_placeholder(tf.float32, shape=(None, features_par[2][1])),
    'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
    'labels_mask': tf.placeholder(tf.int32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
}

with tf.device('/gpu:0'):
    
    # Create model
    model = model_func(placeholders, input_dim=features_par[2][1], logging=False)

    # Define model evaluation function
    def evaluate(features, support, labels, mask, all_phrase, placeholders):
        t_test = time.time()
        feed_dict_val = construct_feed_dict(features, support, labels, mask, all_phrase, placeholders)
        outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
        return outs_val[0], outs_val[1], (time.time() - t_test)

    # Initialize session
    # sess = tf.Session(config=tf.ConfigProto(
    #       allow_soft_placement=True, log_device_placement=False))
    sess = tf.Session(config=tf.ConfigProto(
      allow_soft_placement=True, log_device_placement=False))
    # Init variables
    sess.run(tf.global_variables_initializer())

    cost_val = []
    #

    for epoch in range(0,num_batch):

        t = time.time()
        # Construct feed dictionary
        current_ind = np.sort(np.asarray(rand_ind[round(epoch*(rand_ind.shape[0]/num_batch)):round((epoch+1)*(rand_ind.shape[0]/num_batch))]))    
        
        matched_word_ind = [i for i, item in enumerate(all_size[:,0]) if item in current_ind]

        all_phrase_par = all_phrase[matched_word_ind,:]
        unique_all_phrase_par=[]; 
        [unique_all_phrase_par.append(tuple(r)) for r in all_phrase_par if tuple(r) not in unique_all_phrase_par];
        unique_all_phrase_par = np.asarray(unique_all_phrase_par)
        unique_all_phrase_par = np.sort(unique_all_phrase_par[np.nonzero(unique_all_phrase_par)])
        for order in range(0,len(unique_all_phrase_par)):
            match_order_ind = np.where(all_phrase_par[:,0] == unique_all_phrase_par[order])[0]
            all_phrase_par[match_order_ind,:] = order+1
              
        y_train_par = y_train[current_ind,:]
        train_mask_par = train_mask[current_ind]

        features_par = tuple(features_feed[epoch*3:(epoch+1)*3])
        if FLAGS.model == 'gcn':
            support_par = [tuple(support_feed[epoch])]
        elif FLAGS.model == 'gcn_cheby':
            support_par = list(tuple(support_feed[epoch]))

        feed_dict = construct_feed_dict(features_par, support_par, y_train_par, train_mask_par, all_phrase_par, placeholders)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})

        # Training step
        outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)

        current_ind = np.sort(np.asarray(rand_ind[round((num_batch-1)*(rand_ind.shape[0]/num_batch)):round(num_batch*(rand_ind.shape[0]/num_batch))]))
        
        matched_word_ind = [i for i, item in enumerate(all_size[:,0]) if item in current_ind]

        all_phrase_val = all_phrase[matched_word_ind,:]
        unique_all_phrase_val=[]; 
        [unique_all_phrase_val.append(tuple(r)) for r in all_phrase_val if tuple(r) not in unique_all_phrase_val];
        unique_all_phrase_val = np.asarray(unique_all_phrase_val)
        unique_all_phrase_val = np.sort(unique_all_phrase_val[np.nonzero(unique_all_phrase_val)])
        
        for order in range(0,len(unique_all_phrase_val)):
            match_order_ind = np.where(all_phrase_val[:,0] == unique_all_phrase_val[order])[0]
            all_phrase_val[match_order_ind,:] = order+1
              
        y_train_val = y_train[current_ind,:]
        train_mask_val = train_mask[current_ind]
        
        features_val = tuple(features_feed[(num_batch-1)*3:(num_batch)*3])
        if FLAGS.model == 'gcn':
            support_val = [tuple(support_feed[num_batch-1])]
        elif FLAGS.model == 'gcn_cheby': 
            support_val = list(tuple(support_feed[num_batch-1]))
            
        # Validation
        cost, acc, duration = evaluate(features_val, support_val, y_train_val, train_mask_val, all_phrase_val, placeholders)
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
test_cost, test_acc, test_duration = evaluate(features_test, support_test, y_test, test_mask, all_phrase_test, placeholders)
print("Test set results:", "cost=", "{:.5f}".format(test_cost),
      "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))

print(model.outputs)

feed_dict_test_sing = construct_feed_dict(features_test, support_test, y_test, test_mask, all_phrase_test, placeholders)
outs_test_out = sess.run([model.outputs], feed_dict=feed_dict_test_sing)[0]

y_test_arg = y_test.argmax(axis=1)
outs_test_out_arg = outs_test_out.argmax(axis=1)


print(sklearn.metrics.precision_score(y_test_arg, outs_test_out_arg, average='micro'))
print(sklearn.metrics.recall_score(y_test_arg, outs_test_out_arg, average='micro'))
print(sklearn.metrics.f1_score(y_test_arg, outs_test_out_arg, average='micro'))

cm_tep = sklearn.metrics.confusion_matrix(y_test_arg, outs_test_out_arg)
np.save("/home/yld8809/cm_tep", cm_tep)