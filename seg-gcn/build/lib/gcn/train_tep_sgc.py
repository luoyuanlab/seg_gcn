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

def unique_ind(a):   
    b = np.ascontiguousarray(a).view(np.dtype((np.void, a.dtype.itemsize * a.shape[1])))
    _, idx = np.unique(b, return_index=True)
    idx = np.sort(idx)
    return idx

def cmPRF(cm, ncstart=1): # cm confusion matrix # jump over "noun"
    
    # calculate precision, recall and f-measure given result output
    # ncstart controls whether to include 
    nc, nc2 = cm.shape
    assert nc==nc2
    pres = np.zeros(nc); recs = np.zeros(nc); f1s = np.zeros(nc)

    tp_a = 0; fn_a = 0; fp_a = 0
    for c in range(ncstart,nc):
        tp = cm[c,c]; tp_a += tp
        mask = np.ones(nc,dtype=bool)
        mask[c] = 0 
        fn = np.sum( cm[c, mask] ); fn_a += fn
        fp = np.sum( cm[mask, c] ); fp_a += fp
        if tp+fp == 0:
            pre = 1
        else:
            pre = tp / (tp+fp)
        if tp+fn == 0:
            rec = 1
        else:
            rec = tp / (tp+fn)
        if pre+rec == 0:
            f = 0
        else:
            f = 2*pre*rec / (pre+rec)
        pres[c] = pre; recs[c] = rec; f1s[c] = f
    if tp_a+fp_a == 0:
        mipre = 1
    else:
        mipre = tp_a / (tp_a+fp_a)
    if tp_a+fn_a == 0:
        mirec = 1
    else:
        mirec = tp_a / (tp_a+fn_a)
    if mipre+mirec == 0:
        mif = 0
    else:
        mif = 2*mipre*mirec / (mipre+mirec)
    return (pres, recs, f1s, mipre, mirec, mif, cm)

flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay',  0, 'Weight for L2 loss on Fourier matrix.')
flags.DEFINE_float('weight_decay_wb', 0 , 'Weight for L2 loss on W and b matrix.')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')
flags.DEFINE_integer('early_stopping', 100, 'Tolerance for early stopping (# of epochs).')



label1 = np.load('/home/yld8809/all_rel/tep_all_train.npy')[:,[2,8]]
label1_info = np.float32(np.load('/home/yld8809/all_rel/tep_all_train.npy')[:,2:8])
idx_label1 = unique_ind(label1_info)
label1 = label1[idx_label1,:]

label2 = np.load('/home/yld8809/all_rel/tep_all_test.npy')[:,[2,8]]
label2_info = np.float32(np.load('/home/yld8809/all_rel/tep_all_test.npy')[:,2:8])
idx_label2 = unique_ind(label2_info)
label2 = label2[idx_label2,:]

unique_word_index = np.vstack((label1_info,label2_info))
idx_all = unique_ind(unique_word_index)
features = np.concatenate([np.load('/home/yld8809/tep_features_padded_train.npy'),np.load('/home/yld8809/tep_features_padded_test.npy')])
features = features[idx_all]
unique_word_index = unique_word_index[idx_all]

adj = np.concatenate([np.load('/home/yld8809/tep_adj_padded_train.npy'),np.load('/home/yld8809/tep_adj_padded_test.npy')])
adj = adj[idx_all]


all_zeros_ind = np.empty(shape=[0])
for i in range(0,features.shape[0]):
    if (np.all(features[i].todense()==0) == False):
        all_zeros_ind = np.hstack((all_zeros_ind,i))
        
all_zeros_ind = all_zeros_ind.astype(int)
train_ind_valid = np.where(np.in1d(all_zeros_ind, range(0,label1.shape[0])))[0]
test_ind_valid = np.where(np.in1d(all_zeros_ind, range(label1.shape[0],label1.shape[0]+label2.shape[0])))[0]-label1.shape[0]

features = features[all_zeros_ind]
adj = adj[all_zeros_ind]
unique_word_index = unique_word_index[all_zeros_ind]

label1 = label1[train_ind_valid]
label2 = label2[test_ind_valid]

label = np.vstack((label1,label2))

label = lb.fit_transform(label[:,1])
# make tenp the first label
label = np.hstack((label[:,np.asarray(np.where(lb.classes_ == 'TenP')).reshape(-1)],label[:,np.asarray(np.where(lb.classes_ != 'TenP')).reshape(-1)]))

#sum_weight = np.sum(label, axis=0)
#class_weight = label.sum()/sum_weight
#class_weight = class_weight/label.sum()
#class_weight = class_weight.reshape(-1,1)
class_weight = np.ones((label.shape[1],1))

rand_ind = np.asarray(range(0,label1.shape[0]))

np.random.shuffle(rand_ind)
num_batch = 50




num_coe = adj[0].shape[0]
# Define placeholders
placeholders = {
    'eigvec': tf.placeholder(tf.float32, shape=tf.TensorShape([None, adj[0].shape[0], num_coe])),
    'all_phrase': tf.placeholder(tf.float32, shape=(None, adj[0].shape[0], 1)),                                      
    'features': tf.placeholder(tf.float32, shape=(None, adj[0].shape[0], features[0].shape[1])),
    'labels': tf.placeholder(tf.float32, shape=(None, label.shape[1])),
    'labels_mask': tf.placeholder(tf.int32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'weights': tf.placeholder(tf.float32, shape=tf.TensorShape([label.shape[1],1]))
}

#features_sum = np.empty(shape=[0,300])
#current_ind_list = np.asarray(range(0,label1.shape[0]))
#for epoch in range(0,current_ind_list.shape[0]):
#    current_ind = current_ind_list[epoch]
#    features_sum = np.vstack((features_sum, features[current_ind].todense()))
#    
#mean_vec = np.mean(features_sum,axis=0)
#std_vec = np.std(features_sum,axis=0)
#    
    
epoch_val = int(num_batch/10*9-1)
current_ind_list = rand_ind[range(round((rand_ind.shape[0]/num_batch)*epoch_val),rand_ind.shape[0])]

features_val_feed = np.zeros(shape=[current_ind_list.shape[0], adj[0].shape[0], features[0].shape[1]],dtype=np.float32)
eigvec_val = np.zeros(shape=[current_ind_list.shape[0], adj[0].shape[0], num_coe],dtype=np.float32)
all_phrase_val = np.zeros(shape=[current_ind_list.shape[0], adj[0].shape[0], 1],dtype=np.float32)
y_val = np.zeros(shape=[current_ind_list.shape[0], label.shape[1]],dtype=np.float32)
val_mask = np.full((current_ind_list.shape[0],1), True, dtype=bool).reshape(-1)

for epoch in range(0,current_ind_list.shape[0]):
    current_ind = current_ind_list[epoch]

    features_val_feed[epoch,:,:] = features[current_ind].todense()

    adj_val = adj[current_ind]

    eigvec_val[epoch,:,:], _ = SGC(adj_val, num_coe)


    temp_embedding_ind = unique_word_index[current_ind,:]
    
    word_ind_sorted = np.asarray([int(temp_embedding_ind[2]),int(temp_embedding_ind[3]),int(temp_embedding_ind[4]),int(temp_embedding_ind[5])])
    word_ind_sorted = np.sort(word_ind_sorted)
    
    if word_ind_sorted[0] != 0:
        all_phrase_val[epoch, 0:word_ind_sorted[0],0] = 1
        
    all_phrase_val[epoch, int(temp_embedding_ind[2]):int(temp_embedding_ind[3]+1),0] = 2
    
    if word_ind_sorted[1]+1 != word_ind_sorted[2]:
        all_phrase_val[epoch, int(word_ind_sorted[1]+1):int(word_ind_sorted[2]),0] = 3
        
    all_phrase_val[epoch, int(temp_embedding_ind[4]):int(temp_embedding_ind[5]+1),0] = 4
    
    if word_ind_sorted[3] != adj[0].shape[0]-1:
        all_phrase_val[epoch, int(word_ind_sorted[3]+1):int(adj[0].shape[0]),0] = 5
    

    y_val[epoch,:] = np.float32(label[current_ind,:]).reshape(1,-1)
    
feed_dict_val = construct_feed_dict_sgc(features_val_feed, eigvec_val, y_val, val_mask, all_phrase_val,  class_weight, placeholders)

with tf.device('/cpu:0'):
    
    # Create model
    model = GCN(placeholders, input_dim=features[0].shape[1], logging=False)
    

    sess = tf.Session(config=tf.ConfigProto(
      allow_soft_placement=True, log_device_placement=False))
    
    # Init variables
    sess.run(tf.global_variables_initializer())

    cost_val_f1 = []
    cost_val_loss = []
    
    for epoch_h in range(0,50):   
        t = time.time()
        shuffle_ind_per_epoch = np.asarray(range(0,int(num_batch/10*9-1)))
        np.random.shuffle(shuffle_ind_per_epoch)
    
        for epoch_m in shuffle_ind_per_epoch:
            
            
            current_ind_list = rand_ind[range(int(round((rand_ind.shape[0]/num_batch)*epoch_m)),int(round((rand_ind.shape[0]/num_batch)*(epoch_m+1))))]
            
            features_train_feed = np.zeros(shape=[current_ind_list.shape[0], adj[0].shape[0], features[0].shape[1]],dtype=np.float32)
            eigvec_par = np.zeros(shape=[current_ind_list.shape[0], adj[0].shape[0], num_coe],dtype=np.float32)
            all_phrase_par = np.zeros(shape=[current_ind_list.shape[0], adj[0].shape[0], 1],dtype=np.float32)
            y_train = np.zeros(shape=[current_ind_list.shape[0], label.shape[1]],dtype=np.float32)
            train_mask = np.full((current_ind_list.shape[0],1), True, dtype=bool).reshape(-1)
            
            for epoch in range(0,current_ind_list.shape[0]):
                current_ind = current_ind_list[epoch]

                features_train_feed[epoch,:,:] = features[current_ind].todense()

                adj_par = adj[current_ind]

                eigvec_par[epoch,:,:], _ = SGC(adj_par, num_coe)

                
                temp_embedding_ind = unique_word_index[current_ind,:]
                
                word_ind_sorted = np.asarray([int(temp_embedding_ind[2]),int(temp_embedding_ind[3]),int(temp_embedding_ind[4]),int(temp_embedding_ind[5])])
                word_ind_sorted = np.sort(word_ind_sorted)

                
                if word_ind_sorted[0] != 0:
                    all_phrase_par[epoch, 0:word_ind_sorted[0],0] = 1

                all_phrase_par[epoch, int(temp_embedding_ind[2]):int(temp_embedding_ind[3]+1),0] = 2

                if word_ind_sorted[1]+1 != word_ind_sorted[2]:
                    all_phrase_par[epoch, int(word_ind_sorted[1]+1):int(word_ind_sorted[2]),0] = 3

                all_phrase_par[epoch, int(temp_embedding_ind[4]):int(temp_embedding_ind[5]+1),0] = 4

                if word_ind_sorted[3] != adj[0].shape[0]-1:
                    all_phrase_par[epoch, int(word_ind_sorted[3]+1):int(adj[0].shape[0]),0] = 5


                y_train[epoch,:] = np.float32(label[current_ind,:]).reshape(1,-1)

                

            feed_dict = construct_feed_dict_sgc(features_train_feed, eigvec_par, y_train, train_mask, all_phrase_par, class_weight, placeholders)
            feed_dict.update({placeholders['dropout']: FLAGS.dropout})

            # Training step
            outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)
            
            # val step
        outs_val = sess.run([model.loss, model.accuracy, model.outputs], feed_dict=feed_dict_val)
        

        cm = sklearn.metrics.confusion_matrix(y_val.argmax(axis=1), outs_val[2].argmax(axis=1))
        cost_val_f1.append(cmPRF(cm,ncstart=1)[5])
        cost_val_loss.append(outs_val[0])
        
            # Print results
        print("Epoch:", '%04d' % (epoch_h), "train_loss=", "{:.5f}".format(outs[1]),
                  "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(outs_val[0]), "val_F1=", "{:.5f}".format(cmPRF(cm,ncstart=1)[5]),"time=", "{:.5f}".format(time.time() - t))


print("Optimization Finished!")

current_ind_list = np.asarray(range(label1.shape[0],label.shape[0]))

features_test_feed = np.zeros(shape=[current_ind_list.shape[0], adj[0].shape[0], features[0].shape[1]],dtype=np.float32)
adj_test = np.zeros(shape=[current_ind_list.shape[0], adj[0].shape[0], adj[0].shape[0]],dtype=np.float32)
eigvec_test = np.zeros(shape=[current_ind_list.shape[0], adj[0].shape[0], num_coe],dtype=np.float32)
all_phrase_test = np.zeros(shape=[current_ind_list.shape[0], adj[0].shape[0], 1],dtype=np.float32)
y_test = np.zeros(shape=[current_ind_list.shape[0], label.shape[1]],dtype=np.float32)
test_mask = np.full((current_ind_list.shape[0],1), True, dtype=bool).reshape(-1)

for epoch in range(0,current_ind_list.shape[0]):
    current_ind = current_ind_list[epoch]

    features_test_feed[epoch,:,:] = features[current_ind].todense()

    adj_test = adj[current_ind]

    eigvec_test[epoch,:,:],_ = SGC(adj_test, num_coe)


    temp_embedding_ind = unique_word_index[current_ind,:]
    
    word_ind_sorted = np.asarray([int(temp_embedding_ind[2]),int(temp_embedding_ind[3]),int(temp_embedding_ind[4]),int(temp_embedding_ind[5])])
    word_ind_sorted = np.sort(word_ind_sorted)
    
    if word_ind_sorted[0] != 0:
        all_phrase_test[epoch, 0:word_ind_sorted[0],0] = 1
        
    all_phrase_test[epoch, int(temp_embedding_ind[2]):int(temp_embedding_ind[3]+1),0] = 2
    
    if word_ind_sorted[1]+1 != word_ind_sorted[2]:
        all_phrase_test[epoch, int(word_ind_sorted[1]+1):int(word_ind_sorted[2]),0] = 3
        
    all_phrase_test[epoch, int(temp_embedding_ind[4]):int(temp_embedding_ind[5]+1),0] = 4
    
    if word_ind_sorted[3] != adj[0].shape[0]-1:
        all_phrase_test[epoch, int(word_ind_sorted[3]+1):int(adj[0].shape[0]),0] = 5
    
    y_test[epoch,:] = np.float32(label[current_ind,:]).reshape(1,-1)


            
# Testing
feed_dict_test_sing = construct_feed_dict_sgc(features_test_feed, eigvec_test, y_test, test_mask, all_phrase_test, class_weight,
                                              placeholders)

outs_test_out = sess.run([model.outputs], feed_dict=feed_dict_test_sing)[0]
    
                            
y_test_arg = y_test.argmax(axis=1)
outs_test_out_arg = outs_test_out.argmax(axis=1)

cm = sklearn.metrics.confusion_matrix(y_test_arg, outs_test_out_arg)


print("f1:", "{:.5f}".format(cmPRF(cm,ncstart=1)[5]))

np.save("/home/yld8809/cm_tep_sgc", cm)
np.save("/home/yld8809/cost_val_f1_tep_sgc",cost_val_f1)
np.save("/home/yld8809/cost_val_loss_tep_sgc",cost_val_loss)