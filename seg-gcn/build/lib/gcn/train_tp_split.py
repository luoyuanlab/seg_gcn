from __future__ import division
from __future__ import print_function

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

flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.') # use 0.001 for now please!!!!
flags.DEFINE_float('dropout', 0.0, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('epsilon', 1e-8, 'epsilon')
flags.DEFINE_float('weight_decay', 2e-1, 'Weight for L2 loss on Fourier matrix.')
flags.DEFINE_float('weight_decay_wb', 1e-1, 'Weight for L2 loss on W and b matrix.')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')
flags.DEFINE_integer('early_stopping', 35, 'Tolerance for early stopping (# of epochs).')

label1 = np.load('/home/yld8809/all_rel/tp_all_train.npy')[:,[2,8]]
label1_info = np.float32(np.load('/home/yld8809/all_rel/tp_all_train.npy')[:,2:8])

label2 = np.load('/home/yld8809/all_rel/tp_all_test.npy')[:,[2,8]]
label2_info = np.float32(np.load('/home/yld8809/all_rel/tp_all_test.npy')[:,2:8])

unique_word_index = np.vstack((label1_info,label2_info))
features = np.concatenate([np.load('/home/yld8809/tp_features_padded_train.npy'),np.load('/home/yld8809/tp_features_padded_test.npy')])


adj = np.concatenate([np.load('/home/yld8809/tp_adj_padded_train.npy'),np.load('/home/yld8809/tp_adj_padded_test.npy')])


sentence_length = np.concatenate([np.load('/home/yld8809/sentence_length_tuple_train_tp.npy'),np.load('/home/yld8809/sentence_length_tuple_test_tp.npy')])


label = np.vstack((label1,label2))

label = lb.fit_transform(label[:,1])
# make tenp the first label
label = np.hstack((label[:,np.asarray(np.where(lb.classes_ == 'TrnP')).reshape(-1)],label[:,np.asarray(np.where(lb.classes_ != 'TrnP')).reshape(-1)]))

#sum_weight = np.sum(label, axis=0)
#class_weight = label.sum()/sum_weight
#class_weight = class_weight/label.sum()
#class_weight = class_weight.reshape(-1,1)
class_weight = np.ones((label.shape[1],1))
class_weight[0]=1

noise_index = np.where(label[0:label1.shape[0],0]==1)
true_index = np.where(label[0:label1.shape[0],0]!=1)
noise_index = noise_index[0]
true_index = true_index[0]
np.random.shuffle(noise_index)
rand_ind = np.hstack((true_index,noise_index[0:int(true_index.shape[0]*0.3)]))

#rand_ind = np.asarray(range(0,label1.shape[0]))                         
np.random.shuffle(rand_ind)
num_batch = 50


#features_sum = np.empty(shape=[0,300])
#current_ind_list = np.asarray(range(0,label1.shape[0]))
#for epoch in range(0,current_ind_list.shape[0]):
#    current_ind = current_ind_list[epoch]
#    features_sum = np.vstack((features_sum, features[current_ind].todense()))
#    
#mean_vec = np.mean(features_sum,axis=0)
#std_vec = np.std(features_sum,axis=0)



#    in progress
max_padding_size_1 = 0
max_padding_size_2 = 0
max_padding_size_3 = 0
max_padding_size_5 = 0
for epoch in range(0,label.shape[0]):
    current_ind = epoch
    
    temp_embedding_ind = unique_word_index[current_ind,:]
    
    word_ind_sorted = np.asarray([int(temp_embedding_ind[2]),int(temp_embedding_ind[3]),int(temp_embedding_ind[4]),int(temp_embedding_ind[5])])
    word_ind_sorted = np.sort(word_ind_sorted)
    
    if word_ind_sorted[0] != 0:
        max_padding_size_1 = max(max_padding_size_1,word_ind_sorted[0]+1)
        
    max_padding_size_2 = max(max_padding_size_2, int(temp_embedding_ind[3])-int(temp_embedding_ind[2])+1)
    
    if word_ind_sorted[1]+1 != word_ind_sorted[2]:
        max_padding_size_3 = max(max_padding_size_3,int(word_ind_sorted[2])-int(word_ind_sorted[1]-1))
        
    max_padding_size_2 = max(max_padding_size_2, int(temp_embedding_ind[5])-int(temp_embedding_ind[4]+1))
    
    if word_ind_sorted[3] != adj[0].shape[0]-1:
        max_padding_size_5 = max(max_padding_size_5, int(adj[0].shape[0])-int(word_ind_sorted[3])-1)
        
max_padding_size_1 = int(max_padding_size_1)
max_padding_size_2 = int(max_padding_size_2)
max_padding_size_3 = int(max_padding_size_3)
max_padding_size_5 = int(max_padding_size_5)
starting_late = max_padding_size_1 #how much to include for first component
ending_early = max_padding_size_5 #when to end for last component
#    in progress         

flags.DEFINE_integer('hidden_layer1', features[0].shape[1], 'hidden layer 1.')
for epoch in range(0,label.shape[0]):
    current_ind = epoch
    
    temp_embedding_ind = unique_word_index[current_ind,:]
    
    word_ind_sorted = np.asarray([int(temp_embedding_ind[2]),int(temp_embedding_ind[3]),int(temp_embedding_ind[4]),int(temp_embedding_ind[5])])
    word_ind_sorted = np.sort(word_ind_sorted)    
    max_size = max_padding_size_1+max_padding_size_2+max_padding_size_3+max_padding_size_2+max_padding_size_5
    
    features_mat = np.zeros(shape=[max_size,features[0].shape[1]]) 
    adj_underlying = adj[current_ind]
    sentence_length_underlying = sentence_length[current_ind]
    dep_mat = np.zeros(shape=[max_size,adj_underlying.shape[0]]) 
    
    # add diagnal as 1
    adj_neightborhood = sp.eye(adj_underlying.shape[0])
    adj_neightborhood.setdiag(30,0)
#    adj_neightborhood = sp.eye(adj_underlying.shape[0])
#    adj_neightborhood.setdiag(4,0)

    
    adj_underlying = adj_underlying.todense()
    adj_neightborhood = adj_neightborhood.todense()
    
#    if sentence_length_underlying < 50:
    adj_underlying = adj_underlying + adj_neightborhood
#    else:
#        adj_underlying = adj_neightborhood
    

#    adj_underlying = adj_underlying + np.diag(np.array(adj_underlying.sum(1))) + np.eye(adj_underlying.shape[0])

    
    features_underlying = features[current_ind].todense()
    
    if word_ind_sorted[0] != 0:
        dep_mat[(max_padding_size_1-word_ind_sorted[0]):max_padding_size_1, :] = adj_underlying[0:word_ind_sorted[0], :]
        features_mat[(max_padding_size_1-word_ind_sorted[0]):max_padding_size_1, :] = features_underlying[0:word_ind_sorted[0],:]
        
    dep_mat[int(max_padding_size_1):int(max_padding_size_1+word_ind_sorted[1]-word_ind_sorted[0]+1),:] = adj_underlying[int(word_ind_sorted[0]):int(word_ind_sorted[1]+1), :]
    features_mat[int(max_padding_size_1):int(max_padding_size_1+word_ind_sorted[1]-word_ind_sorted[0]+1), :] = features_underlying[int(word_ind_sorted[0]):int(word_ind_sorted[1]+1),:]
    
    if word_ind_sorted[1]+1 != word_ind_sorted[2]:
        current_padding_loca = max_padding_size_1 + max_padding_size_2
        dep_mat[int(current_padding_loca):int(current_padding_loca+word_ind_sorted[2]-word_ind_sorted[1]-1),:] = adj_underlying[int(word_ind_sorted[1])+1:int(word_ind_sorted[2]), :]
        features_mat[int(current_padding_loca):int(current_padding_loca+word_ind_sorted[2]-word_ind_sorted[1]-1), :] = features_underlying[int(word_ind_sorted[1])+1:int(word_ind_sorted[2]),:]
        
    current_padding_loca = max_padding_size_1 + max_padding_size_2 + max_padding_size_3   
    dep_mat[int(current_padding_loca):int(current_padding_loca+word_ind_sorted[3]-word_ind_sorted[2]+1),:] = adj_underlying[int(word_ind_sorted[2]):int(word_ind_sorted[3]+1), :]
    features_mat[int(current_padding_loca):int(current_padding_loca+word_ind_sorted[3]-word_ind_sorted[2]+1), :] = features_underlying[int(word_ind_sorted[2]):int(word_ind_sorted[3]+1),:]
    
    if word_ind_sorted[3] != adj_underlying.shape[0]-1:
        current_padding_loca = max_padding_size_1 + max_padding_size_2 + max_padding_size_3 + max_padding_size_2
        dep_mat[int(current_padding_loca):int(current_padding_loca+adj_underlying.shape[0]-word_ind_sorted[3]-1),:] = adj_underlying[int(word_ind_sorted[3]+1):int(adj_underlying.shape[0]), :]
        features_mat[int(current_padding_loca):int(current_padding_loca+adj_underlying.shape[0]-word_ind_sorted[3]-1), :] = features_underlying[int(word_ind_sorted[3]+1):int(adj_underlying.shape[0]),:]

# another round
    dep_mat_final = np.zeros(shape=[max_size,max_size])
    
    if word_ind_sorted[0] != 0:
        dep_mat_final[:, (max_padding_size_1-word_ind_sorted[0]):max_padding_size_1] = dep_mat[:, 0:word_ind_sorted[0]]
        
    dep_mat_final[:,int(max_padding_size_1):int(max_padding_size_1+word_ind_sorted[1]-word_ind_sorted[0]+1)] = dep_mat[:,int(word_ind_sorted[0]):int(word_ind_sorted[1]+1)]
    
    if word_ind_sorted[1]+1 != word_ind_sorted[2]:
        current_padding_loca = max_padding_size_1 + max_padding_size_2
        dep_mat_final[:, int(current_padding_loca):int(current_padding_loca+word_ind_sorted[2]-word_ind_sorted[1]-1)] = dep_mat[:, int(word_ind_sorted[1])+1:int(word_ind_sorted[2])]
        
    current_padding_loca = max_padding_size_1 + max_padding_size_2 + max_padding_size_3   
    dep_mat_final[:,int(current_padding_loca):int(current_padding_loca+word_ind_sorted[3]-word_ind_sorted[2]+1)] = dep_mat[:,int(word_ind_sorted[2]):int(word_ind_sorted[3]+1)]
    
    if word_ind_sorted[3] != adj[0].shape[0]-1:
        current_padding_loca = max_padding_size_1 + max_padding_size_2 + max_padding_size_3 + max_padding_size_2
        dep_mat_final[:,int(current_padding_loca):int(current_padding_loca+adj_underlying.shape[0]-word_ind_sorted[3]-1)] = dep_mat[:,int(word_ind_sorted[3]+1):int(adj_underlying.shape[0])]
        
    features[current_ind] = scipy.sparse.coo_matrix(features_mat)
    adj[current_ind] = scipy.sparse.coo_matrix(dep_mat_final)
    

# Define placeholders
placeholders = {
    'eigvec': tf.placeholder(tf.float32, shape=tf.TensorShape([None, adj[0].shape[0], adj[0].shape[0]])),
    'all_phrase': tf.placeholder(tf.float32, shape=(None, adj[0].shape[0], 1)),                                      
    'features': tf.placeholder(tf.float32, shape=(None, adj[0].shape[0], features[0].shape[1])),
    'labels': tf.placeholder(tf.float32, shape=(None, label.shape[1])),
    'labels_mask': tf.placeholder(tf.int32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'weights': tf.placeholder(tf.float32, shape=tf.TensorShape([label.shape[1],1]))
}
                               
epoch_val = int(num_batch/10*9-1)
current_ind_list = rand_ind[range(round((rand_ind.shape[0]/num_batch)*epoch_val),rand_ind.shape[0])]

features_val_feed = np.zeros(shape=[current_ind_list.shape[0], adj[0].shape[0], features[0].shape[1]],dtype=np.float32)
eigvec_val = np.zeros(shape=[current_ind_list.shape[0], adj[0].shape[0], adj[0].shape[0]],dtype=np.float32)
all_phrase_val = np.zeros(shape=[current_ind_list.shape[0], adj[0].shape[0], 1],dtype=np.float32)
y_val = np.zeros(shape=[current_ind_list.shape[0], label.shape[1]],dtype=np.float32)
val_mask = np.full((current_ind_list.shape[0],1), True, dtype=bool).reshape(-1)

for epoch in range(0,current_ind_list.shape[0]):
    current_ind = current_ind_list[epoch]

    features_val_feed[epoch,:,:] = features[current_ind].todense()

    temp_embedding_ind = unique_word_index[current_ind,:]
    
    word_ind_sorted = np.asarray([int(temp_embedding_ind[2]),int(temp_embedding_ind[3]),int(temp_embedding_ind[4]),int(temp_embedding_ind[5])])
    word_ind_sorted = np.sort(word_ind_sorted)
    
    all_phrase_val[epoch, (max_padding_size_1-starting_late):max_padding_size_1,0] = 1
    
    if word_ind_sorted[0] == temp_embedding_ind[2]:
        all_phrase_val[epoch, max_padding_size_1:(max_padding_size_1+max_padding_size_2),0] = 2
        all_phrase_val[epoch, (max_padding_size_1+max_padding_size_2+max_padding_size_3):(max_padding_size_1+max_padding_size_2+max_padding_size_3+max_padding_size_2),0] = 4
    else:
        all_phrase_val[epoch, max_padding_size_1:(max_padding_size_1+max_padding_size_2),0] = 4
        all_phrase_val[epoch, (max_padding_size_1+max_padding_size_2+max_padding_size_3):(max_padding_size_1+max_padding_size_2+max_padding_size_3+max_padding_size_2),0] = 2
        
    
    all_phrase_val[epoch, (max_padding_size_1+max_padding_size_2):(max_padding_size_1+max_padding_size_2+max_padding_size_3),0] = 3
        
    
    all_phrase_val[epoch, (max_padding_size_1+max_padding_size_2+max_padding_size_3+max_padding_size_2):(max_padding_size_1+max_padding_size_2+max_padding_size_3+max_padding_size_2+ending_early),0] = 5
    

    y_val[epoch,:] = np.float32(label[current_ind,:]).reshape(1,-1)
    

    eigvec_val[epoch,:,:] = preprocess_adj(adj[current_ind])
    
feed_dict_val = construct_feed_dict_sgc(features_val_feed, eigvec_val, y_val, val_mask, all_phrase_val,  class_weight, placeholders)

current_ind_list = np.asarray(range(label1.shape[0],label.shape[0]))

features_test_feed = np.zeros(shape=[current_ind_list.shape[0], adj[0].shape[0], features[0].shape[1]],dtype=np.float32)
adj_test = np.zeros(shape=[current_ind_list.shape[0], adj[0].shape[0], adj[0].shape[0]],dtype=np.float32)
eigvec_test = np.zeros(shape=[current_ind_list.shape[0], adj[0].shape[0], adj[0].shape[0]],dtype=np.float32)
all_phrase_test = np.zeros(shape=[current_ind_list.shape[0], adj[0].shape[0], 1],dtype=np.float32)
y_test = np.zeros(shape=[current_ind_list.shape[0], label.shape[1]],dtype=np.float32)
test_mask = np.full((current_ind_list.shape[0],1), True, dtype=bool).reshape(-1)

for epoch in range(0,current_ind_list.shape[0]):
    current_ind = current_ind_list[epoch]

    features_test_feed[epoch,:,:] = features[current_ind].todense()

    temp_embedding_ind = unique_word_index[current_ind,:]
    
    word_ind_sorted = np.asarray([int(temp_embedding_ind[2]),int(temp_embedding_ind[3]),int(temp_embedding_ind[4]),int(temp_embedding_ind[5])])
    word_ind_sorted = np.sort(word_ind_sorted)
    
    all_phrase_test[epoch, (max_padding_size_1-starting_late):max_padding_size_1,0] = 1
    
    if word_ind_sorted[0] == temp_embedding_ind[2]:
        all_phrase_test[epoch, max_padding_size_1:(max_padding_size_1+max_padding_size_2),0] = 2
        all_phrase_test[epoch, (max_padding_size_1+max_padding_size_2+max_padding_size_3):(max_padding_size_1+max_padding_size_2+max_padding_size_3+max_padding_size_2),0] = 4
    else:
        all_phrase_test[epoch, max_padding_size_1:(max_padding_size_1+max_padding_size_2),0] = 4
        all_phrase_test[epoch, (max_padding_size_1+max_padding_size_2+max_padding_size_3):(max_padding_size_1+max_padding_size_2+max_padding_size_3+max_padding_size_2),0] = 2
        
    
    
    all_phrase_test[epoch, (max_padding_size_1+max_padding_size_2):(max_padding_size_1+max_padding_size_2+max_padding_size_3),0] = 3
        
    
    
    all_phrase_test[epoch, (max_padding_size_1+max_padding_size_2+max_padding_size_3+max_padding_size_2):(max_padding_size_1+max_padding_size_2+max_padding_size_3+max_padding_size_2+ending_early),0] = 5
    
    y_test[epoch,:] = np.float32(label[current_ind,:]).reshape(1,-1)
    
    eigvec_test[epoch,:,:] = preprocess_adj(adj[current_ind])
    
# Testing
feed_dict_test_sing = construct_feed_dict_sgc(features_test_feed, eigvec_test, y_test, test_mask, all_phrase_test, class_weight,
                                              placeholders)

with tf.device('/cpu:0'):
    
    # Create model
    model = GCN(placeholders, input_dim=features[0].shape[1], logging=False)
    

    sess = tf.Session(config=tf.ConfigProto(
      allow_soft_placement=True, log_device_placement=False))
    
    # Init variables
    sess.run(tf.global_variables_initializer())

    cost_val_f1 = []
    cost_val_loss = []
    test_f1 = []
    test_loss = []
    
    for epoch_h in range(0,100):   
        t = time.time()
        shuffle_ind_per_epoch = np.asarray(range(0,int(num_batch/10*9-1)))
        np.random.shuffle(shuffle_ind_per_epoch)
        
        count_m = 0
        for epoch_m in shuffle_ind_per_epoch:
            
            count_m = count_m + 1
  
            current_ind_list = rand_ind[range(int(round((rand_ind.shape[0]/num_batch)*epoch_m)),int(round((rand_ind.shape[0]/num_batch)*(epoch_m+1))))]
            
            features_train_feed = np.zeros(shape=[current_ind_list.shape[0], adj[0].shape[0], features[0].shape[1]],dtype=np.float32)
            eigvec_train = np.zeros(shape=[current_ind_list.shape[0], adj[0].shape[0], adj[0].shape[0]],dtype=np.float32)
            all_phrase_par = np.zeros(shape=[current_ind_list.shape[0], adj[0].shape[0], 1],dtype=np.float32)
            y_train = np.zeros(shape=[current_ind_list.shape[0], label.shape[1]],dtype=np.float32)
            train_mask = np.full((current_ind_list.shape[0],1), True, dtype=bool).reshape(-1)
            
            for epoch in range(0,current_ind_list.shape[0]):
                current_ind = current_ind_list[epoch]

                features_train_feed[epoch,:,:] = features[current_ind].todense()
                
                temp_embedding_ind = unique_word_index[current_ind,:]
                
                word_ind_sorted = np.asarray([int(temp_embedding_ind[2]),int(temp_embedding_ind[3]),int(temp_embedding_ind[4]),int(temp_embedding_ind[5])])
                word_ind_sorted = np.sort(word_ind_sorted)

                
                all_phrase_par[epoch, (max_padding_size_1-starting_late):max_padding_size_1,0] = 1

                if word_ind_sorted[0] == temp_embedding_ind[2]:
                    all_phrase_par[epoch, max_padding_size_1:(max_padding_size_1+max_padding_size_2),0] = 2
                    all_phrase_par[epoch, (max_padding_size_1+max_padding_size_2+max_padding_size_3):(max_padding_size_1+max_padding_size_2+max_padding_size_3+max_padding_size_2),0] = 4
                else:
                    all_phrase_par[epoch, max_padding_size_1:(max_padding_size_1+max_padding_size_2),0] = 4
                    all_phrase_par[epoch, (max_padding_size_1+max_padding_size_2+max_padding_size_3):(max_padding_size_1+max_padding_size_2+max_padding_size_3+max_padding_size_2),0] = 2


                
                all_phrase_par[epoch, (max_padding_size_1+max_padding_size_2):(max_padding_size_1+max_padding_size_2+max_padding_size_3),0] = 3


                
                all_phrase_par[epoch, (max_padding_size_1+max_padding_size_2+max_padding_size_3+max_padding_size_2):(max_padding_size_1+max_padding_size_2+max_padding_size_3+max_padding_size_2+ending_early),0] = 5



                y_train[epoch,:] = np.float32(label[current_ind,:]).reshape(1,-1)
                
                eigvec_train[epoch,:,:] = preprocess_adj(adj[current_ind])

            feed_dict = construct_feed_dict_sgc(features_train_feed, eigvec_train, y_train, train_mask, all_phrase_par, class_weight, placeholders)
            feed_dict.update({placeholders['dropout']: FLAGS.dropout})

            # Training step
            outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)
            
            # val step
            if (count_m % 10 == 0):
                outs_val_out = sess.run([model.loss, model.accuracy, model.outputs], feed_dict=feed_dict_val)


                cm = sklearn.metrics.confusion_matrix(y_val.argmax(axis=1), outs_val_out[2].argmax(axis=1))

                cost_val_f1.append(cmPRF(cm,ncstart=1)[5])
                cost_val_loss.append(outs_val_out[0])

                    # Print results
        #        if cost_val_loss[-1] == min(cost_val_loss):
                outs_test_out = sess.run([model.loss, model.accuracy, model.outputs], feed_dict=feed_dict_test_sing)
                cm = sklearn.metrics.confusion_matrix(y_test.argmax(axis=1), outs_test_out[2].argmax(axis=1))

                test_f1.append(cmPRF(cm,ncstart=1)[5])
                test_loss.append(outs_test_out[0])  
                print("Epoch:", '%04d' % (epoch_h), "train_loss=", "{:.5f}".format(outs[1]),
                              "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost_val_loss[-1]), "val_F1=", "{:.5f}".format(cost_val_f1[-1]),"time=", "{:.5f}".format(time.time() - t),"test_f1=", "{:.5f}".format(test_f1[-1]), "test_loss=", "{:.5f}".format(test_loss[-1]))
    #        else:
    #            print("Epoch:", '%04d' % (epoch_h), "train_loss=", "{:.5f}".format(outs[1]),
    #                      "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost_val_loss[-1]), "val_F1=", "{:.5f}".format(cost_val_f1[-1]),"time=", "{:.5f}".format(time.time() - t))

        outs_val_out = sess.run([model.loss, model.accuracy, model.outputs], feed_dict=feed_dict_val)


        cm = sklearn.metrics.confusion_matrix(y_val.argmax(axis=1), outs_val_out[2].argmax(axis=1))

        cost_val_f1.append(cmPRF(cm,ncstart=1)[5])
        cost_val_loss.append(outs_val_out[0])

        # Print results
        #        if cost_val_loss[-1] == min(cost_val_loss):
        outs_test_out = sess.run([model.loss, model.accuracy, model.outputs], feed_dict=feed_dict_test_sing)
        cm = sklearn.metrics.confusion_matrix(y_test.argmax(axis=1), outs_test_out[2].argmax(axis=1))

        test_f1.append(cmPRF(cm,ncstart=1)[5])
        test_loss.append(outs_test_out[0]) 
        
        print("Epoch:", '%04d' % (epoch_h), "train_loss=", "{:.5f}".format(outs[1]),
              "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost_val_loss[-1]), "val_F1=", "{:.5f}".format(cost_val_f1[-1]),"time=", "{:.5f}".format(time.time() - t),"test_f1=", "{:.5f}".format(test_f1[-1]), "test_loss=", "{:.5f}".format(test_loss[-1]))
        if epoch_h > FLAGS.early_stopping and cost_val_loss[-1] > np.mean(cost_val_loss[-(FLAGS.early_stopping+1):-1]):
            outs_test_out = sess.run([model.loss, model.accuracy, model.outputs], feed_dict=feed_dict_test_sing)
            cm = sklearn.metrics.confusion_matrix(y_test.argmax(axis=1), outs_test_out[2].argmax(axis=1))

            test_f1.append(cmPRF(cm,ncstart=1)[5])
            test_loss.append(outs_test_out[0]) 
            print("Early stopping...","test_loss=", "{:.5f}".format(test_loss[-1]),"test_f1=", "{:.5f}".format(test_f1[-1]))
            break
            
print("Optimization Finished!")


    
np.save("/home/yld8809/cm_tp_gcn", cm)
np.save("/home/yld8809/cost_val_f1_tp_gcn",cost_val_f1)
np.save("/home/yld8809/cost_val_loss_tp_gcn",cost_val_loss)
np.save("/home/yld8809/test_f1_tp_gcn",test_f1)
np.save("/home/yld8809/test_loss_tp_gcn",test_loss)