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


## test
test_ind = np.asarray(range(label1.shape[0], label.shape[0]))
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

with tf.device('/cpu:0'):
    
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
    sess = tf.Session()
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