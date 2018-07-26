
       
# Some preprocessing
features = scipy.sparse.coo_matrix(features)
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
#    'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
#    'all_phrase': tf.placeholder(tf.float32, shape=(all_phrase.shape[0], all_phrase.shape[1])),
#    'size_unique_word_index': tf.placeholder(tf.int32),                                         
#    'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
#    'labels': tf.placeholder(tf.float32, shape=(y_train.shape[0], y_train.shape[1])),
#    'labels_mask': tf.placeholder(tf.int32),
#    'dropout': tf.placeholder_with_default(0., shape=()),
#    'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
    'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'all_phrase': tf.placeholder(tf.float32, shape=(all_phrase.shape[0], all_phrase.shape[1])),                            
    'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
    'labels': tf.placeholder(tf.float32, shape=(y_train.shape[0], y_train.shape[1])),
    'labels_mask': tf.placeholder(tf.int32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
}

with tf.device("/gpu:0"):
# Create model
    model = model_func(placeholders, input_dim=features[2][1], logging=False)


# Initialize session

sess = tf.Session(config=tf.ConfigProto(
    allow_soft_placement=True, log_device_placement=False))


# Define model evaluation function
def evaluate(features, support, labels, mask, all_phrase, placeholders):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(features, support, labels, mask, all_phrase, placeholders)
    outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], (time.time() - t_test)


# Init variables
sess.run(tf.global_variables_initializer())

cost_val = []
acc_val = []

# Train model
for epoch in range(FLAGS.epochs):

    t = time.time()
    # Construct feed dictionary
    feed_dict = construct_feed_dict(features, support, y_train, train_mask, all_phrase, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})

    # Training step
    outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)

    # Validation
    cost, acc, duration = evaluate(features, support, y_val, val_mask, all_phrase, placeholders)
    cost_val.append(cost)
    acc_val.append(acc)
    
    # Print results
    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
          "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost),
          "val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))

    if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
        print("Early stopping...")
        break

print("Optimization Finished!")

# Testing
test_cost, test_acc, test_duration = evaluate(features, support, y_test, test_mask, all_phrase, placeholders)
print("Test set results:", "cost=", "{:.5f}".format(test_cost),
      "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))

print(model.outputs)




   
        self.all_phrase = placeholders['all_phrase']
        self.raw_dim = raw_dim
        self.size_unique_word_index = size_unique_word_index

        if self.logging:
                self._log_vars()
                
    def _call(self, inputs):
        
        x = inputs
        output = np.empty(shape=[0, self.raw_dim])
        
        for sent_ind in range(0,self.size_unique_word_index):