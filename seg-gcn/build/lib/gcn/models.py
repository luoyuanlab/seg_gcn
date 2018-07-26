from gcn.layers import *
from gcn.metrics import *
import numpy as np
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS


class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.placeholders = {}

        self.layers = []
        self.activations = []

        self.inputs = None
        self.outputs = None

        self.loss = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

        # Build sequential layer model
        self.activations.append(self.inputs)
        for layer in self.layers:
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)
        self.outputs = self.activations[-1]

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss)

    def predict(self):
        pass

    def _loss(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError

    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, "tmp/%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = "tmp/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)


class MLP(Model):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(MLP, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
        
        
        # Cross entropy error
        self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])

    def _build(self):
        self.layers.append(Dense(input_dim=self.input_dim,
                                 output_dim=FLAGS.hidden1,
                                 placeholders=self.placeholders,
                                 act=tf.nn.relu,
                                 dropout=True,
                                 sparse_inputs=True,
                                 logging=self.logging))

        self.layers.append(Dense(input_dim=FLAGS.hidden1,
                                 output_dim=self.output_dim,
                                 placeholders=self.placeholders,
                                 act=lambda x: x,
                                 dropout=True,
                                 logging=self.logging))

    def predict(self):
        return tf.nn.softmax(self.outputs)


class GCN(Model):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(GCN, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.all_phrase = placeholders['all_phrase']
#        self.size_unique_word_index = placeholders['size_unique_word_index']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer()

        self.build()

    def _loss(self):
        # Weight decay loss
#        self.loss += FLAGS.weight_decay*tf.nn.l2_loss(self.layers[0].vars['weights'])
#        for var in self.layers[0].vars.values():
#            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

#        self.loss += FLAGS.weight_decay*(tf.norm(self.layers[0].vars['weights'],ord=1)**2)/2



        self.loss += FLAGS.weight_decay_1*(tf.nn.l2_loss(self.layers[0].vars['weights']))
        self.loss += FLAGS.weight_decay_wb*self.layers[-1].l2_loss_wb 
    
#         self.loss += FLAGS.weight_decay_lstm*(tf.nn.l2_loss(tf.trainable_variables()[3]))
#         self.loss += FLAGS.weight_decay_lstm*(tf.nn.l2_loss(tf.trainable_variables()[4])) 
#         self.loss += FLAGS.weight_decay_lstm*(tf.nn.l2_loss(tf.trainable_variables()[5]))
#         self.loss += FLAGS.weight_decay_lstm*(tf.nn.l2_loss(tf.trainable_variables()[6]))         
#        self.loss += FLAGS.weight_decay_2*(tf.nn.l2_loss(self.layers[1].vars['weights']))
#        self.loss += FLAGS.weight_decay*tf.nn.l2_loss(self.layers[0].vars['bias'])
        # w, b for l2
        
        # Cross entropy error
#        self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
#                                                  self.placeholders['labels_mask'])        

#        class_weight = tf.constant([[0.25,0.75]])
        class_weight = self.placeholders['weights']
        weight_per_label = tf.transpose( tf.matmul(self.placeholders['labels'], class_weight) ) 
        
        xent = tf.multiply(weight_per_label, tf.nn.softmax_cross_entropy_with_logits(labels = self.placeholders['labels'], logits = self.outputs))
                      
        self.loss += tf.reduce_mean(xent) #shape 1
              

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])

    def _build(self):
        
        
        

#         self.layers.append(GraphConvolution(input_dim=int(FLAGS.embedding_dim_0),
#                                  output_dim=int(FLAGS.embedding_dim_1),
#                                  placeholders=self.placeholders,
#                                  act=tf.tanh,
#                                  dropout=False,
#                                  logging=self.logging))        
        

        
        self.layers.append(GraphConvolution(input_dim=int(FLAGS.embedding_dim_0),
                                 output_dim=int(FLAGS.embedding_dim_1),
                                 placeholders=self.placeholders,
                                 act=tf.tanh,
                                 dropout=True,
                                 logging=self.logging))
        
        
        self.layers.append(LSTM_layer_dual(
                                 placeholders=self.placeholders,
                                 act=tf.tanh,
                                    num_hidden_f = FLAGS.hidden_dim_f,
                                    num_hidden_b = FLAGS.hidden_dim_b,
                                    num_layers_LSTM = 1,
                                 dropout=True,
                                 logging=self.logging))  
                           
        
        # self.layers.append(s_to_d())        
        self.layers.append(embedding_pooling(raw_dim=int(FLAGS.hidden_dim_f+FLAGS.hidden_dim_b)*5,
                                            all_phrase = self.all_phrase,
                                            placeholders = self.placeholders,
                                            act=lambda x: x,
                                            logging=self.logging))     
        # fully connected layer
        self.layers.append(dim_red(         raw_dim=int(FLAGS.hidden_dim_f+FLAGS.hidden_dim_b)*5,
                                            num_class=self.output_dim,
                                            placeholders=self.placeholders, 
                                            act=lambda x: x,
                                            dropout=False, 
                                            logging=self.logging))
        
    def predict(self):
        return tf.nn.softmax(self.outputs)