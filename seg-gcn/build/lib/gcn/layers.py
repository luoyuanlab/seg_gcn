from gcn.inits import *
import tensorflow as tf
import numpy as np

flags = tf.app.flags
FLAGS = flags.FLAGS

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}

        
def body_ep(ind, output, all_phrase_all, x):

    ind = tf.add(ind,1)
    
    all_phrase = all_phrase_all[ind,:,:]
    word_embedding_all = x[ind,:,:]
    
    match_ind1 = tf.equal(all_phrase[:,0], 1)
    match_ind2 = tf.equal(all_phrase[:,0], 2)
    match_ind3 = tf.equal(all_phrase[:,0], 3)
    match_ind4 = tf.equal(all_phrase[:,0], 4)
    match_ind5 = tf.equal(all_phrase[:,0], 5)
    
    def f1(word_embedding_all_input,match_ind): 
        x_sele = tf.boolean_mask(word_embedding_all_input,match_ind)

        x1_tr = tf.transpose(x_sele)

        x1_tr_abs = x1_tr

        location_abs_max1 = tf.cast(tf.argmax(x1_tr_abs, 1),dtype=tf.int32)

        idx_flattened1 = tf.range(0, tf.shape(x1_tr)[0]) * tf.shape(x1_tr)[1] + location_abs_max1

        x_sele = tf.gather(tf.reshape(x1_tr, [-1]), idx_flattened1)  # use flattened indices
        
        return x_sele
    
    x1 = tf.cond(tf.equal(tf.reduce_sum(tf.cast(match_ind1,dtype=tf.int32)),0), lambda: tf.zeros([x.get_shape().as_list()[2]]), lambda: f1(word_embedding_all,match_ind1))
# tf.truncated_normal([self.eigvec_dim], stddev=0.1) 
# x.get_shape().as_list()[2]
    
    x2 = tf.cond(tf.equal(tf.reduce_sum(tf.cast(match_ind2,dtype=tf.int32)),0), lambda: tf.zeros([x.get_shape().as_list()[2]]), lambda: f1(word_embedding_all,match_ind2))
    
    x3 = tf.cond(tf.equal(tf.reduce_sum(tf.cast(match_ind3,dtype=tf.int32)),0), lambda: tf.zeros([x.get_shape().as_list()[2]]), lambda: f1(word_embedding_all,match_ind3))
        
    x4 = tf.cond(tf.equal(tf.reduce_sum(tf.cast(match_ind4,dtype=tf.int32)),0), lambda: tf.zeros([x.get_shape().as_list()[2]]), lambda: f1(word_embedding_all,match_ind4))
    
    x5 = tf.cond(tf.equal(tf.reduce_sum(tf.cast(match_ind5,dtype=tf.int32)),0), lambda: tf.zeros([x.get_shape().as_list()[2]]), lambda: f1(word_embedding_all,match_ind5))    
#    x1_tr = tf.transpose(x1)
#    x2_tr = tf.transpose(x2)
    
#    x1_tr_abs = x1_tr
#    x2_tr_abs = x2_tr
    
#    location_abs_max1 = tf.cast(tf.argmax(x1_tr_abs, 1),dtype=tf.int32)
#    location_abs_max2 = tf.cast(tf.argmax(x2_tr_abs, 1),dtype=tf.int32)
    
#    idx_flattened1 = tf.range(0, tf.shape(x1_tr)[0]) * tf.shape(x1_tr)[1] + location_abs_max1
#    idx_flattened2 = tf.range(0, tf.shape(x2_tr)[0]) * tf.shape(x2_tr)[1] + location_abs_max2

#    x1 = tf.gather(tf.reshape(x1_tr, [-1]), idx_flattened1)  # use flattened indices
#    x2 = tf.gather(tf.reshape(x2_tr, [-1]), idx_flattened2)  # use flattened indices
#==============================================================================
#    x1 = tf.reduce_max(x1, reduction_indices=[0])
#    x2 = tf.reduce_max(x2, reduction_indices=[0])
#==============================================================================

    x_concat = tf.concat([x1,x2,x3,x4,x5],0) 
    x_concat = tf.reshape(x_concat, [1, output.get_shape().as_list()[1]])  
    output = tf.concat([output,x_concat],0)

    return ind, output, all_phrase_all, x

def condition_ep(ind, output, all_phrase_all, x,):

    return tf.less(ind,tf.shape(all_phrase_all)[0]-1)

def body_sgc(ind, x, dropout, eigvec, theta_diag, output_all):

    ind = tf.add(ind,1)
    
    x_par = x[ind,:,:]
    eigvec_par = eigvec[ind,:,:]
    
    x_par = tf.nn.dropout(x_par, 1-dropout)

    # convolve
    pre_sup = dot(eigvec_par, theta_diag,
                      sparse=False)
        
    pre_sup = dot(pre_sup, tf.transpose(eigvec_par),
                      sparse=False)
        
    output = dot(pre_sup, x_par,
                      sparse=False)


            
#    x[ind,:,:] = output
    output_all = tf.concat([output_all,output],0)
    
    return ind, x, dropout, eigvec, theta_diag, output_all

def condition_sgc(ind, x, dropout, eigvec, theta_diag, output_all):

    return tf.less(ind,tf.shape(x)[0]-1)

def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


def sparse_dropout(x, keep_prob, noise_shape):
    """Dropout for sparse tensors."""
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1./keep_prob)


def dot(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res


class Layer(object):
    """Base layer class. Defines basic API for all layer objects.
    Implementation inspired by keras (http://keras.io).

    # Properties
        name: String, defines the variable scope of the layer.
        logging: Boolean, switches Tensorflow histogram logging on/off

    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
        _log_vars(): Log all variables
    """

    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.sparse_inputs = False
        
        
    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            if self.logging and not self.sparse_inputs:
                tf.summary.histogram(self.name + '/inputs', inputs)
            outputs = self._call(inputs)
            
            if self.logging:
                tf.summary.histogram(self.name + '/outputs', outputs)
            return outputs
            
    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])


class Dense(Layer):
    """Dense layer."""
    def __init__(self, input_dim, output_dim, placeholders, dropout=0., sparse_inputs=False,
                 act=tf.nn.relu, bias=False, featureless=False, **kwargs): ####################################
        super(Dense, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = glorot([input_dim, output_dim],
                                          name='weights')
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()
            
        
    def _call(self, inputs):
        x = inputs

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)
                
        # transform
        output = dot(x, self.vars['weights'], sparse=self.sparse_inputs)

        # bias
        if self.bias:
            output += self.vars['bias']
            
        self.check = output
        return self.act(output)


class GraphConvolution(Layer):
    """Graph convolution layer."""
    def __init__(self, input_dim, output_dim, placeholders, act=tf.nn.relu, dropout=False,**kwargs):
        super(GraphConvolution, self).__init__(**kwargs)
        
        

        self.support = placeholders['eigvec'] #wrong name I know
        
#        sent_length = placeholders['eigvec'].get_shape().as_list()[1]

        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.vars['weights'] = uniform([input_dim, output_dim], name='weights')
#        self.vars['bias'] = zeros([1,output_dim], name='bias')
#        self.vars['constant'] = tf.Variable(tf.constant(0.1, shape=[sent_length, output_dim]), name="constant")
        self.act = act
        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs
        
        

        # convolve
        weights_rep = tf.reshape(tf.tile(self.vars['weights'],[tf.shape(x)[0],1]),[tf.shape(x)[0],self.input_dim,self.output_dim])
#        bias_rep = tf.reshape(tf.tile(self.vars['bias'],[tf.shape(x)[1],1]),[tf.shape(x)[1],self.output_dim])
#        bias_rep = tf.reshape(tf.tile(bias_rep,[tf.shape(x)[0],1]),[tf.shape(x)[0],tf.shape(x)[1],self.output_dim])
#        constant_mat = tf.reshape(tf.tile(self.vars['constant'],[tf.shape(x)[0],1]),[tf.shape(x)[0],tf.shape(x)[1],self.output_dim])
        pre_sup = dot(x, weights_rep, sparse=False)
                
        output = dot(self.support, pre_sup, sparse=False)
        
#        output += bias_rep
#        output = tf.concat([output,x],2)
        

        return self.act(output)
class batch_normalization(Layer):
    """Graph convolution layer with drop."""
    def __init__(self, placeholders,**kwargs):
        super(batch_normalization, self).__init__(**kwargs)

        self.phase = placeholders['is_train']

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        output = tf.contrib.layers.batch_norm(inputs, 
                                  center=True, scale=True, 
                                  is_training=self.phase)
        return output
        
class GraphConvolution_drop(Layer):
    """Graph convolution layer with drop."""
    def __init__(self, input_dim, output_dim, placeholders, act=tf.nn.relu, dropout=False,**kwargs):
        super(GraphConvolution_drop, self).__init__(**kwargs)

        self.support = placeholders['eigvec'] #wrong name I know
#        sent_length = placeholders['eigvec'].get_shape().as_list()[1]

        if dropout:
            self.dropout = placeholders['dropout_gcn']
        else:
            self.dropout = 0.
            
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.vars['weights'] = uniform([input_dim, output_dim], name='weights')
#        self.vars['bias'] = zeros([1,output_dim], name='bias')
#        self.vars['constant'] = tf.Variable(tf.constant(0.1, shape=[sent_length, output_dim]), name="constant")
        self.act = act
        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs
        # convolve
        weights_rep = tf.reshape(tf.tile(self.vars['weights'],[tf.shape(x)[0],1]),[tf.shape(x)[0],self.input_dim,self.output_dim])
#        bias_rep = tf.reshape(tf.tile(self.vars['bias'],[tf.shape(x)[1],1]),[tf.shape(x)[1],self.output_dim])
#        bias_rep = tf.reshape(tf.tile(bias_rep,[tf.shape(x)[0],1]),[tf.shape(x)[0],tf.shape(x)[1],self.output_dim])
#        constant_mat = tf.reshape(tf.tile(self.vars['constant'],[tf.shape(x)[0],1]),[tf.shape(x)[0],tf.shape(x)[1],self.output_dim])
        pre_sup = dot(x, weights_rep, sparse=False)
                
        output = dot(self.support, pre_sup, sparse=False)
        
        output = tf.nn.dropout(output, 1-self.dropout, noise_shape = [1, 1, self.output_dim])
#        output += bias_rep
#        output = tf.concat([output,x],2)
        return self.act(output)

class DenseLayer(Layer):
    """Dense layer."""
    def __init__(self, input_dim, output_dim, placeholders, act=tf.nn.relu, dropout=False,**kwargs):
        super(DenseLayer, self).__init__(**kwargs)
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.vars['weights'] = uniform([input_dim, output_dim], name='weights')
        
        self.act = act
        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs
        # convolve
        weights_rep = tf.reshape(tf.tile(self.vars['weights'],[tf.shape(x)[0],1]),[tf.shape(x)[0],self.input_dim,self.output_dim])
        
        output = dot(x, weights_rep, sparse=False)
                

        return self.act(output)


class spectral_graph_convolution(Layer):
    """Graph convolution layer."""
    def __init__(self, output_dim, placeholders, dropout=False,
                 sparse_inputs=False, act=tf.nn.relu, bias=False,
                 featureless=False, **kwargs):
        super(spectral_graph_convolution, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.output_dim = output_dim
        self.eigvec = placeholders['eigvec']
        self.eigvec_dim = placeholders['eigvec'].get_shape().as_list()[1]
        self.sparse_inputs = sparse_inputs

        self.bias = bias
        
        # helper variable for sparse dropout
#        self.num_features_nonzero = placeholders['num_features_nonzero']
        
        self.vars['theta'] = tf.Variable(tf.truncated_normal([self.eigvec_dim], stddev=0.1), name='theta')
#        self.vars['theta'] = glorot_onedim([self.eigvec_dim], name='theta')
#        self.vars['theta'] = glorot([self.eigvec_dim,output_dim], name='theta')
#        self.vars['theta'] = tf.Variable(tf.truncated_normal([self.eigvec_dim,output_dim], stddev=0.1), name="theta")

                
        if self.logging:
            self._log_vars()

    def _call(self, inputs):
#        x = inputs
        
        self.theta_diag = tf.reshape(tf.tile(tf.diag(self.vars['theta']),[tf.shape(inputs)[0],1]),[tf.shape(inputs)[0],self.eigvec_dim,self.eigvec_dim])
#        theta_diag = tf.reshape(tf.tile(self.vars['theta'],[tf.shape(inputs)[0],1]),[tf.shape(inputs)[0],self.eigvec_dim,self.output_dim])
#        pre_sup = tf.multiply(self.eigvec,self.vars['theta'])

        
#        pre_sup = dot(pre_sup, tf.transpose(self.eigvec, perm=[0, 2, 1]),
#                          sparse=False)
            
        pre_sup1 = dot(tf.transpose(self.eigvec, perm=[0, 2, 1]), inputs,
                          sparse=False)
        
#        pre_sup2 = tf.multiply(theta_diag,pre_sup1)
        pre_sup2 = dot(self.theta_diag,pre_sup1,
                          sparse=False)
    
        output = dot(self.eigvec, pre_sup2,
                      sparse=False)
        
        #output_all = tf.convert_to_tensor(np.empty(shape=[0, tf.shape(self.eigvec)[1], self.output_dim],dtype=np.float32))
        
        #ind = tf.constant(-1)    
        
        #ind, x, self.dropout, self.eigvec, theta_diag, output_all = \
        #tf.while_loop(condition_sgc, body_sgc, [ind, x, self.dropout, self.eigvec, theta_diag, output_all])        

        
        self.check = output
        return self.act(output)

class LSTM_layer(Layer):
    """LSTM layer."""
    def __init__(self, placeholders, act=lambda x: x, num_hidden_f = 200,num_hidden_b = 100, dropout=False, **kwargs):
        super(LSTM_layer, self).__init__(**kwargs)

        self.cell_f1 = tf.nn.rnn_cell.LSTMCell(num_hidden_f,forget_bias=0.0)
        self.cell_b1 = tf.nn.rnn_cell.LSTMCell(num_hidden_b,forget_bias=0.0)
        
        self.num_hidden = num_hidden_f + num_hidden_b
        
        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.
            
#         self.cell_f2 = tf.nn.rnn_cell.LSTMCell(num_hidden)
#         self.cell_b2 = tf.nn.rnn_cell.LSTMCell(num_hidden)
        
        self.act = act
        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs
        # convolve
        
        (output_fw1, output_bw1), _ = tf.nn.bidirectional_dynamic_rnn(self.cell_f1, self.cell_b1, x,scope='BLSTM_'+ str(1), dtype=tf.float32)
        output1 = tf.concat([output_fw1, output_bw1], axis=2)
        
#         (output_fw2, output_bw2), _ = tf.nn.bidirectional_dynamic_rnn(self.cell_f2, self.cell_b2, output1, scope='BLSTM_'+ str(2), dtype=tf.float32)
        
#         output2 = tf.concat([output_fw2, output_bw2], axis=2)
        

        output1 = tf.nn.dropout(output1, 1-self.dropout, noise_shape = [1, 1, self.num_hidden])

            
        return self.act(output1)

class LSTM_layer_nodrop(Layer):
    """LSTM layer no drop."""
    def __init__(self, placeholders, act=lambda x: x, num_hidden = 24, dropout=False, **kwargs):
        super(LSTM_layer_nodrop, self).__init__(**kwargs)

        self.cell_f1 = tf.nn.rnn_cell.LSTMCell(num_hidden,forget_bias=0.0)
        self.cell_b1 = tf.nn.rnn_cell.LSTMCell(num_hidden,forget_bias=0.0)
        
        self.num_hidden = num_hidden
        
            
#         self.cell_f2 = tf.nn.rnn_cell.LSTMCell(num_hidden)
#         self.cell_b2 = tf.nn.rnn_cell.LSTMCell(num_hidden)
        
        self.act = act
        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs
        # convolve
        
        (output_fw1, output_bw1), _ = tf.nn.bidirectional_dynamic_rnn(self.cell_f1, self.cell_b1, x,scope='BLSTM_'+ str(1), dtype=tf.float32)
        output1 = tf.concat([output_fw1, output_bw1], axis=2)
        
            
        return self.act(output1)
    
class LSTM_layer_dual(Layer):
    """LSTM_layer_dual layer."""
    def __init__(self, placeholders, act=lambda x: x, num_hidden_f = 150,num_hidden_b = 150,num_layers_LSTM=1, dropout=False,  **kwargs):
        super(LSTM_layer_dual, self).__init__(**kwargs)

#         self.cell_f1 = tf.nn.rnn_cell.LSTMCell(num_hidden_f,forget_bias=0.0)
#         self.cell_b1 = tf.nn.rnn_cell.LSTMCell(num_hidden_b,forget_bias=0.0)
#         self.stacked_cell_f1 = tf.nn.rnn_cell.MultiRNNCell([self.cell_f1] * 2)
#         self.stacked_cell_b1 = tf.nn.rnn_cell.MultiRNNCell([self.cell_b1] * 2)

        num_layers = num_layers_LSTM
    
        self.cell_f = []
        for i in range(num_layers):
            self.cell_f.append(tf.nn.rnn_cell.LSTMCell(num_hidden_f,forget_bias=0.0))
        self.cell_f = tf.nn.rnn_cell.MultiRNNCell(self.cell_f)
        
        self.cell_b = []
        for i in range(num_layers):
            self.cell_b.append(tf.nn.rnn_cell.LSTMCell(num_hidden_b,forget_bias=0.0))
        self.cell_b = tf.nn.rnn_cell.MultiRNNCell(self.cell_b)
        
        self.num_hidden = num_hidden_f+num_hidden_b
        
        if dropout:
            self.dropout = placeholders['dropout_lstm']
        else:
            self.dropout = 0.
            
#         self.cell_f2 = tf.nn.rnn_cell.LSTMCell(num_hidden)
#         self.cell_b2 = tf.nn.rnn_cell.LSTMCell(num_hidden)
        
        self.act = act
        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs
        # convolve
        
        (output_fw1, output_bw1), _ = tf.nn.bidirectional_dynamic_rnn(self.cell_f, self.cell_b, x, scope='BLSTM_'+ str(1), dtype=tf.float32)
        output1 = tf.concat([output_fw1, output_bw1], axis=2)
        output1 = tf.nn.dropout(output1, 1-self.dropout, noise_shape = [1, 1, self.num_hidden])     
        return self.act(output1)

class LSTM_layer_mono(Layer):
    """LSTM_layer_dual layer."""
    def __init__(self, placeholders, act=lambda x: x, num_hidden = 300, num_layers_LSTM = 1, dropout=False,  **kwargs):
        super(LSTM_layer_mono, self).__init__(**kwargs)

#         self.cell_f1 = tf.nn.rnn_cell.LSTMCell(num_hidden_f,forget_bias=0.0)
#         self.cell_b1 = tf.nn.rnn_cell.LSTMCell(num_hidden_b,forget_bias=0.0)
#         self.stacked_cell_f1 = tf.nn.rnn_cell.MultiRNNCell([self.cell_f1] * 2)
#         self.stacked_cell_b1 = tf.nn.rnn_cell.MultiRNNCell([self.cell_b1] * 2)

        num_layers = num_layers_LSTM
    
        self.cell_f = []
        for i in range(num_layers):
            self.cell_f.append(tf.nn.rnn_cell.LSTMCell(num_hidden,forget_bias=0.0))
        self.cell_f = tf.nn.rnn_cell.MultiRNNCell(self.cell_f)
        
        self.num_hidden = num_hidden
        
        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.
            
#         self.cell_f2 = tf.nn.rnn_cell.LSTMCell(num_hidden)
#         self.cell_b2 = tf.nn.rnn_cell.LSTMCell(num_hidden)
        
        self.act = act
        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs
        # convolve
        
        output1, _ = tf.nn.dynamic_rnn(self.cell_f, x, dtype=tf.float32)
        
        output1 = tf.nn.dropout(output1, 1-self.dropout, noise_shape = [1, 1, self.num_hidden])     
        return self.act(output1)    
    
class LSTM_layer_multi_drop(Layer):
    """LSTM_layer_dual layer."""
    def __init__(self, placeholders, act=lambda x: x, num_hidden_f = 150,num_hidden_b = 150,num_layers_LSTM=1, dropout=False,  **kwargs):
        super(LSTM_layer_multi_drop, self).__init__(**kwargs)

#         self.cell_f1 = tf.nn.rnn_cell.LSTMCell(num_hidden_f,forget_bias=0.0)
#         self.cell_b1 = tf.nn.rnn_cell.LSTMCell(num_hidden_b,forget_bias=0.0)
#         self.stacked_cell_f1 = tf.nn.rnn_cell.MultiRNNCell([self.cell_f1] * 2)
#         self.stacked_cell_b1 = tf.nn.rnn_cell.MultiRNNCell([self.cell_b1] * 2)

        num_layers = num_layers_LSTM
    
        self.num_hidden = num_hidden_f+num_hidden_b
        
        if dropout:
            self.dropout = placeholders['dropout_lstm']
        else:
            self.dropout = 0.0
                        
        self.cell_f = []
        for i in range(num_layers):
            cell = tf.nn.rnn_cell.LSTMCell(num_hidden_f,forget_bias=0.0)
            self.cell_f.append(tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=1-self.dropout))
        
        self.cell_f = tf.nn.rnn_cell.MultiRNNCell(self.cell_f)
        
        self.cell_b = []
        for i in range(num_layers):
            cell = tf.nn.rnn_cell.LSTMCell(num_hidden_f,forget_bias=0.0)
            self.cell_b.append(tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=1-self.dropout))
        
        self.cell_b = tf.nn.rnn_cell.MultiRNNCell(self.cell_b)
        

            
#         self.cell_f2 = tf.nn.rnn_cell.LSTMCell(num_hidden)
#         self.cell_b2 = tf.nn.rnn_cell.LSTMCell(num_hidden)
        
        self.act = act
        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs
        # convolve
        
        (output_fw1, output_bw1), _ = tf.nn.bidirectional_dynamic_rnn(self.cell_f, self.cell_b, x, scope='BLSTM_'+ str(1), dtype=tf.float32)
        output1 = tf.concat([output_fw1, output_bw1], axis=2)
#         output1 = tf.nn.dropout(output1, 1-self.dropout, noise_shape = [1, 1, self.num_hidden])     
        return self.act(output1)
    
class spectral_graph_convolution_nondiag(Layer):
    """Graph convolution layer."""
    def __init__(self, output_dim, placeholders, dropout=False,
                 sparse_inputs=False, act=tf.nn.relu, bias=False,
                 featureless=False, **kwargs):
        super(spectral_graph_convolution_nondiag, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.eigvec = placeholders['eigvec']
        self.theta_raw = placeholders['eigval']
        self.eigvec_dim = placeholders['eigvec'].get_shape().as_list()[1]
        self.sparse_inputs = sparse_inputs

        self.bias = bias

        # helper variable for sparse dropout
#        self.num_features_nonzero = placeholders['num_features_nonzero']
        

        self.vars['theta'] = tf.Variable(tf.diag(tf.constant(0, shape=[self.eigvec_dim])), name="theta")
        
        if self.bias:
             self.vars['bias'] = tf.Variable(tf.zeros([output_dim],tf.float32), name='bias')
                
        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs
        
        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)

        # convolve
        pre_sup = dot(self.eigvec, self.vars['theta'],
                      sparse=False)
        
        pre_sup = dot(pre_sup, tf.transpose(self.eigvec),
                      sparse=False)
        
        output = dot(pre_sup, x,
                      sparse=False)

        # bias
        if self.bias:
            output += self.vars['bias']
            

        return self.act(output)
    
class embedding_pooling(Layer):
    """embedding pooling layer."""
    def __init__(self, raw_dim, all_phrase, placeholders, act=tf.nn.relu, **kwargs):
        super(embedding_pooling, self).__init__(**kwargs)

        self.all_phrase = all_phrase
        self.raw_dim = raw_dim
        self.act = act
        
        if self.logging:
                self._log_vars()
                
    
    def _call(self, inputs):
        x = inputs
        
        ind = tf.constant(-1)    
        output = tf.convert_to_tensor(np.empty(shape=[0, self.raw_dim],dtype=np.float32))

        ind, output, self.all_phrase, x = \
        tf.while_loop(condition_ep, body_ep, [ind, output, self.all_phrase, x],\
                      shape_invariants=[ind.get_shape(), tf.TensorShape([None, self.raw_dim]),\
                                        self.all_phrase.get_shape(), x.get_shape()])
        self.check = output
        return self.act(output)

class dim_red(Layer):
    """dimension reduction layer."""
    def __init__(self, raw_dim, num_class, placeholders, act=lambda x: x, dropout=False, **kwargs):
        super(dim_red, self).__init__(**kwargs)
        
        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.
            
        self.raw_dim = raw_dim
        self.num_class = num_class
        self.act = act
        self.vars['weights_w'] = tf.Variable(tf.truncated_normal([self.raw_dim, self.num_class], stddev=0.1), name="W")
        
        self.vars['bias_b'] = tf.Variable(tf.constant(0.1, shape=[self.num_class]), name="b")
                
    def _call(self, inputs):
        x = inputs

        # dropout
        x = tf.nn.dropout(x, 1-self.dropout)
        
        self.l2_loss_wb = tf.nn.l2_loss(self.vars['weights_w'])
        self.l2_loss_wb += tf.nn.l2_loss(self.vars['bias_b'])
        
        output = tf.nn.xw_plus_b(x, self.vars['weights_w'], self.vars['bias_b'], name="scores")
        self.check = output
        
        return self.act(output)  
    
class dim_red_nob(Layer):
    """dimension reduction layer."""
    def __init__(self, raw_dim, num_class, placeholders, act=lambda x: x, dropout=True, **kwargs):
        super(dim_red_nob, self).__init__(**kwargs)
        
        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.
            
        self.raw_dim = raw_dim
        self.num_class = num_class
        self.act = act
        self.vars['weights_w'] = tf.Variable(tf.truncated_normal([self.raw_dim, self.num_class], stddev=0.1), name="W")
        
                
    def _call(self, inputs):
        x = inputs

        # dropout
        x = tf.nn.dropout(x, 1-self.dropout)
        
        self.l2_loss_wb = tf.nn.l2_loss(self.vars['weights_w'])
        
        output =  dot(x, self.vars['weights_w'], sparse=False)
        
        self.check = output
        
        return self.act(output)      
        
class s_to_d(Layer):
    """dimension reduction layer."""
    def __init__(self, **kwargs):
        super(s_to_d, self).__init__(**kwargs)

        if self.logging:
                self._log_vars()
                
    def _call(self, inputs):
        x = inputs

        output = tf.sparse_tensor_to_dense(
            x,
            default_value=0,
            validate_indices=True,
            name=None
        )
        self.check = output
        
        return output         