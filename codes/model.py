import os
import tensorflow as tf
import configparser as cp
from dataset import batch
#list of arguments to be given in this for the module to works

class initializer:
    def __init__(self, shape, scope=None, dtype=None):
        #TODO:- check if the shape of the weight is one or not
        assert len(shape) == 2
        self.scope = scope
        #TODO:- Check if this thing is correct for self.W and self.b
        #self.W = tf.truncated_normal(0.1, shape)
        #self.b = tf.constant(0.1, shape[1])
        with tf.variable_scope('weights_' + self.scope):
            self.W = tf.get_variable(
                'weights',
                shape,
                # initializer=tf.truncated_normal_initializer()  # TODO: Tune value (fct of input size: 1/sqrt(input_dim))
                dtype=dtype
            )
            self.b = tf.get_variable(
                'bias',
                shape[1],
                initializer=tf.constant_initializer(),
                dtype=dtype
            )
    def get_weight(self):
        return self.W, self.b

    def __call__(self, X):
        #TODO:- check if name scope is necessary
        with tf.name_scope(self.scope):
            return tf.matmul(X, self.W) + self.b
    
#list of args
#learning rate 0.001
#test True or False that will determine what will happen with our model 
#maxLenDeco
#maxLenEnco

class RNNModel:
    def __init__(self,text_data):
        config = cp.ConfigParser()
        self.DirName='/'.join(os.getcwd().split('/')[:-1]);
        config.read(self.DirName+"/Database/Config.ini");
        self.test = config['General'].getboolean('test')
        self.textdata = text_data       #this will keep the text data object
        self.dtype = tf.float32
        self.encoder = None
        self.decoder = None
        self.dropout = config['General'].getfloat('dropout')
        self.decoder_target = None
        self.decoder_weight = None
        self.loss_fct = None
        self.opt_op = None
        self.outputs = None
        self.softmaxSamples = int(config.get('Model', 'softmaxSamples'))
        self.hiddenSize = int(config.get('Model', 'hiddenSize'))
        self.numLayers = int(config.get('Model', 'numLayers'))
        self.maxLenEnco = int(config.get('Dataset', 'maxLength'))
        self.maxLenDeco = self.maxLenEnco + 2 #Todo: will see if it needs to be in config
        self.embeddingSize = int(config.get('Model', 'embeddingSize'))
        self.learningRate = float(config.get('Model', 'learningRate'))
        self.build_network()           #this is done to compute the graph

    def build_network(self):
        outputProjection = None
        # Sampled softmax only makes sense if we sample less than vocabulary size.
        if 0 < self.softmaxSamples < self.textdata.vocab_size():
            outputProjection = initializer(
                (self.hiddenSize, self.textdata.vocab_size()),
                scope='softmax_projection',
                dtype=self.dtype
            )

            def sampledSoftmax(inputs, labels):
                labels = tf.reshape(labels, [-1, 1])  # Add one dimension (nb of true classes, here 1)

                # We need to compute the sampled_softmax_loss using 32bit floats to
                # avoid numerical instabilities.
                local_weight = tf.cast(tf.transpose(outputProjection.W), tf.float32)
                local_bias = tf.cast(outputProjection.b, tf.float32)
                local_inputs = tf.cast(inputs, tf.float32)

                return tf.cast(
                    tf.nn.sampled_softmax_loss(
                        local_weight,  # Should have shape [num_classes, dim]
                        local_bias,
                        local_inputs,
                        labels,
                        self.softmaxSamples,
                        self.textdata.vocab_size()),
                    self.dtype)

        with tf.device('/gpu:0'):
            enc_dec_cell = tf.contrib.rnn.BasicLSTMCell(self.hiddenSize,
                                                    state_is_tuple=True)
            if not self.test:  # TODO: Should use a placeholder instead
                enc_dec_cell = tf.contrib.rnn.DropoutWrapper(enc_dec_cell,
                                                         input_keep_prob=1.0,
                                                         output_keep_prob=self.dropout)
            enc_dec_cell = tf.contrib.rnn.MultiRNNCell([enc_dec_cell] * self.numLayers,state_is_tuple=True)

            self.encoder  = [tf.placeholder(tf.int32, [None, ]) for _ in range(self.maxLenEnco)]

            #TODO:- check if operation name is not necessary for placeholders
            self.decoder  = [tf.placeholder(tf.int32,[None, ],name='inputs') for _ in range(self.maxLenDeco)]
            self.decoder_weights=[tf.placeholder(tf.float32,[None,],name='weights') for _ in range(self.maxLenDeco)];
            self.decoder_targets  = [tf.placeholder(tf.int32, [None, ],name='targets') for _ in range(self.maxLenDeco)]

            decoder_outputs, states = tf.contrib.legacy_seq2seq.embedding_rnn_seq2seq(
                self.encoder,  # List<[batch=?, inputDim=1]>, list of size args.maxLength
                self.decoder,  # For training, we force the correct output (feed_previous=False)
                enc_dec_cell,
                self.textdata.vocab_size(),
                self.textdata.vocab_size(),  # Both encoder and decoder have the same number of class
                embedding_size=self.embeddingSize,  # Dimension of each word
                output_projection=outputProjection.getWeights() if outputProjection else None,
                feed_previous=bool(self.test)
                )
        #print(self.test)

        if self.test:
            #print(decoder_outputs)
            if not outputProjection:
                self.outputs = decoder_outputs
            else:
                self.outputs = [outputProjection(output) for output in decoder_outputs]
        
        else:
            #this is when we are not testing on our model but train our system
            with tf.device('/gpu:0'):
                self.loss_fct = tf.contrib.legacy_seq2seq.sequence_loss(
                    decoder_outputs,
                    self.decoder_targets,
                    self.decoder_weights,
                    self.textdata.vocab_size(),
                    softmax_loss_function= sampledSoftmax if outputProjection else None
                )
                tf.summary.scalar('loss', self.loss_fct)

                opt = tf.train.AdamOptimizer(
                    learning_rate=self.learningRate,
                    beta1=0.9,
                    beta2=0.999,
                    epsilon=1e-08
                )
                self.opt_op = opt.minimize(self.loss_fct);

    def step(self, batch):
        feed_dict = {}
        ops = None
        #TODO:- Remove args dependecy if possible
        if not self.test:  # Training
            feed_dict = {self.encoder[i]: batch.var_encoder[i]
                         for i in range(self.maxLenEnco)}
            feed_dict.update({self.decoder[i]: batch.var_decoder[i]
                              for i in range(self.maxLenDeco)})
            feed_dict.update({self.decoder_targets[i]: batch.var_target[i]
                              for i in range(self.maxLenDeco)})
            feed_dict.update({self.decoder_weights[i]: batch.var_weight[i]
                              for i in range(self.maxLenDeco)})
            ops = (self.opt_op, self.loss_fct)
        else:  # Testing (batchSize == 1)
            feed_dict = {self.encoder[i]: batch.var_encoder[i]
                         for i in range(self.maxLenEnco)}
            feed_dict[self.decoder[0]]  = [self.textdata.var_token]
            ops = tuple([self.outputs])
        # Return one pass operator
        return ops, feed_dict
