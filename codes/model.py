import tensorflow as tf

#list of arguments to be given in this for the module to works

class initializer:
    def _init_(self,shape,scope=None,dtype=None):
        assert len(shape)==1
        #check if the shape of the weight is one or not
        self.scope=scope
        self.W=tf.truncated_normal(0.1,shape);
        self.b=tf.constant(0.1,shape[1]);
    def get_weight(self):
        return self.W,self.b;
    def _call_(self,X):
        return tf.matmul(X,self.W)+self.b;
    
#list of args
#learning rate 0.001
#test True or False that will determine what will happen with our model 
#maxLenDeco
#maxLenEnco

class model:
    def _init_(self,args,textdata):
        self.textdata=textdata;#this will keep the data of the text data
        self.args=args;
        self.dtype=float32;
        self.encoder=None;
        self.decoder=None;
        self.decoder_target=None;
        self.decoder_weight=None;
        self.build_network();#this is done to compute the graph
        
    def build_network(self):
        encoDecoCell = tf.contrib.rnn.BasicLSTMCell(self.args.hiddenSize, state_is_tuple=True);
        with tf.name_scope('placeholder_encoder'):
            self.encoderInputs  = [tf.placeholder(tf.int32,   [None, ]) for _ in range(self.args.maxLenEnco)]
        with tf.name_scope('placeholder_decoder'):
            self.decoderInputs  = [tf.placeholder(tf.int32,   [None, ]) for _ in range(self.args.maxLenDeco)]
            self.decoderWeights=[tf.placeholder(tf.float32,[None,]) for _ in range(self.args.maxLenDeco)];
            self.decoderTarget  = [tf.placeholder(tf.int32,   [None, ]) for _ in range(self.args.maxLenDeco)]
        decoderOutputs, states = tf.contrib.legacy_seq2seq.embedding_rnn_seq2seq(
            self.encoderInputs,  # List<[batch=?, inputDim=1]>, list of size args.maxLength
            self.decoderInputs,  # For training, we force the correct output (feed_previous=False)
            encoDecoCell,
            self.textData.getVocabularySize(),
            self.textData.getVocabularySize(),  # Both encoder and decoder have the same number of class
            embedding_size=self.args.embeddingSize,  # Dimension of each word
            output_projection=outputProjection.getWeights() if outputProjection else None,
            feed_previous=bool(self.args.test)  # When we test (self.args.test), we use previous output as next input (feed_previous)
        );
        if self.args.test:
            if not outputProjection:
                self.outputs = decoderOutputs
            else:
                self.outputs = [outputProjection(output) for output in decoderOutputs]
        else:#this is when we are not testing on our model but train our system
            self.lossFct = tf.contrib.legacy_seq2seq.sequence_loss(
            decoderOutputs,
            self.decoderTargets,
            self.decoderWeights,
            self.textData.getVocabularySize(),
            softmax_loss_function= sampledSoftmax if outputProjection else None  # If None, use default SoftMax
        );
            tf.summary.scalar('loss', self.lossFct)  # Keep track of the cost

            #initialise optimizer
            opt=tf.AdamOptimizer(args.learning_rate);
            optop=opt.minimize(self.lossFct);
    def step(self):
        print("hello");
