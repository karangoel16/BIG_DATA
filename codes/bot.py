import configparser
#import datetime  # Chronometer
import os
import tensorflow as tf
import numpy as np
import math
import csv
#Todo:- ask karan to have first character of class name as capital
from dataset import dataset
from model import RNNModel

DirName='/'.join(os.path.dirname(os.path.realpath(__file__)).split('/')[:-1])
config = configparser.ConfigParser()
config.read(DirName+"/Database/Config.ini");


#this is our main frame work where all the training takes place 
class Bot:
    "Bot framework which integrates all the component"
#intialising all the parameters needed in the code
    def __init__(self):
        #TODO:- Instead of using command line args we will go for config only
        # set the appropriate values from config compared to what was use as args
        # in original code

        self.text_data = dataset()  # Dataset
        self.model = None  # Sequence to sequence model
        self.verbose = None
        # Tensorflow utilities for convenience saving/logging
        self.writer = None
        self.saver = None
        self.model_dir = ''
        self.global_step = 0
        self.DirName = DirName	
        self.session = None
        self.model_tag = None
        # Filename and directories constants
        # Todo:- Move following to seperate config files
        self.root_dir = DirName#'/'.join(os.getcwd().split('/')[:-1])
        self.MODEL_DIR_BASE = config.get('Bot', 'modelDirBase')
        self.MODEL_NAME_BASE = config.get('Bot', 'modelNameBase')
        self.MODEL_EXT = config.get('Bot', 'modelExt')
        self.CONFIG_FILENAME = config.get('Bot', 'configFilename')
        self.CONFIG_VERSION = config.get('Bot', 'configVersion')
        self.TEST_IN_NAME = os.path.join(DirName,config.get('Bot', 'testInName'))
        self.TEST_OUT_SUFFIX = os.path.join(DirName,config.get('Bot', 'testOutSuffix'))
        self.SENTENCES_PREFIX = ['Q: ', 'A: ']
        self.reset = None
        self.create_dataset = None
        self.device = config.get('General', 'device')
        self.twitter = config['General'].getboolean('twitter')

#some more parameters are loaded in the function all are read from the config file and then it is used accordingly
    
    def load_config(self):
        #Todo:- Load all the required values from the required configs
        self.keep_all = config['General'].getboolean('keepall')
        self.epochs = int(config.get('General', 'epochs'))
        self.current_epoch = 0
        self.learning_rate =float(config.get('Model', 'learningRate'))
        self.save_ckpt_at = int(config.get('General', 'saveCkptAt'))
        self.batch_size = int(config.get('General', 'batchSize'))
        self.global_step = int(config.get('General', 'globalStep'))
        self.max_length = int(config.get('Dataset', 'maxLength'))
        self.watson_mode = config['Bot'].getboolean('watsonMode')
        self.auto_encode = config['Bot'].getboolean('autoEncode')
        self.attention = config['Bot'].getboolean('attention')
        self.corpus = config.get('Bot', 'corpus') #Todo:- Fix this hardcode
        self.dataset_tag = ""
        self.hidden_size = int(config.get('Model', 'hiddenSize'))
        self.num_layers =  int(config.get('Model', 'numLayers'))
        self.embedding_size = int(config.get('Model', 'embeddingSize'))
        self.init_embeddings = config['Bot'].getboolean('initEmbeddings')
        self.softmax_samples = int(config.get('Model', 'softmaxSamples'))
        self.embedding_source = config.get("Bot", "embeddingSource")
        self.model_tag = None
        self.test = config['General'].getboolean('test')
        self.file = config['General'].getboolean('file')
        #print(self.init_embeddings)

#Here main is called , from where it is bifurcated according to the inputs that we get from the config file

    def main(self):

        print("SmartGator Intelligent chatbot")

        self.root_dir = os.getcwd() 

        #self.text_data = dataset()
        self.load_config()
        self.load_model_params()

        print(self.text_data)
        with tf.device(self.get_device()):
            self.model = RNNModel(self.text_data)

        #print (self._get_summary_name())
        #init_op = tf.global_variables_initializer()
        self.writer = tf.summary.FileWriter(self._get_summary_name())
        self.saver = tf.train.Saver(max_to_keep=200, write_version=tf.train.SaverDef.V1)

        self.session = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False)
        )

        print("Initializing tf variables")
        self.session.run(tf.global_variables_initializer())
        #print(self.test)
        self.manage_previous_model(self.session)
        if self.init_embeddings:
            self.load_embedding(self.session)
        if self.twitter:
            return 
        elif self.test: 
            self.interactive_main(self.session);
        elif self.file:
            #try:
            with open(self.TEST_IN_NAME,"r") as f:
                    #try:
                with open(self.TEST_OUT_SUFFIX,'w') as output:
                    for line in f:
                        print(self.predict_daemon(line));
                    #except:
                        #print("Writing in file is a problem")
            #except:
            #    print("Open file error")
        else:
            self.train_model(self.session)

        self.session.close()
        print("Say Bye Bye to SmartGator! ;)")
    
    # Implementtion done, Testing remains
    def train_model(self, session):
        merged_summaries = tf.summary.merge_all()

        #Todo:- See if we can try making it possible to restore from previous run
        if self.global_step == 0:
            self.writer.add_graph(session.graph)

        print('Training begining (press Ctrl+C to save and exit)...')
        csv_name = self.root_dir + self._get_csv_name()
        print("Will be writing data to:", csv_name)
        with open(csv_name, "a") as f:
            print("opened th csv file")
            wr = csv.writer(f, quoting=csv.QUOTE_ALL)
            row_data = []
            try:
                if self.current_epoch == self.epochs:
                    #TODO: User input if neccessary
                    print("Current epoch is same as total required epochs")
                    return

                #wr.writerow(["global_step", "epoch", "loss", "perplexity"])
                for epoch in range(self.current_epoch, self.epochs):

                    print(
                          "\n----- Epoch {}/{} ; (lr={}) -----".format(
                            epoch+1,
                            self.epochs,
                            self.learning_rate
                            )
                          )
                    batches = self.text_data.getBatches()
                    local_step = 0

                    for curr_batch in batches:
                        ops, feed_dict = self.model.step(curr_batch)
                        assert len(ops) == 2
                        _, loss, summary = session.run(ops + tuple([merged_summaries]), feed_dict)
                        self.writer.add_summary(summary, self.global_step)
                        self.global_step += 1
                        local_step += 1

                        if self.global_step % 100 == 0:
                            perplexity = math.exp(float(loss)) if loss < 300 else float("inf")
                            print("----- Step %d/%d -- Loss %.2f -- Perplexity %.2f -- GlobalStep %d" %  (local_step, len(batches), loss, perplexity, self.global_step))

                            row_data.append([self.global_step, epoch, loss, perplexity])
                        #Save checkpoint
                        if self.global_step % self.save_ckpt_at == 0:
                            self._save_session(session)
                            print("Writing data to csv")
                            wr.writerows(row_data)
                            print("Write complete")
                            row_data = []
                    self.current_epoch += 1
            except (KeyboardInterrupt, SystemExit):
                print("Saving state and Exiting the program")
                wr.writerows(row_data)
                print("Written data to file")

        self._save_session(session)

    def _get_csv_name(self):
        if self.init_embeddings:
            return "/data_word2vec.csv"
        elif self.attention:
            return "/data_attention.csv"
        else:
            return "/data_tokenizer.csv"

    def predict_test_set(self, session):
        with open(os.path.join(self.root_dir, self.TEST_IN_NAME), 'r') as f:
            lines = f.readlines()

        model_list = self._get_model_list()
        if not modelList:
            print('Please Train the model first before predicting values')
            return

        for model in model_list:
            print('Restoring previous model from {}'.format(model))
            self.saver.restore(session, model)
            print('Testing...')

            save_name = model_name[:-len(self.MODEL_EXT)] + self.TEST_OUT_SUFFIX
            with open(save_name, 'w') as f:
                ignored_sen = 0
                for line in lines:
                    question = line[:-1]

                    answer = self.predict_single(question)
                    if not answer:
                        ignored_sen += 1
                        continue

                    prediction = '{x[0]}{0}\n{x[1]}{1}\n\n'.format(question, self.text_data.sequence2str(answer, clean=True), x=self.SENTENCES_PREFIX)
                    f.write(prediction)
                print('Prediction finished, {}/{} sentences ignored (too long)'.format(ignored_sen, len(lines)))

    def _get_model_list(self):
        model_list = [os.path.join(self.model_dir, f) for f in os.listdir(self.model_dir) if f.endswith(self.MODEL_EXT)]
        return sorted(model_list)

    def interactive_main(self,session):
        print('Initiating interactive mode .. ')
        print('Enter your query or press ENTER to quit!')

        while True:
            question = input(self.SENTENCES_PREFIX[0])
            if question == '' or question == 'exit':
                break

            question_seq = []
            answer = self.predict_single(question, question_seq)
            if not answer:
                print('Out of my scope .. ask something simpler!')
                continue

            print('{}{}'.format(self.SENTENCES_PREFIX[1],self.text_data.sequence2str(answer,cl=True)))

            if self.verbose:
                print(self.text_data.batch_seq2str(question_seq,clean=True,reverse=True))
                print(self.text_data.sequence2str(answer))

            print()

    def interactive_main_twitter(self,session,question):
        #print('Initiating interactive mode .. ')
        #print('Enter your query or press ENTER to quit!')

        #while True:
        #    question = input(self.SENTENCES_PREFIX[0])
        #    if question == '' or question == 'exit':
        #        break

        question_seq = []
        answer = self.predict_single(question, question_seq)
        if not answer:
           return 'Out of my scope .. ask something simpler!'

        return self.text_data.sequence2str(answer,cl=True)

    def predict_single(self, question, question_seq=None):
        #print(self.text_data.test_())
        batch = self.text_data.sentence2enco(question)
        if not batch:
            return None
        #if questionSeq is not None:  # If the caller want to have the real input
        #    questionSeq.extend(batch.encoderSeqs)

        # Run the model
        ops, feedDict = self.model.step(batch)
        output = self.session.run(ops[0], feedDict)  # TODO: Summarize the output too (histogram, ...)
        #print(output)
        answer = self.text_data.deco2sentence(output)
        return answer

    def predict_daemon(self,sentence):
            print(sentence)
            answer=self.predict_single(sentence)
            print(answer)
            return self.text_data.sequence2str(answer,cl = True)

    def close_daemon(self):
        print("Daemon Existing .. ")
        self.session.close()
        print("Done.")

    def load_embedding(self,session):
        # TODO :- see if we need this load embedding model as of now
        with tf.variable_scope("embedding_rnn_seq2seq/rnn/embedding_wrapper",reuse=True):
            embedding_in = tf.get_variable("embedding")
        with tf.variable_scope("embedding_rnn_seq2seq/embedding_rnn_decoder",reuse=True):
            embedding_out = tf.get_variable("embedding")

        variables = tf.get_collection_ref(tf.GraphKeys.TRAINABLE_VARIABLES)
        variables.remove(embedding_in)
        variables.remove(embedding_out)

        # leave if restoring a model #
        if self.global_step != 0:
            return

        # Define new model here #
        # TO DO 406-434#
        embeddings_path = os.path.join('/tmp', self.embedding_source)
        
        embeddings_format = os.path.splitext(embeddings_path)[1][1:]
        print("Loading pre-trained word embeddings from %s " % embeddings_path)
        with open(embeddings_path, "rb") as f:
            header = f.readline()
            vocab_size, vector_size = map(int, header.split())
            binary_len = np.dtype('float32').itemsize * vector_size
            initW = np.random.uniform(-0.25,0.25,(len(self.text_data.var_word_id), vector_size))
            for line in range(vocab_size):
                word = []
                while True:
                    ch = f.read(1)
                    if ch == b' ':
                        word = b''.join(word).decode('utf-8')
                        break
                    if ch != b'\n':
                        word.append(ch)
                if word in self.text_data.var_word_id:
                    if embeddings_format == 'bin':
                        vector = np.fromstring(f.read(binary_len), dtype='float32')
                    elif embeddings_format == 'vec':
                        vector = np.fromstring(f.readline(), sep=' ', dtype='float32')
                    else:
                        raise Exception("Unkown format for embeddings: %s " % embeddings_format)
                    initW[self.text_data.var_word_id[word]] = vector
                else:
                    if embeddings_format == 'bin':
                        f.read(binary_len)
                    elif embeddings_format == 'vec':
                        f.readline()
                    else:
                        raise Exception("Unkown format for embeddings: %s " % embeddings_format)

        # PCA Decomposition to reduce word2vec dimensionality
        if self.embedding_size < vector_size:
            U, s, Vt = np.linalg.svd(initW, full_matrices=False)
            S = np.zeros((vector_size, vector_size), dtype=complex)
            S[:vector_size, :vector_size] = np.diag(s)
            initW = np.dot(U[:, :self.embedding_size], S[:self.embedding_size, :self.embedding_size])

        # Initialize input and output embeddings
        session.run(embedding_in.assign(initW))
        session.run(embedding_out.assign(initW))

    def manage_previous_model(self,session):
        model_name = self._get_model_name()

        if os.listdir(self.model_dir):
            if self.reset:
                print('Reseting by destroyinh previous model at {}'.format(model_name))

            elif os.path.exists(model_name):
                print('Restoring previous model from {}'.format(model_name))
                self.saver.restore(session,model_name)

            elif self._get_model_list():
                print('Conflicting previous models found.')
                raise RuntimeError('Check previous models in \'{}\'or try with keep_all flag)'.format(self.model_dir))

            else:
                print('No previous model found. Cleaning for sanity .. at{}'.format(self.model_dir))
                self.reset = True

            if self.reset:
                file_list = [os.path.join(self.model_dir,f) for f in os.listdir(self.model_dir)]
                for f in file_list:
                    print('Removing {}'.format(f))
                    os.remove(f)

        else:
            print('Nothing apriorily exists, starting fresh from direwctory: {}'.format(self.model_dir))

#this function is to save model after every few iteration
    
    def _save_session(self,session):
        print('Chkpnt reached: saving model .. ')
        self.save_model_params()
        self.saver.save(session,self._get_model_name())
        print('Model saved.')

    def load_model_params(self):
        #TO DO 494-556#
        self.model_dir = os.path.join(self.root_dir,self.MODEL_DIR_BASE)
        if self.model_tag:
            self.model_dir += '-'+self.model_dir

        config_name = os.path.join(self.model_dir,self.CONFIG_FILENAME)
        if not self.reset and not self.create_dataset and os.path.exists(config_name):
            config = configparser.ConfigParser()
            config.read(config_name)

            current_version = config['General'].get('version')
            if current_version != self.CONFIG_VERSION:
                raise UserWarning('Current configuration version {0} different from {1}.Do manual changes on \'{2}\''.format(current_version,self.CONFIG_VERSION,config_name))

            self.global_step = config['General'].getint('global_step')
            self.max_length = config['General'].getint('max_length')
            self.watson_mode = config['General'].getboolean('watson_mode')
            self.auto_encode = config['General'].getboolean('auto_encode')
            self.corpus = config['General'].get('corpus')
            self.dataset_tag = config['General'].get('dataset_tag', '')
            self.hidden_size = config['Network'].getint('hidden_size')
            self.num_layers = config['Network'].getint('num_layers')
            self.embedding_size = config['Network'].getint('embedding_size')
            self.init_embeddings = config['Network'].getboolean('init_embeddings')
            self.softmax_samples = config['Network'].getint('softmax_samples')
            self.current_epoch = config['General'].getint('epoch')
            # Print the restored params
            print()
            print('Warning: Restoring parameters:')
            print('global_step: {}'.format(self.global_step))
            print('maxLength: {}'.format(self.max_length))
            print('watsonMode: {}'.format(self.watson_mode))
            print('autoEncode: {}'.format(self.auto_encode))
            print('corpus: {}'.format(self.corpus))
            print('datasetTag: {}'.format(self.dataset_tag))
            print('hiddenSize: {}'.format(self.hidden_size))
            print('numLayers: {}'.format(self.num_layers))
            print('embeddingSize: {}'.format(self.embedding_size))
            print('initEmbeddings: {}'.format(self.init_embeddings))
            print('softmaxSamples: {}'.format(self.softmax_samples))
            print('current_epoch: {}'.format(self.current_epoch))
            print()

    def save_model_params(self):
        config = configparser.ConfigParser()
        general = {
            "version": self.CONFIG_VERSION,
            "global_step": str(self.global_step),
            "max_length": str(self.max_length),
            "watson_mode": str(self.watson_mode),
            "auto_encode":  str(self.auto_encode),
            "corpus": self.corpus,
            "dataset_tag": self.dataset_tag,
            "epoch": self.current_epoch
        }
        config["General"] = general
        network = {
            "hidden_size": str(self.hidden_size),
            "num_layers": str(self.num_layers),
            "embedding_size": str(self.embedding_size),
            "init_embeddings": str(self.init_embeddings),
            "softmax_samples": str(self.softmax_samples)
        }
        config["Network"] = network
        # Keep track of the learning params (but without restoring them)
        config['Training (won\'t be restored)'] = {}
        config['Training (won\'t be restored)']['learning_rate'] = str(self.learning_rate)
        config['Training (won\'t be restored)']['batch_size'] = str(self.batch_size)

        with open(os.path.join(self.model_dir, self.CONFIG_FILENAME), 'w') as configFile:
            config.write(configFile)

    def _get_summary_name(self):
        return self.model_dir

    def _get_model_name(self):
        model_name = os.path.join(self.model_dir,self.MODEL_NAME_BASE)
        if self.keep_all:
            mdoel_name += '-' + str(self.global_step)

        return model_name + self.MODEL_EXT

#this is used to define the place where and model will be instantiated
    
    def get_device(self):
        if 'cpu' in self.device:
            return self.device
        elif 'gpu' in self.device:
            return self.device
        elif self.device is None:
            return None
        else:
            print('Warning: Error detected in device name: {}, switch to default device'.format(self.device))
            return None


if __name__=="__main__":
	bot = Bot()
	bot.main()    
