import configparser
import os
import tensorflow as tf
import numpy as np
import math
import csv
import argparse
from time import time
from dataset import dataset
from model import RNNModel

# Directory name of parent dir # 
DirName='/'.join(os.path.dirname(os.path.realpath(__file__)).split('/')[:-1])
config = configparser.ConfigParser()
config.read(DirName+"/Database/Config.ini");


# Main class file which containf all the methods for various model initialisation, training and testing # 
class Bot:
    
    # Intialising all the parameters needed in the code #
    def __init__(self):
        
        self.model = None
        self.verbose = None

        # Tensorflow utilities for convenience saving/logging #
        self.writer = None
        self.saver = None
        self.model_dir = ''
        self.global_step = 0
        self.DirName = DirName	
        self.session = None
        self.model_tag = None

        # Initialization Params are provided using Config.ini stored in Database/ #
        self.root_dir = DirName
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

    # Runtime param: provided using Config.ini #    
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
        self.file_ = config['General'].getboolean('file')

    # Some command line argument params #
    def load_args(self):
        parser = argparse.ArgumentParser(description='SmartGator config arguments.')
        parser.add_argument("--test", "-t", dest="test", action="store_true",
                            help="Argument to run the code in test mode.")
        parser.add_argument("--reset", "-r", dest="reset", action="store_true",
                            help="To remove previously saved model and start fresh.")
        parser.add_argument("-w", dest="word2vec", action="store_true",
                            help="to run the code in word2vec mode.")
        parser.add_argument("-a", dest="attention", action="store_true",
                            help="to run the code in attention mechanism code.")
        parser.add_argument("-d", dest="device", action="store", default=None,
                            help='gpu/cpu device like /gpu:0|/gpu:1|/cpu:0.')
        self.args = parser.parse_args()

    def update_settings(self):
        if self.args.test:
            self.test = True
        if self.args.reset:
            self.reset = True
        if self.args.word2vec:
            self.init_embeddings = True
        if self.args.attention:
            self.attention = True
        if self.args.device:
            self.device = self.args.device

    ######################### Main function of class Bot ##########################
    # As per the input/configuration provided, corresponding task is accomplished #        
    def main(self):

        print("SmartGator Intelligent chatbot")

        self.root_dir = os.getcwd() 
        self.load_config()
        self.load_model_params()
        self.load_args()
        self.update_settings()
        self.text_data = dataset(self.args)

        # RNN Model Initialized #
        self.model = RNNModel(self.text_data, self.args)

        # Handlers to write and save learned models #
        self.writer = tf.summary.FileWriter(self._get_summary_name())
        self.saver = tf.train.Saver(max_to_keep=200, write_version=tf.train.SaverDef.V1)

        self.session = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False)
        )

        print("Initializing tf variables")
        self.session.run(tf.global_variables_initializer())

        # If a previous model exists load it and procedd from last run step #
        self.manage_previous_model(self.session)
        
        # If using word2vec model we need to laod word vectors #
        if self.init_embeddings:
            self.load_embedding(self.session)

        # Twitter Interface up or not #
        if self.twitter:
            return

        # Batch Testing #
        elif self.file_:
            try:
                with open(self.TEST_IN_NAME,"r") as f:
                    try:
                        with open(self.TEST_OUT_SUFFIX,'w') as output:
                            for line in f:
                                output.write(self.predict_daemon(line[:-1])+"\n")
                    except:
                        print("Writing in file is a problem")
            except:
                print("Open file error")

        # Else if in CLI testing mode #
        elif self.test:
             self.interactive_main(self.session);

        # Else in training mode #
        else:
            self.train_model(self.session)

        self.session.close()
        print("Say Bye Bye to SmartGator! ;)")
    
    # Function to train the chatbot #
    def train_model(self, session):
        merged_summaries = tf.summary.merge_all()

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
                    start_time = time()
                    for curr_batch in batches:
                        ops, feed_dict = self.model.step(curr_batch)
                        assert len(ops) == 2
                        _, loss, summary = session.run(ops + tuple([merged_summaries]), feed_dict)
                        self.writer.add_summary(summary, self.global_step)
                        self.global_step += 1
                        local_step += 1

                        if self.global_step % 100 == 0:
                            end_time = time()
                            time_consumed = int(end_time - start_time)
                            perplexity = math.exp(float(loss)) if loss < 300 else float("inf")
                            print("----- Step %d/%d -- Loss %.2f -- Perplexity %.2f -- GlobalStep %d -- TimeConsumed %d sec" %  (local_step, len(batches), loss, perplexity, self.global_step, time_consumed))
                            row_data.append([self.global_step, epoch, loss, perplexity])
                            start_time = time()
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

    # List all the available models #
    def _get_model_list(self):
        model_list = [os.path.join(self.model_dir, f) for f in os.listdir(self.model_dir) if f.endswith(self.MODEL_EXT)]
        return sorted(model_list)

    # Command line interface for bot testing #
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

    # Function to interface with twitter module .. receive query and send reply #
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

    # Function to predict output sequence for single input sequence         #
    # Used in interactive_main_twitter, interactive_main                    #
    def predict_single(self, question, question_seq=None):
        #print(self.text_data.test_())
        #print(question)
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

    # Used in Batch testing to call predict_single on every input #
    def predict_daemon(self,sentence):
            print(sentence)
            question_seq=[]
            answer=self.predict_single(str(sentence),question_seq)
            #print(answer)
            return self.text_data.sequence2str(answer,cl = True)

    def close_daemon(self):
        print("Daemon Exiting .. ")
        self.session.close()
        print("Done.")

    # Load word embeddings for word2vec model #
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

        # Change location accordingly #
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

        # PCA Decomposition to reduce word2vec dimensionality #
        if self.embedding_size < vector_size:
            U, s, Vt = np.linalg.svd(initW, full_matrices=False)
            S = np.zeros((vector_size, vector_size), dtype=complex)
            S[:vector_size, :vector_size] = np.diag(s)
            initW = np.dot(U[:, :self.embedding_size], S[:self.embedding_size, :self.embedding_size])

        # Initialize input and output embeddings #
        session.run(embedding_in.assign(initW))
        session.run(embedding_out.assign(initW))

    # If previous model exsits, we relaunch the training from last run step #
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


    # Save model after every few iteration #
    def _save_session(self,session):
        print('Chkpnt reached: saving model .. ')
        self.save_model_params()
        self.saver.save(session,self._get_model_name())
        print('Model saved.')

    # Loads model params during testing and model restoration #
    def load_model_params(self):
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

    # Saves current state of model when training is interrupted or finished #
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

    # Define the place where model will be instantiated #
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
