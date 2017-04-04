import configparser
#import datetime  # Chronometer
import os
import tensorflow as tf
import numpy as np
import math
#Todo:- ask karan to have first character of class name as capital
from codes.dataset import dataset
from codes.model import RNNModel

class Bot:
    "Bot framework which integrates all the component"

    def __init__(self):
        #TODO:- Instead of using command line args we will go for config only
        # set the appropriate values from config compared to what was use as args
        # in original code

        self.text_data = None  # Dataset
        self.rnn_model = None  # Sequence to sequence model

        # Tensorflow utilities for convenience saving/logging
        self.writer = None
        self.saver = None
        self.model_dir = ''
        self.global_step = 0

        self.session = None

        # Filename and directories constants
        # Todo:- Move following to seperate config files
        self.root_dir
        self.MODEL_DIR_BASE = 'save/model'
        self.MODEL_NAME_BASE = 'model'
        self.MODEL_EXT = '.ckpt'
        self.CONFIG_FILENAME = 'params.ini'
        self.CONFIG_VERSION = '0.4'
        self.TEST_IN_NAME = 'data/test/samples.txt'
        self.TEST_OUT_SUFFIX = '_predictions.txt'
        self.SENTENCES_PREFIX = ['Q: ', 'A: ']

    def load_config(self):
        #Todo:- Load all the required values from the required configs
        self.keep_all = False
        self.ecpochs = 30
        self.learning_rate = 1e-4
        self.save_ckpt_at = 1000

    def main(self, **kwargs):
        #Todo:- sample call for Bot().main(rootdir="..", model="..", ....)

    def _save_session(self, session):
        self.saver.save(session, self._model_name())

    # Implementtion done, Testing remains
    def train_model(self, session):
        merged_summaries = tf.summary.merge_all()

        #Todo:- See if we can try making it possible to restore from previous run
        if self.global_step == 0:
            self.writer.add_graph(session.add_graph)

        print('Training begining (press Ctrl+C to save and exit)...')

        try:
            for epoch in range(self.epochs):
                print(
                      "\n----- Epoch {}/{} ; (lr={}) -----".format(
                        e+1,
                        self.epochs,
                        self.learning_rate
                        )
                      )
                batches = dataset.next_batch()

                for curr_batch in batches:
                    ops, feed_dict = self.rnn_model.step()
                    assert len(ops) == 2
                    _, loss, summary = session.run(ops + tuple([merged_summaries]), feed_dict)
                    self.writer.add_summary(summary, self.global_step)
                    self.global_step += 1

                    if self.globStep % 100 == 0:
                        perplexity = math.exp(float(loss)) if loss < 300 else float("inf")
                        print("----- Step %d -- Loss %.2f -- Perplexity %.2f" % (self.globStep, loss, perplexity))

                    #Save checkpoint
                    if self.global_step % self.save_ckpt_at == 0:
                        self._save_session(session)

        except (KeyboardInterrupt, SystemExit):
            print("Saving state and Exiting the program")

        self._save_session(session)

    def prediction_set(self, session):
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

    # Implementation done, Testing remains
    def _model_name(self):
        model_name = os.path.join(self.model_dir, self.MODEL_NAME_BASE)
        if self.keep_all:
            model_name += '-' + str(self.global_step)
        return model_name + self.MODEL_EXT

    def predict_single(self, question, questionSeq=None):

        batch = self.text_data.sentence2enco(question)
        if not batch:
            return None
        #if questionSeq is not None:  # If the caller want to have the real input
        #    questionSeq.extend(batch.encoderSeqs)

        # Run the model
        ops, feedDict = self.model.step(batch)
        output = self.session.run(ops[0], feedDict)  # TODO: Summarize the output too (histogram, ...)
        answer = self.text_data.deco2sentence(output)

        return answer

    def predict_daemon(self,sentence):
        return self.text_data.sequence2str(
            self.predict_single(sentence),
            clean = True
            )

    def close_daemon(self):
        print("Daemon Existing .. ")
        self.session.close()
        print("Done.")

    def load_embedding(self,session):
        with tf.variable_scope("embedding_rnn_seq2seq/RNN/EmbeddingWrapper",reuse=True):
            embedding_in = tf.get_variable("embedding")
        with tf.variable_scope("embedding_rnn_seq2seq/embedding_rnn_decoder",reuse=True):
            embedding_out = tf.get_variable("embedding")

        variables = tf.get_collection_ref(tf.GraphKeys.TRAINABLE_VARIABLES)
        variable.remove(embedding_in)
        variables.remove(embedding_out)

        # leave if restoring a model #
        if self.globStep != 0:
            return

        # Define new model here #
        # TO DO 406-434#

    def manage_previous_mode(self,session):
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

    def _save_session(self.session):
        tqdm.write('Chkpnt reached: saving model .. ')
        self.save_model_params()
        self.saver.save(session,self._get_model_name())
        tqdm.write('Model saved.')

    def _get_model_list(self):
        return [os.path.join(self.model_dir,f) for f in os.listdir(self.model_dir) if f.endswith(self.MODEL_EXT)]

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

            # Print the restored params
            print()
            print('Warning: Restoring parameters:')
            print('globStep: {}'.format(self.glob_step))
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
            print()

    def save_model_params(self):
        #TO DO 560-586#

    def _get_summary_name(self):
        return self.model_dir

    def _get_model_name(self):
        model_name = os.path.join(self.model_dir,self.MODEL_NAME_BASE)
        if self.keep_all:
            mdoel_name += '-' + str(self.global_step)

        return model_name + self.MODEL_EXT

    def get_device(self):
        if self.get_device == 'cpu':
            return '/cpu:0'
        elif self.get_device == 'gpu':
            return '/gpu:0'
        elif self.get_device is None:
            return None
        else:
            print('Warning: Error detected in devoce name: {}, switch to default devicde'.format(self.get_device))
            return None


    
