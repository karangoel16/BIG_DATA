import os
import nltk
import pickle
import csv
import random
import configparser as cp
from cornell import cornell_data
from scotus import scotus
from ubuntu import ubuntu
import numpy as np

class batch:
    """Struct containing batches info
    """
    def __init__(self):
        self.var_encoder = []
        self.var_decoder = []
        self.var_target = []
        self.var_weight = []
        
class dataset:
    '''
        We need the DirName that is the dir where our code is present so that we can run the set on it
    '''
    def __init__(self):
        '''
            args
                1.Dataset to be loaded
                2.corpus to load 
                3.maxlength
                4.
            variable
                1.word to id dictionary
                2.id to word dictionary
                3.padding 
                4.ending of line statement
                5.unique
                6.pad
                7.unknown
                8.Sample Training
            
        '''
        self.DirName='/'.join(os.path.dirname(os.path.realpath(__file__)).split('/')[:-1]);
        #print(self.DirName)
        if not os.path.exists(self.DirName):
            print('INCORRECT PATH ENTERED FOR THE CORPUS');
            return ;
        Config = cp.ConfigParser();
        config_file = self.DirName + "/Database/Config.ini"
        #print(config_file);
        Config.read(config_file);
        self.choice=int(Config.get('Dataset','choice'));
        self.batch_size=int(Config.get('General','batchSize'));
        self.var_pad=-1;
        self.var_eos=-1;
        #self.var_unique=-1;
        self.var_unknown=-1;
        self.var_token=-1;
        self.var_sam_train=[];
        self.var_word_id={};#this is to compute the number to word
        self.var_id_word={};#this is to compute the word to number
        self.var_max_length=int(Config.get('Dataset','maxLength'));
        self.maxLenEnco=self.var_max_length;
        self.maxLenDeco=self.maxLenEnco+2;
        self.test=Config['General'].getboolean('test')
        self.watson=Config['Bot'].getboolean('watsonMode')
        self.autoencode=Config['Bot'].getboolean('autoEncode')
        dict_temp={};
        self.var_corpus_dict=self.DirName+"/Database/file_dict"+Config.get('Dataset','maxLength')+".p"
#we will save all the values in the dictionary in one go and will save this file
        self.load_data();
        print('Conversation loaded.')
        #except:
        #    print("Not able to connect to the database (check github)");
        #    return;
    def conv_set(self,conversation):
        #print(len(conversation))
        '''
            here we extract the lines and then from lines gives token to the sentence which are correct 
        '''
        for i in range(len(conversation["lines"])-1):
            #print(i);
            #print(conversation["lines"][i])
            var_user_1=conversation["lines"][i];#this is for user 1
            
            var_user_2=conversation["lines"][i+1];#this is for user 2
            
            var_user_1_word=self.token_(var_user_1["text"]);
            
            var_user_2_word=self.token_(var_user_2["text"],True);
            if var_user_1_word and var_user_2_word:
                #we will call the functions from here , we have checked that the conversation going on is legitimite
                self.var_sam_train.append([var_user_1_word,var_user_2_word]);
                #print(var_user_1_word)
                #print(self.sequence2str(var_user_1_word))
            
    def token_(self,line,var_target=False):
        
        var_word=[];
        
        var_sent_token=nltk.sent_tokenize(line);
        for t in range(len(var_sent_token)):
            
            if not var_target:
                t=len(var_sent_token)-1-t;#we will rotate the array for the false
        
            var_token=nltk.word_tokenize(var_sent_token[t]);
            
            if len(var_word)+len(var_token)<=self.var_max_length:
                
                var_temp=[];
                
                for token in var_token:
                
                    var_temp.append(self.word_id(token));
                
                if var_target:
                
                    var_word=var_word+var_temp;
                
                else:
                    var_word=var_temp+var_word; 
            
            else:
                
                break;#when the length goes above the maxlength then we break the charachter
        
        return var_word;
    
    def load_data(self):
        exist_dataset=False;#if the data file does not exist
        if os.path.exists(self.var_corpus_dict):
            exist_dataset=True;
        if not exist_dataset:
            with open((self.DirName+"/Database/CorpusData.csv")) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if reader.line_num==(self.choice+1) or self.choice==4:
                        dict_temp=row;
                        self.var_corpus_name=dict_temp['CorpusName'];
                        #self.var_corpus_dict=self.DirName+dict_temp['Dictionary_Add'];
                        self.var_corpus_loc=self.DirName+dict_temp['Corpus Unique Path'];
                        print(self.var_c)
                        break;  
            path=self.var_corpus_loc;
            if self.var_corpus_name=='cornell':
                t=cornell_data(path);
            elif self.var_corpus_name=='ubuntu':
                t=ubuntu(path);
            elif self.var_corpus_name=='scotus':
                t=scotus(path);
            else:
                print("Not a valid option");
            self.create_corpus(t.getconversation());
            self.save_dataset();
        else:
            #we need to load data set here
            self.load_dataset();#this is place where we will load the dataset
            
            
    def create_corpus(self,conversations):
        self.var_pad = self.word_id('<pad>')  # Padding (Warning: first things to add > id=0 !!)
        self.var_token = self.word_id('<go>')  # Start of sequence
        self.var_eos = self.word_id('<eos>')  # End of sequence
        self.var_unknown = self.word_id('<unknown>')  # Word dropped from vocabulary
        for conversation in conversations:
            self.conv_set(conversation);
            
            
    def word_id(self,word,add=True):
        word=word.lower();#to convert word into the lower charachter of words
        if word in self.var_word_id:
            return self.var_word_id[word];
        else:
            if add:
                word_len=len(self.var_word_id);
                self.var_word_id[word]=word_len;#this is to add the dictionary of the word in the list to encode
                self.var_id_word[word_len]=word;#this is to add the dictionary of the word to decode
            else:
                self.var_word_id[word]=self.var_unknown;
                word_len=self.var_unknown;
        return word_len;#this is to add the word back into the dictionary

    def save_dataset(self):
        path=self.var_corpus_dict;
        #print(path)
        with open(path,'wb') as f:
            data={'word_id':self.var_word_id,
                  'id_word':self.var_id_word,
                  'sample':self.var_sam_train,
                  '<pad>':self.var_pad,
                  '<unknown>':self.var_unknown,
                  '<eos>':self.var_eos,
                  '<go>':self.var_token
                 };
            pickle.dump(data,f,-1);
        #except:
        #    print("Error in save dataset");

    def load_dataset(self):
        path=self.var_corpus_dict;
        try:
            with open(path,"rb") as f:
                data=pickle.load(f);
                self.var_word_id=data['word_id'];
                self.var_id_word=data['id_word'];
                self.var_sam_train=data['sample']
                self.var_pad=data['<pad>'];
                self.var_token=data['<go>'];
                self.var_eos=data['<eos>'];
                self.var_unknown=data['<unknown>'];
        except:
            print("Error in load dataset");

#def sent2enco(self,sent):
        
#This program is to return all values to the chatbot where it will call again
    def id_seq(self,var_dec):
        var_seq=[];
        for i in var_dec:
            var_seq.append(np.argmax(var_dec))
        return var_seq;
    def test(self):
        print("Dataset Test Confirmed");
    def sequence2str(self,seq,cl=False,reverse=False):
        if not seq:
            return None;
        if not cl:
            return ' '.join([self.var_id_word[idq] for idq in seq]);
        var_sent=[];
        for word_id in seq:
            if word_id == self.var_eos:
                break;
            elif word_id != self.var_pad and word_id != self.var_token:
                var_sent.append(self.var_id_word[word_id])
        if reverse:  # Reverse means input so no <eos> (otherwise pb with previous early stop)
            var_sent.reverse()
        return self.detokenize(var_sent);
    
    def sentence2enco(self,var_sent):
        if var_sent == "":
            return None;#this is to check if the sentence which we have sent has some value in the string or not
        var_tokens=nltk.word_tokenize(var_sent);
        if len(var_tokens) > self.var_max_length:
            return None; #since we have not trained our data set on this datas
        word_id=[];
        for t in var_tokens:
            word_id.append(self.word_id(t,False));#we do not want new element should be added to the dictionary as this might not be correct word too
        '''We need to create batch so that it can be sent to the model inside'''
        var_batch=self.create_batch([[word_id,[]]])
        return var_batch
    def vocab_size(self):
        return len(self.var_word_id)
    def sample_size(self):
        return len(self.var_sam_train)      
    def create_batch(self,var_samples):
        var_batch=batch()
        var_batch_size=len(var_samples)
        for i in range(var_batch_size):
            var_sample=var_samples[i]
            #TODO Mode check:
            if not self.test and self.watson:
                var_sample = list(reversed(var_sample))
            if not self.test and self.autoencode:
                k=random.randint(0,1);
                var_sample=(var_sample[k],var_sample[k])
            var_batch.var_encoder.append(list(reversed(var_sample[0])))
            var_batch.var_decoder.append([self.var_token]+var_sample[1]+[self.var_token])
            #print(len(var_batch.var_encoder[i]))
            #print(len(var_batch.var_decoder[i]))
            #print(self.var_max_length)
            var_batch.var_target.append(var_batch.var_decoder[-1][1:])
            assert len(var_batch.var_encoder[i])<=self.maxLenEnco
            assert len(var_batch.var_decoder[i])<=self.maxLenDeco
            var_batch.var_encoder[i] = [self.var_pad]*(self.maxLenEnco-len(var_batch.var_encoder[i]))+var_batch.var_encoder[i]
            var_batch.var_decoder[i] = [self.var_pad]*(self.maxLenDeco-len(var_batch.var_decoder[i]))+var_batch.var_decoder[i]
            var_batch.var_target[i]=var_batch.var_target[i] +[self.var_pad]*(self.maxLenDeco-len(var_batch.var_target[i]))
            var_batch.var_weight.append([1.0]*len(var_batch.var_target[i]+[0.0]*(self.maxLenDeco-len(var_batch.var_target[i]))))
            ##need to write more code here
        var_encoders=[];
        #print(var_batch.var_encoder)
        for i in range(self.maxLenEnco):
            var_encoder=[]
            for j in range(var_batch_size):
                #print(var_batch.var_encoder[j][i])
                var_encoder.append(var_batch.var_encoder[j][i])
            var_encoders.append(var_encoder)
        var_batch.var_encoder=var_encoders
        #print(var_batch.var_encoder)
        var_decoders=[]
        var_targets=[]
        var_weights=[]
        for i in range(self.maxLenDeco):
            var_decoder=[]
            var_target=[]
            var_weight=[]
            for j in range(var_batch_size):
                #print(var_batch.var_encoder[j][i])
                var_decoder.append(var_batch.var_decoder[j][i])
                var_target.append(var_batch.var_target[j][i])
                var_weight.append(var_batch.var_weight[j][i])
            var_decoders.append(var_decoder)
            var_targets.append(var_target)
            var_weights.append(var_weight)
        var_batch.var_decoder=var_decoders
        var_batch.var_weight=var_weights
        var_batch.var_target=var_targets
        return var_batch   
    
    def getBatches(self):
        """Prepare the batches for the current epoch
        Return:
            list<Batch>: Get a list of the batches for the next epoch
        """
        random.shuffle(self.var_sam_train);

        batches = []

        def genNextSamples():
            """ Generator over the mini-batch training samples
            """
            for i in range(0, len(self.var_sam_train), self.batch_size):
                yield self.var_sam_train[i:min(i + self.batch_size,  len(self.var_sam_train))]

        for samples in genNextSamples():
            var_batch = self.create_batch(samples)#this function is yet to be implemented
            batches.append(var_batch)
        return batches
   
    def detokenize(self, tokens):
        return " ".join(tokens).strip().capitalize()
        return ''.join([
            ' ' + t if not t.startswith('\'') and
                       t not in string.punctuation
                    else t
            for t in tokens]).strip().capitalize()
    
    def batch_seq2str(self,var_batch,seq_id=0):
        var_seq=[]
        for i in range(len(var_batch)):
            var_seq.append(var_batch[i][seq_id])
        return self.sequence2str(var_seq);
    
    def deco2sentence(self,decoder_output):
        var_sequence=[]
        if decoder_output==None:
            return None    
        for out in decoder_output:
            var_sequence.append(np.argmax(out))
        return var_sequence;
if __name__ == "__main__":        
    t=dataset();#we have to enter the path Name    
    print(t.getBatches())
