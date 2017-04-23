import os
import nltk
import pickle
import csv
import random
import configparser as cp
from cornell import cornell_data
from scotus import scotus
from ubuntu import ubuntu
from opensub import OpensubsData
import collections
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
    def __init__(self, args):
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
        self.vocabularySize=int(Config.get('General','Vsize'))#this is for the size of the vocab
        self.filterVocab = int(Config.get('General','Fvocab'))
        self.var_word_id={};#this is to compute the number to word
        self.var_id_word={};#this is to compute the word to number
        self.idCount={}
        self.var_max_length=int(Config.get('Dataset','maxLength'));
        self.maxLenEnco=self.var_max_length;
        self.maxLenDeco=self.maxLenEnco+2;
        self.test=Config['General'].getboolean('test')
        self.watson=Config['Bot'].getboolean('watsonMode')
        self.autoencode=Config['Bot'].getboolean('autoEncode')
        dict_temp={};
        self.var_corpus_dict=self.DirName+"/Database/file_dict"+str(self.choice)+Config.get('Dataset','maxLength')+str(self.vocabularySize)+".pkl"
        if self.args.test:
            self.test = True
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
        q=[];
        exist_dataset=False;#if the data file does not exist
        if os.path.exists(self.var_corpus_dict):
            exist_dataset=True;
        if not exist_dataset:
            with open((self.DirName+"/Database/CorpusData.csv")) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if reader.line_num==(self.choice+1) or self.choice==5:
                        dict_temp=row;
                        self.var_corpus_name=dict_temp['CorpusName'];
                        #self.var_corpus_dict=self.DirName+dict_temp['Dictionary_Add'];
                        self.var_corpus_loc=self.DirName+dict_temp['Corpus Unique Path'];
                        #print(self.var_c)
                        path=self.var_corpus_loc;
                        if self.var_corpus_name=='cornell':
                            print('cornell')
                            t=cornell_data(path);
                        elif self.var_corpus_name=='ubuntu':
                            print('ubuntu')
#                            path="/tmp"
                            t=ubuntu(path);
                        elif self.var_corpus_name=='scotus':
                            print('scotus')
                            t=scotus(path);
                        elif self.var_corpus_name=='open':
                            print('open subtitle')
                            t=OpensubsData(path);
                        else:
                            print("Not a valid option");
                        q.extend(t.getconversation())
                        #print(t.getconversation())
            self.create_corpus(q);
            #this is to filter the sentences and keywords so that we can reduce the vocab size
            self.filterFromFull()
            self.save_dataset();
        else:
            #we need to load data set here
            self.load_dataset();#this is place where we will load the dataset
        assert self.var_pad == 0    
            
    def create_corpus(self,conversations):
        self.var_pad = self.word_id('<pad>')  # Padding (Warning: first things to add > id=0 !!)
        self.var_token = self.word_id('<go>')  # Start of sequence
        self.var_eos = self.word_id('<eos>')  # End of sequence
        self.var_unknown = self.word_id('<unknown>')  # Word dropped from vocabulary
        #print(self.var_pad)
        for conversation in conversations:
            self.conv_set(conversation);
            
            
    def word_id(self,word,add=True):
        word=word.lower();#to convert word into the lower charachter of word
        if not add:
            wordId = self.var_word_id.get(word, self.var_unknown)
        # Get the id if the word already exist
        elif word in self.var_word_id:
            wordId = self.var_word_id[word]
            self.idCount[wordId] += 1
        # If not, we create a new entry
        else:
            wordId = len(self.var_word_id)
            self.var_word_id[word] = wordId
            self.var_id_word[wordId] = word
            self.idCount[wordId] = 1
#this is to add the word back into the dictionary
        return wordId;
    def save_dataset(self):
        path=self.var_corpus_dict;
        #print(path)
        with open(path,'wb') as f:
            data={'word2id':self.var_word_id,
                  'id2word':self.var_id_word,
                  'idCount':self.idCount,
                  'trainingSamples':self.var_sam_train,
                 };
            pickle.dump(data,f,-1);
        #except:
        #    print("Error in save dataset");

    def load_dataset(self):
        path=self.var_corpus_dict;
        try:
            with open(path,"rb") as f:
                data=pickle.load(f);
                self.var_word_id=data['word2id'];
                self.var_id_word=data['id2word'];
                self.idCount=data.get('idCount',None);
                self.var_sam_train=data['trainingSamples']
                self.var_pad=self.var_word_id['<pad>'];
                self.var_token=self.var_word_id['<go>'];
                self.var_eos=self.var_word_id['<eos>'];
                self.var_unknown=self.var_word_id['<unknown>'];  
        except:
            print("Error in load dataset");

#def sent2enco(self,sent):
    def filterFromFull(self):
        """ Load the pre-processed full corpus and filter the vocabulary / sentences
        to match the given model options
        """

        def mergeSentences(sentences, fromEnd=False):
            """Merge the sentences until the max sentence length is reached
            Also decrement id count for unused sentences.
            Args:
                sentences (list<list<int>>): the list of sentences for the current line
                fromEnd (bool): Define the question on the answer
            Return:
                list<int>: the list of the word ids of the sentence
            """
            # We add sentence by sentence until we reach the maximum length
            merged = []

            # If question: we only keep the last sentences
            # If answer: we only keep the first sentences
            if fromEnd:
                sentences = reversed(sentences)
            for sentence in sentences:
                #print(sentence)
                # If the total length is not too big, we still can add one more sentence
                if len(merged) + 1  <= self.var_max_length: #TOD:check change here
                    if fromEnd:  # Append the sentence
                        merged = [sentence] + merged
                    else:
                        merged = merged + [sentence]
                else:  # If the sentence is not used, neither are the words
                    for w in sentence:
                        self.idCount[w] -= 1
            return merged

        newSamples = []

        # 1st step: Iterate over all words and add filters the sentences
        # according to the sentence lengths
        for inputWords, targetWords in self.var_sam_train:
            #print(inputWords)
            #print(targetWords)
            inputWords = mergeSentences(inputWords, fromEnd=True)
            targetWords = mergeSentences(targetWords, fromEnd=False)

            newSamples.append([inputWords, targetWords])
        words = []


        # 2nd step: filter the unused words and replace them by the unknown token
        # This is also where we update the correnspondance dictionaries
        specialTokens = {  # TODO: bad HACK to filter the special tokens. Error prone if one day add new special tokens
            self.var_pad,
            self.var_token,
            self.var_eos,
            self.var_unknown
        }
        newMapping = {}  # Map the full words ids to the new one (TODO: Should be a list)
        newId = 0
        print(self.vocabularySize)
        selectedWordIds = collections \
            .Counter(self.idCount) \
            .most_common(self.vocabularySize or None)  # Keep all if vocabularySize == 0
        selectedWordIds = {k for k, v in selectedWordIds if v > self.filterVocab}
        selectedWordIds |= specialTokens

        for wordId, count in [(i, self.idCount[i]) for i in range(len(self.idCount))]:  # Iterate in order
            if wordId in selectedWordIds:  # Update the word id
                newMapping[wordId] = newId
                word = self.var_id_word[wordId]  # The new id has changed, update the dictionaries
                del self.var_id_word[wordId]  # Will be recreated if newId == wordId
                self.var_word_id[word] = newId
                self.var_id_word[newId] = word
                newId += 1
            else:  # Cadidate to filtering, map it to unknownToken (Warning: don't filter special token)
                newMapping[wordId] = self.var_unknown
                del self.var_word_id[self.var_id_word[wordId]]  # The word isn't used anymore
                del self.var_id_word[wordId]

        # Last step: replace old ids by new ones and filters empty sentences
        def replace_words(words):
            valid = False  # Filter empty sequences
            for i, w in enumerate(words):
                words[i] = newMapping[w]
                if words[i] != self.var_unknown:  # Also filter if only contains unknown tokens
                    valid = True
            return valid

        self.var_sam_train.clear()

        for inputWords, targetWords in newSamples:
            valid = True
            valid &= replace_words(inputWords)
            valid &= replace_words(targetWords)
            valid &= targetWords.count(self.var_unknown) == 0  # Filter target with out-of-vocabulary target words ?

            if valid:
                self.var_sam_train.append([inputWords, targetWords])  # TODO: Could replace list by tuple

        self.idCount.clear()  # Not usefull anymore. Free data   
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
            var_batch.var_decoder[i] = var_batch.var_decoder[i]+[self.var_pad]*(self.maxLenDeco-len(var_batch.var_decoder[i]))
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
    print(t.vocab_size())
