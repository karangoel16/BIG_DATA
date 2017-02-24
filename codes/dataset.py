import os
import nltk
import pickle
import pandas as pd
import csv
import random
from cornell import cornell_data
from scotus import scotus
from ubuntu import ubuntu

class dataset:
    
    '''
        We need the DirName that is the dir where our code is present so that we can run the set on it
    '''
    def __init__(self,DirName):
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
        if not os.path.exists(DirName):
            print('INCORRECT PATH ENTERED FOR THE CORPUS');
            return ;
        self.DirName=DirName;
        #try:
            #corpus_data=read.table((DirName+"/Database/CorpusData.csv"));
        print("Enter the number of the database you want to read?");
        print("1. Ubuntu");
        print("2. Scotus");
        print("3. Cornel");
        choice=int(input());
        dict_temp={};
        '''
            This is place where we add the different corpus
        '''
        with open((DirName+"/Database/CorpusData.csv")) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if reader.line_num==(choice+1):
                    dict_temp=row;
                    break;
        self.var_corpus_name=dict_temp['CorpusName'];
        self.var_corpus_dict=dict_temp['Dictionary_Add'];
        self.var_corpus_loc=dict_temp['Corpus Unique Path'];
        '''We have to take maximum length from the user as the arguments in the final draft of the program'''
        self.var_max_length=10;
        self.var_word_id={};#this is to compute the number to word
        self.var_id_word={};#this is to compute the word to number
        self.var_pad=-1;
        self.var_eos=-1;
        self.var_unique=-1;
        self.var_unknown=-1;
        self.var_token=-1;
        self.var_sam_train=[];
        self.load_data(); 
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

    def load_dataset(self):
        with open(os.path.join(self.DirName,self.var_corpus_dict),"rb") as f:
            data=pickle.load(f);
            self.var_word_id=data['word_id'];
            self.var_id_word=data['id_word'];
            self.var_sam_train=data['sample']
            self.var_pad=data['<pad>'];
            self.var_token=data['<go>'];
            self.var_eos=data['<eos>'];
            self.var_unknown=data['<unknown>'];
    
    def next_batch(self):
        random.shuffle(self.var_sam_train);
        
t=dataset('/home/karan/Documents/GIT_HUB/BIG_DATA');#we have to enter the path Name    
