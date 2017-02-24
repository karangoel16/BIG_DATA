import os
import nltk
import pickle
from Collections import OrderedDict
import pandas as pd
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
            
        '''
        if not os.path.exists(DirName):
            print('INCORRECT PATH ENTERED FOR THE CORPUS');
            return ;
        '''We have to take maximum length from the user as the arguments in the final draft of the program'''
        self.var_max_length=10;
        self.var_word_id={};#this is to compute the number to word
        self.var_id_word={};#this is to compute the word to number
        self.var_pad=-1;
        self.var_eos=-1;
        self.var_unique=-1;
        self.var_unknown=-1;
        self.var_conversation=[];#this is to keep conversation from everywhere
        #print(DirName+"/Corpus/Cornel/");
        self.var_cornell=cornell_data(DirName+"/Corpus/Cornel/");#this is where we keep all the cornell data
        #self.var_ubuntu=ubuntu(DirName+"/Corpus/ubuntu/");#this is where we keep all the ubuntu data
        #self.var_scotus=scotus(DirName+"/Corpus/scotus/");#this is where we get all the scotus data
        self.var_conversation.append(self.var_cornell.getconversation());#this will change the conversation to this model
        #self.var_conversation.append(self.var_scotus.getconversation());#this will append the conversation to this place
    
    def conv_set(self):
        
        for i in range(len(self.conversation)):
        
            var_user_1=self.conversation["lines"][i];#this is for user 1
            
            var_user_2=self.conversation["lines"][i+1];#this is for user 2
            
            var_user_1_word=self.token_(var_user_1);
            
            var_user_2_word=self.token_(var_user_2);
            
            #if var_user_1_word and var_user_2_word:
                #we will call the functions from here , we have checked that the conversation going on is legitimite
            
    def token_(self,line,var_target):
        
        var_word=[];
        
        var_sent_token=nltk.sent_tokenize(line);
        
        for t in len(var_sent_token):
            
            if not var_target:
                t=len(var_sent_token)-1-t;#we will rotate the array for the false
        
            var_token=nltk.word_tokenize(var_sent_token[t]);
            
            if len(var_word)+len(var_token)<=self.var_max_length:
                
                var_temp=[];
                
                for token in var_token:
                
                    var_temp.append(self.var_word_id(token));
                
                if var_target:
                
                    word=word+var_temp;
                
                else:
                    word=var_temp+word; 
            
            else:
                
                break;#when the length goes above the maxlength then we break the charachter
        
        return word;
    
    def load_data(self):
        exist_dataset=False;#if the data file does not exist
        if not os.path.exists(os.path.join(self.DirName,"test"):
            exist_dataset=True;
        else:
            t=os.path.join(self.DirName,)
    def word_id(self,word,add):
        word=word.lower();#to convert word into the lower charachter of words
        if word in self.var_word_id:
            return self.var_word_id[word];
        else:
            if add:
                word_len=self.var_word_id;
                self.var_word_id[word]=word_len;#this is to add the dictionary of the word in the list to encode
                self.var_id_word[word_len]=word;#this is to add the dictionary of the word to decode
            else:
                self.var_word_id[word]=self.var_unknown;
                word_len=self.var_unknown;
        return word_len;#this is to add the word back into the dictionary

    def save_dataset(self):
        with open(os.path.join(DirName,"test"),'wb') as f:
            data={'word_id':seld.var_word_id,
                  'id_word':self.var_id_word,
                 };
            pickle.dump(data,f,-1);

    def load_dataset(self):
        with open(os.path.join(DirName,"test","rb") as f:
                  data=pickle.load(f);
                  self.var_word_id=data['word_id'];
                  self.var_id_word=data['id_word'];
     
    def batch(self):
        
        
        
t=dataset('/home/karan/Documents/GIT_HUB/BIG_DATA');#we have to enter the path Name    
