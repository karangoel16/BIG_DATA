import os
from cornell import cornell_data
from scotus import scotus
from ubuntu import ubuntu

class dataset:
    def __init__(self,DirName):
        if not os.path.exists(DirName):
            print('INCORRECT PATH ENTERED FOR THE CORPUS');
            return ;
        self.var_conversation=[];#this is to keep conversation from everywhere
        #print(DirName+"/Corpus/Cornel/");
        self.var_cornell=cornell_data(DirName+"/Corpus/Cornel/");#this is where we keep all the cornell data
        #self.var_ubuntu=ubuntu(DirName+"/Corpus/ubuntu/");#this is where we keep all the ubuntu data
        self.var_scotus=scotus(DirName+"/Corpus/scotus/");#this is where we get all the scotus data
        self.var_conversation.append(self.var_cornell.getconversation());#this will change the conversation to this model
        self.var_conversation.append(self.var_scotus.getconversation());#this will append the conversation to this place
t=dataset('/home/karan/Documents/GIT_HUB/BIG_DATA');#we have to enter the path Name    