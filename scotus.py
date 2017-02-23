import os
import sys
import argparse

class scotus:
    def __init__(self,dirName):
        '''
            dirName is the place where we will get ubuntu dataset
        '''
        try:
            test=sys.argv[1];
        except:
            test="normal";
        self.max_dir=10;
        self.conversation=[];#this is the data dictionary
        dir=os.path.join(dirName,'');
        #this is to call for how many directories you want to 
        num_sub=0;
        if(not os.path.exists(dir)):
            print("Incorrect Directory")
            return ;
        else:
            for f in os.scandir(dir):
                self.conversation.append({'line':self.loadlines(f)});
            print(self.conversation);
                
    #the function has been kept with similar names in all the corpus to keep uniformity
    def loadlines(self,filename):
        '''
            this function takes all the data from the user and is able to return a dictionary with the string
            of the lines spoken in that particular file
        '''
        lines=[];#this will keep the lines in the data set
        with open(filename.path,'r') as f:
            #this will open the lines to extract the lines from the user
            for line in f:
                #point 1: we are able to reach to the line level now we need to extract the lines from the user
                values = line.split(":");#this will split the lines in tabs
                lines.append({"text":values[1]});
        return lines;
    def getconversation(self):
        return self.conversation;
