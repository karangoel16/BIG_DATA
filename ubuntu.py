import os

'''
	this program is written with the motive that the function will be able to extract the conversation from the ubuntu corpus which is available to us
	from the extractor 
	Extractor Reference:https://github.com/rkadlec/ubuntu-ranking-dataset-creator

'''

class ubuntu:
    def __init__(self,dirName):
        '''
            dirName is the place where we will get ubuntu dataset
        '''
        self.max_dir=10;
        self.conversation=[];#this is the data dictionary
        dir=os.path.join(dirName,'dialogs');
        #this is to call for how many directories you want to 
        num_sub=0;
        if(not os.path.exists(dir)):
            print("Incorrect Directory")
            return 0;
        else:
            for sub in os.scandir(dir):#this is to open the directory and scan the sub directory
                if sub.is_dir():
                    for f in os.scandir(sub.path):#this is to scan all the files within the ubuntu data file
                        if f.name.endswith('.tsv'):
                            #now we have to add the code to read the conversation from the file
                            #now we can append in the lines from the loadfunction to the conversation which we can load later
                            self.conversation.append({"line":self.loadlines(f)});
                            return ;
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
                values = line.split("\t");#this will split the lines in tabs
                lines.append({"text":values[3]});
        return lines;
    def getconversation(self):
        return self.conversation;
        

