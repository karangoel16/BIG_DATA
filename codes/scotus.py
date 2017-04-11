import os
import sys

class scotus:
    def __init__(self,dirName):
        dirName=os.path.join(dirName,"")
        print(dirName)
    #here we will check if we do not enter any dirName or directory doesn't exist we simply come out of the cornell
        if not os.path.exists(dirName):
            print("PATH ERROR")
            return ;
        '''if not dirName:
            return ;#this will return the function
        try:
            test=sys.argv[1];
        except:
            test="normal";
        self.max_dir=10;'''
        self.conversation=[];#this is the data dictionary
        dir=os.path.join(dirName,'')
        #this is to call for how many directories you want to 
        num_sub=0;
        print(dir)
        if(not os.path.exists(dir)):
            print("FILE NOT FOUND")
            return ;
        else:
            self.conversation.append({'lines':self.loadlines(os.path.join(dirName, "scotus"))});

    def loadlines(self,filename):
        '''
            this function takes all the data from the user and is able to return a dictionary with the string
            of the lines spoken in that particular file
        '''
        lines = []

        with open(filename, 'r') as f:
            for line in f:
                l = line[line.index(":")+1:].strip()  # Strip name of speaker.

                lines.append({"text": l})

        return lines

    def getconversation(self):
        return self.conversation;

if __name__=="__main__":
	t=scotus('/cise/homes/kgoel/Downloads/BIG_DATA/Corpus/SCOTUS/');
	print(t.getconversation());
