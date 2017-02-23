import os
class cornell_data:
    def __init__(self,dirName):
        self.lines={};
        self.conversation={};
        LINE_FIELD=["lineID","characterID","movieID","character","text"]
        CONVERSATION_FIELD=["character1ID","character2ID","movieID","utteranceIDs"];
        self.lines = self.loadlines(os.path.join(os.getcwd()+dirName,'movie_lines.txt') , LINE_FIELD)
        self.conversatation=self.loadconversation(os.path.join(os.getcwd()+dirName,'movie_conversations.txt'),CONVERSATION_FIELD);
   
    def loadlines(self,filename,fields):#done with preparing the data of cornel
        '''Here pathname is the file name for the file '''
        lines={};
        try:
            with open(filename,'r',encoding='iso-8859-15') as f:
                for line in f:
                    values=line.split(' +++$+++ ');#here the delimiter is +++$+++
                    lineObj={};
                    for i,field in enumerate(fields):
                        lineObj[field]=values[i];
                    lines[lineObj['lineID']]=lineObj;
        except FileNotFoundError:
            print("FILE NOT FOUND");
        return lines;
    
    def loadconversation(self,filename,fields):
        conversation=[];
        try:
            with open(filename,'r',encoding='iso-8859-15') as f:
                for line in f:
                    values=line.split(' +++$+++ ');#this will split line with the help of the delimiter
                    convObj={};#we will use this to get conversation in order
                    for i,field in enumerate(fields):
                        convObj[field]=values[i];
                    lineIds = eval(convObj["utteranceIDs"]);
                    convObj["lines"] = []
                    for lineID in lineIds:
                        print(self.lines[lineID])
                        convObj["lines"].append(self.lines[lineID]);
                    conversation.append(convObj);
        except FileNotFoundError:
            print('File Not Found')
        return conversation;
    def getconversation(self):
        return self.conversation;
        
