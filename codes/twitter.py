import tweepy
from tweepy.streaming import StreamListener
from tweepy import Stream
import csv
import configparser as cp
import json
'''
Global declaration of variables
'''
Config = cp.ConfigParser()
DirName='/home/karan/Documents/GIT_HUB/BIG_DATA'
Config.read(DirName+"/Database/config.ini");
C_key=Config.get('Twitter','C_key');
C_token=Config.get('Twitter','C_token');
A_key=Config.get('Twitter','A_key');
A_token=Config.get('Twitter','A_token');
auth=tweepy.OAuthHandler(C_key,C_token);
auth.set_access_token(A_key,A_token);
api=tweepy.API(auth);
class bot_stream(StreamListener):
    def post(self,username,tid):
        api.update_status('@'+username+' i am yet to begin answering!!!!',in_reply_to_status_id=tid);
    def on_data(self,data):
        json_load = json.loads(data)
        #print(json_load['text']);
        user_data=json_load['user']
        self.post( user_data['screen_name'],json_load['id']);
    def on_error(self,status):
        print(status)
class bot_handler:
    def __init__(self):
        l=bot_stream();
        stream = tweepy.Stream(auth, l)
        stream.filter(track=['@SmartGatorAI'])
t=bot_handler();
#t.post("test tweet");

