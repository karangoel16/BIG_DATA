import tweepy
import os
from tweepy.streaming import StreamListener
from tweepy import Stream
import csv
import configparser as cp
import json
from bot import Bot
'''
Global declaration of variables
'''
Config = cp.ConfigParser()
DirName='/'.join(os.path.dirname(os.path.realpath(__file__)).split('/')[:-1])
Config.read(os.path.join(DirName,'Database/twitter.ini'));
C_key=Config.get('Twitter','C_key');
C_secret=Config.get('Twitter','C_token');
A_key=Config.get('Twitter','A_key');
A_token=Config.get('Twitter','A_token');
auth=tweepy.OAuthHandler(C_key,C_secret);
auth.set_access_token(A_key,A_token);
api=tweepy.API(auth);
class bot_stream(StreamListener):
    def __init__(self):
        self.t=Bot()
        self.t.main()
        print("Bot Ready")
    def post(self,username,line,tid):
        message='@'+username+' '+self.t.interactive_main_twitter(self.t.session,line)
        print(message)
        api.update_status(message,in_reply_to_status_id=tid);
    def on_data(self,data):
        json_load = json.loads(data)
        print(json_load['text']);
        user_data=json_load['user']
        self.post( user_data['screen_name'],json_load['text'],json_load['id']);
    def on_error(self,status):
        print(status)
class bot_handler:
    def __init__(self):
        self.l=bot_stream()
        try:
            stream = tweepy.Stream(auth, self.l)
            stream.filter(track=['@SmartGatorAI'])
        except KeyboardInterrupt:
            print('closing session')
            self.l.t.session.close()
if __name__=="__main__":
	t=bot_handler();
#t.post("test tweet");

