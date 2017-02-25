import tweepy
import csv
import configparser as cp
class bot_handler:
    def __init__(self,DirName):
        Config = cp.ConfigParser()
        Config.read(DirName+"/Database/config.ini");
        self.C_token=Config.get('Twitter','C_token');
        self.C_key=Config.get('Twitter','C_key');
        self.A_key=Config.get('Twitter','A_key');
        self.A_token=Config.get('Twitter','A_token');
        print(self.A_token);
        self.auth=tweepy.OAuthHandler(self.C_key,self.C_token);
        self.auth.set_access_token(self.A_key,self.A_token);
        self.api=tweepy.API(self.auth);
    def post(self,string):
        self.api.update_status(string);
    def get_last_tweet(self):
        tweet = self.api.user_timeline(id = self.client_id, count = 1)[0]
t=bot_handler('/home/karan/Documents/GIT_HUB/BIG_DATA');
t.post("test tweet");

