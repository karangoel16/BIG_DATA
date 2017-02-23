import tweepy
class bot_handler:
	def __init__(self,CONSUMER_KEY,CONSUMER,SECRET,ACCESS_KEY,ACCESS_SECRET):
		print(ACCESS_SECRET)	;	
		self.CONSUMER_KEY=CONSUMER_KEY;
		self.CONSUMER_SECRET=CONSUMER_SECRET;
		self.ACCESS_KEY=ACCESS_KEY;
		self.ACCESS_SECRET=ACCESS_SECRET;
		self.auth=tweepy.OAuthHandler(self.CONSUMER_KEY,self.CONSUMER_SECRET);
		self.auth.set_access_token(self.ACCESS_KEY,self.ACCESS_SECRET);
		self.api=tweepy.API(self.auth);
	def post(self,string):
		self.api.update_status(string);
	def get_last_tweet(self):
    		tweet = self.api.user_timeline(id = self.client_id, count = 1)[0]
    		print(tweet.text)
con_key=input("Consumer key");
con_sec=input("Consumer secret");
acc_key=input("ACCESS_KEY");
acc_sec=input("ACCESS_SECRET");
t=bot_handler(con_key,con_sec,acc_key,acc_sec);
string=input("Enter your tweet");
t.post(string);

