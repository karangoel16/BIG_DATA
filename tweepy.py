import tweepy

auth=tweepy.OAuthHandler(CONSUMER_KEY,CONSUMER_SECRET);
auth.set_access_token(ACCESS_KEY,ACCESS_SECRET);
api=tweepy.API(auth);
api.update_status("Hi testing tweepy, for python!!!");
def get
