# BIG_DATA
things needed to run this code
1. nltk
2. tensorflow
3. os
4. gzip
5. configparser

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

Need to save another file for twitter.ini in database with A_token,A_key,C_key,C_token
run the iteration and you could make changes in the dataset accordingly

Inputs required from the user , we need to set the following stuff
1. Maxlength - required for the sentences 
2. Choice - if we enter 5 in the config file then corpus will be created with all the corpus and if we give particular choice then that particular file will be implemented
3. initEmbedding- True/False
4. Twitter - True/False
5. Test - CLI interface (True/False)

Twitter and Test can't run togther and if both are false then we will run training , if we have model saved it will start running the code again from the point we have left 

We have implemented seq2seq model , and have trained it on various models we will be working and changing this model to different version.


