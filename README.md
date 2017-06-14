Libraries needed to run this project
* nltk
* tensorflow
* gzip
* configparser

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

Need to save another file for twitter.ini in database with A_token,A_key,C_key,C_token run the iteration and you could make changes in the dataset accordingly

Inputs required from the user , we need to set the following stuff

Maxlength - required for the sentences

Choice - if we enter 5 in the config file then corpus will be created with all the corpus and if we give particular choice 
then that particular file will be implemented

initEmbedding- True/False

Twitter - True/False

Test - CLI interface (True/False)

##Twitter and Test can't run togther and if both are false then we will run training by default. If we have an incomplete trained model, it will start running the code again from the point we have left.

hiddenlayer size: (Number of cells required by the user)

embeddingSize 

embeddingSource: Name of the of the pretrained vector 

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

Link to few preTrained Embedding vectors are :-

1. Google - https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing

2. Fasttext - https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md

Note :- To include embeddings we need to set initEmbeddings to be true

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

To see the loss graph follow the following steps: 

Go into codes folder and then from there call the following command:
```
tensoboard --logdir save/
```
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
