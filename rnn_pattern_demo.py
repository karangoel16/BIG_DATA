
# coding: utf-8

# **Teaching a computer to add (using memorization)**
# The goal here is to take advantage of Recurrent Neural Networks, for more background see my blog post at http://projects.rajivshah.com/blog/2016/04/05/rnn_addition/  This code was partially derived from https://github.com/yankev/tensorflow_example

# In[1]:

#Import basic libraries
import numpy as np
import tensorflow as tf
import random
#from tensorflow.models.rnn import rnn_cell
#from tensorflow.models.rnn import rnn
#from tensorflow.models.rnn import seq2seq
from numpy import sum
import matplotlib.pyplot as plt
#%matplotlib inline  


# In[2]:

#Defining some hyper-params
num_units = 50       #this is the parameter for input_size in the basic LSTM cell
input_size = 1      
batch_size = 50    
seq_len = 15
drop_out = 0.6 


# In[3]:

#Creates our random sequences
def function_ap(n,d,a):
    X=[];
    for _ in range(n):
        X.append(a+d);
        a=a+d;
    return X;
def gen_data(min_length=5, max_length=15, n_batch=50):
    
    #X = np.concatenate([np.random.randint(10,size=(n_batch, max_length, 1))],
    #                   axis=-1)
    X=[function_ap(16,2,random.randint(1,10)) for i in range(n_batch)];
    y = np.zeros((n_batch,))
    # Compute masks and correct values
    for n in range(n_batch):
        # Randomly choose the sequence length
        #length = np.random.randint(min_length, max_length)
        #X[n, length:, 0] = 0
        #Sum the dimensions of X to get the target value
        #y[n] = np.sum(X[n, :, 0]*1)
        y[n]=X[n][-1];
        X[n]=X[n][0:-1];
    y = np.array(y)
    X = np.array(X)
    X.shape = [X.shape[0],X.shape[1],1]
    return (X,y)


# In[4]:

### Model Construction
num_layers = 2
cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)
cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers)
cell = tf.nn.rnn_cell.DropoutWrapper(cell,output_keep_prob=drop_out)

#create placeholders for X and y
inputs = [tf.placeholder(tf.float32,shape=[batch_size,1]) for _ in range(seq_len)]
result = tf.placeholder(tf.float32, shape=[batch_size])
initial_state = cell.zero_state(batch_size, tf.float32)

outputs, states = tf.nn.seq2seq.rnn_decoder(inputs, initial_state, cell, scope ='rnnln')
outputs2 = outputs[-1]

W_o = tf.Variable(tf.random_normal([num_units,input_size], stddev=0.01))     
b_o = tf.Variable(tf.random_normal([input_size], stddev=0.01))

outputs3 = tf.matmul(outputs2, W_o) + b_o

cost = tf.pow(tf.sub(tf.reshape(outputs3, [-1]), result),2)

train_op = tf.train.RMSPropOptimizer(0.005, 0.2).minimize(cost) 


# In[5]:

### Generate Validation Data
tempX,y_val = gen_data(5,seq_len,batch_size)
X_val = []
print(tempX.shape)
for i in range(seq_len):
    X_val.append(tempX[:,i,:])


# ##Run this cell to see what the inputs look like  
# print (tempX[1])  
# print (y_val[1])

# In[6]:

##Session
sess = tf.Session()
sess.run(tf.initialize_all_variables())
train_score =[]
val_score= []
x_axis=[]


# In[7]:

num_epochs=10000
for k in range(1,num_epochs):

    #Generate Data for each epoch
    tempX,y = gen_data(5,seq_len,batch_size)
    X = []
    for i in range(seq_len):
        X.append(tempX[:,i,:])

    #Create the dictionary of inputs to feed into sess.run
    temp_dict = {inputs[i]:X[i] for i in range(seq_len)}
    temp_dict.update({result: y})

    _,c_train = sess.run([train_op,cost],feed_dict=temp_dict)   #perform an update on the parameters

    val_dict = {inputs[i]:X_val[i] for i in range(seq_len)}  #create validation dictionary
    val_dict.update({result: y_val})
    c_val = sess.run([cost],feed_dict = val_dict )            #compute the cost on the validation set
    if (k%100==0):
        train_score.append(sum(c_train))
        val_score.append(sum(c_val))
        x_axis.append(k)

# In[8]:

#print ("Final Train cost: {}, on Epoch {}".format(train_score[-1],k))
#print ("Final Validation cost: {}, on Epoch {}".format(val_score[-1],k))
line1=plt.plot(train_score,label='Training Value')
line2=plt.plot(val_score,label='Validation Value')
plt.legend(loc=3)
plt.ylabel('Accuracies->');
plt.xlabel('Iterations(x100)->');
#plt.plot(train_score, 'r-', val_score, 'b-')
plt.show()


# In[9]:

##This part generates a new validation set to test against
val_score_v =[]
num_epochs=1

for k in range(num_epochs):

    #Generate Data for each epoch
    tempX,y = gen_data(5,seq_len,batch_size)
    X = []
    for i in range(seq_len):
        X.append(tempX[:,i,:])

    val_dict = {inputs[i]:X[i] for i in range(seq_len)}
    val_dict.update({result: y})
    outv, c_val = sess.run([outputs3,cost],feed_dict = val_dict ) 
    val_score_v.append([c_val])
print ("Validation cost: {}, on Epoch {}".format(c_val,k))



# In[10]:

##Target
tempX[3],y[3]


# In[11]:

#Prediction
outv[3]


# In[ ]:



