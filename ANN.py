
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


dataset = pd.read_csv('Churn_Modelling.csv')
dataset.head()


# In[3]:


dataset.info()


# In[4]:


X = dataset.iloc[:,3:13].values


# In[5]:


y = dataset.iloc[:,13].values


# In[6]:


from sklearn.model_selection import train_test_split


# In[7]:


from sklearn.preprocessing import LabelEncoder,OneHotEncoder


# In[8]:


# we have some object type data and hence need to convert
# them into int
labelencoder1 = LabelEncoder()
# for geography
X[:,1] = labelencoder1.fit_transform(X[:,1])


# In[9]:


# we have some object type data and hence need to convert
# them into int
labelencoder2 = LabelEncoder()
# for gender
X[:,2] = labelencoder2.fit_transform(X[:,2])


# In[10]:


# we need to create dummy values for geography and drop the
# the unwanted column out of it 

onehotencoder = OneHotEncoder(categorical_features=[1])


# In[11]:


X = onehotencoder.fit_transform(X).toarray()


# In[12]:


# removing the first dummy class
X = X [:,1:]


# In[13]:


X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.30, random_state=101)


# In[14]:


# feature scaling  
from sklearn.preprocessing import StandardScaler


# In[15]:


sc = StandardScaler()


# In[16]:


X_train = sc.fit_transform(X_train)


# In[17]:


X_test = sc.transform(X_test)


# In[18]:


# step 2  : making ANN
# importing keras and its modules
import keras 


# In[19]:


# to initialise the ANN
from keras.models import Sequential
# dense model to build layers of ann
from keras.layers import Dense


# In[20]:


# you can initialise a ANN in 2 ways 
# either def sequence of layers 
# or def by a graph

# object of sequencial
classifier = Sequential()


# In[21]:


# adding 2 layers : input layer and first hidden layer
# units = no of hidden layers
# kernal_initializer = initilaise weights using function
# activation = activation function
# input_dim = no of features in the input 
classifier.add(Dense(units=6,kernel_initializer='uniform',
                     activation= 'relu',input_dim=11))


# In[22]:


# we will add one more hidden layer even though its not 
#  neccesarry 
#  we are adding it so that we can learn how to add one more
#  layer 
#  and deep learning has many hiiden layers in ANN
classifier.add(Dense(units=6,kernel_initializer='uniform',
                     activation= 'relu'))


# In[23]:


# adding output layer 
classifier.add(Dense(units=1,kernel_initializer='uniform',
                     activation= 'sigmoid'))


# In[24]:


# compile the ANN by applying stochastic GD
# optimizer = the algo we need to use to find the optimal 
# weights ...there are many we wld use ADAM
# loss = SGD is based on lost function we needs to be optimised
# 
classifier.compile(optimizer='adam',
                   loss='binary_crossentropy',
                   metrics=['accuracy'])


# In[25]:


# make connection with the training set using fit method to fit
# the training data
# no of epochs 
 classifier.fit(X_train,y_train,epochs=50,batch_size=10)


# In[26]:


# we get the probablities an dto convert them into true and false
# we need to convert depending upon the threshold whether 
# its true or false ..we take treashold to 50%
y_pred  = classifier.predict(X_test)


# In[27]:


y_pred = (y_pred> 0.5)


# In[28]:


from sklearn.metrics import confusion_matrix,classification_report


# In[29]:


print(confusion_matrix(y_test,y_pred))


# In[30]:


print(classification_report(y_test,y_pred))


# In[ ]:


# '''
# Suppose we need to know if this customer wld leave or stay 
# in the bank

# Geography: France
# Credit Score: 600
# Gender: Male
# Age: 40 years old
# Tenure: 3 years
# Balance: $60000
# Number of Products: 2
# Does this customer have a credit card ? Yes
# Is this customer an Active Member: Yes
# Estimated Salary: $50000

# '''


# In[31]:


new_pred = classifier.predict(sc.transform(np.array(
    [[0.0,0.0,600.0,1.0,40.0,3.0,60000.0,
      2.0,1.0,1.0,50000.0]])))
new_pred = (new_pred>0.5)
print(new_pred)


# In[ ]:




