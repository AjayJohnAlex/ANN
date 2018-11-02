
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


X = dataset.iloc[:,3:13].values


# In[4]:


y = dataset.iloc[:,13].values


# In[5]:


from sklearn.model_selection import train_test_split


# In[6]:


from sklearn.preprocessing import LabelEncoder,OneHotEncoder


# In[7]:


# we have some object type data and hence need to convert
# them into int
labelencoder1 = LabelEncoder()
# for geography
X[:,1] = labelencoder1.fit_transform(X[:,1])


# In[8]:


# we have some object type data and hence need to convert
# them into int
labelencoder2 = LabelEncoder()
# for gender
X[:,2] = labelencoder2.fit_transform(X[:,2])


# In[9]:


# we need to create dummy values for geography and drop the
# the unwanted column out of it 

onehotencoder = OneHotEncoder(categorical_features=[1])


# In[10]:


X = onehotencoder.fit_transform(X).toarray()


# In[11]:


# removing the first dummy class
X = X [:,1:]


# In[12]:


X.shape


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


# implementing k 4 cross validation to make better pred
# keras classifier wrapper and it expects a function to 
# returned as its builds the architecture of ANN
from keras.wrappers.scikit_learn import KerasClassifier
# k4 models 
from sklearn.model_selection import cross_val_score
# to initialise the ANN
from keras.models import Sequential
# dense model to build layers of ann
from keras.layers import Dense


# In[23]:


# the classifer is local in fn
def build_classifier():
    
    # you can initialise a ANN in 2 ways 
    # either def sequence of layers 
    # or def by a graph

    # object of sequencial
    classifier = Sequential()
    # adding 2 layers : input layer and first hidden layer
    # units = no of hidden layers
    # kernal_initializer = initilaise weights using function
    # activation = activation function
    # input_dim = no of features in the input 
    classifier.add(Dense(units=6,kernel_initializer='uniform',
                         activation= 'relu',input_dim=11))
    
    # we will add one more hidden layer even though its not 
    #  neccesarry 
    #  we are adding it so that we can learn how to add one more
    #  layer 
    #  and deep learning has many hiiden layers in ANN
    classifier.add(Dense(units=6,kernel_initializer='uniform',
                         activation= 'relu'))
    
    # adding output layer 
    classifier.add(Dense(units=1,kernel_initializer='uniform',
                         activation= 'sigmoid'))
    
    # compile the ANN by applying stochastic GD
    # optimizer = the algo we need to use to find the optimal 
    # weights ...there are many we wld use ADAM
    # loss = SGD is based on lost function we needs to be optimised
    # 
    classifier.compile(optimizer='adam',
                       loss='binary_crossentropy',
                       metrics=['accuracy'])
    
    return classifier


# In[24]:


# new classifier
classifier = KerasClassifier(
    build_fn=build_classifier,batch_size =10,epochs = 100)


# In[21]:


# now we use cross value score from sklearn 
# k4 classification is used to get a relevant pred
# it wld return  10 accuracy

# accuracies = cross_val_score(estimator=classifier,X=X_train,y=y_train,cv=10,n_jobs = 1)


# In[22]:


mean = accuracies.mean()
variance = accuracies.std()
print('mean is ',mean)
print('variance is ',variance)


# In[ ]:




