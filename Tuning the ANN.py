
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


X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.30, random_state=101)


# In[13]:


# feature scaling  
from sklearn.preprocessing import StandardScaler


# In[14]:


sc = StandardScaler()


# In[15]:


X_train = sc.fit_transform(X_train)


# In[16]:


X_test = sc.transform(X_test)


# In[17]:


# implementing k 4 cross validation to make better pred
# keras classifier wrapper and it expects a function to 
# returned as its builds the architecture of ANN
from keras.wrappers.scikit_learn import KerasClassifier
# k4 models 
from sklearn.model_selection import GridSearchCV
# to initialise the ANN
from keras.models import Sequential
# dense model to build layers of ann
from keras.layers import Dense


# In[18]:


# the classifer is local in fn
def build_classifier(optimizer):
    
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
    classifier.compile(optimizer=optimizer,
                       loss='binary_crossentropy',
                       metrics=['accuracy'])
    
    return classifier


# In[19]:


# new classifier
classifier = KerasClassifier(build_fn=build_classifier)


# In[20]:


# dictionary for checking the best value for batch_size
# epochs and best optimiser

parametrs  = {
    'batch_size' : [25,32],
    'epochs': [100,500],
    'optimizer':['adam','rmsprop']
}


# In[21]:


grid_search = GridSearchCV(estimator=classifier,
                           param_grid=parametrs,
                           scoring = 'accuracy',
                          cv =10)


# In[ ]:


grid_search = grid_search.fit(X_train,y_train)


# In[ ]:


best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_
print(best_parameters)
print(best_accuracy)

