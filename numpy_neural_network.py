# Multilayer Perceptron (MLP)
import numpy as np
import pandas as pd

# # Importing titanic dataset to get X and Y 
# df = pd.read_csv('../datasets/train.csv')
# # print(df.info())
# new_df =df[['Pclass','Age','SibSp','Parch','Fare','Survived']]
# print(new_df.head())
# df = df.dropna(inplace=True)
# X = new_df.drop('Survived',axis=1)
# y = new_df['Survived']
# print(X.shape)
# print(y.shape)
# Initialise the Input and Output array

input_array = np.array([[1,0,1,0],[0,0,1,1],[1,1,1,1]])
output_array = np.array([[0],[1],[0]])

# inititailse the variables
learning_rate = 0.1
epochs = 15000
input_neurons = input_array.shape[1]
output_neurons = 1
hidden_layer_neurons = 20

# initialise the weights for hidden layer and output layer 

weight_hidden_layer = np.random.uniform(size=(input_neurons,hidden_layer_neurons))
# print(weight_hidden_layer.shape,'weight_hidden_layer\n')
bias_hidden_layer = np.random.uniform(size=(1,hidden_layer_neurons))
# print(bias_hidden_layer.shape,'bias_hidden_layer\n')
weight_output_layer = np.random.uniform(size=(hidden_layer_neurons,output_neurons))
# print(weight_output_layer.shape,'weight_output_layer\n')
bias_output_layer  = np.random.uniform(size=(1,output_neurons))
# print(bias_output_layer.shape,'bias_output_layer\n')

# we are using sigmoid as the activation function
def sigmoid(x):
    return 1/(1 + np.exp(-x))

# derivative of sigmoid function
def derivative_sigmoid(x):
    return x*(1-x)
# calculating the forward and backward propagation for given no of epochs 

for i in range(epochs):
    
    # Full Batch Gradient Descent is used. 

    # forward propagation
    hidden_layer_input = input_array.dot(weight_hidden_layer) + bias_hidden_layer
    # print(hidden_layer_input.shape,'hidden_layer_input\n')
    activation_hidden_layer = sigmoid(hidden_layer_input)
    # print(activation_hidden_layer.shape,'activation_hidden_layer\n')
    output_layer_input = activation_hidden_layer.dot(weight_output_layer) + bias_output_layer
    # print(output_layer_input.shape,'output_layer_input\n')
    output = sigmoid(output_layer_input)
    # print(output.shape,'output\n')

    # backpropagation
    # Cost Function : Mean Sqaure Error
    error = output_array - output
    # print(error.shape,'error\n')
    slope_output_layer = derivative_sigmoid(output)
    # print(slope_output_layer.shape,'slope_output\n')
    slope_hidden_layer = derivative_sigmoid(activation_hidden_layer)
    # print(slope_hidden.shape,'slope_hidden\n')
    delta_at_output = error * slope_output_layer
    # print(delta_at_output.shape,'delta_at_output\n')
    error_at_hidden = delta_at_output.dot(weight_output_layer.T)
    # print(error_at_hidden.shape,'error_at_hidden\n')
    delta_hidden_layer = error_at_hidden * slope_hidden_layer
    # print(delta_hidden_layer.shape,'delta_hidden_layer\n')

    weight_output_layer += activation_hidden_layer.T.dot(delta_at_output) * learning_rate
    # print(weight_output_layer.shape,'weight_output_layer\n')
    bias_output_layer += np.sum(delta_at_output,axis=0) * learning_rate
    # print(bias_output_layer.shape,'bias_output_layer\n')
    weight_hidden_layer += input_array.T.dot(delta_hidden_layer) * learning_rate
    # print(weight_hidden_layer.shape,'weight_hidden_layer\n')
    bias_hidden_layer += np.sum(delta_hidden_layer,axis=0) * learning_rate
    # print(bias_hidden_layer.shape,'bias_hidden_layer\n')

# print(input_array.shape)
# print(output_array.shape)
print("Input Values\n",input_array)
print("Output values\n",output_array)
print("\n predicted output\n",output)
