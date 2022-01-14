# -*- coding: utf-8 -*-
"""CNN DefiIA.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/10xYGYK5mhSswCNJp3xGNNck-mYymSbR0

# Time series forecasting using convolutional neural networks
"""

import numpy as np 
import pandas as pd 
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow as tf
import time

# from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import mean_absolute_percentage_error

#import h5py  # Output network weights in HDF5 format during checkpointing

from tensorflow import keras
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.layers import Input, Dropout, Dense, Embedding, Flatten, MaxPooling1D,RepeatVector,TimeDistributed, Bidirectional
from tensorflow.keras.layers import SimpleRNN, LSTM, TimeDistributed,Masking,Conv1D,InputLayer

"""##  Preprocessing"""

Y_train=pd.read_csv("Y_train_completed.csv", parse_dates=[3], infer_datetime_format=True) #Train data with some nans filled
Y_test=pd.read_csv("Y_test_completed.csv") #Test data with some nans filled
#After some transformations, we merge train and test into a same dataset
Y_test['Day']=Y_test['Day']+730
Y_test['number_sta']=Y_test['Station']
del Y_test['Station']
Y_train['date']=pd.to_datetime(Y_train['date'])
Y_train['date'].min()
Y_train['Day'] = (Y_train['date']-Y_train['date'].min()).dt.days
Y_train['precip']=Y_train['Ground_truth']
del Y_train['Ground_truth']
df=[Y_train,Y_test]
data=pd.concat(df)
data=data.reset_index()
data

"""# Construction of the array containing all accumulated daily rainfalls
### Columns represent the 325 stations and lines are the 730+363=1093 dates
"""

# #This is the code that helped construct this array
# from numpy import hstack
# sta1=data['number_sta'].unique()
# sta1=sorted(sta1)
# dataset=np.empty((1093,len(sta1)))
# dataset[:]=np.nan
# for i in range(len(sta1)):
#     for j in np.array(data[data['number_sta']==sta1[i]]['Day'],dtype = int):
#         dataset[j,i]=float(data[(data.number_sta==sta1[i]) & (data.Day==j)]['precip'])

# print(dataset.shape)

dataset=np.load('datas')

"""### log transformation of the dataset """

dataset=np.log(1+dataset)
dataset

"""### We fill the nans with the values -0.01 so that the Neural Network can ignore that value"""

dataset=pd.DataFrame(dataset)
dataset=dataset.fillna(-0.01)
dataset=np.array(dataset)

"""# Construction of  X_train,Y_train,X_test,Y_test"""

# Split a multivariate sequence into samples
def convert_data_multiplesparallelseries(data, n_in=3, n_out=1):
    X, y = [], []
    # input sequence (t-n, ... t-1)
    for i in range(len(data)):
        # find the end of this pattern
        end_x = i + n_in
        end_y = end_x + n_out
        # check if we are beyond the sequence
        if end_y > len(data):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = data[i:end_x, :], data[end_x:end_y, :]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

n_steps_in, n_steps_out = 60, 1 #Values 0-59 predict value 60, values 1-60 predict value 61 etc.

X, y = convert_data_multiplesparallelseries(dataset, n_steps_in, n_steps_out)
if n_steps_out==1:
    y = y.reshape(y.shape[0], y.shape[-1])
# Summarize the data
for i in range(3):
    print("-"*15)
    print("X:", X[i])
    print("y:", y[i])
print("...")

n_features = X.shape[-1] # in order to capture the multi-variate dimension

X_train, X_test = X[:730], X[-363:] #We retrieve train and test datasets 

y_train, y_test = y[:730], y[-363:] #We retrieve train and test datasets

print("X_train:",X_train.shape,"; X_test:", X_test.shape)
print("y_train:",y_train.shape,"   ; y_test:", y_test.shape)

"""# Implementation of a convolutional neural network"""

start=time.time()
model = Sequential()
model.add(Masking(mask_value=-0.01, input_shape=(n_steps_in, n_features))) #ignores value -0.01
model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(n_steps_in, n_features)))
model.add(Conv1D(filters=32, kernel_size=2,activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(600, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(n_features))
model.summary()
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, batch_size=80, epochs=400,verbose=0)
end=time.time()
print("time to run neural network: ", end-start," seconds")

"""# Results

### MAPE calculations
"""

# Get the predicted values
y_pred = model.predict(X_test).reshape(y_test.shape[0],y_test.shape[1])
#do an inverse log transform->exponential -1
y_pred1=np.exp(y_pred) #we add 1 to the values to be able to calculate MAPE
threshold=np.ones((y_pred1.shape[0],y_pred1.shape[1])) #we set a treshold for values smaller than 1
y_pred1=np.maximum.reduce([y_pred1,threshold])
y_test1=y_test+1


# Mean Absolute Percentage Error (MAPE)
MAPE = np.mean((np.abs(np.subtract(y_test1, y_pred1)/ y_test1))) * 100
print(f'Mean Absolute Percentage Error (MAPE): {np.round(MAPE, 2)} %')

y_pred=y_pred1-1

from numpy import hstack
sta1=data['number_sta'].unique() #array of the 325 station numbers

import matplotlib.patches as mpatches
#We plot values for the 10th station
for i in [80,324]:

    test=plt.plot(y_test[:,i],color='b')
    blue_patch = mpatches.Patch(color='blue', label='y_test station '+str(sta1[i]))
    red_patch = mpatches.Patch(color='red', label='y_pred station '+str(sta1[i]))
    pred=plt.plot(y_pred[:,i],color='r')
    plt.legend([test,pred],handles=[blue_patch,red_patch])
    plt.show()

"""### loading kaggle's Y_test """

Y_test=pd.read_csv("Baseline_observation_test.csv")

Y_test.sort_values('Id')
Y_test[['number_sta','day']] = Y_test.Id.str.split("_",expand=True)
Y_test

###We compute arg value of the station in list of stations

Y_test['rank']=Y_test['day']
number_sta=np.array(Y_test['number_sta'])
sta1=sta1.tolist()
number_sta
for i in Y_test.index:
    Y_test['rank'][i]=sta1.index(int(number_sta[i]))
# Y_test

Y_test

"""### We search for prediction i of Y_test in our y_pred according to station number and day
### (85140 rows, can take some time)
"""

Y_test['pred']=Y_test['Prediction']
for i in Y_test.index:
    Y_test['pred'][i]=y_pred[int(Y_test['day'][i]),int(Y_test['rank'][i])]

"""### We compute new MAPE"""

y_test1=np.array(Y_test['Prediction'])+1
y_pred1=np.array(Y_test['pred'])+1
# Mean Absolute Percentage Error (MAPE)
MAPE = np.mean((np.abs(np.subtract(y_test1, y_pred1)/ y_test1))) * 100
print(f'Mean Absolute Percentage Error (MAPE): {np.round(MAPE, 2)} %')

Y_testfi=Y_test[['Id','pred']]
Y_testfi['Prediction']=Y_testfi['pred']+1
del Y_testfi['pred']
print(Y_testfi)

"""### load data into file to submit on Kaggle"""

#Y_testfi.to_csv("predictionCNNlast.csv",index=False)




