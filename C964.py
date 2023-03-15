pip install yfinance

pip install --upgrade mplfinance


import math
import pandas_datareader as df
import numpy as np
import pandas as pd
import datetime
import yfinance as yf
import mplfinance as mpl
import seaborn as sns
from scipy.stats import norm
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from math import sqrt
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
plt.style.use('seaborn-deep')


stock = input("Enter ticker: ")

st = yf.Ticker(stock)
data =  st.history(period='1y', rounding= True)[['Open', 'High', 'Low', 'Close']]
#data

"""
mpl.plot(
    data,
    type="candle", 
    figsize=(20,10),
    title = f"{stock} Price",  
    style="nightclouds",
    volume= False
    )
"""

features=4

data_1= data.filter(items=['Open', 'High', 'Low', 'Close'])

dataset= data_1.values

training_length= math.ceil(len(dataset)* 0.75)

datascaler=MinMaxScaler(feature_range=(0,10))

scaled_data= datascaler.fit_transform(dataset)

#scaled_data

pl=14

training_data= scaled_data[0:training_length,:]

xtrain=[]
ytrain=[]

for i in range(pl, len(training_data)):
    xtrain.append(training_data[i - pl:i, 0:scaled_data.shape[1]])
    ytrain.append(training_data[i, 1]) 

xtrain= np.array(xtrain)
ytrain= np.array(ytrain)
#ytrain.shape

xtrain= np.reshape(xtrain, (xtrain.shape[0], xtrain.shape[1], features))

model= Sequential()
model.add(LSTM(50, return_sequences= True, input_shape= (xtrain.shape[1],features)))
model.add(LSTM(10, return_sequences= False))
model.add(Dense(10))
model.add(Dense(1))


model.compile(optimizer='nadam', loss='mean_squared_error')

model.fit(xtrain ,ytrain, batch_size=1, epochs=10)

testing_data= scaled_data[training_length-pl:, :]

xtest= []
ytest= []

for i in range(pl, len(testing_data)): 
  xtest.append(testing_data[i-pl:i, 0:scaled_data.shape[1]])
  ytest.append(testing_data[i,1])
  #ytest

xtest= np.array(xtest)
xtest = np.reshape(xtest,(xtest.shape[0], xtest.shape[1],features))
#xtest.shape

predictions= model.predict(xtest)
prediction_copies = np.repeat(predictions, scaled_data.shape[1], axis=-1)
predictions1 = datascaler.inverse_transform(prediction_copies)[:,0]
#predictions1

ytest= np.array(ytest)

ytest = np.reshape(ytest,(ytest.shape[0], 1))

y_copies = np.repeat(ytest, scaled_data.shape[1], axis=-1)
ytest1 = datascaler.inverse_transform(y_copies)[:,0]
ytest1 = ytest1.round(decimals=2, out=None)
#ytest1

r2_scaled = r2_score(ytest, predictions)
round(r2_scaled, 4)

r2 = r2_score(ytest1, predictions1)
round(r2, 4)

mae = mean_squared_error(ytest1, predictions1)
round(mae, 4)

mae_scaled = mean_squared_error(ytest, predictions)
round(mae_scaled, 4)

rmse_scaled = sqrt(mean_squared_error(ytest, predictions))
round(rmse_scaled, 4)

rmse = sqrt(mean_squared_error(ytest1, predictions1))
round(rmse, 4)


ytests= np.array(ytest)
#ytests

predicttest = predictions[:,0]
#predicttest

"""
plt.figure(figsize=(10,10))
plt.scatter(ytest, predicttest, c='blue')
plt.yscale('linear')
plt.xscale('linear')
#'linear', 'log', 'symlog', 'logit', 'function', 'functionlog'
p1 = max(max(predicttest), max(ytest))
p2 = min(min(predicttest), min(ytest))
plt.plot([p1, p2], [p1, p2])
plt.xlabel('ytest', fontsize=40)
plt.ylabel('Predictions', fontsize=40)
plt.axis('auto')
plt.show()
"""

"""
plt.figure(figsize=(10,10))
plt.plot((abs(ytests - predictions)*10), 'bo')
plt.plot([0,62],[0,0],'c--')
plt.plot([0,62],[5,5],'y--')
plt.plot([0,62],[10,10],'r--')
plt.plot([0,62],[15,15],'m--')
plt.ylabel('precent from actual', fontsize=30)
plt.xlabel('predictions', fontsize=30)
plt.show()
"""

"""
plt.figure(figsize=(10,10))
plt.plot(ytest1, 'go-')
plt.plot(predictions1, 'bs--')
plt.ylabel('Price', fontsize=30)
plt.xlabel('Days', fontsize=30)
plt.show()
"""

"""
data = (abs(ytests - predictions)*10)
bin = np.arange(0,16,0.25) 
plt.hist(data, bins = bin, edgecolor='blue')
plt.ylabel('Number of Predictions', fontsize=15)
plt.xlabel('Precent from actual price', fontsize=15)
plt.show()
"""

"""
train= data[:training_length]
observed= data[training_length:]
predicted = data[training_length:]
predicted['predicted']= predictions1
plt.figure(figsize=(16,8))
plt.title=('model')
plt.xlabel('date', fontsize=18)
plt.ylabel('closeing price', fontsize=18)
plt.plot(train['Close'], 'black')
plt.plot(observed['Close'],'green')
plt.plot(predicted['predicted'], 'red')
plt.legend(['train','observed', 'predicted'])
plt.show()
"""

st = yf.Ticker(stock)
data2 =  st.history(period='14d', rounding= True)[[ 'Open', 'High', 'Low', 'Close']]
#data2

scaled_data1= datascaler.fit_transform(data2)
scaled_data1=np.array(scaled_data1)
scaled_data1 = np.reshape(scaled_data1,(1, scaled_data1.shape[0], features))

predictions2= model.predict(scaled_data1)
#predictions2

prediction_copies = np.repeat(predictions2, scaled_data.shape[1], axis=-1)
predictions2 = datascaler.inverse_transform(prediction_copies)[:,0]

tprice=predictions2[0]
tprice=round(tprice, 2)
tprice



st = yf.Ticker(stock)
coda =  st.history(period='1d', rounding= True)[['Close']]



coda=coda.values

if coda > tprice:
  print("SELL")
elif coda < tprice:
  print('BUY')
elif coda == tprice: 
  print('HOLD')
