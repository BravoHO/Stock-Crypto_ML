
import requests
import time
import datetime
import pandas as pd
import yfinance as yf
import math
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, RepeatVector, TimeDistributed
import matplotlib.pyplot as plt

ticker = input("Enter your ticker: ").upper()
print("")

epochcount = 25
print("\nPlease wait ~"+str(round(epochcount*7/60,2))+" mins, bot will begin training now")
ticker2 = ticker
ticker = yf.Ticker(ticker)
df = ticker.history(period="max")
data = df.filter(['Close'])

dataset=data.values

training_data_len = math.ceil(len(dataset) * 0.8)

scaler = MinMaxScaler(feature_range=(0,1))

scaled_data = scaler.fit_transform(dataset)

train_data = scaled_data[0:len(dataset),:]

x_train = []
y_train = []

nnn = len(train_data)

i=60

while i < nnn:
  x_train.append(train_data[i-60:i,0])
  y_train.append(train_data[i, 0])
  i+=1
x_train = np.array(x_train)
y_train = np.array(y_train)
x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
"""
model = Sequential()
model.add(LSTM(50, return_sequences=True,input_shape=(x_train.shape[1],1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error',metrics=['accuracy'])
model.fit(x_train, y_train, batch_size = 32, epochs =epochcount)
"""
model = Sequential()
model.add(LSTM(50, activation = 'tanh',return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(rate=0.2))
model.add(LSTM(50, activation = 'tanh', return_sequences=False))
model.add(Dropout(rate=0.2))
model.add(Dense(15))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error',metrics=['accuracy'])
model.fit(x_train, y_train, batch_size = 32, epochs =epochcount)

'''
model = Sequential()
model.add(LSTM(128, activation = 'tanh', input_shape=(x_train.shape[1], 1)))
model.add(Dropout(rate=0.2))
model.add(RepeatVector(x_train.shape[1]))
model.add(LSTM(128, activation = 'tanh', return_sequences=True))
model.add(Dropout(rate=0.2))
model.add(TimeDistributed(Dense(x_train.shape[2])))
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")
model.fit(x_train,
                    y_train,
                    epochs=10,
                    batch_size=32,
                    validation_split=0.1,
                    callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, mode='min')],
                    shuffle=False)
               '''
test_data = scaled_data[training_data_len - 60:,:]

x_test = []
y_test = dataset[training_data_len:, :]

i = 60
while i < len(test_data):
  x_test.append(test_data[i-60:i,0])

  i+=1

x_test = np.array(x_test)

x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

rmse=np.sqrt(np.mean(((predictions- y_test)**2)))

train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions



ticker = yf.Ticker(ticker2)
df = ticker.history(period="max")
new_df = df.filter(['Close'])
last_60_days = new_df[-60:].values
last_60_days_scaled = scaler.transform(last_60_days)

X_test = []
X_test.append(last_60_days_scaled)
X_test = np.array(X_test)
X_test = np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))

averages = 0
counter = 0
print("\nPerforming 1000 predictions and aggregating them.\n")
cumul = 1000
for k in range(cumul):
  pred_price = model.predict(X_test)
  pred_price = scaler.inverse_transform(pred_price)
  averages+=pred_price[0][0]
  counter+=1
  if counter % 100 == 0:
    print(str(counter)+"/"+str(cumul))



pred_price = averages/cumul
status = "Up"
if pred_price < last_60_days[59][0]:
  status = "Down"
"""
print("The model has predicted the stock will go "+status+" tomorrow.")
print("The model has predicted a closing price of: $"+str(pred_price[0][0])+" tomorrow.")"""
percentage = round((pred_price/last_60_days[59][0])*100-100,2)
print(ticker)
print("Predicted movement:",status,"*IMPORTANT*")
print("Predicted change: "+str(round(pred_price-last_60_days[59][0],2)))
print("Predicted change%: "+str(percentage)+"%")
print("Predicted price:",pred_price,"from",last_60_days[59][0])

print(last_60_days[59][0])

plt.ion
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.plot(valid['Predictions'][-5:])
plt.legend(['Train','Val','Predictions'],loc='lower right')
plt.show()