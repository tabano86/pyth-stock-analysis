import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
import pandas as pd
from pandas.plotting import register_matplotlib_converters

from src.utils.StockRestService import get_json_data, TimePeriod

register_matplotlib_converters()

raw_dataset = get_json_data("AMD", TimePeriod.FIVE_YEARS)
raw_dataset['date'] = raw_dataset.index
dataset = pd.DataFrame(index=range(0, len(raw_dataset)), columns=['date', 'close'])

for i in range(0, len(raw_dataset)):
    dataset['date'][i] = raw_dataset['date'][i]
    dataset['close'][i] = raw_dataset['close'][i]

# setting index
dataset.index = dataset['date']
dataset.drop('date', axis=1, inplace=True)

# train and validation
split_date = np.datetime64('today') - np.timedelta64(60, 'D')
train = dataset[(dataset.index < split_date)]
valid = dataset[(dataset.index >= split_date)]

# converting dataset into x_train and y_train
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset.values)

x_train, y_train = [], []
for i in range(60, len(train)):
    x_train.append(scaled_data[i - 60:i, 0])
    y_train.append(scaled_data[i, 0])
x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, epochs=2, batch_size=32)

# predicting 246 values, using past 60 from the train data
inputs = dataset[len(dataset) - len(valid) - 60:].values
inputs = inputs.reshape(-1, 1)
inputs = scaler.transform(inputs)

X_test = []
for i in range(60, inputs.shape[0]):
    X_test.append(inputs[i - 60:i, 0])
X_test = np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
closing_price = model.predict(X_test)
closing_price = scaler.inverse_transform(closing_price)

predictions = pd.DataFrame(index=range(0, len(closing_price)), columns=['close', 'date'])

for i in range(0, len(closing_price)):
    predictions['close'][i] = closing_price[i]
    predictions['date'][i] = split_date + np.timedelta64(i, 'D')

# setting index
predictions.set_index('date', inplace=True)

# for plotting
# train = dataset[:len(closing_price)]
# valid = dataset[len(closing_price):]

plt.plot(predictions['close'], label='prediction data')
plt.plot(dataset['close'], label='train data')
plt.plot(valid['close'], label='valid data')
plt.legend()
plt.show()
