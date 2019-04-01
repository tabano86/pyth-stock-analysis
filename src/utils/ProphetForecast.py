import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fbprophet import Prophet
from pandas.plotting import register_matplotlib_converters

from src.utils.StockRestService import get_json_data, TimePeriod

register_matplotlib_converters()

raw_dataset = get_json_data("AMD", TimePeriod.FIVE_YEARS)
raw_dataset['date'] = raw_dataset.index
dataset = pd.DataFrame(index=range(0, len(raw_dataset)), columns=['date', 'close'])

for i in range(0, len(raw_dataset)):
    dataset['date'][i] = raw_dataset['date'][i]
    dataset['close'][i] = raw_dataset['close'][i]

dataset.rename(columns={'close': 'y', 'date': 'ds'}, inplace=True)

# train and validation
split_date = np.datetime64('today') - np.timedelta64(60, 'D')
train = dataset[(dataset['ds'] <= split_date)]
valid = dataset[(dataset['ds'] > split_date)]

model = Prophet(daily_seasonality=True)
model.fit(train)

# predictions
close_prices = model.make_future_dataframe(periods=len(valid))
forecast = model.predict(close_prices)

# set date indexes for plotting
forecast.set_index('ds', inplace=True)
train.index = train['ds']
valid.index = valid['ds']

plt.plot(forecast['yhat'], label='forecast')
plt.plot(train['y'], label='train')
plt.plot(valid['y'], label='valid')
plt.legend()
plt.show()
