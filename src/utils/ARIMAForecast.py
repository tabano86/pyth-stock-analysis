import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.plotting import register_matplotlib_converters
from pmdarima import auto_arima

from src.utils.StockRestService import get_json_data, TimePeriod

register_matplotlib_converters()


# create a differenced series
def get_difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return np.array(diff)


# invert differenced value
def get_inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]


def forecast(ticker: str):
    DAYS_IN_YEAR = int(365)

    raw_dataset = get_json_data(ticker, TimePeriod.FIVE_YEARS)
    FORECAST_DAYS = int(60)
    TRAINING_DAYS = len(raw_dataset)

    raw_dataset['date'] = raw_dataset.index

    dataset = pd.DataFrame(data=raw_dataset[['close', 'date']])
    dataset.set_index('date', inplace=True)
    dataset = dataset.resample('D').ffill().reset_index()
    dataset.set_index('date', inplace=True)
    split_date = np.datetime64('today') - np.timedelta64(TRAINING_DAYS, 'D')
    dataset['diff'] = dataset['close'].diff(DAYS_IN_YEAR)
    # dataset = dataset[DAYS_IN_YEAR:]
    train = dataset[(dataset.index < split_date)]
    valid = dataset[(dataset.index >= split_date)]
    model = auto_arima(y=train['diff'].values,
                       start_p=1,
                       start_q=1,
                       test='adf',
                       seasonal_test='seas',
                       max_p=5,
                       max_q=5,
                       m=7,
                       start_P=0,
                       seasonal=True,
                       d=None,
                       D=1,
                       trace=True,
                       error_action='ignore',
                       suppress_warnings=True,
                       stepwise=True,
                       )

    model_fit = model.fit(y=train['diff'].values)
    forecast = model_fit.predict(n_periods=FORECAST_DAYS)
    output = pd.DataFrame(None, columns=['yhat', 'close', 'date'])

    for i in range(0, len(forecast)):
        ref_date = split_date + np.timedelta64(i, 'D')
        ref_yhat = forecast[i]
        ref_close = ref_yhat + (dataset[(dataset.index == (ref_date - np.timedelta64(1, 'D')))])['close'][0]
        output = output.append({'yhat': ref_yhat, 'close': ref_close, 'date': ref_date}, ignore_index=True)
    output.set_index('date', inplace=True)

    output['movingAverage'] = output['close'].rolling(window=20, center=False,
                                                      min_periods=1).mean()
    output['std'] = output['close'].rolling(window=20, center=False,
                                            min_periods=1).std()
    output['upperBand'] = output['movingAverage'] + (output['std'] * 2)
    output['lowerBand'] = output['movingAverage'] - (output['std'] * 2)

    train['training close'] = train['close']
    valid['valid close'] = valid['close']
    # output = pd.merge(output, train['training close'], how='left', left_index=True, right_index=True)
    # output = pd.merge(output, valid['valid close'], how='right', left_index=True, right_index=True)

    plot_output = output[(output.index >= np.datetime64('today') - np.timedelta64(TRAINING_DAYS, 'D'))]

    plt.fill_between(x=plot_output.index,
                     y1=plot_output['upperBand'].values,
                     y2=plot_output['lowerBand'].values,
                     color='grey', alpha=0.7,
                     interpolate=True)

    plt.plot(plot_output['lowerBand'], color='blue', lw=2, alpha=0.7)
    plt.plot(plot_output['upperBand'], color='blue', lw=2, alpha=0.7)
    # plt.plot(plot_output['training close'], label='Training Data')
    # plt.plot(plot_output['valid close'], label='Actual Data')
    plt.plot(plot_output['close'], label='Forecast Data')
    plt.grid(axis='both', which='major')
    plt.legend()
    plt.autoscale()
    plt.show()


forecast('AMD')
