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


def forecast(ticker: str, as_of_date: np.datetime64 = None):
    DAYS_IN_YEAR = int(365)
    FORECAST_DAYS = int(60)

    raw_dataset = get_json_data(ticker, TimePeriod.FIVE_YEARS)

    raw_dataset['date'] = raw_dataset.index

    dataset = pd.DataFrame(data=raw_dataset[['close', 'date']])
    dataset.set_index('date', inplace=True)
    dataset = dataset.resample('D').ffill().reset_index()
    dataset.set_index('date', inplace=True)

    if as_of_date is None:
        split_date = dataset.index.max()
    else:
        split_date = as_of_date

    dataset['diff'] = dataset['close'].diff(DAYS_IN_YEAR)

    train = dataset[(dataset.index < split_date)]
    train = train[DAYS_IN_YEAR:]

    model = auto_arima(y=train['diff'].values,
                       start_p=1,
                       start_q=1,
                       test='adf',
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
        ref_date = split_date + np.timedelta64(i + 1, 'D')
        ref_yhat = forecast[i]
        ref_close = ref_yhat + train[(ref_date - np.timedelta64(DAYS_IN_YEAR, 'D') == train.index)]['close'].item()
        output = output.append({'yhat': ref_yhat, 'close': ref_close, 'date': ref_date}, ignore_index=True)
    output.set_index('date', inplace=True)

    output['movingAverage'] = output['close'].rolling(window=20, center=False,
                                                      min_periods=0).mean()
    output['std'] = output['close'].rolling(window=20, center=False,
                                            min_periods=0).std()
    output['upperBand'] = output['movingAverage'] + (output['std'] * 2)
    output['lowerBand'] = output['movingAverage'] - (output['std'] * 2)

    train['training close'] = train['close']
    # output = pd.merge(output, train['training close'], how='left', left_index=True, right_index=True)

    return output[(output.index >= split_date - np.timedelta64(FORECAST_DAYS, 'D'))]
