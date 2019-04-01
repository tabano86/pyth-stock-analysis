from enum import Enum

import pandas as pd


class TimePeriod(Enum):
    FIVE_YEARS = '5y'
    TWO_YEARS = '2y'
    ONE_YEAR = '1y'
    YEAR_TO_DATE = 'ytd'
    SIX_MONTHS = '6m'
    THREE_MONTHS = '3m'
    ONE_MONTH = '1m'
    ONE_DAY = '1d'


def get_json_data(ticker: str, time_period: TimePeriod) -> pd.DataFrame:
    url = "https://api.iextrading.com/1.0/stock/" + ticker + "/chart/" + time_period.value
    response = pd.read_json(url)
    print("The response contains {0} properties".format(len(response)) + "\n")
    df = response
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)
    print(response.head())
    return response
