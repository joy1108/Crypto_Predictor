import requests
import pandas as pd
from datetime import datetime, timedelta

def getDataApi(dat, endt):
    """
    Retrieves data from the Coinbase API for the given cryptocurrency symbol and end date.
    """
    api_url = "https://api.pro.coinbase.com"
    bar_size = "86400"

    delta = timedelta(hours=24)
    time_start = endt - delta * 300

    time_start = time_start.isoformat()
    endt = endt.isoformat()

    parameters = {
        "start": time_start,
        "end": endt,
        "granularity": bar_size,
    }

    data = requests.get(f"{api_url}/products/{dat}/candles",
                        params=parameters,
                        headers={"content-type": "application/json"})
    return data

def getAllData(dat):
    current_time = datetime.now()
    one_day = timedelta(days=1)
    data_df = pd.DataFrame(columns=["low", "high", "open", "close", "volume"])

    while True:
        api_data = getDataApi(dat, current_time)
        formatted_data = formatData(api_data)
        if not formatted_data.empty:
            data_df = data_df.append(formatted_data)
            current_time -= one_day * 300
        else:
            break

    return data_df

def formatData(data):
    my_df = pd.DataFrame(data.json(), columns=["time", "low", "high", "open", "close", "volume"])
    my_df['date'] = pd.to_datetime(my_df['time'], unit='s')
    my_df.set_index("date", inplace=True)
    my_df.drop(["time"], axis=1, inplace=True)
    return my_df

def getListCoins():
    currencies = []
    name_mapping = {}
    api_url = "https://api.pro.coinbase.com/currencies"
    res = requests.get(api_url).json()

    for currency in res:
        if currency['details']['type'] == 'crypto':
            symbol = currency['id'] + "-USD"
            currencies.append(symbol)
            name = currency['name']
            name_mapping[symbol] = name

    currencies.sort()
    tuple_currencies = tuple(currencies)

    return tuple_currencies, name_mapping


def getFinalData(sym, period="1 DAY"):
    df_origin = getAllData(sym)

    if period == "1 DAY":
        return df_origin
    else:
        if period == "1 WEEK":
            df = df_origin.groupby(pd.Grouper(freq='1W'))
        elif period == "2 WEEKS":
            df = df_origin.groupby(pd.Grouper(freq='2W'))
        elif period == "1 MONTH":
            df = df_origin.groupby(pd.Grouper(freq='M'))

        lst_c = []
        for group in df:
            data = convertData(group)
            lst_c.append(data)

        final = pd.concat(lst_c)
        final = final[::-1]
        return final

def convertData(tup):
    index = tup[0]
    dataf = tup[1]
    col = dataf.columns
    high,low,vol = dataf['high'].max(), dataf['low'].min(), dataf['volume'].sum()
    close,open = dataf['close'].iloc[-1], dataf['open'].iloc[0]
    df = pd.DataFrame([low, high, open, close, vol], columns=[index], index=col)
    return df.T