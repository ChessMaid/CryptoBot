###############################################################################
############################### IMPORTS #######################################
###############################################################################

#external imports
import requests

import numpy as np
import pandas as pd

###############################################################################

# internal imports
from utilities.decorators import deprecated

###############################################################################

# typing (external | internal)

###############################################################################
###############################################################################
###############################################################################

# check for missing data in dataframe
@deprecated
def __check_df(df, step, inplace=True, print_diff=False):
    
    if not inplace:
        df = df.copy()
    
    df.index = df.index.map(
        lambda date : date.timestamp()
    )
    
    df.sort_index()
    
    start, stop = int(min(df.index)), int(max(df.index))
    
    target = pd.Series(range(start, stop + step, step))
    
    difference = target[~target.isin(df.index.values)]
    
    for missing in difference:
        new_col = df.loc[missing-step]
        new_col.name = missing
        
        df = df.append(new_col)
        df.sort_index(inplace=True)
    
    df.index = df.index.map(
        lambda stamp : pd.to_datetime(stamp, unit='s')
    )
    
    if print_diff:
        return df, difference
    else:
        return df

# historical prices: name, time_step, start, end, 
@deprecated
def __get_historical_prices(market, resolution, start, end, limit=5000):
    """
    Requests historical data from FTX.
    
    Parameters
    ----------
    market : str
        The market from which data is supposed to be drawn, e.g. "BTC-PERP".
    resolution : int
        Resolution of requested candles in seconds. Valid are:
            "15", "60", "300", "900", "3600", "14400", "86400"
    start : float
        Start time of request in seconds since The Epoch.
    end : float
        End time of request in seconds since The Epoch.
    limit : int, optional
        Maximum number of candles to be requested. 
        The maximum and default are "5000".

    Raises
    ------
    NameError
        Raises error when request was unsuccesful twice in a row.

    Returns
    -------
    dict
        Returns dictionary of np.ndarrays corresponding to the data received.

    """

    req_string = "https://ftx.com/api/markets/"

    req_string += market

    req_string += '/candles?resolution='

    req_string += str(resolution)

    req_string += "&limit="

    req_string += str(limit)

    req_string += "&start_time="

    req_string += str(start)

    req_string += "&end_time="

    req_string += str(end)

    # request data:
    response = requests.get(req_string)

    # convert to json
    response = response.json()

    # test if request was successful
    if response['success'] != True:
        # try again:
        response = requests.get(req_string)
        response = response.json()

        # test for success again:
        if response['success'] != True:
            print(response)
            raise NameError('GET request failed twice')

    # if successful: extract data from json
    response = response['result']

    data = dict()

    # if repsonse is empty return empty dict
    if response == list():
        return dict()

    for key in response[0]:
        data[key] = np.array([response[i][key] for i in range(len(response))])

    return data

@deprecated
def get_historical_prices(market, resolution, start, end, check=True):
    """
    Requests historical data from FTX.
    
    Parameters
    ----------
    market : str
        The market from which data is supposed to be drawn, e.g. "BTC-PERP".
    resolution : int
        Resolution of requested candles in seconds. Valid are:
            "15", "60", "300", "900", "3600", "14400", "86400"
    start : float
        Start time of request in seconds since The Epoch.
    end : float
        End time of request in seconds since The Epoch.
        
    Raises
    ------
    NameError
        Raises error when a request was unsuccesful twice in a row.

    Returns
    -------
    dict
        Returns dictionary of np.ndarrays corresponding to the data received.

    """
    
    
    num_calls = (end - start + 1)/(resolution * 5000)
    
    rem = (num_calls - int(num_calls)) * 5000
    
    unit_start = start + resolution * rem
    
    request_list = []
    
    if rem != 0:
        request_list.append(
            __get_historical_prices(
                market      = market,
                resolution  = resolution,
                start       = start,
                end         = unit_start - 1,
                limit       = 5000
            )
        )
    
    for call in range(0,int(num_calls)):
        request_list.append(
            __get_historical_prices(
                market      = market,
                resolution  = resolution,
                start       = unit_start + call * resolution * 5000,
                end         = unit_start + (call + 1) * resolution * 5000 - 1,
                limit       = 5000
            )
        )
    
    data = {}
    for key in request_list[-1].keys():
        data[key] = np.concatenate(
            [request[key] for request in request_list if request != {}]
        )
        
    df = pd.DataFrame.from_dict(data)
    
    df.set_index("startTime", inplace=True)
    
    df.index = df.index.map(
        lambda date : pd.to_datetime(date[:-6].replace("T", " "))    
    )
    
    if check:
        df = __check_df(df, int(resolution), inplace=True)
        
    return df

@deprecated
def get_heikin_ashi(market, resolution, start, end, limit="5000"):
    raise NotImplementedError("This function is no longer useable, use get_historical_prices in combination with add_heikin_ashi")

@deprecated
def get_heikin_ashi_tiled(market, resolution, start, end, limit="5000"):
    raise NotImplementedError("This function is no longer useable, use get_historical_prices is combination with add_heikin_ashi")

###############################################################################

def add_heikin_ashi(data: dict):
    """
    Takes candle data and calculates the corresponding heikin ashi candles

    Parameters
    ----------
    data : dict
        Candle information.

    Returns
    -------
    data : dict
        Original data with heikin ashi open and close data added.

    """

    # split data
    close = data['close']
    open  = data['open']
    high  = data['high']
    low   = data['low']

    # calculate heikin ashi close values
    close_ha = (close + open + high + low)/4

    # recursively calculate heikin ashi open values
    open_ha = [open[0]]
    for i in range(1,len(open)):
        open_ha.append((open_ha[i-1]+close_ha[i-1])/2)

    open_ha = np.array(open_ha)

    # add new information to data
    data['open_ha']  = open_ha
    data['close_ha'] = close_ha

    return data

###############################################################################
    
@deprecated
def weighted_box_avg(data, traceback, weights=None):
    """
    Parameters
    ----------
    data : list or np.ndarray
        A timeseries.
    traceback : int
        Window size.
    weights : list, optional
        The number of weights has to equal the traceback. The default is None.

    Raises
    ------
    ValueError
        Asserts that the number of weigths equals the traceback. 

    Returns
    -------
    np.ndarray
        Windowed average of original timeseries.
        If weights are given this average is weigthed. 

    """

    if weights != None and len(weights) != traceback:
        raise ValueError("unequal number of weights versus traceback")

    data_length = len(data)

    # stack data
    data_stack = np.tile(data, (len(data),1))

    if weights != None:
        # get weigth matrix with fat diagonal
        weight_mat = np.zeros([data_length,data_length])
        for i in range(traceback):
            ones = np.ones([data_length-i,data_length-i])
            np.fill_diagonal(weight_mat[i:,:], weights[traceback-i-1]*ones)

        # weigth data
        data_stack = np.multiply(data_stack, weight_mat)

    # tril data
    first_tril  = np.tril(data_stack)
    second_tril = np.tril(data_stack, -traceback)

    # get row wise sum
    sums = np.sum(first_tril - second_tril, axis=-1)

    # get valid data
    sums = sums[traceback-1:]

    # divide by traceback elementwise
    moving_avg = sums/traceback

    return moving_avg

################################################################################

# exponentieller durchschnitt:
    # smoothing factor = r == alpha/(traceback+1)
    # r*close + avg_yesterday*(1-r)

@deprecated
def exp_avg(data, traceback, alpha=2, shift=1):
    """
    Parameters
    ----------
    data : list or np.ndarray
        A timeseries.
    traceback : int
        Window size.
    alpha : float, optional
        Scales the smoothing factor. The default is 2.
    shift : float, optional
        Shifts the smoothing factor. The default is 1.

    Returns
    -------
    np.ndarray
        Exponential avergae of orginal timeseries.

    """
    # get exponential rate
    r = alpha/(traceback + shift)

    # initialize avergae list
    exp_avg_list = np.array(data[:traceback])
    exp_avg_list = np.concatenate([exp_avg_list, [np.sum(data[:traceback])/traceback]])

    # append averages to list
    for i in range(traceback+1,len(data)):
        exp_avg_list = np.concatenate([exp_avg_list, [r*data[i] + exp_avg_list[i-1]*(1-r)]])

    return exp_avg_list

################################################################################


# relative strength index:
# RSI:
    # up = avg(max(change(src), 0), len)
    # down = avg(-min(change(src), 0), len)
    # rsi = 100 - (100 / (1 + up / down)))

@deprecated
def rsi(data, traceback=14):
    """
    Parameters
    ----------
    data : list or np.ndarray
        A timeseries.
    traceback : int, optional
        Window size. The default is 14.

    Returns
    -------
    np.ndarray
        Returns the RSI-series for the orginal timeseries.

    """

    length = len(data)

    data_bin = np.array([data[i-traceback-1:i] for i in range(traceback+1, len(data))])

    upper = np.concatenate([data_bin, np.zeros((length-traceback-1,1))], axis=-1)
    lower = np.concatenate([np.zeros((length-traceback-1,1)), data_bin], axis=-1)
    
    change = upper - lower
    change = change[:,1:15]

    change_max =  np.maximum(change, np.zeros(np.shape(change)))
    change_min = -np.minimum(change, np.zeros(np.shape(change)))

    up   = np.mean(change_max, axis=-1)
    down = np.mean(change_min, axis=-1)
    
    final = 1 + np.divide(up, down)
    final = 100 - np.divide(100, final)

    return np.concatenate([[np.nan] * (traceback+1), final])

###############################################################################
    
def get_markets():
    """Returns a list of available markets"""    
    m = requests.request('GET', 'https://ftx.com/api/markets')
    return m.json()['result']

def get_market_keys():
    """Returns a list of market property keys"""
    return get_markets()[0].keys()    

###############################################################################