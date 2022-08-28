###############################################################################
############################### IMPORTS #######################################
###############################################################################

#external imports
import numpy as np
import pandas as pd
import bybit
import requests


###############################################################################

# internal imports

###############################################################################

# typing (external | internal)
from pandas import DataFrame

###############################################################################
###############################################################################
###############################################################################


class Connector(object):
    def __init__(self):
        pass
    
    def _check_df(self, df: DataFrame, step: int, return_diff: bool=False) -> DataFrame:
        """takes a datafram indexed by timestamps and checks for missing rows"""
            
        start, stop = int(min(df.index)), int(max(df.index))
        
        target = pd.Series(range(start, stop + step, step))
        
        difference = target[~target.isin(df.index.values)]
        
        for missing in difference:
            new_col = df.loc[missing-step]
            new_col.name = missing
            
            df = df.append(new_col)
            df.sort_index(inplace=True)
        
            return df, difference
        else:
            return df
        
class FTXConnector(Connector):
    def __init__(self, check: bool=True):
        super().__init__()
        self.check = check
        
    # historical prices: name, time_step, start, end, 
    def _get_historical_prices(self, market: str, resolution: int, end:float, limit: int=5000) -> dict:
        """request data with last candle closing at <<end>>"""
        
        start = end - limit * resolution
    
        req_string = "https://ftx.com/api/markets/"
        req_string += market
        req_string += '/candles?resolution='
        req_string += str(resolution)
        req_string += "&limit="
        req_string += str(limit)
        req_string += "&start_time="
        req_string += str(start)
        req_string += "&end_time="
        req_string += str(end - resolution)
    
        # request data:
        response = requests.get(req_string).json()   
    
        # extract data from json
        result = response['result']
        
        return result
    
    def get_historical_prices(self, market: str, resolution: int, end: float, limit: int) -> DataFrame:
        """
        Requests candle data from FTX-Api.

        Parameters
        ----------
        market : str
            Example: "BTCUSD".
        resolution : int
            Either 60, 300, 900, 3600, 14400 or 86400 seconds.
        end : float
            Given as seconds since The Epoch.
        limit : int
            How many candles to request.

        Returns
        -------
        df : DataFrame
            DataFrame containing candle data indexed by dates.

        """
        
        assert resolution in [60, 300, 900, 3600, 14400, 86400]
    
        end = end - end % resolution
        
        num_calls = limit // 5000
    
        rem = limit % 5000
        
        unit_end = end - rem * resolution
        
        responses = []
        
        if rem != 0:
            print("fetching REM")
            responses.append(
                self._get_historical_prices(
                    market      = market,
                    resolution  = resolution,
                    end         = end,
                    limit       = rem
                )
            )
        
        for call in range(num_calls):
            print("fetching FULL")
            responses.append(
                self._get_historical_prices(
                    market     = market,
                    resolution = resolution,
                    end        = unit_end - call * resolution * 5000
                )
            )
        
        data = []
        for response in responses:
            data += response
        
        # turn recevied candles into dataframe
        df = pd.DataFrame(data)
        
        # drop column of dates
        df.drop(["startTime"], axis='columns', inplace=True)
        
        # rename "time" column to "startTime"
        df.rename({'time' : 'startTime'}, axis='columns', inplace=True)
        
        # set index to "startTime"
        df.set_index('startTime', inplace=True)
        
        # sort according to dates
        df.sort_index(inplace=True)
        
        # normalize timestamps to nice values
        df.index = df.index.map(
            lambda stamp : round(stamp/(1000*resolution)) * resolution    
        )
        
        # check dataframe for missing values
        if self.check:
            df = self._check_df(df, resolution)
        
        # turns timestamps to dates
        df.index = df.index.map(
            lambda stamp : pd.to_datetime(stamp, unit='s')    
        )
        
        return df

class ByBitConnector(Connector):
    def __init__(self, test: bool=False, check: bool=True):
        super().__init__()
        self.client = bybit.bybit(test=test)
        self.test   = test
        self.check  = check
        
    def _get_historical_prices(self, market: str, resolution: int, end: float, limit: int=200) -> dict:
        """request data with last candle closing at <<end>>"""
        
        start = end - limit * resolution
            
        if resolution == 86400:
            mod_resolution = "D"
        else:
            mod_resolution = str(resolution//60)
            
        response = self.client.Kline.Kline_get(
            **{
            'symbol'   : market,
            'interval' : mod_resolution,
            'from'     : start,
            'limit'    : limit
        }).result()[0]["result"]
            
        return response
    
    def get_historical_prices(self, market: str, resolution: int, end: float, limit: int) -> DataFrame:
        """
        Requests candle data from ByBit-Api.

        Parameters
        ----------
        market : str
            Example: "BTCUSD".
        resolution : int
            Either 300, 900, 1800, 3600, 14400 or 86400 seconds.
        end : float
            Close time of last candle, given as seconds since The Epoch.
        limit : int
            How many candles to request.

        Returns
        -------
        df : DataFrame
            DataFrame containing candle data indexed by dates.

        """
        
        assert resolution in [300, 900, 1800, 3600, 14400, 86400]
        
        end = end - end % resolution
        
        num_calls = limit // 200
    
        rem = limit % 200
        
        unit_end = end - rem * resolution
        
        responses = []
        
        if rem != 0:
            responses.append(
                self._get_historical_prices(
                    market      = market,
                    resolution  = resolution,
                    end         = end,
                    limit       = rem
                )
            )
    
        for call in range(num_calls):
            responses.append(
                self._get_historical_prices(
                    market     = market,
                    resolution = resolution,
                    end        = unit_end - call * resolution * 200
                )
            )
        
        data = []
        for response in responses:
            data += response
        
        # turn recevied candles into dataframe
        df = pd.DataFrame(data)
        
        # remove unnecessary columns
        df.drop(['symbol', 'interval'], axis='columns', inplace=True)
        
        # turn all entries to floats
        df = df.apply(lambda x : np.float32(x))
        
        # rename timestamp column
        df.rename({'open_time' : 'startTime'}, axis='columns', inplace=True)
        
        # set timestamps as index
        df.set_index('startTime', inplace=True)
        
        # sort according to dates
        df.sort_index(inplace=True)
        
        # # normalize timestamps to nice values
        df.index = df.index.map(
            lambda stamp : round(stamp/resolution) * resolution    
        )
        
        # check dataframe for missing values
        if self.check:
            df = self._check_df(df, resolution)
        
        # turns timestamps to dates
        df.index = df.index.map(
            lambda stamp : pd.to_datetime(stamp, unit='s')    
        )
        
        return df
        
        
    
if __name__ == "__main__":
    
    import time
    
    # NOW = time.time()
    # RES = 3600
    # LIMIT = 500
    
    # connector  = ByBitConnector(test=False, check=True)
    
    # s = time.time()
    # df1 = connector.get_historical_prices(
    #     "BTCUSD",
    #     RES,
    #     NOW,
    #     LIMIT
    # )
    # e = time.time()
    
    # print(df1)
    # print(e-s)
    
    # connector2 = FTXConnector()
    
    # s = time.time()
    # df2 = connector2.get_historical_prices(
    #     "BTC-PERP",
    #     RES,
    #     NOW,
    #     limit=LIMIT
    # )
    # e = time.time()
    
    # print(df2)
    # print(e-s)
  
    # import matplotlib.pyplot as plt
    # import dataprovider as dp
    
    # trans = dp.HeikinAshiTransformer(["high", "low", "open", "close"])
    
    # df1 = df1[-50:]

    # df1_mod = trans.transform(df1)
    
    # plt.figure()
    # plt.plot(df1["open"], label="open")
    # plt.plot(df1["close"], label="close")
    # plt.plot(df1_mod["open_ha"], label="ha_open")
    # plt.plot(df1_mod["close_ha"], label="ha_close")
    # plt.legend()
    
    connector = ByBitConnector(test=False, check=False)
    
    now = time.time()
    now = now - now % 900
    print(now)
    df = connector._get_historical_prices("BTCUSD", 900, now, limit=150)
    print(pd.DataFrame(df))
    
    
    
    
    
    
    