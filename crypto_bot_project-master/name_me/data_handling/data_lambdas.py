###############################################################################
############################### IMPORTS #######################################
###############################################################################

#external imports
import pandas as pd
import numpy as np

import time

###############################################################################

# internal imports
from ..data_handling import get_historical_prices as ghp

###############################################################################

# typing (external | internal)
from typing import Callable

from pandas import DataFrame

###############################################################################
###############################################################################
###############################################################################

class DataLambda(Callable[[], DataFrame]):
    def __init__(self, func: Callable[[], DataFrame]):
        self.func = func
    
    def __call__(self):
        return self.func()
    
    @staticmethod
    def get_lambda(profile: dict, end: float, check: bool=True):
    
        api  = profile['api']
        test = profile['use_testnet']
        
        if api == "FTX":
            if test: raise NotImplementedError("Testnet for FTX not support!")
            connector = ghp.FTXConnector(check=check)
        elif api == "ByBit":
            connector = ghp.ByBitConnector(test=test, check=check)
        
        resolution = profile['resolution']
        subs       = profile['subs']
        
        market = profile['market']
        limit  = profile['limit']
        
        sub_limits = [
            resolution//sub * limit for sub in subs    
        ]
        
        end -= end % resolution
        
        def func() -> DataFrame:
            # fetch data for main resolution
            main_df = connector.get_historical_prices(
                market, resolution, end, limit
            )
            
            # fetch data for each sub resolution
            sub_dfs = [
                connector.get_historical_prices(
                    market, 
                    sub,
                    end, 
                    sub_limit
                ) for sub, sub_limit in zip(subs, sub_limits)
            ]
            
            # for each sub resolution prepare the corresponding dataframe for concat
            SPLITS = []
            for sub, sub_df in zip(subs, sub_dfs):
                # how often does the sub resolution fit into the main resolution?
                q = resolution // sub
                
                # split sub dataframe into q pieces
                splits = []
                for i in range(q):
                    # take every q-th entry starting at i
                    choice = sub_df[i::q]
                    
                    # rename columns
                    choice.columns = choice.columns.map(
                        lambda col : col + f'_{sub}_{i+1}:{q}'    
                    )
                    
                    # round timestamp down to correspnding main timestamp
                    def norm(date):
                        stamp  = date.timestamp()
                        stamp -= stamp % resolution
                        return pd.to_datetime(stamp, unit='s')
                    
                    choice.index = choice.index.map(norm)
                    
                    
                    splits.append(choice)
                    
                SPLITS.append(splits)
              
            # concat the splits for each sub into one dataframe
            sub_dfs = [
                pd.concat(splits, axis=1) for splits in SPLITS    
            ]
        
            # concat main df and sub dfs from splits
            df = pd.concat([main_df] + sub_dfs, axis=1)
            
            return df
    
        return DataLambda(func)
    
    @staticmethod
    def get_test_lambda(profile: dict, debug: bool=False):
        
        limit      = profile['limit']
        resolution = profile['resolution']
        
        def func() -> DataFrame:
        
            sin_freqs = [0.8,2.3,2.8,20,75,300]
            cos_freqs = [0.5, 1.7,2.1,35, 85,145,290, 66]
            
            sin_coeffs = [0.7,0.4,0.8,0.5,0.09,0.07]
            cos_coeffs = [-1.5, 0.6,1,0.15, 0.03, 0.02, 0.02, 0.01]
            
            x = np.linspace(0, 1, limit+1)
            
            sin_waves = [
                coeff * np.sin(
                    2 * np.pi * freq * x
                ) for coeff, freq in zip(sin_coeffs, sin_freqs)
            ]
            
            cos_waves = [
                coeff * np.cos(
                    2 * np.pi * freq * x
                ) for coeff, freq in zip(cos_coeffs, cos_freqs)
            ]
            
            s = np.zeros(np.shape(x))
            
            for wave in sin_waves:
                s += wave
            for wave in cos_waves:
                s += wave
                
            
            s += 1.5 * abs(np.min(s))
            
            df = pd.DataFrame(
                index = pd.to_datetime(
                    time.time() + np.arange(0, resolution * limit, resolution),
                    unit='s'
                )
            )
            
            df['open']  = s[:-1]
            df['close'] = s[1:]
            df['high'] = 1.1 * (df['open'] + df['close']) / 2
            df['low'] = 0.9 * (df['open'] + df['close']) / 2
            
            if debug:
                return x, sin_waves, cos_waves, df
            else:
                return df
        
        return DataLambda(func)

if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
    
    from profiles import profile_manager as pm
    
    profile = pm.get_profile_by_file("./profile_files/rl_training.json")
    
    # real_lam = DataLambda.get_lambda(profile, time.time(), check=True)
    test_lam = DataLambda.get_test_lambda(profile, debug=True)
    
    # df = real_lam()
    
    # print(df)
    
    x, sins, coss, df = test_lam()
    
    plt.figure()
    for wave in sins:
        plt.plot(wave)
    
    plt.figure()
    for wave in coss:
        plt.plot(wave)
        
    plt.figure()
    plt.plot(df['close'])
    
    plt.figure()
    plt.scatter(x[200:275], df['close'][200:275])
    
    
    
    
    
    
    