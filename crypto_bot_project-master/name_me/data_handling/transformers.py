###############################################################################
############################### IMPORTS #######################################
###############################################################################

# external imports
from itertools import combinations

import numpy as np
import pandas as pd

###############################################################################

# internal imports
from ..utilities.decorators import DEPRECATED

###############################################################################

# typing (external | internal)
from typing import Optional, List, Callable
from pandas import DataFrame
from numpy  import ndarray

###############################################################################
############################### TRANSFORMER ###################################
###############################################################################
        
    # PARENT

class DataTransformer(object):
    """base class for data transformers. abstracts the transform method"""
    def transform(self, df: DataFrame) -> DataFrame:
        pass
    
###############################################################################
        
    # BASE LEVEL TRANSFORMER

class LambdaTransformer(DataTransformer):
    """performs a predefined lambda action to each column or each entry of the dataframe"""
    def __init__(self, action: Callable, entrywise: bool=False, name: str=None):
        self.action    = action
        self.entrywise = entrywise
        self.name      = name

    def transform(self, df: DataFrame) -> DataFrame:
        
        if self.entrywise:
            # apply to each column to each entry of column
            df = df.apply(lambda col : col.apply(self.action))
            
        else:
            # apply lambda to all columns of dataframe
            df = df.apply(self.action)
        
        # rename columns
        if self.name is not None:
            df.rename(columns = {
                    col : col + f"_{self.name}" for col in df.columns
                }, inplace=True
            )
        
        return df
    
class CopyTransformer(DataTransformer):
    """copies the columns of a dataframe specified by the given rule"""
    def __init__(self, rule: Callable):
        self.rule = rule
    
    def transform(self, df: DataFrame) -> DataFrame:
        keys = list(filter(self.rule, df.columns))
        return df[keys].copy()
    
class TimeSeriesSplitter(DataTransformer):
    """takes a dataframe and splits each column according to lookback"""
    def __init__(self, lookback: int, blowup_nan: bool=False, rename: bool=True):
        self.lookback   = lookback
        self.blowup_nan = blowup_nan
        self.rename     = rename
        
    def transform(self, df: DataFrame) -> DataFrame:
        
        # get length and number of series in the dataframe
        series_length, num_series = df.shape
        
        # function that splits a series according to lookback
        def column_split(col):
            
            dropped = col.dropna() 
            
            dropped_len = dropped.size
            
            dropped = np.array(dropped)
            
            num_nans = col.size - (dropped_len - self.lookback + 1)
            
            if self.blowup_nan:
                split = [[np.nan] * self.lookback] * num_nans
            else:
                split = [np.nan] * num_nans
                
            for t in range(self.lookback, dropped.size + 1):
                split.append(dropped[(t - self.lookback):t])
            
            return split
        
        # apply function to each column of dataframe
        df = df.apply(lambda col : column_split(col), axis=0)
            
        # rename columns
        if self.rename:
            df.rename(
                    columns = {col : col + f"_split{self.lookback}" for col in df.columns},
                    inplace=True
            )
            
        return df
    
class SkipSplitter(DataTransformer):
    """takes a dataframe and splits each column according to lookback and skips ahead"""
    def __init__(self, lookback: int, skip: int=1, blowup_nan: bool=False, rename: bool=True):
        self.lookback   = lookback
        self.skip       = skip
        self.blowup_nan = blowup_nan
        self.rename     = rename
        
    def transform(self, df: DataFrame) -> DataFrame:
        
        def column_split(col):
            """splits a series according to lookback"""
            col = np.array(col)
            
            (col_len,)     = col.shape
                
            pivot_indicies = reversed(
                range(col_len-1, -1, -self.skip)
            )
            
            if self.blowup_nan:
                filler = [np.nan] * self.lookback
            else:
                filler = np.nan
            
            
            split = []
            for index in pivot_indicies:
                if index - self.lookback + 1 < 0:
                    split.append(filler)                    
                else:
                    split.append(
                        col[index-self.lookback+1:index+1]
                    )

            return split
        
        df_new = pd.DataFrame(
            columns=df.columns,
            index = df.index[-1::-self.skip][::-1]
        )
        
        for col in df.columns:
            df_new[col] = column_split(df[col])
            
        # rename columns
        if self.rename:
            df_new.rename(
                    columns = {col : col + f"_skip{self.skip}split{self.lookback}" for col in df_new.columns},
                    inplace=True
            )
            
        return df_new
    
@DEPRECATED
class GAFTransformer(DataTransformer):
    """transforms a sequence of data to the GAF(gramian angular field) images"""
    def __init__(self, method = "summation", sample_range = (-1, 1)):
        
        from pyts.image import GramianAngularField

        self.GAFM = GramianAngularField(method=method, sample_range=sample_range)

    def transform(self, df):
        
        # apply lambda to all columns
        df = df.apply(
            lambda col : np.split(
                self.GAFM.transform(
                    np.concatenate(col.array, axis=0)
                ), len(df), axis=0
            ), axis=0
        )
        
        # rename columns
        df = df.rename(columns = {col : col + "_gaffed" for col in df.columns})
        
        return df
    
class MovingMin(DataTransformer):
    """takes a dataframe and calculates the moving min for each each column"""
    def __init__(self, lookback: int):
        self.lookback = lookback
        
    def transform(self, df: DataFrame) -> DataFrame:
        # find mins
        df = df.rolling(self.lookback).min()
        
        # rename columns
        df = df.rename(
            columns = {col : col + f"_min{self.lookback}" for col in df.columns}
        )
        
        return df
    
class MovingMax(DataTransformer):
    """takes a dataframe and calculates the moving max for each each column"""
    def __init__(self, lookback: int):
        self.lookback = lookback
        
    def transform(self, df: DataFrame) -> DataFrame:
        # find mins
        df = df.rolling(self.lookback).max()
        
        # rename columns
        df = df.rename(
            columns = {col : col + f"_max{self.lookback}" for col in df.columns}
        )
        
        return df
    
class MovingMean(DataTransformer):
    """takes a dataframe and calculates the moving mean for each column"""
    def __init__(self, lookback: int):
        self.lookback = lookback
        
    def transform(self, df: DataFrame) -> DataFrame:
        
        # calculate means
        df = df.rolling(self.lookback).mean()
        
        # rename columns
        df.rename(
            columns = {
                col : col + f"_ma{self.lookback}" for col in df.columns
            }, inplace=True
        )
        
        return df
    
class WeightedMovingMean(DataTransformer):
    """takes a dataframe and computes a weighted mean of each column"""
    def __init__(self, lookback: int, weights: ndarray):
        self.lookback = lookback

        self.splitter = TimeSeriesSplitter(self.lookback, blowup_nan=True)    
            
        self.weights  = weights
        
    def transform(self, df: DataFrame) -> DataFrame:
        
        # split data according to lookback
        df = self.splitter.transform(df)
        
        # calculate mean from split using weights
        df = df.apply(lambda col : np.mean(
               np.multiply(np.stack(col), self.weights),
               axis=1
            ), axis=0
        )
    
        # rename columns
        df.rename(
            columns = {
                col : col[:-(6+len(str(self.lookback)))] + f"_wma{self.lookback}" for col in df.columns
            }, inplace=True
        )
        
        return df
    
class RSI(DataTransformer):
    """takes a dataframe and computes the rsi version of each column"""
    def __init__(self, lookback: int):
        self.lookback = lookback
        
        self.mean = MovingMean(self.lookback)
        
    def transform(self, df: DataFrame) -> DataFrame:
        
        df_diff = df.diff()
        
        df_max = df_diff.applymap(lambda x :  max(x, 0))
        df_min = df_diff.applymap(lambda x : -min(x, 0))
        
        avg_up   = self.mean.transform(df_max)
        avg_down = self.mean.transform(df_min)
        
        df = 100 - 100 * avg_up / (avg_up + avg_down)
        
        return df.rename(
            columns = {
                col : col[:-(3+len(str(self.lookback)))] + f"_rsi{self.lookback}" for col in df.columns
            }
        )
    
class ExpAvg(DataTransformer):
    """takes a dataframe and computes the exponential average version of each column"""
    def __init__(self, alpha: float):
        self.alpha = alpha
        
    def transform(self, df: DataFrame) -> DataFrame:
        
        df = df.copy()
        
        prev = df.iloc[0]
        for i in range(1, len(df)):
            temp       = self.alpha * df.iloc[i] + (1 - self.alpha) * prev
            prev       = df.iloc[i]
            df.iloc[i] = temp
            
        df = df.rename(
            columns = {
                col : col + f"_eav{self.alpha:.2f}" for col in df.columns
            }
        )
        
        return df 
    
class HeikinAshiTransformer(DataTransformer):
    """takes a dataframe with HLOC data and returns heikin ashi candles"""
    def __init__(self, hloc: List[str]):
        self.hloc = hloc
    
    def transform(self, df: DataFrame) -> DataFrame:
        
        h_key, l_key, o_key, c_key = self.hloc
        
        H = df[h_key].fillna(method="ffill").fillna(0)
        L = df[l_key].fillna(method="ffill").fillna(0)
        O = df[o_key].fillna(method="ffill").fillna(0)
        C = df[c_key].fillna(method="ffill").fillna(0)

        # calculate heikin ashi close values
        close_ha = (H + L + O + C)/4
    
        # recursively calculate heikin ashi open values
        new_open = O[0]
        open_ha  = [new_open]
        for i in range(1,len(O)):
            new_open = (new_open + close_ha[i-1])/2
            open_ha.append(new_open)
    
        open_ha = np.array(open_ha)
    
        df_ha = pd.DataFrame.from_dict({
            o_key + '_ha' : open_ha,
            c_key + '_ha' : close_ha
        })
        
        df_ha.index = df.index
        
        return df_ha
    
class Normalizer(DataTransformer):
    """Normalizes all columns of the given dataframe"""
    def __init__(self, means: Optional[dict] = {}, stds: Optional[dict] = {}):
        self.means = means
        self.stds  = stds
    
    def transform(self, df: DataFrame) -> DataFrame:
        
        self.means = {
            col : np.mean(df[col]) for col in df.columns    
        }
        
        self.stds = {
            col : np.std(df[col]) for col in df.columns    
        }
        
        return df.apply(
            lambda col : (col - self.means[col.name])/self.stds[col.name],
            axis=0   
        )
    
    def get_params(self) -> dict:
        return {"means" : self.means, "stds" : self.stds}
    
    def set_params(self, params: dict) -> None:
        self.means, self.stds = params["means"], params["stds"]
    
    def mimic(self, df: DataFrame) -> DataFrame:
        if self.means == {} or self.stds == {}:
            raise AttributeError("Means and Deviations not set!")
        return df.apply(
            lambda col : (col - self.means[col.name])/self.stds[col.name],
            axis=0
        )
        
@DEPRECATED
class AllTaFeaturesTransformer(DataTransformer):
    """returns all features of the ta library"""
    def __init__(self, hlocv, drop_percentage=0.05, verbose=False):
        raise RuntimeError("Don't use this u fuckhead")
        self.hlocv           = hlocv
        self.drop_percentage = drop_percentage
        self.verbose         = verbose
        
        self.high, self.low, self.open, self.close, self.volume = hlocv
        
    def transform(self, df):
        
        from ta import add_all_ta_features
        
        df = df[self.hlocv].copy()
        
        df = add_all_ta_features(
            df     = df,
            open   = self.open,
            high   = self.high,
            low    = self.low,
            close  = self.close, 
            volume = self.volume
        )
        
        
        ser     = lambda col : df[col]
        percent = lambda ser : np.sum(np.isnan(ser))/len(ser)
        rule    = lambda col : percent(ser(col)) > self.drop_percentage
        
        faulty_keys = list(filter(rule, df.columns))
        
        if self.verbose:
            for pair in zip(faulty_keys,[percent(ser(key)) for key in faulty_keys]):
                print(pair)
        
        df.drop(
            self.hlocv + faulty_keys,
            axis='columns',
            inplace=True
        )
        
        return df
    
class CompareColumnsTransformer(DataTransformer):
    """takes a dataframe and compares all pairs of rows specifed by rule"""
    def __init__(self, rule: Callable):
        self.rule = rule
        
    def transform(self, df: DataFrame) -> DataFrame:
        
        keys = list(filter(self.rule, df.columns))
        pairs = list(set(combinations(keys, 2)))
        
        d = {}
        for key1, key2 in pairs:
            if key1 != key2:
                d[f"({key1}<{key2})"] = df[key1] < df[key2]
                
        comp_df = pd.DataFrame.from_dict(d)
        comp_df.index = df.index
                
        return pd.concat([df, comp_df], axis='columns')    
    
class AddTimeOfDay(DataTransformer):
    """takes a dataframe and prepens the time of day in seconds of the index"""
    def __init__(self):
        pass
    
    def transform(self, df: DataFrame) -> DataFrame:
        df.insert(0, 'time_of_day', df.index.map(
            lambda date : date.timestamp() % (24 * 60 *60)   
        ))
        return df
    
###############################################################################
        
    # META TRANSFORMERS
    
class MetaTransformer(DataTransformer):
    """class describing the interweaving of transformers"""
    def __init__(self, transformers: List[DataTransformer]):
        self.transformers = transformers
        self.num          = len(transformers)
        
class IgnoreKeysTransformer(MetaTransformer):
    """ignores all keys specified by a rule when applying a given transformation"""
    def __init__(self, rule: Callable, transformer: DataTransformer, IgnoreAllExceptSpecified: bool=False):
        self.rule        = rule
        self.transformer = transformer
        self.IAES        = IgnoreAllExceptSpecified
        
    def transform(self, df: DataFrame) -> DataFrame:
        
        keys = list(filter(self.rule, df.columns))
        
        if self.IAES:
            used_keys    = keys
            ignored_keys = df.columns.drop(keys)
        else:
            used_keys    = df.columns.drop(keys)
            ignored_keys = keys
        
        # pick data to be transformed and get column names
        picked = df[used_keys]
        
        # transform and get new names
        trans_picked = self.transformer.transform(picked)
        
        # get new length of dataframe
        new_length = min(len(df), len(trans_picked))
        
        # cut frame and replace old data with new data
        df = df[ignored_keys][-new_length:]
        
        df[trans_picked.columns] = trans_picked[-new_length:]
        
        # warnings.warn("Inner transformation might not rename columns, possibly leading to them being overwritten")
        
        return df
    
class FilterKeysTransformer(MetaTransformer):
    """only uses keys specified by rule when applying a given transformation"""
    def __init__(self, rule: Callable, transformer: DataTransformer, IgnoreSpecified: bool=False):
        self.transformer = transformer
        self.rule        = rule
        self.IS          = IgnoreSpecified
        
    def transform(self, df: DataFrame) -> DataFrame:
        
        keys = list(filter(self.rule, df.columns))
        
        if self.IS:
            used_keys = df.columns.drop(keys)
        else:
            used_keys = keys
            
        # pick data to be transformed and get column names
        picked = df[used_keys]
        
        # transform
        return self.transformer.transform(picked)

class KeepKeysTransformer(MetaTransformer):
    """applies a given transformer while retaining the specified columns"""
    def __init__(self, rule: Callable, transformer: DataTransformer):
        self.rule        = rule
        self.transformer = transformer
        
    def transform(self, df: DataFrame) -> DataFrame:
        
        keys = list(filter(self.rule, df.columns))
        
        # pick keys that will be kept
        pick = df[keys].copy()
        
        # transform all data
        df = self.transformer.transform(df)
        
        return pd.concat([pick, df], axis='columns')

class SequentialTransformer(MetaTransformer):
    """takes a list of transformers and composes them"""
    def __init__(self, transformers: List[DataTransformer]):
        super().__init__(transformers)

    def transform(self, df: DataFrame) -> DataFrame:
        for i in range(self.num):
            df = self.transformers[i].transform(df)
            
        return df
    
class ParallelTransformer(MetaTransformer):
    """takes a list of tranformers and applies them in parallel"""
    def __init__(self, transformers: List[DataTransformer]):
        super().__init__(transformers)
        
    def transform(self, df: DataFrame) -> DataFrame:
        return pd.concat(
            [T.transform(df) for T in self.transformers],
            axis='columns'
        )
###############################################################################
    
class StructureTransformer(DataTransformer):
    """class describing structural changes of dataframes"""
    def __init__(self):
        pass
    
    def transform(self, df: DataFrame) -> DataFrame:
        pass
    
class PickKeysTransformer(StructureTransformer):
    """picks the given keys and returns the corresponding dataframe"""
    def __init__(self, keys: List[str]):
        self.keys = keys

    def transform(self, df: DataFrame) -> DataFrame:
        return df[self.keys]
    
class DropColumnsTransformer(StructureTransformer):
    """drops the specified columns of a dataframe"""
    def __init__(self, keys: List[str]):
        self.keys = keys
        
    def transform(self, df: DataFrame) -> DataFrame:
        return df.drop(self.keys, axis=1)
    
class DropNaNsTransformer(StructureTransformer):
    """drops all rows from a dataframe containing NaN values in the specified columns"""
    def __init__(self, columns: Optional[List[str]]=None, consider_arrays: bool=False):
        self.columns         = columns
        self.consider_arrays = consider_arrays
    
    def transform(self, df: DataFrame) -> DataFrame:
        if self.consider_arrays:
            
            if self.columns is None:
                pick = df
            else:
                pick = df[self.columns]
            
            mask = pick.apply(
                lambda row : np.any(
                    row.apply(
                        lambda entry : np.any(np.isnan(entry))
                    ), axis=0
                ), axis=1
            )
            
            return df[~mask]
            
        else:
            return df.dropna()
    
class StackAlongRowsTransformer(StructureTransformer):
    """concat all arrays in a row into one"""
    def __init__(self):
        pass
    
    def transform(self, df: DataFrame) -> DataFrame:
        return pd.DataFrame(
            df.apply(
                lambda row : np.stack(row, axis=0),
                axis=1    
            )
        ).rename(
            columns = { 0 : "rows_stacked" }
        )
            
class RenameColumnsTransformer(StructureTransformer):
    """renames rows according to a rule specified"""
    def __init__(self, rule: Callable):
        self.rule = rule
        
    def transform(self, df: DataFrame) -> DataFrame:
        return df.rename(
            columns = {
                col : self.rule(col) for col in df.columns
            }
        )
        
class AddShiftsTransformer(StructureTransformer):
    """adds shifted version of all columns to a dataframe"""
    def __init__(self, shifts: List[int]):
        self.shifts = shifts
        
    def transform(self, df: DataFrame) -> DataFrame:
        df_list = [df]
        for shift in self.shifts:
            name = f"_{'prev' if shift > 0 else 'next'}{abs(shift)}"
            
            shifted_df = df.shift(shift)
            
            shifted_df.rename(
                columns = {col : col + name for col in df.columns},
                inplace=True
            )
            
            df_list.append(shifted_df)

        return pd.concat(df_list, axis='columns', join='outer')
    
###############################################################################