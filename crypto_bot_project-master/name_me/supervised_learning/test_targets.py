###############################################################################
############################### IMPORTS #######################################
###############################################################################

#external imports
import numpy as np
np.warnings.filterwarnings('ignore')

###############################################################################

# internal imports
from utilities import plotting as pltt

###############################################################################
###############################################################################
###############################################################################

class Target(object):
    def __init__(self, rule):
        self.rule = rule
        
    def get_labels(self, df, slope):
        return self.rule(df, slope)

class TargetEnvironment(object):
    def __init__(self, provider, profile):
        super().__init__()
        
        # save profile and extract needed date
        self.profile  = profile
        
        self.lev             = self.profile['test_lev']
        self.res             = self.profile['resolution']
        self.funding         = self.profile['funding']
        self.reduced_funding = 1 - (1 - self.funding)**(self.res/(8*60*60))
        self.fees            = self.profile['fees']
        
        # get data
        self.provider = provider
        self.df       = self.provider.get_data().dropna()
        
        self.quantile = self.profile["cone_quantile"]
        self.slope = np.abs(
            self.df['__CLOSE_QUOT'] - 1
        ).quantile(q=self.quantile)
        
    def __run_labels(self, labels):
        
        df = pd.DataFrame(index = self.df.index)
        
        df['T']   = labels
        df['|T|'] = np.abs(df['T'])
        
        df['O'] = self.df['__OPEN']
        df['C'] = self.df['__CLOSE']
        
        df['q']      = df['C'] / df['O']
        df['q^']     = df['q'] - 1
        
        df = df.dropna()
        
        r = [1]
        last_r = np.inf
        for i in range(1, len(df.index)):
            gain    = df['T'].iloc[i] * df['q^'].iloc[i]
            fees    = np.abs(df['T'].iloc[i] - df['T'].iloc[i-1] / last_r) * self.fees
            funding = df['|T|'].iloc[i] * self.funding
            
            new_r  = 1 + self.lev * (gain - fees - funding)
            last_r = new_r
            
            r.append(new_r)
        
        df['r'] = r
        
        return df['r'].apply(lambda x : max(x, 0))
    
    def test_target(self, target: Target):
        
        labels = target.get_labels(self.df, self.slope).shift(1)
        
        factors = self.__run_labels(labels)
    
        product = factors.prod()
        mean    = product**(1/len(factors))
        
        perct_of_candles_traded = np.mean(labels != 0)
        
        return product, mean, perct_of_candles_traded
    
    def debug_plot(self, target):
        
        labels = target.get_labels(self.df, self.slope).shift(1)
        facs   = self.__run_labels(labels)
        
        df = self.df
        # df['__CLOSE'] = dp.ExpAvg(0.5).transform(
        #     dp.ExpAvg(0.5).transform(df[['__CLOSE']])
        # )
        # df['__OPEN'] = df['__CLOSE'].shift(-1)
        
        df     = df.loc[facs.index]
        labels = labels.loc[facs.index]
        
        
        pltt.debug_plot(self.profile, df.iloc[:200], labels.iloc[:200], facs.iloc[:200])

    
    
if __name__ == "__main__":
    
    from ..datatransforming import transformers
    from ..datafetching import data_lambdas as dls
    from ..profiles import profile_manager as pm
    from ..datafetching import dataprovider as dp
    import pandas as pd
    import time
    
    
    CONFIG_FILE_NAME = './configs/target_testing.json'
        
    profile = pm.get_profile_by_file(CONFIG_FILE_NAME)
    
    lam = dls.get_lambda(
        profile = profile,
        end     = time.time(),
        check   = True
    )
    
    transformer = transformers.get_transformer(
        profile["transformer_uuid"]
    )
       
            
    provider = dp.DataProvider(
        lam, transformer
    )

    envir = TargetEnvironment(provider, profile)
    
    def full(df, slope):
        
        def __cone(entry): return 0 if abs(entry) < slope else entry
        
        return np.sign((df['__CLOSE_QUOT'] - 1).shift(-1).apply(__cone))
    
    def close_ha(df, slope):
        def __cone(entry): return 0 if abs(entry) < slope else entry
        return np.sign((df['__CLOSE_HA_QUOT'] - 1).shift(-1).apply(__cone))
    
    def close_ma(df, n, slope):
        def __cone(entry): return 0 if abs(entry) < slope else entry
        
        return np.sign((df[f'__CLOSE_MA{n}_QUOT'] - 1).shift(-1).apply(__cone))
    
    def mix(df, n, slope):
        mask = np.abs(full(df, slope))
        smooth = close_ma(df, n, slope)
        return np.multiply(smooth, mask)
    
    def exp(df, alpha, shift, slope):
        def __cone(entry): return 0 if abs(entry) < slope else entry
        
        avg = dp.ExpAvg(alpha).transform(df[['__CLOSE']])
        avg = avg.apply(
            lambda col : col.div(col.shift(1)) - 1
        )[avg.columns[0]].apply(
            __cone
        )
        
        return np.sign(avg).shift(shift)
    
    def smooth_exp(df, slope, alpha=0.5, shift=-1):
        def smooth(df: pd.DataFrame) -> pd.Series:
            labels = df['label']
            for i in reversed(range(1,df.shape[0]-1)):
                prv = df.iloc[i-1]
                nxt = df.iloc[i+1]
                if prv['label'] == nxt['label']:
                    if prv['__OPEN'] > nxt['__CLOSE'] and prv['label'] == -1:
                        labels.iloc[i] = -1
                    elif prv['__OPEN'] < nxt['__CLOSE'] and prv['label'] == 1:
                        labels.iloc[i] = +1
            return labels
                    
        def __cone(entry): return 0 if abs(entry) < slope else entry
        def cone(series: pd.Series) -> pd.Series:
            return series.apply(
                lambda entry : abs(entry) < slope
            )
        
        labels = dp.ExpAvg(alpha).transform(df[['__CLOSE']])
        labels = labels.apply(
            lambda col : col.div(col.shift(1)) - 1
        )[labels.columns[0]]
        labels = labels.apply(
            __cone    
        )
        labels = np.sign(labels)
        
        df['label'] = labels
        
        df['label'] = smooth(df)
        
        return df['label'].shift(shift)
    
    def maexp(df, alpha, n, slope):
        def __cone(entry): return 0 if abs(entry) < slope else entry
        
        avg = dp.ExpAvg(alpha).transform(df[['__CLOSE']])
        avg = avg.rolling(n).mean()
        avg = avg.apply(
            lambda col : col.div(col.shift(1)) - 1
        )[avg.columns[0]].apply(
            __cone
        )
        
        return np.sign(avg).shift(-2)
    
    def test(df, slope):
        avg = df['__CLOSE']
        
        quot01 = avg.div(avg.shift(1)) - 1
        quot12 = quot01.shift(1)
        quot23 = quot01.shift(2)
        
        up   = np.sign(quot01) == 1
        down = np.sign(quot01) == -1
        
        fst_sign_momentum = np.sign(quot01) == np.sign(quot12)
        snd_sign_momentum = np.sign(quot01) == np.sign(quot23)
        
        sign_momentum = fst_sign_momentum & snd_sign_momentum
        
        fst_up_momentum   = quot01 > quot12 
        fst_down_momentum = quot01 < quot12
        
        snd_up_momentum   = quot12 > quot23
        snd_down_momentum = quot12 < quot23
        
        up_momentum   =   fst_up_momentum &   snd_up_momentum
        down_momentum = fst_down_momentum & snd_down_momentum
        
        slope = np.abs(quot01) > slope
        
        long  = up   & up_momentum   & sign_momentum & slope
        short = down & down_momentum & sign_momentum & slope
        
        target = pd.Series(data=[0]*df.shape[0], index=df.index)
        
        target[ long] =  1
        target[short] = -1
        
        print(np.sum(target != 0), len(target))
        
        return target
        
    def three_last(df, slope):
        C = df['__CLOSE']
        O = df['__OPEN']
        
        quot01 = C.div(C.shift(1)) - 1
        quot12 = quot01.shift(1)
        quot23 = quot01.shift(2)
        
        up   = np.sign(quot01) == 1
        down = np.sign(quot01) == -1
        
        fst_sign_momentum = np.sign(quot01) == np.sign(quot12)
        snd_sign_momentum = np.sign(quot01) == np.sign(quot23)
        
        sign_momentum = fst_sign_momentum & snd_sign_momentum
        
        slope = np.abs(quot01) > slope
        
        long  = up   & sign_momentum 
        short = down & sign_momentum 
        
        target = pd.Series(data=[0]*df.shape[0], index=df.index)
        
        target[ long] =  1
        target[short] = -1
        
        print(np.sum(target != 0), len(target))
        
        return target
        
        
    SLOPE = envir.slope
    FULL  = Target(full)
    
    Close_ha = Target(close_ha)
    Close_ma = Target(lambda df, slope : close_ma(df, 2, slope))
    Mixed    = Target(lambda df, slope : mix(df, 2, slope))
    Exp      = Target(lambda df, slope : exp(df, 0.5, -1, slope))
    SExp     = Target(lambda df, slope : smooth_exp(df, slope, 0.8, -1))
    MaExp    = Target(lambda df, slope : maexp(df, 0.8, 2, slope))
    
    Test      = Target(test)
    ThreeLast = Target(three_last)
    
    print("\nFull:")
    print(envir.test_target(FULL))
    
    print("\nTargets:")
    print("HA:    ", envir.test_target(Close_ha))
    print("MA:    ", envir.test_target(Close_ma))
    print("MIX:   ", envir.test_target(Mixed))
    # print("EXP:   ", envir.test_target(Exp))
    # print("SEXP:   ", envir.test_target(SExp))
    # print("MAEXP: ", envir.test_target(MaExp))
    
    print('\nStrategies:')
    print('Test:      ', envir.test_target(Test))
    print('ThreeLast: ', envir.test_target(ThreeLast))
    
    # envir.debug_plot(Exp)
    # envir.debug_plot(SExp)

    # N = 16
    # S = 2
    
    # search = []
    # cnt = 0
    # for alpha in np.linspace(1/N, 1-1/N, N-1):
    #     for shift in range(-S, 1):
    #         if cnt % 5 == 0:
    #             print(f'@ iteration {cnt}')
    #         cnt += 1
    #         target = Target(lambda df, slope : exp(df, alpha, shift, slope))
    #         result = envir.test_target(target)
                                
    #         search.append((result, alpha, shift))
            
    # # sort wrt search result, best one at position 0
    # search.sort(key = lambda tup : tup[0][0], reverse = True)
    
    # print(search)
    
    
    
    
    
    
    
    
    