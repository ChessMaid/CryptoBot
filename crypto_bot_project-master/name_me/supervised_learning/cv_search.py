###############################################################################
############################### IMPORTS #######################################
###############################################################################

#external imports
from os import error
import numpy as np
np.warnings.filterwarnings('ignore')

import pandas as pd
import random
import time
from ..datafetching import dataprovider as dp
from ..datatransforming import transformers 

from enum import Enum
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split as split

from joblib import dump, load
from uuid import uuid4
from pymitter import EventEmitter


###############################################################################

# internal imports
from ..utils import stat_analysis as stat
from ..utils import plotting as pltt

###############################################################################
###############################################################################
###############################################################################

class Wrapper(BaseEstimator):
    def __init__(self, model, data, profile):
        super().__init__()
        
        self.model    = model
        self.df       = data
        self.profile  = profile
        
        self.res             = self.profile['resolution']
        self.funding         = self.profile['funding']
        self.reduced_funding = 1 - (1 - self.funding)**(self.res/(8*60*60))        
        
        # get price factors and cone slope
        self.df['%-diff'] = (self.df['__CLOSE_QUOT'] - 1).shift(-1)
        self.quantile     = self.profile["cone_quantile"]
        self.slope        = np.abs(self.df['%-diff']).quantile(q=self.quantile)
        
        # choose target for prediction
        if self.profile['target'] == 'ha_close':
            self.df['labels'] = (self.df['__CLOSE_HA_QUOT'] - 1).shift(-1)
            self.df['labels'] = np.sign(self.df['labels'].apply(self.__cone))
            
        if self.profile['target'] == 'ha_open':
            self.df['labels'] = (self.df['__OPEN_HA_QUOT'] - 1).shift(-2)
            self.df['labels'] = np.sign(self.df['labels'].apply(self.__cone))
            
        if self.profile['target'] == 'close_ma2':
            self.df['labels'] = (self.df['__CLOSE_MA2_QUOT'] - 1).shift(-1)
            self.df['labels'] = np.sign(self.df['labels'].apply(self.__cone))
        
        if self.profile['target'] == '%diff':
            self.df['labels'] = self.df['__CLOSE'].diff().shift(-1)
            self.df['labels'] = np.sign(self.df['labels'].apply(self.__cone))
        if self.profile['target'] == 'mix':
            mask = np.sign((self.df['__CLOSE_QUOT'] - 1).shift(-1).apply(self.__cone))
            smooth = np.sign((self.df['__CLOSE_MA2_QUOT'] - 1).shift(-1).apply(self.__cone))
            
            self.df['labels'] = np.multiply(smooth, mask)
        
        # drop all NaN rows
        self.df = self.df.dropna()
        
        # extract labels, weights and factors from dataframe
        self.labels  = self.df['labels']
        
        self.perct_diffs = self.df['%-diff']
        
        
        # delete private and temporary columns
        private_columns   = [col for col in self.df.columns if '__' in col]
        temporary_columns = ['labels', '%-diff']
        
        self.df_debug_copy = self.df.copy()
        
        self.df.drop(
            private_columns + temporary_columns,
            axis='columns',
            inplace=True
        ) 
        
        # split dataset into train and validation sets
        self.x_train, self.x_test, self.y_train, self.y_test = split(
            self.df,
            self.labels,
            test_size = 1 - self.profile['train_split'],
            shuffle   = self.profile['shuffle_dataset']
        )
        
    def __cone(self, slope):
        if abs(slope) < self.slope:
            return 0
        else:
            return slope

    def __get_stays_and_switches(self, preds):
        """for given predictions determines where we stay in current position"""
        
        stays    = ((preds == preds.shift(1)) & (preds != 0)).fillna(True)
        stays    = stays.apply(lambda b : 1 if b else 0)
        
        switches = np.abs(preds.diff())
        
        return stays, switches
        
        
    def __run_predictions(self, preds):
        
        lev = self.profile['train_lev']
        
        stays, switches = self.__get_stays_and_switches(preds)
        
        
        funding = self.reduced_funding * stays
        fees    = self.profile['fees'] * switches
        
        fees.index    = preds.index
        funding.index = preds.index
        
        perct_diffs = self.perct_diffs[preds.index]
        
        factors = (1 + lev * (preds * perct_diffs - fees - funding))
        factors = factors.fillna(1)
        factors = factors.apply(
            lambda r : 0 if r <= 0 else r    
        )
        
        return factors
    
    def __get_mean(self, preds, am_gm='gm'):
        
        factors = self.__run_predictions(preds)

        if am_gm == 'gm':
            return factors.prod() ** (1/len(factors)), factors
        elif am_gm == 'am':
            return factors.mean(), factors
        else:
            raise ValueError(f'{am_gm} is not a valid mean!')
        
    def get_params(self):
        return self.model.get_params()
        
    def set_params(self, params):
        self.model.set_params(**params)
    
    def fit(self, data, labels):
        self.model.fit(data, labels)
    
    def predict(self, x):
        preds = pd.Series(self.model.predict(x))
        preds.index = x.index
        return preds
        
    def score(self, preds):
        return self.__get_mean(preds, am_gm='gm')

    def save(self, path):
        dump(self.model, path)

    def load(self, path):
        self.model = load(path)
        
    def debug_plot(self):
        self.fit(self.x_train, self.y_train)
    
        preds = self.predict(self.x_test)
        facs  = self.__run_predictions(preds)
        
        print(f"%-correct: {np.mean(preds == self.y_test)}")
        print(f"Score: {facs.prod()**(1/len(facs))}")
    
        pltt.debug_plot(
            self.profile,
            self.df_debug_copy.loc[preds.index],
            preds,
            self.__get_stays_and_switches(preds),
            facs
        )


class CV_Search(object):
    """takes a model and performs a random search and scores via cross validation"""
    def __init__(self, model, grid, n_iter=10, n_cv_sets=5, verbose=0):
        self.model       = model
        self.orig_params = model.get_params()
        self.grid        = grid
        self.n_iter      = n_iter
        self.n_cv_sets   = n_cv_sets
        self.verbose     = verbose
           
        # get data and labels and resize to fit cross validation
        self.df         = self.model.df.copy()
        self.labels     = self.model.labels.copy()
        
        self.length     = len(self.df)
        self.val_length = self.length // self.n_cv_sets
        
        self.length = self.val_length * self.n_cv_sets
        self.df     = self.df[-self.length:]
        self.labels = self.labels[-self.length:]

        self.event_emitter = EventEmitter()
        
        if self.model.profile["shuffle_dataset"]:
            # shuffle data and labels in parallel
            self.df     = self.df.sample(frac=1)
            self.labels = self.labels.loc[self.df.index]
        
        # split into cv sets
        self.cv_sets = []
        for k in range(self.n_cv_sets):
            
            start = k * self.val_length
            end   = start + self.val_length
            
            val_index   = self.df.index[start:end]
            train_index = self.df.index[~self.df.index.isin(val_index)]
            
            self.cv_sets.append((
                self.df.loc[train_index], self.labels.loc[train_index],
                self.df.loc[val_index],   self.labels.loc[val_index]
            ))
            
    
    def __sample_grid_entry(self, entry):
        if type(entry) == list:
            return random.sample(entry, 1)[0]
        else:
            return entry.rvs(1)[0]
    
    def __sample_grid(self):
        return {
            k : self.__sample_grid_entry(v) for k, v in self.grid.items()    
        }
    
    def __set_params(self, **params):
        self.model.set_params(params)
        
    def __get_params(self):
        return self.model.get_params()
        
    def __reset_model(self):
        self.model.set_params(self.orig_params)
        
    def __fit_model(self, data, labels):
        s = time.time()
        self.model.fit(data, labels)
        e = time.time()
        return e-s
        
    def __predict(self, data):
        return self.model.predict(data)
        
    def __percent_correct(self, preds, labels):
        return np.mean(preds == labels)
        
    def __score(self, preds):
        return self.model.score(preds)
    
    def __run_cv_set(self, cv_set, model_params):
        
        x_train, y_train, x_val, y_val = cv_set
        
        self.__reset_model()
        self.__set_params(**model_params)
        
        if self.verbose >= 3:
            print('    Fitting model.', end=' ')
        
        fit_time = self.__fit_model(x_train, y_train)            

        if self.verbose >= 3:
            print(f'Done! Time elapsed: {fit_time:.3f} seconds')
        
        train_preds = self.__predict(x_train)
        
        train_perct = self.__percent_correct(
            train_preds, self.labels[train_preds.index]
        )
        
        train_score, train_facs = self.__score(
            train_preds
        )
        
        val_preds = self.__predict(x_val)
        
        val_perct = self.__percent_correct(
            val_preds, self.labels[val_preds.index]
        )
        
        val_score, val_facs = self.__score(
              val_preds  
        )
            
        if self.verbose >= 4:
            print(f'    Validation percent and score: {val_perct:.3f}, {val_score:.5f}')
        
        return train_perct, train_score, train_facs, val_perct, val_score, val_facs
        
    def __run_iteration(self, model_params):
        
        train_percents = []
        train_scores   = []
        train_facs     = []
        val_percents   = []
        val_scores     = []
        val_facs       = []
        
        for cv_set, k in zip(self.cv_sets, range(self.n_cv_sets)):
            
            self.event_emitter.emit('set_run', current=k, max=self.n_cv_sets)
            
            if self.verbose >= 2:
                print(f'  Running cv-set number {k+1} out of {self.n_cv_sets}')
            
            tp, ts, tf, vp, vs, vf = self.__run_cv_set(cv_set, model_params)
            
            train_percents.append(tp)
            train_scores.append(  ts)
            train_facs.append(    tf)
            val_percents.append(  vp)
            val_scores.append(    vs)
            val_facs.append(      vf)
            
        return {
            'train_percts'  : train_percents, 'train_scores' : train_scores,
            'train_factors' : train_facs,
            'val_percts'    : val_percents,   'val_scores'   : val_scores,
            'val_factors'   : val_facs
        }
            
    
    def run(self):
        
        results = []
        params  = []
        
        for k in range(self.n_iter):

            self.event_emitter.emit('iteration_run', current=k, max=self.n_iter)
            
            if self.verbose >= 1:
                print(f'Starting {k}-{"st" if k%10==1 else "nd" if k%10==2 else "rd" if k%10==3 else "th"} iteration')
            
            model_params = self.__sample_grid()
            
            params.append(model_params)
            results.append(self.__run_iteration(model_params))
        
        
        df = pd.DataFrame(results, index=range(self.n_iter))
        
        mean = lambda col : col.apply(
            lambda entry : np.mean(np.array(entry))    
        )
        
        df["min_val_score"] = df["val_scores"].apply(
            lambda entry : np.min(entry)    
        )
        
        df['train%']      = mean(df['train_percts'])
        df['train_score'] = mean(df['train_scores'])
        df['val%']        = mean(df['val_percts'])
        df['val_score']   = mean(df['val_scores'])
        
        df['params'] = params
        
        var = lambda col : col.apply(
            lambda entry : stat.var(pd.concat(entry))
        )
        
        df['train_var'] = var(df['train_factors'])
        df['val_var']   = var(df['val_factors'])
        
        skew = lambda col : col.apply(
            lambda entry : stat.skew(pd.concat(entry))
        )
        
        df['train_skew'] = skew(df['train_factors'])
        df['val_skew']   = skew(df['val_factors'])
        
        return df                            

def get_profile_provider(profile):
    from datafetching import data_lambdas as dls
    
    data_lambda = dls.get_single_lambda(profile, time.time())
    transformer = transformers.get_transformer('614e94a2-3a35-4341-ba25-ad2b4a8709e8')
            
    return dp.DataProvider(data_lambda, transformer)

class Classifier(Enum):
    GRADIENT_BOOSTED_CLASSIFIER = "GRADIENT_BOOSTED_CLASSIFIER",
    RANDOM_FOREST_CLASSIFIER = "RANDOM_FOREST_CLASSIFIER",
    SUPPORT_VECTOR_MACHINE = "SUPPORT_VECTOR_MACHINE",

def get_model_and_grid(classifier: Classifier):
    # models
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.svm import SVC  

    # random distributions  
    from scipy.stats import uniform, randint

    # grids
    forest_grid = {
      'n_estimators'   : randint(low=10, high=250),
      'criterion'      : ['entropy', 'gini'],
      'max_depth'      : randint(low=25, high=50),
      'max_features'   : [None, 'sqrt', 'log2'],
      'bootstrap'      : [False, True],
      'max_leaf_nodes' : randint(low=10, high=100),
      'ccp_alpha'      : uniform(loc=0, scale=10 * 0.00005)
    }
    
    boosted_grid = {
      'learning_rate'     : uniform(loc=0.1, scale=0.50),
      'n_estimators'      : randint(low=10, high=50),
      'subsample'         : uniform(loc=0.75, scale=0.25),
      'criterion'         : ['friedman_mse', 'mse'],
      'max_depth'         : randint(low=5, high=8),
      'max_features'      : [None, 'sqrt', 'log2'],
      'ccp_alpha'         : uniform(loc=0, scale=10 * 0.00005)
    }
    
    svm_grid = {
        'C'      : uniform(loc=0, scale=2),
        'kernel' : ['linear', 'poly', 'rbf', 'sigmoid'],
        'degree' : randint(low=2, high=8),
        'gamma'  : ['scale', 'auto'],
        'coef0'  : uniform(loc=-1, scale=2),
    }
    
    if classifier == Classifier.GRADIENT_BOOSTED_CLASSIFIER:
        return GradientBoostingClassifier(loss = 'deviance'), boosted_grid
    elif classifier == Classifier.RANDOM_FOREST_CLASSIFIER:
        return RandomForestClassifier(), forest_grid
    elif classifier == Classifier.SUPPORT_VECTOR_MACHINE:
        return SVC(), svm_grid
    else:
        raise ValueError('classifier type not supported')
    
if __name__ == "__main__":
            
    from profiles import profile_manager as pm
    
    CONFIG_FILE_NAME = './configs/cv_search.json'
        
    profile = pm.get_profile_by_file(CONFIG_FILE_NAME)

    provider = get_profile_provider(profile)
    model, grid = get_model_and_grid(Classifier.GRADIENT_BOOSTED_CLASSIFIER)
    wrapped = Wrapper(model, provider.get_data(), profile)
            
    ### AREA 51
    search = CV_Search(
        model     = wrapped,
        grid      = grid,
        n_iter    = 1,
        n_cv_sets = 10,
        verbose   = 4
    )
    ### AREA 51 END
    
    save = input('Save best model? [Y/n]:') == 'Y'
    filename = str(uuid4()) + '.joblib'    

    if save:
        cust_filename = input(f'Filename [{filename}]:')
        filename = (cust_filename if cust_filename != '' else filename) + '.joblib'

    plot = input('Plot best model? [Y/n]:') == 'Y'
    print()
    
    result = search.run().sort_values(['val_score'], ascending=False)
    
    print(result[["train%", "train_score", "val%", "val_score", "min_val_score"]])
    
    if save:
        wrapped.set_params(result['params'][0])
        wrapped.fit(wrapped.df, wrapped.labels)
        wrapped.save(path=f'./models/{filename}')

    if plot:
        wrapped.set_params(result['params'][0])
        wrapped.debug_plot()
        
        
        
        
