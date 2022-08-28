###############################################################################
############################### IMPORTS #######################################
###############################################################################

#external imports

import torch
import torch.nn as nn

import sklearn.metrics as metrics

import numpy as np
import pandas as pd

import random

###############################################################################

# internal imports
from data_handling import dataprovider as dp
from data_handling import transformers as tr

from supervised_learning import test_targets as t

# import dueling_net as dn

###############################################################################

# typing (external | internal)
from typing import List, Tuple, Union

from numpy  import ndarray
from pandas import Index, Series
from torch  import Tensor

from data_handling.dataprovider import DataProvider
from data_handling.transformers import Normalizer

from supervised_learning.test_targets import Target

###############################################################################
###############################################################################
###############################################################################

# TODO
# CV search:
    # buffer between train and val set!

# TRAINING:
    # DONE! class weights to balance data set 
    # duplicate data according to weights
    # DONE! shuffle for each epoch
    # Leave out some batches? (https://datascience.stackexchange.com/questions/24511/why-should-the-data-be-shuffled-for-machine-learning-tasks)
    # Dropout (https://machinelearningmastery.com/dropout-for-regularizing-deep-neural-networks/)

# TESTING:
    # class weights for testing means

###############################################################################
###############################################################################
###############################################################################

# with convolutional networks
class NN(nn.Module):
    def __init__(self,
            nneurons   : List[int],
            afuncs     : List[str],
            drop_ps    : Union[None, List[float]] = None,
            logsoftmax : bool = True):
        super().__init__()
        
        self.nneurons   = nneurons
        self.afuncs     = afuncs
        self.drop_ps    = drop_ps
        self.logsoftmax = logsoftmax
        
        # build linear layers with in and outputs specified by nneurons
        self.linear_layers = []
        for num_in, num_out in zip(self.nneurons[:-1], self.nneurons[1:]):
            self.linear_layers.append(
                nn.Linear(num_in, num_out)
            )
        
        # build dropout layers
        if self.drop_ps is not None:
            self.dropout_layers = []
            for p in self.drop_ps:
                self.dropout_layers.append(
                    nn.Dropout(p)    
                )
        
        # build layers corresponding to activation functions speified by afuncs
        self.activation_layers = []
        for func in self.afuncs:
            self.activation_layers.append(
                self.__get_afunc_from_str(func)    
            )
            
        # check if we have the right amount of dropout layers
        if self.drop_ps is not None:
            assert len(self.linear_layers) == len(self.dropout_layers)
           
        # intermingle layers: ll, afunc, ll, afunc, ..., afunc, ll
        # (drops before each linear layer)
        self.layers = []
        llayers = self.linear_layers[::-1]
        alayers = self.activation_layers[::-1]
        if self.drop_ps is not None:
            dlayers = self.dropout_layers[::-1]
        even = True
        while True:
            if even:
                if len(llayers) == 0:
                    break
                else:
                    if self.drop_ps is not None:
                        self.layers.append(dlayers.pop())
                    self.layers.append(llayers.pop())
            else:
                if len(alayers) == 0:
                    break
                else:
                    self.layers.append(alayers.pop())
                    
            even ^= True
            
        # possibly add a logsoftmax layer:
        if self.logsoftmax:
            self.layers.append(nn.LogSoftmax(dim=1))
            
        # combine all layers into one stream
        self.stream = nn.Sequential(*self.layers)
        
        self.epochs_trained = 0
        
    def __get_afunc_from_str(self, func: str) -> nn.modules.activation:
        if func == "relu":  return nn.ReLU()
        if func == "prelu": return nn.PReLU()
        raise NotImplementedError(f"Activation function {func} not supported!")
        
    def __num_trainable_params(self) -> int:
        params = filter(lambda p : p.requires_grad, self.parameters())
        return sum([np.prod(p.size()) for p in params])
    
    def __repr__(self) -> str:
        return self.stream.__repr__() + f" with a total number of trainable parameters: {self.__num_trainable_params()}"
        
    def forward(self, x: Tensor) -> Tensor:
        return self.stream(x)

    def save(self, path: str) -> None:
        nn_params = {
            'nneurons'          : self.nneurons,
            'afuncs'            : self.afuncs,
            'drop_ps'           : self.drop_ps,
            'logsoftmax'        : self.logsoftmax,
            'state_dict'        : self.stream.state_dict()
        }
        
        params = {
            'nn_params'         : nn_params,
            'normalizer_params' : self.normalizer_params
        }
        
        torch.save(params, path)
                

    def load(self, path: str) -> None:
        # When implementing: Dont forget loading of dropout!
        raise RuntimeError('This class is not meant to support loading of models!')


class NNClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def predict(self, X: Tensor) -> int:
        with torch.no_grad():
            return self.model(X).argmax(dim=1).item() - 1
        
    def multi_predict(self, Xs: Tensor) -> Tensor:
        with torch.no_grad():
            return self.model(Xs).argmax(dim=1) - 1

    def save(self, path: str) -> None:
        raise RuntimeError('This class is not meant to support saving of models!')

    def load(self, params: dict) -> None:
        self.model = NN(
            nneurons   = params['nneurons'],
            afuncs     = params['afuncs'],
            drop_ps    = params['drop_ps'],
            logsoftmax = params['logsoftmax']    
        ).stream
        
        self.model.load_state_dict(params['state_dict'])
        
        self.model.eval()
        
class ModelLoader(object):
    def __init__(self):
        pass
    
    def load_normalizer(path :str) -> Normalizer:
        params = torch.load(path)
        
        Normalizer = tr.Normalizer()
        Normalizer.set_params(
            params['normalizer_params']
        )
        
        return Normalizer
    
    def load_nnet(path: str) -> Tuple[NNClassifier, Normalizer]:
        params = torch.load(path)
        
        Classifier = NNClassifier()
        Classifier.load(params['nn_params'])

        return Classifier
            
class DataSet(object):
    """holds a dataset and allows partitioning into batches"""
    def __init__(self, data: Tuple[Tensor, ...], index: Index):
        self.raw_data = data # never shuffled
        self.data     = data # possibly shuffled
        self.index    = index
        self.length   = len(self.data)
        
    def set_batch_size(self, batch_size: int, stack: bool) -> None:
        """sets the batch size and batch type of the datatset"""
        self.batch_size = batch_size
        self.stack      = stack
        
    def make_batches(self) -> None:
        """build batches from the given data set"""
        # set alias to make the following code more readable
        try: b, stack = self.batch_size, self.stack
        except: raise AttributeError('Batch size not set!')
        
        # loop to get data to split according to batch size and stack
        if stack:
            self.batches = tuple(
                torch.stack(
                    self.data[l:l+b] if l+b < self.length else self.data[l:]
                ) for l in range(0, self.length, b)
            )
        else:
            self.batches = tuple(
                torch.cat(
                    self.data[l:l+b] if l+b < self.length else self.data[l:]
                ) for l in range(0, self.length, b)
            )
        
        
    def get_batches(self) -> Tuple[Tensor, ...]:
        try:    return self.batches
        except: raise AttributeError("Batches not built!")
            
class TrainTestPair(object):
    """holds train Xs, ys and test Xs, ys and allows partitioning into batches"""
    def __init__(self, train_Xs: DataSet, train_ys: DataSet, test_Xs: DataSet, test_ys: DataSet):
        self.train_Xs = train_Xs
        self.train_ys = train_ys
        self.test_Xs  = test_Xs
        self.test_ys  = test_ys
    
    def shuffle(self) -> None:
        """shuffles the train Xs and ys"""
        a = self.train_Xs.raw_data
        b = self.train_ys.raw_data
        
        assert self.train_Xs.length == self.train_ys.length
        
        shuffled = random.sample(tuple(zip(a,b)), self.train_Xs.length)
        
        self.train_Xs.data, self.train_ys.data = zip(*shuffled)
    
    def set_batch_sizes(self, train_batch_size: int, test_batch_size:int) -> None:
        """sets the batch sizes for all contained datasets"""
        self.train_Xs.set_batch_size(train_batch_size, stack=True)
        self.train_ys.set_batch_size(train_batch_size, stack=False)
        self.test_Xs.set_batch_size(test_batch_size,   stack=True)
        self.test_ys.set_batch_size(test_batch_size,   stack=False)
        
    def make_train_batches(self) -> None:
        """builds the train batches as specified by previous call of set_batch_sizes"""
        self.train_Xs.make_batches()
        self.train_ys.make_batches()
        self.train_batches = tuple(
            zip(
                self.train_Xs.get_batches(), self.train_ys.get_batches()
            )
        )
        
    def make_test_batches(self) -> None:
        """builds the test batches as specified by previous call of set_batch_sizes"""
        self.test_Xs.make_batches()
        self.test_ys.make_batches()
        self.test_batches = tuple(
            zip(
                self.test_Xs.get_batches(), self.test_ys.get_batches()
            )
        )
        
    def get_train_batches(self) -> Tuple[Tuple[Tensor, Tensor], ...]:
        """get train batches previously built by call of make_batches"""
        try:    return self.train_batches
        except: raise AttributeError("Train batch not built!")
    
    def get_test_batches(self) -> Tuple[Tuple[Tensor, Tensor], ...]:
        """get test batches previously built by call of make_batches"""
        try:    return self.test_batches
        except: raise AttributeError("Test batch not built!")

class NNEnvir(object):
    def __init__(self, pair: TrainTestPair, optimizer, loss_func):
        self.pair      = pair
        self.optimizer = optimizer
        self.loss_func = loss_func

        self.pair.make_test_batches()        
        self.test_batches  = self.pair.get_test_batches()
        
    def set_model(self, model: NN) -> None:
        """set the model for the environment"""
        self.model = model
        
    def test_model(self) -> dict:
        """run model on test set and return information on performance"""
        
        self.model.eval()
        with torch.no_grad():
            num_points, num_correct = 0, 0
            for X, y in self.test_batches:
            
                preds = self.model(X).argmax(dim=1)
                
                num_points  += len(y)
                num_correct += (preds == y).sum().item()
        self.model.train()
            
        return {
            'accuracy' : num_correct/num_points    
        }
    
    def train_epoch(self) -> None:
        """trains the model using the train set split into batches"""
        # keep track of losses to print every 50 batches
        losses = []
        
        # shuffle train data and build into new batches
        self.pair.shuffle()
        self.pair.make_train_batches()
        
        # loop through batches
        for batch, (X, y) in enumerate(self.pair.get_train_batches()):
            
            # forward propagation
            preds = self.model(X)
            
            loss  = self.loss_func(preds, y)
            
            # remember loss
            losses.append(loss.item())
            
            # reset gradient
            self.optimizer.zero_grad()
            
            # back propagation
            loss.backward()
            
            # adjust model parameters
            self.optimizer.step()
            
            if batch % 50 == 0:
                # print info
                print(f"Batch {batch}, avg loss: {sum(losses)/len(losses)}")
                # reset loss list
                losses = []
                
        self.model.epochs_trained += 1
            
            
    def train(self, num_epochs: int) -> None:
        """trains the model with number of epochs specified"""
        
        # loop through epochs
        for epoch in range(1, num_epochs+1):
            print(f"\nStarting epoch {epoch}:")
            
            # training
            self.train_epoch()
            
            # test
            info = self.test_model()
        
            print(f"Epoch {epoch} ended with test accuracy: {info['accuracy']}")
        
    def get_confusion_matrix(self) -> ndarray:
        """returns the confusion matrix of the model on the test dataset"""
        
        self.model.eval()
        with torch.no_grad():
            # get all labels and predictions
            labels = []
            preds  = []
            for (X, y) in self.test_batches:
                labels.append(y)
                preds.append(
                    self.model(X).argmax(dim=1)
                )
        self.model.train()
        
        # concat labels and predicitons into one tensor respectively
        preds  = torch.cat(preds)
        labels = torch.cat(labels)
    
        # return confusion matrix
        return metrics.confusion_matrix(labels, preds)
    
    def get_classification_report(self) -> str:
        """returns a sklearn classification report of the model"""
        # get all labels and predictions
        
        self.model.eval()
        with torch.no_grad():
            labels = []
            preds  = []
            for (X, y) in self.test_batches:
                labels.append(y)
                preds.append(
                    self.model(X).argmax(dim=1)
                )
        self.model.train()
        
        # concat labels and predicitons into one tensor respectively
        preds  = torch.cat(preds)
        labels = torch.cat(labels)
        
        # return report
        return metrics.classification_report(
            labels, preds
        )
    
class NNCVSearch(object):
    """builds different models and tests them via cross validation"""
    def __init__(
            self,
            provider   : DataProvider,
            target     : Target,
            normalizer : Normalizer,
            nn_grid    : dict,
            optim_grid : dict,
            profile    : dict
        ):
        self.provider   = provider
        self.target     = target
        self.normalizer = normalizer
        self.nn_grid    = nn_grid
        self.optim_grid = optim_grid
        self.profile    = profile
        
        # extract data from profile
        self.n_iter    = self.profile['num_iterations']
        self.n_cv_sets = self.profile['num_cv_sets']
        
        # get data
        self.df = self.provider.get_data()
        
        # get slope for target labels
        self.quantile = self.profile["cone_quantile"]
        self.slope    = np.abs(
            self.df['__CLOSE_QUOT'] - 1
        ).quantile(q=self.quantile)
        
        # get labels
        self.df["labels"] = self.target.get_labels(self.df, self.slope)
        
        # drop nans
        self.df = self.df.dropna()
        
        # extract labels
        self.labels = self.df["labels"]
        self.df.drop(["labels"], axis='columns', inplace=True)
        
        # cut off unwanted part of df and labels
        self.val_length = self.df.shape[0] // self.n_cv_sets
        self.length     = self.val_length * self.n_cv_sets
        
        self.df     = self.df[    -self.length:]
        self.labels = self.labels[-self.length:]
        
        # drop private columns
        privs = [col for col in self.df.columns if "__" in col]
        self.df.drop(privs, axis='columns', inplace=True)
        
        # normalize data
        self.df_copy = self.df.copy()
        self.df = self.normalizer.transform(self.df)
        
        # retype
        self.df     = self.df.astype(np.float32)
        self.labels = self.labels.astype(np.int64)
        
        if self.profile['weighted']:
            # get weights for balanced training
            self.weights = torch.tensor(
                self.labels.value_counts().astype(np.float32).values
            )
        else:
            self.weights = None
        
        # remember input shape and occuring labels
        self.in_shape = self.df.shape[1]
        self.names    = list(np.unique(self.labels))
        
        # split data and labels into list of tensors
        self.Xs = tuple(
            T.flatten() for T in torch.tensor(self.df.values).split(1, dim=0)
        )
        self.ys = torch.tensor(self.labels.values).split(1, dim=0)
        
        # remember indicies
        self.X_inds = self.df.index
        self.y_inds = self.df.index
        
        # split into cv sets:
        self.cv_pairs = []
        for k in range(self.n_cv_sets):
            
            # get bounds for k-th validation set
            start =  k    * self.val_length
            end   = (k+1) * self.val_length
            
            # build testpair with k-th validation set
            pair = TrainTestPair(
                DataSet(
                    self.Xs[:start]+self.Xs[end:],
                    pd.Index.union(self.X_inds[:start], self.X_inds[end:])
                ),
                DataSet(
                    self.ys[:start]+self.ys[end:],
                    pd.Index.union(self.y_inds[:start], self.y_inds[end:])
                ),
                DataSet(self.Xs[start:end], self.X_inds[start:end]),
                DataSet(self.ys[start:end], self.y_inds[start:end])
            )
            
            # set batch sizes
            pair.set_batch_sizes(
                train_batch_size = profile['train_batch_size'],
                test_batch_size  = profile['test_batch_size']
            )
            
            self.cv_pairs.append(pair)
            
    def __sample_grid_entry(self, entry):
        if type(entry) == list:
            return random.sample(entry, 1)[0]
        else:
            return entry.rvs(1)[0]
    
    def __sample_grids(self) -> Tuple[dict, dict]:
        nlayers = self.__sample_grid_entry(self.nn_grid['nlayers'])
        
        nneurons = [self.in_shape] + [
            self.__sample_grid_entry(
                self.nn_grid['nneurons']
            ) for _ in range(nlayers-1)    
        ] + [3]
        
        afuncs = [
            self.__sample_grid_entry(
                self.nn_grid['afuncs']    
            ) for _ in range(nlayers-1)
        ]
        
        drop_ps = [
            self.__sample_grid_entry(
                self.nn_grid['drop_ps']    
            ) for _ in range(nlayers)
        ]
        
        nn_params = {
            'nneurons' : nneurons,
            'afuncs'   : afuncs,
            'drop_ps'  : drop_ps
        } 
        
        optim_params = {
            k : self.__sample_grid_entry(v) for k, v in self.optim_grid.items()    
        }
        
        return nn_params, optim_params
            
    def __get_model(self, params: dict) -> NN:
        nn = NN(**params)
        nn.normalizer_params = self.normalizer.get_params()
        return nn
            
    def __get_optimizer(self, optim_params, nn: NN) -> torch.optim:
        if optim_params['name'] == 'SGD': 
            return torch.optim.SGD(
                nn.parameters(), lr = optim_params['lr']        
            )
        elif optim_params['name'] == 'Adam': 
            return torch.optim.Adam(
                nn.parameters(), lr = optim_params['lr']        
            )
        else:
            raise NotImplementedError(
                f"Optimizer '{optim_params['name']}' is not supported!"
            )
            
    def __get_loss_func(self):
        if self.profile["loss_func"] == "CrossEntropyLoss":
            return nn.CrossEntropyLoss(weight=self.weights)
        if self.profile["loss_func"] == "NLLLoss":
            return nn.NLLLoss(weight=self.weights)
        else:
            raise NotImplementedError(
                f"Loss '{self.profile['loss_func']}' is not supported!"
            )
    
    def __build_envir(self, pair: TrainTestPair, nn_params: dict, optim_params: dict) -> NNEnvir:
        
        nn        = self.__get_model(nn_params)
        optimizer = self.__get_optimizer(optim_params, nn)
        loss_func = self.__get_loss_func()
        
        envir = NNEnvir(
            pair, optimizer, loss_func
        )
        
        envir.set_model(nn)
        
        return envir  
    
    def __prepare_iteration(self) -> List[NNEnvir]:
        
        nn_params, optim_params = self.__sample_grids()
        
        return [
            self.__build_envir(
                pair, nn_params, optim_params
            ) for pair in self.cv_pairs    
        ]
    
    def run_iteration(self, epochs: int):
        
        for envir in self.__prepare_iteration():
            envir.train(epochs)
            
            Xs = torch.stack(envir.pair.test_Xs.data)
            ys = torch.cat(envir.pair.test_ys.data)
            
            preds = envir.model(Xs).argmax(dim=1)
            
            
            
            ps, rs, fs, ss = metrics._classification.precision_recall_fscore_support(
                ys, preds
            )
            
            blueprint = '  {}  ' + '{:>9.2f}' * 3 + ' {:>9}\n'
            
            mapping = {0 : "short", 1 : " none", 2 : " long"}
            names   = [mapping[l] for l in self.names]
            
            report = "\n         precision   recall  f-score   support\n"
            for n, p, r, f, s in zip(names, ps, rs, fs, ss):
                report += blueprint.format(n, p, r, f, s)
            
            df = pd.DataFrame(zip(ps, rs, fs, ss))
            df.columns = ["precision", "recall", "f-score", "support"]
            df.index = names
            
            print(df)
            
            # print(report)
    
    def temp(self, pair: TrainTestPair) -> NNEnvir:
        
        nn_params, optim_params = self.__sample_grids()
        
        return self.__build_envir(pair, nn_params, optim_params)
    
    
class MarketEnvir(object):
    def __init__(self, provider: DataProvider, target: Target, profile: dict):
        self.provider = provider
        self.target   = target
        self.profile  = profile
        
        self.lev     = self.profile['lev']
        self.fees    = self.profile["fees"]
        self.res     = self.profile['resolution']
        self.funding = 1 - (1 - self.profile["funding"])**(self.res/(8*60*60))
        
        self.df        = self.provider.get_data()
        self.rel_quots = self.df["__CLOSE_QUOT"] - 1
        
    def __get_inpositions_and_switches(self, preds: Series) -> Tuple[Series, Series]:
        """get stays and switches of given predictions"""
        
        inpositions = (preds != 0).astype(np.int64)
        
        switches    = np.abs(preds.diff())
        switches[0] = 2 # worst case
        switches    = switches.astype(np.int64)
        
        return inpositions, switches
    
    def __get_fees_and_funding(self, preds: Series) -> Tuple[Series, Series]:
        """for given predictions calculates the fees and funding"""
        
        inpositions, switches = self.__get_inpositions_and_switches(preds)
        
        fees    = switches * self.fees
        funding = inpositions * self.funding
        
        return fees, funding
    
    def __run_predictions(self, preds: Series) -> Series:
        
        lev = self.profile['lev']
        
        fees, funding = self.__get_fees_and_funding(preds)
        
        assert np.all(preds.index == fees.index   )
        assert np.all(preds.index == funding.index)
        
        rel_quots = self.rel_quots[preds.index]
        
        factors = 1 + lev * (preds * rel_quots - fees - funding)
        factors = factors.fillna(1).apply(lambda x : max(x, 0))
        
        return factors
    
    def __get_predictions(self, model: NNClassifier, dataset: DataSet) -> Series:
        """takes a model and a dataset containing Xs and predicts on it"""
        
        Xs = torch.stack(dataset.data)
        
        preds = model.multi_predict(Xs)
        
        return pd.Series(preds, index=dataset.index)
    
    def test_model(self, model: NNClassifier, dataset: DataSet):
        """takes a model and a test dataset of Xs and tests on the market"""
        
        df = pd.DataFrame(index = self.df.index)
        
        df['preds'] = self.__get_predictions(model, dataset)
        df['T']     = df['preds'].shift(1)
        df['|T|']   = np.abs(df['T'])
        
        df['O'] = self.df['__OPEN']
        df['C'] = self.df['__CLOSE']
        
        df["q"]      = df['C'] / df['O']
        df["q^"]     = df["q"] - 1
        
        df["l"]      = np.sign(df["q^"])
        df["|l|"]    = np.abs(df["l"])
        
        df = df.dropna()
        
        r = [1]
        last_r = np.inf
        
        for i in range(1, len(df.index)):
            gain    = df["l"].iloc[i] * df["q^"].iloc[i]
            fees    = np.abs(df["l"].iloc[i] - df["l"].iloc[i-1] / last_r) * self.fees
            funding = df["|l|"].iloc[i] * self.funding
            
            new_r  = 1 + self.lev * (gain - fees - funding)
            last_r = new_r
            
            r.append(new_r)     
        df['full_r'] = r
        
        r = [1]
        last_r = np.inf
        for i in range(1, len(df.index)):
            gain    = df["T"].iloc[i] * df["q^"].iloc[i]
            fees    = np.abs(df["T"].iloc[i] - df["T"].iloc[i-1] / last_r) * self.fees
            funding = df["|T|"].iloc[i] * self.funding
            
            new_r  = 1 + self.lev * (gain - fees - funding)
            last_r = new_r
            
            r.append(new_r)
        
        df['r'] = r
        
        df['full_r'].apply(lambda x : max(x,0))
        df['r'].apply(lambda x : max(x, 0))
        
        df = df.dropna()
        
        print(f"full:   {df['full_r'].prod()}, {df['full_r'].prod()**(1/len(df))}")
        print(f"target: {df['r'].prod()}, {df['r'].prod()**(1/len(df))}")
        
        df = df.drop(['O', 'C', 'q', '|T|', '|l|'], axis='columns')
        return df
    
    def debug_print(self):
        pass

        

if __name__ == "__main__":
    from scipy.stats import uniform, randint
    import matplotlib.pyplot as plt
    import time

    from data_handling import transformers
    from data_handling import dataprovider as dp
    from data_handling import data_lambdas as dls
    
    from profiles import profile_manager as pm
    
    
    profile = pm.get_profile_by_file('./configs/nn.json')
    
    # lam = dls.get_test_lambda(profile)
    
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
    
    def exp(df, slope, alpha=0.5, shift=-1):
        def __cone(entry): return 0 if abs(entry) < slope else entry
        
        avg = dp.ExpAvg(alpha).transform(df[['__CLOSE']])
        avg = avg.apply(
            lambda col : col.div(col.shift(1)) - 1
        )[avg.columns[0]].apply(
            __cone
        )
        return np.sign(avg).shift(shift) + 1
    
    def smooth_exp(df, slope, alpha=0.8, shift=-1):
        def smooth(df: pd.DataFrame) -> pd.Series:
            labels = df['__LABEL']
            for i in reversed(range(1,df.shape[0]-1)):
                prv = df.iloc[i-1]
                nxt = df.iloc[i+1]
                if prv['__LABEL'] == nxt['__LABEL']:
                    if prv['__OPEN'] > nxt['__CLOSE'] and prv['__LABEL'] == -1:
                        labels.iloc[i] = -1
                    elif prv['__OPEN'] < nxt['__CLOSE'] and prv['__LABEL'] == 1:
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
        
        df['__LABEL'] = labels
        
        df['__LABEL'] = smooth(df)
        
        return df['__LABEL'].shift(shift) + 1
    
    def close_ma2(df, slope):
        def __cone(entry): return 0 if abs(entry) < slope else entry
        target = np.sign((df['__CLOSE_MA2_QUOT'] - 1).shift(-1).apply(__cone)) + 1
        return target
    
    def full(df, slope):
        def __cone(entry): return 0 if abs(entry) < slope else entry
        target = np.sign((df['__CLOSE_QUOT'] - 1).shift(-1).apply(__cone)) + 1
        return target
 
    # target = t.Target(full)
    # target = t.Target(close_ma2)
    # target = t.Target(exp)
    target = t.Target(smooth_exp)
    
    # grids
    nn_grid = {
        'nlayers'  : randint(low=2, high=10),
        'nneurons' : randint(low=10, high=200), 
        'afuncs'   : ['relu', 'prelu'],
        'drop_ps'  : uniform(loc=0, scale=1)
    }
        
    optim_grid = {
        'name'     : ['SGD', 'Adam'],
        'lr'       : uniform(loc=1e-5, scale=0.00099), #[1e-5, 1e-3]
        'momentum' : uniform(loc=0, scale=1.5)
    }
    
    print("Building search object")
    
    normalizer = dp.Normalizer()
    search = NNCVSearch(
        provider,
        target,
        normalizer,
        nn_grid,
        optim_grid,
        profile,  
    )
    
    ###########################################################################
    
    #   SELF BUILT
    
    net = NN(
        nneurons   = [search.in_shape, 500, 250, 50, 3],
        afuncs     = ["prelu"]*4,
        drop_ps    = [0.6, 0.5, 0.4, 0.3],
        logsoftmax = True
    )
    
    net.normalizer_params = search.normalizer.get_params()
    
    
    optimizer = torch.optim.SGD(
        params   = net.parameters(),
        lr       = 5e-4,
        momentum = 0.7,
        nesterov = True
        
    )
    
    loss_func = nn.NLLLoss()
    
    envir = NNEnvir(
        pair      = search.cv_pairs[2],
        optimizer = optimizer,
        loss_func = loss_func
    )
    
    envir.set_model(net)
    
    
    ###########################################################################
    
    #   AUTO BUILT
    
    # envir = search.temp(search.cv_pairs[2])    
    
    ###########################################################################
    
    print(envir.model)
    
    # envir.train(50)
    
    # matrix = envir.get_confusion_matrix()
    # plt.imshow(matrix)
    # print(envir.get_classification_report())
    
    # envir.model.save("./models/NN0002.pth")
    
    # C = ModelLoader.load_nnet("./models/NN0002.pth")
    
    # market = MarketEnvir(provider, target, profile)
    # df = market.test_model(C, search.cv_pairs[2].test_Xs)
        
