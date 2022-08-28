###############################################################################
############################### IMPORTS #######################################
###############################################################################

# external imports
import torch
import pandas as pd
import numpy as np
import random

###############################################################################

# internal imports

###############################################################################

# typing (external | internal)
from typing import Tuple, Optional
from pandas import DataFrame, Series
from torch import Tensor

from ..data_handling.dataprovider import DataProvider

###############################################################################
###############################################################################
###############################################################################

# logging
from ..logging_wrapper import bot_logger as log

###############################################################################
###############################################################################


class DiscreteActionSpace(object):
    def __init__(self, n: int):
        self.n = n

    def __repr__(self) -> str:
        return f'DisvreteActionSpace(n={self.n})'

    def sample(self) -> int:
        return random.randint(0, self.n-1)


class ContinuousObservationSpace(object):
    def __init__(self, shape: Tuple[int, ...]):
        self.shape = shape

    def __repr__(self) -> str:
        return f'ContinuousObservationSpace(shape={self.shape})'


class Envir(object):
    def __init__(self):
        pass

    def reset(self):
        pass

    def step(self, action):
        pass

    def render(self):
        pass

###############################################################################


class DataIterator(object):
    """describes an interator over a dataframe"""

    def __init__(self, df: DataFrame):
        self.position = 0
        self.df = df
        self.len = len(self.df)

    def __next__(self) -> bool:
        """move forward one step"""
        self.position += 1

        # check if the exists a next entry
        return self.position >= self.len - 1

    def reset(self) -> None:
        """move the head of the iterator to the start"""
        self.position = 0

    def get_current(self, key: Optional[str] = None, lookback: int = 1):
        """get the current row or entry if a key is specifed"""
        if key == None:
            return self.df.iloc[self.position-(lookback-1):self.position+1]
        else:
            return self.df[key].iloc[self.position]

    def move(self, position: int) -> None:
        """moves the head of the iterator to the specifed position"""
        self.position = position

    def peak(self, n: int, key: Optional[str] = None):
        """peaks n entries of the specified key"""
        if self.position + n > self.len:
            raise IndexError('Trying to peak over edge!')
        else:
            if key == None:
                return self.df.iloc[self.position:self.position+n]
            else:
                return self.df[key].iloc[self.position:self.position+n]

###############################################################################
###############################################################################


class MarketCalculator(object):
    def __init__(self, profile: dict):
        """calculates the factor over a candle"""

        # market constant
        self.lev = profile['lev']
        self.fees = profile['fees']
        assert profile['api'] == 'ByBit'  # BECAUSE OF REDUCED FUNDING
        self.funding = 1 - \
            (1 - profile['funding'])**(profile['resolution']/(8*60*60))

    def calc(self, OPEN: float, CLOSE: float, action: int, last_action: int, last_r: float) -> float:

        q = CLOSE / OPEN

        new_direc = action - 1
        last_direc = last_action - 1

        gain = new_direc * (q - 1)
        fees = abs(new_direc - last_direc / last_r) * self.fees
        funding = abs(new_direc) * self.funding

        new_r = max(1 + self.lev * (gain - fees - funding), 0)

        return new_r


class Market(object):
    """class describing the market"""

    def __init__(self, iterator: DataIterator):
        self.iterator = iterator

    def get_current(self) -> tuple:
        OPEN = self.iterator.get_current('__OPEN')
        CLOSE = self.iterator.get_current('__CLOSE')
        return OPEN, CLOSE

###############################################################################
###############################################################################


class TradingEnvir(Envir):
    """class describing the evironment of the agent"""

    def __init__(self, provider: DataProvider, profile: dict):
        super().__init__()

        self.provider = provider
        self.profile = profile

        self.train_phase = True

        self.lev = profile['lev']
        self.fees = profile['fees']
        self.funding = profile['funding']

        self.peak = profile['peak']
        self.lookback = profile['lookback']
        self.max_episode_len = profile['max_episode_len']

        self.train_split = profile['train_split']

        self.position_to_action = {
            'Short': 0, 'None': 1, 'Long': 2
        }
        self.action_to_position = {
            0: 'Short', 1: 'None', 2: 'Long'
        }

        self.df = self.provider.get_data().dropna()

        self.len = len(self.df)
        self.train_len = int(self.len * self.train_split)

        self.train_df = self.df.iloc[:self.train_len]
        self.test_df = self.df.iloc[self.train_len:]

        # lower and upper bounds for start index of episode
        self.low = self.lookback - 1
        self.high = self.train_len - self.max_episode_len - self.peak + 1

        if self.high < self.low:
            raise ValueError("Not enough data!")

        # Calculator for getting returns after actions
        self.calculator = MarketCalculator(self.profile)

        def initializer() -> None:
            # reset attributes which are modified over an episode
            self.step_counter = 0

            self.total_r = 1
            self.last_r = np.inf
            self.last_actions = [1] * self.lookback

            # get start index of episode
            if self.train_phase:
                # draw random start point low <= ... <= high
                self.start = random.randint(self.low, self.high)
            else:
                self.start = self.lookback - 1

            # pad episode infront and behind to have acces to enough data
            padded_start = self.start - (self.lookback - 1)
            padded_end = self.start + self.max_episode_len + self.peak

            # build episode as Iterator
            self.episode = DataIterator(
                self.train_df.iloc[padded_start:padded_end]
                if self.train_phase else
                self.test_df.iloc[padded_start:padded_end]
            )

            # move head past padding
            self.episode.move(self.lookback - 1)

            # fit market to episode
            self.market = Market(
                self.episode,
            )

        self.__internal_reset = initializer
        self.__internal_reset()

        self.action_space = DiscreteActionSpace(n=3)
        self.observation_space = ContinuousObservationSpace(
            shape=self.__get_observation().shape
        )

    def __calc_return(self, action: int) -> float:
        """calculates the reward of the given action"""
        OPEN = self.episode.get_current(key='__OPEN')
        CLOSE = self.episode.get_current(key='__CLOSE')

        return self.calculator.calc(
            OPEN=OPEN,
            CLOSE=CLOSE,
            action=action,
            last_action=self.last_actions[-1],
            last_r=self.last_r
        )

    def __peak_factors(self, action: int) -> Series:
        """peak into futures and evaluate position"""
        OPEN = self.episode.get_current(key='__OPEN')
        CLOSES = self.episode.peak(self.peak, key='__CLOSE')

        factors = CLOSES.apply(
            lambda close: self.calculator.calc(
                OPEN=OPEN,
                CLOSE=close,
                action=action,
                last_action=self.last_actions[-1],
                last_r=self.last_r
            )
        )

        return factors

    def __get_observation(self) -> Tensor:
        """return the current state of the environment"""
        # get observation from market data according to lookback
        observation = self.episode.get_current(
            lookback=self.lookback
        ).copy()

        # insert last (lookback many) actions into observation
        if len(observation.index) != len(self.last_actions[-self.lookback:]):
            print(observation)
            print(observation.index)
            print(self.last_actions)
            print(self.lookback)
        observation['last_actions'] = pd.Series(
            data=self.last_actions[-self.lookback:],
            index=observation.index
        )

        # drop private columns from state
        private_cols = [col for col in observation.columns if '__' in col]
        observation.drop(private_cols, axis='columns', inplace=True)

        return torch.Tensor(observation.values).float().unsqueeze(0)

    def reset(self) -> Tensor:
        self.__internal_reset()
        return self.__get_observation()

    def step(self, action: int) -> Tuple[Tensor, float, bool, str]:
        self.step_counter += 1

        # calculate return over current candle with given action
        r = self.__calc_return(action)

        # reward as log-return
        reward = np.log(r) if r > 0 else -np.inf

        # check if episode is done
        done = False
        done |= (self.step_counter >= self.max_episode_len)
        done |= (r == 0)

        # get to next timestep
        next(self.episode)

        self.total_r = self.total_r * r
        self.last_r = r
        self.last_actions = self.last_actions[1:] + [action]

        print(
            f'ACTION : {self.action_to_position[action]} as step {self.step_counter}')
        print(f'r      : {r:.6f} {"👍" if r >= 1 else "😰"}')
        print(f'reward : {reward:.6f} {"👍" if reward >= 0 else "😰"}')
        if done:
            print('\nEPISODE DONE!')
        print("—————————————————————————————————————————")

        # LOGGER.info(f'ACTION: {self.action_to_position[action]} as step {self.step_counter}')
        # LOGGER.info(f'r     : {r:.6f} {"👍" if r - 1 >= 0 else "😰"}')
        # LOGGER.info(f'reward: {reward:.6f} {"👍" if reward - 1 >= 0 else "😰"}')
        # if done: LOGGER.info('\n EPISODE DONE!')
        # LOGGER.info("—————————————————————————————————————————")

        # next state, reward, done
        return self.__get_observation(), reward, done, 'OK!'

    def train(self) -> None:
        """set environement to train mode"""
        self.train_phase = True

    def eval(self) -> None:
        """set environment to eval mode"""
        self.train_phase = False


if __name__ == "__main__":

    from profiles import profile_manager as pm
    from data_handling import dataprovider as dp
    from data_handling import data_lambdas as dls
    from data_handling import transformer_schemes as schemes

    import time

    profile = pm.get_profile_by_file('./profile_files/rl_training.json')

    lam = dls.DataLambda.get_lambda(
        profile=profile,
        end=time.time(),
        check=True
    )

    transformer = schemes.get_transformer(
        profile["transformer_uuid"]
    )

    provider = dp.DataProvider(
        lam, transformer
    )

    envir = TradingEnvir(provider, profile)

    done = False
    while not done:
        _, _, done, _ = envir.step(envir.action_space.sample())

    print(envir.total_r)
