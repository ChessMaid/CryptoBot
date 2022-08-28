###############################################################################
############################### IMPORTS #######################################
###############################################################################

# external imports

import torch
import torch.nn as nn

import random
import numpy as np

###############################################################################

# internal imports
from ..reinforcement_learning import replay_memory as rm

from ..utilities.decorators import DEPRECATED

###############################################################################

# typing (external | internal)
from typing import List, Tuple, Union
from torch import Size, Tensor
from pandas import DataFrame

from reinforcement_learning.envir import Envir

###############################################################################
###############################################################################
###############################################################################

# logging
from logging_wrapper import bot_logger as log

LOGGER = log.get_logger('training', 'training.log')

###############################################################################
###############################################################################
###############################################################################


@DEPRECATED
class DuellingQfunc(nn.Module):
    def __init__(self,
                 in_shape: tuple,
                 kernel_shapes: List[Tuple[int, int]],
                 profile: dict):
        super().__init__()

        self.in_shape = in_shape
        self.kernel_shapes = kernel_shapes
        self.profile = profile

        self.num_actions = 3

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=2,
                kernel_size=self.kernel_shapes[0]
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=2,
                out_channels=2,
                kernel_size=self.kernel_shapes[1]
            ),
            nn.ReLU()
        )

        # dynamicially find the output shape after
        # passing through convolutional part of the network
        self.conv_output_shape = self.__get_conv_output_shape()
        self.conv_output_size = self.conv_output_shape.numel()

        # models the value of the given state
        self.value_stream = nn.Sequential(
            nn.Linear(self.conv_output_size, 128),
            nn.ReLU(),
            nn.Linear(128, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

        # models the relative advantages of each action in the current state
        self.advantage_stream = nn.Sequential(
            nn.Linear(self.conv_output_size, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_actions)
        )

    def __get_conv_output_shape(self) -> Size:
        """dynamicially compute output shape of conv layer"""
        with torch.no_grad():
            return self.conv(
                torch.zeros((1, 1) + self.in_shape)
            ).shape

    def forward(self, states: Tensor) -> Tensor:

        # pass data through convolutional part
        features = self.conv(states)

        # flatten features for perceptron pass
        batch_size = states.shape[0]
        features = features.view(batch_size, -1)

        # find values of states and advantages of actions
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)

        # calculate Q-values according to https://arxiv.org/pdf/1511.06581.pdf
        Qvals = values + (advantages - advantages.mean())

        return Qvals

    # def save(self, path):
    #     torch.save(self.state_dict(), path)

    # def load(self, path):
    #     self.load_state_dict(torch.load(path))

    def num_trainable_params(self) -> int:
        """returns the number of trainable paramters"""
        params = filter(lambda p: p.requires_grad, self.parameters())
        return np.sum([p.numel() for p in params])


@DEPRECATED
class DoubleQfunc(nn.Module):
    def __init__(self,
                 in_shape: tuple,
                 kernel_shapes: List[Tuple[int, int]],
                 num_actions: int,
                 profile: dict):
        super().__init__()

        # device on which to carry out matrix operations
        self.device = torch.device(profile['device'])

        # hyperparamters
        self.gamma = self.profile['gamma']
        self.target_period = self.profile['target_period']
        self.loss_func = nn.MSELoss()

        # online and target Q functions
        self.online_Q = DuellingQfunc(
            in_shape, kernel_shapes, num_actions, profile)
        self.target_Q = DuellingQfunc(
            in_shape, kernel_shapes, num_actions, profile)

        # copy parameters from online to target networks
        self.target_Q.load_state_dict(self.online_Q.state_dict())

        # counter for when to copy paramters into target Qfunc networks
        self.target_counter = 0

    def update(self, batch_size: int) -> None:
        """samples a batch and updates the network according to the bellman equation"""

        # sample batch from replay memory
        batch = self.replay_buffer.sample(batch_size)

        # unpack batch
        states, actions, rewards, next_states = batch

        # convert batch into tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)

        # calculate current Q
        current_Qvals = self.online_Q(states)

        # next actions according to online Qfunc
        next_actions = torch.argmax(self.online_Q(next_states), dim=1)

        # expected Qvals of actions according to target Qfunc
        expected_Qvals = rewards + self.gamma * \
            self.target_Q(next_states)[next_actions]

        # calculate loss for update
        loss = self.loss_func(current_Qvals, expected_Qvals)

        # update network parameters
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # increment counter for target Qfunc update
        self.target_counter += 1

        # copy parameters
        if self.target_counter % self.target_period == 0:
            self.target_Q.load_state_dict(self.online_Q.state_dict())


if __name__ == "__main__":

    from profiles import profile_manager as pm

    profile = pm.get_profile_by_file('./profile_files/rl_training.json')

    net = DuellingQfunc((8, 30), [(2, 1)]*3, profile)

    print(net.num_trainable_params())
