###############################################################################
############################### IMPORTS #######################################
###############################################################################

#external imports
import torch
import torch.nn as nn

import numpy as np

###############################################################################

# internal imports

###############################################################################

# typing (external | internal)
from typing import Tuple
from torch  import Size, Tensor

###############################################################################
###############################################################################
###############################################################################
    
class DuellingQfunc(nn.Module):
    def __init__(self, in_shape: tuple, num_actions: int):
        self.in_shape    = in_shape
        self.num_actions = num_actions
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels  = 1,
                out_channels = 8, 
                kernel_size  = (4,1),
                stride       = (1,1)
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels  = 8,
                out_channels = 16,
                kernel_size  = (4,1),
                stride       = (1,1)
            ),
            nn.ReLU(),
        )
        
        # dynamicially find the output shape after
        # passing through convolutional part of the network
        self.conv_output_shape = self.__get_conv_output_shape()
        self.conv_output_size  = self.conv_output_shape.numel()
        
        # models the value of the given state
        self.value_stream = nn.Sequential(
            nn.Linear(self.conv_output_size, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
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
                torch.zeros((1,) + self.in_shape)
            ).shape
    
    def forward(self, states: Tensor) -> Tensor:
        
        # pass data through convolutional part
        features = self.conv(states)
        
        # flatten features for perceptron pass
        batch_size = states.shape[0]
        features = features.view(batch_size, -1)
        
        # find values of states and advantages of actions
        values     = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # calculate Q-values according to https://arxiv.org/pdf/1511.06581.pdf
        Qvals = values + (advantages - advantages.mean())
        
        return Qvals
        
    def num_trainable_params(self) -> int:
        """returns the number of trainable paramters"""
        params = filter(lambda p : p.requires_grad, self.parameters())
        return np.sum([p.numel() for p in params])
    
class DoubleQfunc(nn.Module):
    def __init__(self, in_shape: tuple, num_actions: int, profile: dict):
        super().__init__()
        
        self.in_shape    = in_shape
        self.num_actions = num_actions
        self.profile     = profile
        
        # device on which to carry out matrix operations
        self.device = torch.device(
            profile['device']
        )
        
        # hyperparamters
        self.gamma         = profile['gamma']
        self.target_period = profile['target_period']
        self.loss_func     = nn.MSELoss()
        
        # online and target Q functions
        self.online_Q = DuellingQfunc(in_shape, num_actions).to(self.device)
        self.target_Q = DuellingQfunc(in_shape, num_actions).to(self.device)
        
        # copy parameters from online to target networks
        self.target_Q.load_state_dict(self.online_Q.state_dict())

        # optimizer to update online Q parameters        
        self.optimizer = torch.optim.RMSprop(
            params = self.online_Q.parameters(),
            lr     = profile['learning_rate']    
        )
        
        # counter for when to copy paramters into target Qfunc networks
        self.target_counter = 0
        
    def forward(self, state: Tensor) -> Tensor:
        return self.online_Q(state)
    
    def train(self) -> None:
        self.online_Q.train()
        
    def eval(self) -> None:
        self.online_Q.eval()
    
    def update(self, batch: Tuple[Tensor, ...]) -> None:
        """samples a batch and updates the network according to the bellman equation"""

        # unpack batch        
        states, actions, rewards, next_states, dones = batch
        
        # push to device
        states      = states.to(self.device)
        actions     = actions.to(self.device)
        rewards     = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones       = dones.to(self.device)
        
        batch_range = torch.arange(len(states))
        
        # calculate current Q values
        current_Qvals = self.online_Q(states)[batch_range, actions]
            
        # cumpute TD-target
        with torch.no_grad():
            # next actions according to online Qfunc
            next_actions = torch.argmax(self.online_Q(next_states), dim=1)
            
            # expected Qvals of actions according to target Qfunc
            expected_Qvals = rewards + self.gamma * self.target_Q(next_states)[batch_range,next_actions]
        
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
            self.__save(self.profile['save_path'])
            
    def __save(self, path: str) -> None:
        
        params = {
            'online_Q' : self.online_Q.state_dict(),
            'target_Q' : self.target_Q.state_dict()
        }
        
        torch.save(params, path)
        
    def load(self, path: str) -> None:
        params = torch.load(path)
        
        self.online_Q.load_state_dict(params['online_Q'])
        self.target_Q.load_state_dict(params['target_Q'])
    

