###############################################################################
############################### IMPORTS #######################################
###############################################################################

#external imports
import torch
import torch.nn as nn
import torchvision

import numpy as np
import random

###############################################################################

# internal imports
from ..reinforcement_learning import replay_memory as rm

###############################################################################

# typing (external | internal)
from typing import Tuple
from torch  import Size, Tensor

from ..reinforcement_learning.envir import Envir

###############################################################################
###############################################################################
###############################################################################

class EnvWrapper(Envir):
    def __init__(self, env):
        self.env          = env
        self.action_space = self.env.action_space
        
        self.phi = torchvision.transforms.Compose([
            lambda T : torch.tensor(T),
            lambda T : T.permute(2,0,1),
            torchvision.transforms.Grayscale(),
            torchvision.transforms.Resize(size=(110,84)),
            lambda T : T.type(torch.float)
        ])
        
    def reset(self) -> Tensor:
        self.last_4_frames = [self.phi(self.env.reset())] * 4
        return torch.cat(self.last_4_frames).unsqueeze(0)
    
    def __cycle_in_new_frame(self, new_frame: Tensor) -> None:
        self.last_4_frames = self.last_4_frames[1:] + [new_frame]
    
    def step(self, action: int) -> Tuple[Tensor, float, bool, str]:
        next_state, reward, done, info = self.env.step(action)
        
        self.__cycle_in_new_frame(self.phi(next_state))
        
        return torch.cat(self.last_4_frames).unsqueeze(0), reward, done, info
    
    def render(self):
        return self.env.render()
    
    
class DuellingQfunc(nn.Module):
    def __init__(self, in_shape: tuple, num_actions: int):
        self.in_shape    = in_shape
        self.num_actions = num_actions
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels  = 4,
                out_channels = 16, 
                kernel_size  = (8,8),
                stride       = (4,4)
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels  = 16,
                out_channels = 32,
                kernel_size  = (4,4),
                stride       = (2,2)
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
            expected_Qvals = rewards + self.gamma * self.target_Q(next_states)[batch_range,next_actions] * (~dones)
        
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
    

class DuellingAgent:
    def __init__(self, env: Envir, in_shape: tuple, num_actions: int, profile: dict):
        self.env = env
        
        self.profile = profile
        
        self.eps     = self.profile['epsilon']
        self.eps_dec = self.profile['epsilon_decrement']
        self.eps_min = self.profile['epsilon_minimum']
        
        self.dream_size   = self.profile['dream_size']
        self.dream_period = self.profile['dream_period']
        
        self.memory = rm.ReplayMemory(
            self.profile['memory_size'],
        )
        
        self.Q = DoubleQfunc(in_shape, num_actions, self.profile)

    def get_action(self, state: Tensor) -> int:
        if np.random.uniform(0,1) < self.eps:
            self.__decrement_eps()
            action = self.env.action_space.sample()
        
        else:
            self.Q.eval()
            with torch.no_grad():
                qvals = self.Q(state)
            self.Q.train()
            
            action = torch.argmax(qvals, dim=1).item()
            
        self.__decrement_eps()
        return action
    
    def train(self, num_episodes: int):
        
        total_transitions = 0
        
        for ep in range(num_episodes):
            print(f'\nEPISODE: {ep+1}/{num_episodes}')
            print(f'EPSILON: {self.eps}')
            
            state          = self.env.reset()
            done           = False
            cum_reward     = 0
            episode_length = 0
        
            while not done:
                
                if ep % 5 == 0:
                    env.render()
                
                total_transitions += 1
                episode_length    += 1
                
                action = self.get_action(state)
                
                next_state, reward, done, _ = self.env.step(action)
                
                cum_reward += reward
                
                self.memory.store(
                    (
                        state,
                        torch.Tensor([action]).long(),
                        torch.Tensor([reward]).float(),
                        next_state,
                        torch.Tensor([done]).bool()
                    )
                )
                
                state = next_state
        
                if total_transitions % self.dream_period == 0:
                    if self.memory.size >= self.dream_size:
                        dream = self.memory.dream(self.dream_size)
                        
                        self.Q.update(dream)   
                
                # if total_steps % target_update == 0:
                #     agent.soft_update_target(tau)
                    
            print(f'Episode {ep+1} with cumulative reward {cum_reward}.')
            print(f'{episode_length} transitions in this episode.')
            print(f'Total transitions: {total_transitions}')
            
        
    # def soft_update_target(self, tau):
    #     for t_param, m_param in zip(self.target.parameters(), self.model.parameters()):
    #         t_param.data.copy_(tau*m_param.data + (1-tau)*t_param.data)

    def __decrement_eps(self) -> None:
        self.eps = max(self.eps * (1-self.eps_dec), self.eps_min)
        
if __name__ == "__main__":
    
    from profiles import profile_manager as pm
    
    import gym
    
    profile = pm.get_profile_by_file('./profile_files/gym.json')
        
    env = EnvWrapper(gym.make('SpaceInvaders-v0'))
    
    agent = DuellingAgent(env, (4,110,84), env.action_space.n, profile)
    
    agent.Q.load('C:\\Users\\Cedric\\Dropbox\\Crypto\\Code\\crypto_bot_project\\models\\TEMP2.pth')
    
    agent.train(10000)