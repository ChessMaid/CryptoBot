###############################################################################
############################### IMPORTS #######################################
###############################################################################

#external imports
import numpy as np
import random

###############################################################################

# internal imports

###############################################################################

# typing (external | internal)
from typing import Tuple

###############################################################################
###############################################################################
###############################################################################
    
class MDP():
    def __init__(self):
        self.s0 = 0
        self.s1 = 1
        self.s2 = 2
        self.s3 = 3 
        self.s4 = 4
        self.s5 = 5
        
        self.current = self.s0
        
        class OSpace():
            def __init__(self, n: int):
                self.n = n
        
        class ASpace():
            def __init__(self, n: int):
                self.n = n
            def sample(self):
                return random.randint(0, self.n-1)
            
        self.state_space       = [self.s0, self.s1, self.s2, self.s3, self.s4, self.s5]
        self.observation_space = OSpace(len(self.state_space))
        self.action_space      = ASpace(2)
        
    def reset(self) -> None:
        self.current = self.s0
        return self.current
    
    def step(self, action: int) -> Tuple[int, int, bool, str]:
        if self.current is self.s0:
            if action == 0:
                self.current, reward = self.s1, 1
            elif action == 1:
                self.current, reward = self.s3, 5
            else:
                raise RuntimeError('Invalid action!')
        elif self.current is self.s1:
            if action == 0:
                self.current, reward = self.s2, 2
            elif action == 1:
                self.current, reward = self.s4, -1
            else:
                raise RuntimeError('Invalid action!')
        elif self.current is self.s2:
            if action == 0:
                self.current, reward = self.s4, 4
            elif action == 1:
                self.current, reward = self.s5, 3
            else:
                raise RuntimeError('Invalid action!')
        elif self.current is self.s3:
            if action == 0:
                self.current, reward = self.s4, -1
            elif action == 1:
                self.current, reward = self.s4, 1
            else:
                raise RuntimeError('Invalid action!')
        elif self.current is self.s4:
            if action == 0:
                self.current, reward = self.s5, 0
            elif action == 1:
                self.current, reward = self.s5, 2
            else:
                raise RuntimeError('Invalid action!')
        
        done = (self.current is self.s5)
        
        return self.current, reward, done, ''
    
    def render(self):
        pass
    
    

class Agent():
    def __init__(self, env: MDP, num_states: int, num_actions: int, profile: dict):
        self.env = env
        
        self.num_states  = num_states
        self.num_actions = num_actions
        
        self.profile = profile
        
        self.eps     = self.profile['epsilon']
        self.eps_dec = self.profile['epsilon_decrement']
        self.eps_min = self.profile['epsilon_minimum']
        
        self.alpha = profile['learning_rate']
        self.gamma = profile['gamma']
        
        self.dream_size   = self.profile['dream_size']
        self.dream_period = self.profile['dream_period']
        
        self.Q_table = np.zeros((num_states, num_actions))
    
    def get_action(self, state: int) -> int:
        if np.random.uniform(0,1) < self.eps:
            self.__decrement_eps()
            action = self.env.action_space.sample()
        
        else:
            action, = np.unravel_index(
                np.argmax(self.Q_table[state,:]), self.Q_table[state,:].shape
            )
            
        self.__decrement_eps()
        return action
    
    def train(self, num_episodes: int) -> None:
        
        total_transitions = 0
        
        for ep in range(num_episodes):
            print(f'\nEPISODE: {ep+1}/{num_episodes}')
            print(f'EPSILON: {self.eps}')
            
            state          = self.env.reset()
            done           = False
            cum_reward     = 0
            episode_length = 0

            
            # states, actions, rewards, next_states, dones = [], [], [], [], []
            
            # while not done:
                
            #     total_transitions += 1
            #     episode_length    += 1
                
                
            #     action = self.get_action(state)
                
            #     # step environment
            #     next_state, reward, done, _ = self.env.step(action)
                
            #     # remember info
            #     states.append(state)
            #     actions.append(action)
            #     rewards.append(reward)
            #     next_states.append(next_state)
            #     dones.append(done)
                
            #     cum_reward += reward
            #     state = next_state
            
            # # update Q table at every step
            # for t in reversed(range(episode_length)):
                
            #     # get current state and action
            #     state, action = states[t], actions[t]
                
            #     # calculate discountet rewards from this state on
            #     G_t = 0
            #     for reward in rewards[t:][::-1]:
            #         G_t = self.gamma * G_t + reward
                    
            #     self.Q_table[state, action] += self.alpha*(G_t - self.Q_table[state, action])
                
            
            
        
            while not done:
                
                total_transitions += 1
                episode_length    += 1
                
                action = self.get_action(state)
                
                next_state, reward, done, _ = self.env.step(action)
                
                if done: reward = 1
                
                # current Q value
                current = self.Q_table[state, action]
                
                # TD target
                td_target = reward + self.gamma * np.max(self.Q_table[next_state, ...])
                
                # TD difference
                td_difference = td_target - current
                
                # update Q table
                self.Q_table[state, action] += self.alpha * td_difference
                
                cum_reward += reward
                state = next_state
                    
            print(f'Episode {ep+1} with cumulative reward {cum_reward}.')
            print(f'{episode_length} transitions in this episode.')
            print(f'Total transitions: {total_transitions}')

    def __decrement_eps(self) -> None:
        self.eps = max(self.eps * (1-self.eps_dec), self.eps_min)
    

if __name__ == "__main__":
    
    from profiles import profile_manager as pm
    
    profile = pm.get_profile_by_file('./profile_files/rl_training.json')
    
    env = MDP()
    
    agent = Agent(env, env.observation_space.n, env.action_space.n, profile)
    
    agent.train(5000)
    
    print(agent.Q_table)
    
    
    
    
    
    
    