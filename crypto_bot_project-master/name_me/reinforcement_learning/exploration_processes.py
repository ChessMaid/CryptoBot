###############################################################################
############################### IMPORTS #######################################
###############################################################################

#external imports

###############################################################################

# internal imports
import numpy as np
import torch

###############################################################################

# typing (external | internal)
from torch import Tensor

from ..reinforcement_learning.envir import DiscreteActionSpace

###############################################################################
###############################################################################
###############################################################################

    # PARENT

class ExplorationProcess(object):
    def __init__(self, action_space: DiscreteActionSpace, profile: dict):
        self.action_space = action_space
        self.profile      = profile
        
    def get_action(self, Qvals: Tensor, current_transition: int) -> int:
        pass
    
###############################################################################
        
    # EXPLICIT PROCESSES
    
class EpsilonGreedyProcess(ExplorationProcess):
    def __init__(self, action_space: DiscreteActionSpace, profile: dict):
        super().__init__(action_space, profile)
        
        self.epsilon              = self.profile['epsilon']
        self.epsilon_decay_factor = self.profile['epsilon_decay_factor']
        self.minimum_epsilon      = self.profile['minimum_epsilon']
        
    def get_action(self, Qvals: Tensor, current_transition: int) -> int:
        # get epsilon for current position
        self.current_epsilon = max(
            self.epsilon * self.epsilon_decay_factor ** current_transition,
            self.minimum_epsilon
        )
        
        print(f'CURRENT EPSILON: {self.current_epsilon:.3f}.\n')
        
        # act greedy 1-epsilon % of the time
        if np.random.uniform(0, 1) < self.current_epsilon:
            return self.action_space.sample()
        else:
            action = torch.argmax(Qvals, dim=1).item()
            
        return action
            
    
def SoftmaxExploration(ExplorationProcess):
    def __init__(self, action_space: DiscreteActionSpace, profile: dict):
        super().__init__(action_space, profile)
        
    def get_action(self, Qvals: Tensor, current_transition: int) -> int:
        
        
        tau = 1
    
        tempered_Qvals = Qvals / tau
        
        P = torch.softmax(tempered_Qvals)
        
###############################################################################
###############################################################################
###############################################################################

    # tests  
          
if __name__ == "__main__":
    
    from profiles import profile_manager as pm
    
    profile = pm.get_profile_by_file(
        path = './profile_files/exploration_processes/epsilon_greedy.json'
    )
    
    process = EpsilonGreedyProcess(DiscreteActionSpace(3), profile)
    
###############################################################################
###############################################################################
###############################################################################
    
    
    
    