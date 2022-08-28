###############################################################################
############################### IMPORTS #######################################
###############################################################################

# external imports
import time
import torch


###############################################################################

# internal imports
from ..reinforcement_learning import replay_memory as rm
from ..reinforcement_learning import q_funcs as q

###############################################################################

# typing (external | internal)
from torch import Tensor
from typing import List

from ..reinforcement_learning.envir import Envir
from ..reinforcement_learning.exploration_processes import ExplorationProcess
from ..logging_wrapper.bot_logger import *

###############################################################################
###############################################################################
###############################################################################


class DuellingAgent(object):
    def __init__(self, env: Envir, process: ExplorationProcess, profile: dict):
        self.env = env
        self.process = process

        self.profile = profile

        self.dream_size = self.profile['dream_size']
        self.dream_period = self.profile['dream_period']

        self.memory = rm.ReplayMemory(
            self.profile['memory_size'],
        )

        self.Q = q.DoubleQfunc(
            in_shape=self.env.observation_space.shape,
            num_actions=self.env.action_space.n,
            profile=self.profile
        )

        self.total_transitions = 0

    def get_action(self, observation: Tensor) -> int:

        # calculate Q values
        self.Q.eval()
        with torch.no_grad():
            # unsqueeze: batch size = 1
            Qvals = self.Q(observation.unsqueeze(0))
        self.Q.train()

        # let process decide on action, based on Q values and current transition
        action = self.process.get_action(
            Qvals=Qvals,
            current_transition=self.total_transitions
        )

        return action

    def train(self, num_episodes: int) -> List[float]:

        rewards = []
        cum_rewards = []

        for ep in range(num_episodes):
            print(f'\nEPISODE: {ep+1}/{num_episodes}')

            state = self.env.reset()
            done = False
            cum_reward = 0
            episode_length = 0

            while not done:
                # get action according to exploration process
                action = self.get_action(state)

                # apply action to environment
                next_state, reward, done, _ = self.env.step(action)

                # keep track of statistics
                self.total_transitions += 1
                episode_length += 1

                rewards.append(reward)

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

                if self.total_transitions % self.dream_period == 0:
                    if self.memory.size >= self.dream_size:
                        dream = self.memory.dream(self.dream_size)

                        self.Q.update(dream)

                # if total_steps % target_update == 0:
                #     agent.soft_update_target(tau)

            cum_rewards.append(cum_rewards)

            print(f'Episode {ep+1} with cumulative reward {cum_reward}.')
            print(f'{episode_length} transitions in this episode.')
            print(f'Total transitions: {self.total_transitions}')

            time.sleep(3)

        return rewards, cum_rewards

    # def soft_update_target(self, tau):
    #     for t_param, m_param in zip(self.target.parameters(), self.model.parameters()):
    #         t_param.data.copy_(tau*m_param.data + (1-tau)*t_param.data)

###############################################################################
###############################################################################
###############################################################################
