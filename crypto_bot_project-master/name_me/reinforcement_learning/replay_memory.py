###############################################################################
############################### IMPORTS #######################################
###############################################################################

#external imports
import torch


import random

# import numpy as np
# import h5py 
# import os

###############################################################################

# internal imports

###############################################################################

# typing (external | internal)
from typing import Tuple
from torch  import Tensor

###############################################################################
###############################################################################
###############################################################################


class ReplayMemory():
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.size     = 0
        self.head     = 0
        self.content  = []
        
    def store(self, experience: Tuple[Tensor, ...]) -> None:
        if self.size < self.max_size:
            self.content.append(experience)
            self.size += 1
        else:
            self.content[self.head] = experience
            self.head = ((self.head + 1) % self.max_size)
            
    def dream(self, n: int) -> Tensor:
        if n > self.size:
            raise RuntimeError(f'Tried sampling {n} entries from memory of size {self.size}!')
        # randomly choose from memory
        samples = random.sample(self.content, n)
        
        states      = [experience[0] for experience in samples]
        actions     = [experience[1] for experience in samples]
        rewards     = [experience[2] for experience in samples]
        next_states = [experience[3] for experience in samples]
        dones       = [experience[4] for experience in samples]
        
        states      = torch.stack(states,      dim=0)
        actions     = torch.stack(actions,     dim=0)
        rewards     = torch.stack(rewards,     dim=0)
        next_states = torch.stack(next_states, dim=0)
        dones       = torch.stack(dones,       dim=0)
        
        states      = states.float()
        actions     = actions.long()
        rewards     = rewards.float()
        next_states = next_states.float()
        dones       = dones.bool()
        
        return (states, actions, rewards, next_states, dones)


# class Memory(object):
#     def __init__(self, maxsize, arr_shape, filename):
        
#         # point to current slot to be written into
#         self.pointer    = 0
        
#         # number of memories in memory
#         self.fill_level = 0
        
#         # number of new rows added to database if full
#         self.chunk_size = 1
        
#         # max size of database
#         self.maxsize = maxsize
        
#         self.arr_shape = arr_shape
        
#         # names of the SARSD columns used as dataset names
#         self.columns = [
#             "state", "action", "reward", "next state", "done"
#         ]
        
#         # shapes of the entries of the SARSD columns
#         self.entry_shapes = [
#             self.arr_shape, (1,), (1,), self.arr_shape, (1,)
#         ]
        
#         self.filename = filename
#         self.file     = h5py.File(self.filename, "a")
        
#         # create datasets for each SARSD column
#         self.datasets = []
#         for name, shape in zip(self.columns, self.entry_shapes):
#             try:
#                 self.datasets.append(
#                     self.file.create_dataset(
#                         name,
#                         (self.chunk_size,) + shape,
#                         maxshape = (self.maxsize,) + shape,
#                         dtype="f",
#                     )
#                 )
#             except:
#                 self.datasets.append(
#                     self.file[name]
#                 )
                
#     def add_experience(self, tup):
#         """adds a tuple consisting of SARSD into memory"""
        
#         # if fill level is not reached check if we need to add a new chunk
#         if self.fill_level != self.maxsize:
            
#             # check if new chunk has to be added to datasets
#             if self.fill_level % self.chunk_size == 0:
#                 for dataset, shape in zip(self.datasets, self.entry_shapes):
#                     if self.fill_level + self.chunk_size < self.maxsize:
#                         dataset.resize(
#                             (self.fill_level + self.chunk_size,) + shape
#                         )
#                     else:
#                         dataset.resize(
#                             (self.maxsize,) + shape
#                         )
        
#         # add data to datasets
#         for entry, dataset in zip(tup, self.datasets):
#             dataset[self.pointer,...] = entry
        
#         # update pointer and make sure to loop
#         self.pointer = (self.pointer + 1) % self.maxsize
        
#         # update fill level
#         if self.fill_level < self.maxsize:
#             self.fill_level += 1
        
#     def retrieve(self, indicies):
#         """retrieve specified memories from memory"""
        
#         states      = self.datasets[0][indicies,...]
#         actions     = self.datasets[1][indicies,...]
#         rewards     = self.datasets[2][indicies,...]
#         next_states = self.datasets[3][indicies,...]
#         dones       = self.datasets[4][indicies,...]
        
#         return states, actions, rewards, next_states, dones
        
#     def dream(self, n):
#         """uniformly sample memories from memory"""
        
#         # get random indicies between 0 and current fill level
#         indicies = np.sort(
#             np.random.choice(
#                 np.arange(0, self.fill_level, 1), size = n, replace = False
#             )
#         )
        
#         return self.retrieve(indicies)
        
#     def close_connection(self):
#         """close connection to database to free up corresponding file"""
#         self.file.close()
        
#     def open_connection(self):
#         """open connection to database"""
#         self.file = h5py.File(self.filename, "a")
        
#     def delete_database(self):
#         """close file correspnding to database and delete it"""
#         self.close_connection()
#         os.remove(self.filename)
        
#     def __len__(self):
#         return self.fill_level


# if __name__ == '__main__':
    
#     capacity = 10000
#     shape = (37, 48, 48)
#     rng = np.random.default_rng()
    
#     memory = Memory(capacity, shape, "replay_memory.hdf5")
    
#     for i in range(capacity):
#         if (i+1) % 10 == 0:
#             print(f"inserted {i+1} rows")
#         memory.add_experience(
#             (rng.random(shape), 0, 1, rng.random(shape), 0)    
#         )
        
#     s, a, r, ns, d = memory.dream(16)
        
#     print(
#         s.shape, a.shape, r.shape, ns.shape, d.shape    
#     )
    
#     def time(n):
#         for _ in range(n):
#             s, a, r, ns, d = memory.dream(16)
    
#     import cProfile
    
#     cProfile.run("time(16)")
    
#     memory.delete_database()

    
    