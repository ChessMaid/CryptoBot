###############################################################################
############################### IMPORTS #######################################
###############################################################################

#external imports
# import pandas as pd
# import numpy as np
# import torch
import time

###############################################################################

# internal imports
from data_handling import dataprovider as dp
from data_handling import data_lambdas as dls
from data_handling import schemes

from profiles import profile_manager as pm

from reinforcement_learning import agents
from reinforcement_learning import envir
from reinforcement_learning import exploration_processes as exp_p

# import dueling_net as dn

###############################################################################

# typing (external | internal)

###############################################################################
###############################################################################
###############################################################################

if __name__ == '__main__':
        
    profile = pm.get_profile_by_file('./profile_files/rl_training.json')
    
    # data_lambda = dls.DataLambda.get_lambda(
    #     profile = profile,
    #     end     = time.time(),
    #     check   = True
    # )
    
    data_lambda = dls.DataLambda.get_test_lambda(
        profile = profile
    )
    
    transformer = schemes.get_transformer(
        uuid = profile["transformer_uuid"]
    )
            
    provider = dp.DataProvider(
        data_lambda = data_lambda,
        transformer = transformer
    )
    
    envir = envir.TradingEnvir(
        provider = provider,
        profile  = profile
    )
    
    process = exp_p.EpsilonGreedyProcess(
        action_space = envir.action_space,
        profile      = profile    
    )
    
    agent = agents.DuellingAgent(
        env         = envir,
        process     = process,
        profile     = profile
        
    )
    
    rewards, cum_rewards = agent.train(500)  
    
    # print("testing:")
    # tie, toe, vie, voe = stat.test_model(
    #     agent,
    #     env,
    #     profile['num_episodes'],
    #     profile['max_steps'],
    #     profile
    # )
    
    # # stat.analyze(tie, toe, vie, voe)
    
    
    
    
        