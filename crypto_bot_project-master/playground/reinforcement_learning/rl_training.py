from name_me.data_handling import dataprovider as dp
from name_me.data_handling import data_lambdas as dls
from name_me.data_handling import schemes

from name_me.profiles import profile_manager as pm

from name_me.reinforcement_learning import agents
from name_me.reinforcement_learning import envir
from name_me.reinforcement_learning import exploration_processes as exp_p


if __name__ == '__main__':

    profile = pm.get_profile_by_file('./profile_files/rl_training.json')

    data_lambda = dls.DataLambda.get_test_lambda(
        profile=profile
    )
    transformer = schemes.get_transformer(
        uuid=profile["transformer_uuid"]
    )

    provider = dp.DataProvider(
        data_lambda=data_lambda,
        transformer=transformer
    )

    envir = envir.TradingEnvir(
        provider=provider,
        profile=profile
    )

    process = exp_p.EpsilonGreedyProcess(
        action_space=envir.action_space,
        profile=profile
    )

    agent = agents.DuellingAgent(
        env=envir,
        process=process,
        profile=profile
    )

    rewards, cum_rewards = agent.train(500)
