from flask import Blueprint, request

from name_me.data_handling import dataprovider as dp
from name_me.data_handling import data_lambdas as dls
from name_me.data_handling import schemes

from name_me.reinforcement_learning import agents
from name_me.reinforcement_learning import envir as rl_env
from name_me.reinforcement_learning import exploration_processes as exp_p

from threading import Thread
import psutil


BLUEPRINT_NAME = 'rl'
rl_blueprint = Blueprint(
    BLUEPRINT_NAME, __name__, url_prefix=f'/{BLUEPRINT_NAME}')

training_thread_running = False


@rl_blueprint.route('/', methods=['GET'])
def start_trainging():
    global training_thread_running

    print("ttr = True" if training_thread_running else "ttr = False")

    if training_thread_running:
        return "Training already running"

    request_data = request.get_json()
    print(type(request_data))
    profile = request_data['profiles'][0]

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

    envir = rl_env.TradingEnvir(
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

    def exec_training(episodes):
        global training_thread_running
        try:
            agent.train(episodes)
        except Exception as exp:
            print(exp)
        finally:
            training_thread_running = False

    training_thread = Thread(target=exec_training, kwargs={
        'episodes': int(request.args.get('episodes', default=10))
    })

    training_thread.start()

    training_thread_running = True

    return "Training started"
