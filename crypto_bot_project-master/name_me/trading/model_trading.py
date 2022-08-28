###############################################################################
############################### IMPORTS #######################################
###############################################################################

#external imports
import logging
import time
import traceback
from joblib import load
from datetime import datetime
import numpy as np
import torch

###############################################################################

# internal imports
from data_handling import data_lambdas as dls
from data_handling import dataprovider as dp
from data_handling import transformers
from data_handling import framework as fw

from logging_wrapper import bot_logger

from profiles import profile_manager as pm

from supervised_learning import neural_net_classifier as nnc

from trading_models.bybit_models import *

###############################################################################
###############################################################################
###############################################################################


def get_handle_prediction_order(prediction: float, account: BybitAccount, market: BybitMarket, profile: dict) -> BybitLimitOrder:

    if prediction in [-1, 1]:
        return account.get_go_to_order(market, profile['asset'], profile['trade_lev'], PositionSide(prediction > 0))
    elif prediction == 0:
        return account.get_normalize_X1_order(market, profile['asset'])
    else:
        raise RuntimeError('prediction broken')
        
def prepare_data(transformer: dp.DataTransformer, normalizer: dp.Normalizer, profile: dict):
    
    # build provider
    lam = dls.get_lambda(profile, time.time())
    
    provider = dp.DataProvider(
        data_lambda = lam,
        transformer = transformer
    )
    
    # fetch data
    df = provider.get_data()

    # drop private columns
    df.drop([
            col for col in df.columns if '__' in col
        ], axis='columns', inplace=True
    )
    
    # normalize data
    df = normalizer.mimic(df)

    if profile['approach'] == 'sklearn':
        return df.tail(1)
    elif profile['approach'] == 'nn':
        return torch.tensor(df.astype(np.float32).tail(1).values)
    else:
        raise RuntimeError('approach not supported.')

def get_model(profile: dict):
    if profile['approach'] == 'sklearn':
        return load(profile['model_path'])
    elif profile['approach'] == 'nn':
        return nnc.ModelLoader.load_nnet(profile["model_path"])
    else:
        raise RuntimeError('approach not supported.')

def start_bybit_trading(trading_profile: dict, account: BybitAccount, market: BybitMarket, logs_path: str):
    log_formatter           = logging.Formatter("%(asctime)s — %(message)s", "%Y-%m-%d %H:%M:%S")    
    logger                  = bot_logger.get_logger('TRADING_LOGGER', logs_path, log_formatter)
    model                   = get_model(trading_profile)
    transformer             = transformers.get_transformer(trading_profile['transformer_uuid'])
    normalizer              = nnc.ModelLoader.load_normalizer(trading_profile["model_path"])
    current_time            = time.time()
    last_candle_open_time   = current_time - current_time % trading_profile['resolution']
    last_worth              = account.get_decimal_worth_in_market(trading_profile['asset'], market)
    n_iterations            = 0
    
    logger.info(f'**STARTING TRADING ON MARKET {trading_profile["market"]}**')

    while True:
        try:
            # set worth to initial worth in first iteration (prevent double worth fetching)            
            new_worth   = account.get_decimal_worth_in_market(trading_profile['asset'], market) if n_iterations != 0 else last_worth
            worth_diff  = new_worth - last_worth
            last_worth  = new_worth

            if trading_profile['use_testnet']:
                logger.info('Trading on ByBit **TESTNET**!')
            logger.info(f'CURRENT_WORTH: **{new_worth:.2f}$**')

            if n_iterations != 0:
                logger.info(f'WORTH_DIFF: **{worth_diff:.2f} $** => {":flushed:" if worth_diff < 0 else ":money_mouth:"}')

            n_iterations += 1

            logger.info(f'ITERATION **#{n_iterations}**')
            
            X = prepare_data(transformer, normalizer, trading_profile)
            
            prediction = model.predict(X)
            logger.info(f'PREDICTION: **{"SHORT" if prediction == -1 else "LONG" if prediction == 1 else "NONE"}**')

            order = get_handle_prediction_order(prediction, account, market, trading_profile)
            if order.size != 0:
                logger.info(f'CREATING **{order.side.name}** ORDER | SIZE: **{order.size} $**')        
                order.execute(OrderExecutionType(trading_profile['order_type']))

                if order.is_filled:
                    logger.info(f'ORDER FILLED')
                elif order.is_cancelled:
                    logger.info(f'ORDER CANCELLED')
            else:
                logger.info("POSITION ALREADY SATISFIED")

        except Exception as exp:
            logger.error(f'**ERROR**: {exp}')
            stack = traceback.format_tb(exp.__traceback__)
            print('\n'.join(stack))
            logger.error(f'Traceback has been logged to file')
        finally:
            while True:
                if time.time() >= last_candle_open_time + trading_profile['resolution'] - trading_profile['loop_iteration_offset']:
                    last_candle_open_time += trading_profile['resolution']
                    break
                time.sleep(0.5)


if __name__ == '__main__':
    TRADING_PROFILE_PATH    = '../profiles/trading.json';
    TRADING_CREDS_PATH      = '../profiles/bybit_api_credentials.json'
    LOGS_PATH               = '../logs/rf_trading.log'

    trading_profile = pm.get_profile_by_file(TRADING_PROFILE_PATH)
    is_test         = trading_profile['use_testnet']
    factory         = BybitFactory()
    creds           = factory.get_bybit_credentials_by_config(TRADING_CREDS_PATH, is_test)
    account         = factory.get_bybit_account(creds);
    factory.get_market(trading_profile['market'])

    start_bybit_trading(trading_profile, account, market, LOGS_PATH)