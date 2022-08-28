# cryptobot

This package includes simplifications of broker data accesses, data transformation, trading API accesses, and machine learning approaches to automate cryptocurrency trading.

## Structure

```
.
├── README.md
│
├── datafetching
│   ├── data_lambdas
│   ├── dataprovider
│   ├── framework
│   └── get_historical_prices
├── datatransforming
│   └── transformers
├── enviroment
├── logging
│   └── bot_logger
├── logs
├── ml
│   ├── cryptomarketmodel
│   ├── cv_search_parallel
│   ├── cv_search
│   ├── dueling_net
│   ├── duelling_dqn_training
│   ├── gym_test
│   ├── neural_net_classifier
│   ├── replay_memory
│   ├── rl_training
│   └── test_targets
├── profile_files
│   ├── bybit_api_credentials.json.sample
│   ├── cv_search.json.sample
│   ├── logger.json.sample
│   ├── nn.json.sample
│   ├── target_testing.json.sample
│   └── trading.json.sample
├── profiles
│   ├── config_manager
│   └── profile_manager
├── trading
│   ├── ftx_client
│   └── model_trading
├── trading_models
│   └── bybit_models
└── utils
    ├── calc_monthly
    ├── decorators
    ├── plotting
    ├── stat_analysis
    └── support_and_resistance
```
