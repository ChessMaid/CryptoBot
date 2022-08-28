import dataclasses
import enum
from os import symlink
import time
from bravado.client import SwaggerClient
import bybit
from json import dumps, loads
from enum import Enum

from numpy import place, select, string_
from profiles.config_manager import read_config_safe
from dataclasses import dataclass, InitVar, field, replace, asdict
from collections import namedtuple

import logging
# from .. import bot_logger

FORMATTER        = logging.Formatter("%(asctime)s — %(message)s", "%Y-%m-%d %H:%M:%S")    
# LOGGER           = bot_logger.get_logger('TRADING_LOGGER', './logs/rf_trading.log', FORMATTER)

class PriceSide(Enum):
    """ enum for trading-price-side definitions """
    ASK = True
    BID = False

    def __repr__(self) -> str:
        return self.name

class PositionSide(Enum):
    """ enum for position-side definitions """
    LONG    = True
    SHORT   = False

    def __repr__(self) -> str:
        return self.name

class BybitReturnCode(Enum):
    """ enum for bybit-api-returncode definitions """
    SUCCESS          = 0
    REQUEST_FAILED   = 10001
    IP_MISMATCH      = 10010
    ORDER_NOT_EXISTS = 20001
    UNKOWN_ERROR     = 30076

class OrderType(Enum):
    LIMIT   = "Limit"
    MARKET  = "Market"

class OrderExecutionType(Enum):
    MARKET      = "Market"
    LIMIT       = "Limit"
    FORCE_LIMIT = "Force"

@dataclass
class BybitCredentials:
    """ holds bybit api credentials """
    api_key:    str
    secret_key: str
    is_test:    bool

@dataclass
class BybitMarket:
    """ holds information about a specific market on bybit """
    symbol: str

    def __post_init__(self):
        self.__client = bybit.bybit(test=False)

    def get_last_order_book_price(self, side: PriceSide, offset=0.5):
        result = self.__client.Market.Market_orderbook(
            symbol="BTCUSD"
        ).result()[0]
        #LOGGER.debug('fetching last orderbook price: \n' + result.__repr__())
        ask = side == PriceSide.ASK
        return float(result['result'][25 if ask else 0]['price'])

    def get_last_traded_price(self, side: PriceSide, offset=0.5):
        result = self.__client.Market.Market_symbolInfo(
            symbol = self.symbol
        ).result()[0]
        #LOGGER.debug('fetching last traded price: \n' + result.__repr__())
        response = result['result'][0] # fuck you bybit api
        ask = side == PriceSide.ASK
        return float(response["ask_price" if ask else "bid_price"])

    def get_market_price(self):
        result = self.__client.Market.Market_symbolInfo(symbol=self.symbol).result()[0]
        #LOGGER.debug('fetching last market price: \n' + result.__repr__())
        return float(result['result'][0]['mark_price'])

    def get_orderbook(self):
        result = self.__client.Market.Market_orderbook(
            symbol="BTCUSD"
        ).result()[0]
        #LOGGER.debug('fetching orderbook: \n' + result.__repr__())
        order_book = namedtuple('OrderBook', ['buy', 'sell'])
        order_book.buy  = result['result'][:25]
        order_book.sell = result['result'][25:]
        return order_book

@dataclass
class BybitAccount:
    creds:      BybitCredentials
    __client:   SwaggerClient

    def __post_init__(self):
        self.__client =  bybit.bybit(
            test        = creds.is_test, 
            api_key     = creds.api_key, 
            api_secret  = creds.secret_key
        )

    def get_worth(self, asset: str) -> float:
        result = self.__client.Wallet.Wallet_getBalance(coin=asset).result()[0]
        return float(result['result'][asset]['equity'])

    def get_worth_in_market(self, asset: str, market: BybitMarket) -> int:
        worth           = self.get_worth(asset)
        latest_price    = market.get_market_price()
        return int(worth * latest_price)

    def get_decimal_worth_in_market(self, asset: str, market: BybitMarket) -> float:
        worth           = self.get_worth(asset)
        latest_price    = market.get_market_price()
        return float(worth * latest_price)

    def get_worth_in_market_considering_lev(self, asset: str, lev: float, market: BybitMarket):        
        return int(lev * self.get_decimal_worth_in_market(asset, market)) # rounds down    

    def get_positions(self, market: BybitMarket) -> int:
        result          = self.__client.Positions.Positions_myPosition(symbol=market.symbol).result()[0]
        size            = int(result['result']['size'])
        return size if result['result']['side'] == 'Buy' else -size

    def get_go_to_order(self, market: BybitMarket, asset: str, lev: float, side: PositionSide):
        real_worth      = self.get_worth_in_market(asset, market)
        target          = real_worth * lev * (1 if side == PositionSide.LONG else -1)
        curr            = self.get_positions(market)
        diff            = int(target - real_worth - curr)
        order           = BybitLimitOrder(PositionSide(diff > 0), market, self, asset, abs(diff))
        return order

    def get_normalize_X1_order(self, market: BybitMarket, asset: str): 
        return self.get_go_to_order(market, asset, 0, PositionSide.SHORT)

@dataclass
class BybitLimitOrder:
    """ 
    models a limit-order on bybit 
    """
    side:           PositionSide
    market:         BybitMarket     = field(repr=False)
    account:        BybitAccount    = field(repr=False)
    asset:          str             = field(repr=False)
    size:           int
    use_orderbook:  bool            = field(repr=False, default=False)
    __status:       str             = field(repr=False, init=False, default='')
    __id:           str             = field(repr=False, init=False, default='')
    __unfilled:     float           = field(repr=False, init=False)

    def __post_init__(self):
        self.__unfilled = self.size

    @property
    def status(self):
        self.refresh()
        return self.__status

    @property
    def unfilled_size(self) -> float:
        return self.__unfilled

    @property
    def is_filled(self) -> bool:
        return self.unfilled_size == 0 or self.status == "Filled"

    @property
    def is_cancelled(self) -> bool:
        return self.__status == "Cancelled"

    @property
    def id(self) -> str:
        return self.__id

    def execute(self, execution_type: OrderExecutionType, **execution_params):
        if execution_type == OrderExecutionType.MARKET:
            self.place(None, OrderType.MARKET)
        elif execution_type == OrderExecutionType.FORCE_LIMIT:
            self.__force_limit_order(execution_params['start_time'], execution_params)
        else:
            raise RuntimeError('Not implemented.')

    def place(self, price: float, order_type: OrderType) -> None:
        result = self.account.client.Order.Order_new(
                    side          = "Buy" if self.side == PositionSide.LONG else "Sell",
                    symbol        = self.market.symbol,
                    order_type    = order_type.value,
                    qty           = self.unfilled_size,
                    price         = price,
                    time_in_force = "PostOnly"
                ).result()[0]

        #LOGGER.debug('placing order: \n' + result.__repr__())

        return_code = BybitReturnCode(int(result["ret_code"]))
        if return_code != BybitReturnCode.SUCCESS:
            raise RuntimeError(return_code.name)
                    
        self.__id = result["result"]["order_id"]

    def replace(self, price: float) -> None:
        response = self.account.client.Order.Order_replace(
                        symbol    = self.market.symbol,
                        order_id  = self.id,
                        p_r_price = str(price)
                    ).result()[0]

        return_code = BybitReturnCode(int(response["ret_code"]))     

        if return_code != BybitReturnCode.ORDER_NOT_EXISTS:
            if return_code not in [BybitReturnCode.SUCCESS, BybitReturnCode.UNKOWN_ERROR]:
                raise RuntimeError("Order request not successful")
            self.__id = response["result"]["order_id"]
        else:
            self.__unfilled = 0

    def __force_limit_order(self, start_time, **execution_params) -> None:

        default_execution_params = {
            "price_offset"        : 0.5,
            "retry_replace_time"  : 3,
            "max_seconds"         : 290
        }
        execution_params = { **default_execution_params, **execution_params }
        
        price_offset        = execution_params['price_offset']
        last_order_price    = 0
        current_price       = 0
        last_price_method   = None

        if self.use_orderbook:
            last_price_method = lambda: self.market.get_last_order_book_price(PriceSide(self.side.value))
        else:
            last_price_method = lambda: self.market.get_last_traded_price(PriceSide(self.side.value))

        while not self.is_filled and not self.is_cancelled:
            if self.is_cancelled or self.status == "":
                last_order_price = last_price_method() + (-price_offset if self.side.value else price_offset)
                self.place(last_order_price, OrderType.LIMIT)
            elif time.time() >= start_time + execution_params['max_seconds']:
                self.cancel()
            else:
                last_order_price = current_price
                current_price = last_price_method() + (-price_offset if self.side.value else price_offset)
                if current_price != last_order_price:
                    self.replace(current_price)
            
            time.sleep(execution_params['retry_replace_time'])
            
            self.refresh()

    def cancel(self):
        if self.id is not None and self.id != "":
            res = self.account.client.Order.Order_cancel(symbol=self.market.symbol, order_id=self.id).result()
            if BybitReturnCode(res[0]['ret_code']) == BybitReturnCode.SUCCESS:
                self.__status = "Cancelled"

    def refresh(self) -> None:
        if self.id is not None and self.id != "":
            response = self.account.client.Order.Order_query(
                symbol=self.market.symbol, order_id=self.id
            ).result()[0]
            if response is not None:
                refresh_request = response["result"]
                if refresh_request is not None:
                    self.__status = refresh_request["order_status"]
                    if not self.is_cancelled :
                        self.__unfilled = refresh_request["leaves_qty"]
                else:
                    print('result was None', response)
            else:
                print('response was None')                

@dataclass
class BybitFactory:
    default_account: BybitAccount = field(init=False, default=None)

    def get_bybit_account(self, creds: BybitCredentials) -> BybitAccount:
        """
        creates a bybit client for api interaction
        """        
        return BybitAccount(creds)

    def get_bybit_credentials_by_config(self, path: str, test: bool) -> BybitCredentials:
        """
        creates bybit api credentials using a json config 
        with a format like this:
            { 
                'api_key'         : '',
                'secret_key'      : '',
                'test_api_key'    : '',
                'test_secret_key' : ''
            }
        """
        config = read_config_safe(path, configsample={
            'api_key'         : '',
            'secret_key'      : '',
            'test_api_key'    : '',
            'test_secret_key' : ''
        })
        
        if test:
            return self.get_bybit_credentials(config['test_api_key'], config['test_secret_key'], True)
        else:
            return self.get_bybit_credentials(config['api_key'], config['secret_key'], False)
        

    def get_bybit_credentials(self, api_key: str, api_secret: str, is_test: bool) -> BybitCredentials:
        """
        creates bybit api credentials using an api_key and an api_secret
        """
        return BybitCredentials(api_key, api_secret, is_test)

    def get_market(self, symbol: str) -> BybitMarket:
        """
        creates a market-object
        if no client is specified, the default_client is used
        """
        return BybitMarket(symbol)

    def get_limit_order(self, pos_side: PositionSide, size: float, market_symbol: str, asset: str, use_orderbook=False, account=None) -> BybitLimitOrder:
        """
        creates a limit-order-object
        if no client is specified, the default_client is used
        """
        if account is not None:
            market = self.get_market(market_symbol)
            return BybitLimitOrder(pos_side, market, account, asset, size, use_orderbook)
        elif self.default_account is not None:
            market = self.get_market(market_symbol)
            return BybitLimitOrder(pos_side, market, self.default_account, asset, size, use_orderbook)
        else:
            raise RuntimeError('no client specified.')


if __name__ == '__main__':
    fac                     = BybitFactory()
    creds                   = fac.get_bybit_credentials_by_config('./configs/bybit_api_credentials.json')
    account                 = fac.get_bybit_account(creds)
    fac.default_account     = account
    market                  = fac.get_market('BTCUSD')
    order                   = account.get_go_to_order(market, 'BTC', 0, PositionSide.LONG)
    print(order)

    order.execute(OrderExecutionType.MARKET)

    # while True:
    #     print('ASK', market.get_last_traded_price(PriceSide.ASK))
    #     print('BID', market.get_last_traded_price(PriceSide.BID))
    #     time.sleep(4)
    # order                   = fac.get_limit_order(PositionSide.SHORT, 10, market.symbol, 'BTC')
    # order.force(price_offset=0.5, retry_replace_time=3, max_seconds=10)
