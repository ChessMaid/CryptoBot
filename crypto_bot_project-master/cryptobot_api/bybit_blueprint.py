from flask import Blueprint, request
# from name_me.trading.model_trading import start_bybit_trading

BLUEPRINT_NAME = 'bybit'
bybit_blueprint = Blueprint(
    BLUEPRINT_NAME, __name__, url_prefix=f'/{BLUEPRINT_NAME}')


@bybit_blueprint.route('/', methods=['GET'])
def start_trading():
    # TODO: Implement trading call

    # request_data = request.get_json()
    # print(type(request_data))
    # start_bybit_trading(request_data['profiles'][0])
    pass
