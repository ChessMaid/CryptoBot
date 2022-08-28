from cryptobot_api import app
from cryptobot_api.bybit_blueprint import bybit_blueprint as bybit
from cryptobot_api.rl_blueprint import rl_blueprint as rl

import os

from flask.app import Flask


def create_web_service(debug: bool):
    app = Flask(__name__)
    app.debug = debug
    app.register_blueprint(bybit)
    app.register_blueprint(rl)
    return app


def debug_active(default=False):
    if "DEBUG_WERBSERVICE" in os.environ:
        return os.environ["DEBUG_WEBSERVICE"] == "true"
    return default


if __name__ == '__main__':
    app = create_web_service(debug=debug_active())
    app.run(host=os.environ["WEBSERVICE_HOST"],
            port=os.environ["WEBSERVICE_PORT"])
