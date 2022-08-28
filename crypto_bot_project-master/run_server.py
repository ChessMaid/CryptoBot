from cryptobot_api import app
from cryptobot_api.bybit_blueprint import bybit_blueprint as bybit
from cryptobot_api.rl_blueprint import rl_blueprint as rl

from flask.app import Flask

app = Flask(__name__)
app.config['DEBUG'] = False
app.register_blueprint(bybit)
app.register_blueprint(rl)
app.run(host="0.0.0.0")
