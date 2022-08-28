###############################################################################
############################### IMPORTS #######################################
###############################################################################

# external imports
from dataclasses import dataclass
import logging
import sys

from types import SimpleNamespace
from logging.handlers import RotatingFileHandler
from discord_webhook import DiscordWebhook

###############################################################################

###############################################################################
###############################################################################
###############################################################################

DEFAULT_LOG_FORMAT = "%(asctime)s — %(levelname)s — %(message)s"
DEFAULT_FORMATTER = logging.Formatter(DEFAULT_LOG_FORMAT)


class DiscordWebhookHandler(logging.StreamHandler):
    def __init__(self, webhook_url):
        logging.StreamHandler.__init__(self)
        self.webhook_url = webhook_url

    def emit(self, record) -> None:
        if record.levelname != 'DEBUG':
            message = self.format(record)
            if bool(message) and bool(message.strip()) and record is not None:
                DiscordWebhook(self.webhook_url, content=message).execute()


def get_console_handler(formatter=None):
    console_handler = logging.StreamHandler(sys.stdout)

    if formatter is None:
        console_handler.setFormatter(DEFAULT_FORMATTER)
    else:
        console_handler.setFormatter(formatter)

    return console_handler


def get_file_handler(file_name, profile: dict, formatter=None):
    file_handler = RotatingFileHandler(
        file_name,
        maxBytes=profile['maxbytes'],
        backupCount=profile['backup count'],
        encoding=profile['encoding']
    )

    if formatter is None:
        file_handler.setFormatter(DEFAULT_FORMATTER)
    else:
        file_handler.setFormatter(formatter)

    return file_handler


def get_discord_handler(webhook_url, formatter=None):
    discord_handler = DiscordWebhookHandler(webhook_url)

    if formatter is None:
        discord_handler.setFormatter(DEFAULT_FORMATTER)
    else:
        discord_handler.setFormatter(formatter)

    return discord_handler


def get_logger(logger_name, file_name, profile: dict, formatter=None) -> logging.Logger:
    logger = logging.getLogger(logger_name)
    # better to have too much log than not enough
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        if profile['USE_CONSOLE']:
            logger.addHandler(get_console_handler(formatter))
        if profile['USE_FILE']:
            logger.addHandler(get_file_handler(file_name, formatter))
        if profile['USE_DISCORD']:
            logger.addHandler(get_discord_handler(
                profile['DISCORD_WEBHOOK_URL'], formatter))

    # with this pattern, it's rarely necessary to propagate the error up to parent
    logger.propagate = False

    return logger


if __name__ == '__main__':
    logger = get_logger('test', 'test.log')
    print(logger.handlers)
    logger.info('TEST')
