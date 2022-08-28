###############################################################################
############################### IMPORTS #######################################
###############################################################################

# external imports
import os
import json

###############################################################################

# internal imports

###############################################################################
###############################################################################
###############################################################################


def create_config(filename: str, config):
    """    
    create json config file with specified filename and sample object\n

    usage:\n
    \tconf                = {}
    \tconf.some_property  = 'some_value'
    \tconf.other_property = 'other_value'

    \tcreate_config('my_config.json', conf)\n

    """
    with open(filename, 'w') as file:
        json.dump(config, file)


def read_config(filename):
    """
    read config by filename
    """
    with open(filename, 'r') as file:
        return json.load(file)


def read_config_safe(filename, configsample):
    """
    read a config which gets created if it not exists\n
    create a config sample object if you use this
    """
    if not __fileExists(filename):
        create_config(filename, configsample)
    return read_config(filename)


def __fileExists(filename):
    try:
        with open(filename) as _:
            # might cause error later on due to file not being closed
            return True
    except IOError:
        return False
