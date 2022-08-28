###############################################################################
############################### IMPORTS #######################################
###############################################################################

#external imports
import functools
import warnings
import sys
import uuid

###############################################################################

# internal imports

###############################################################################

# typing (external | internal)
from typing import Callable

###############################################################################
###############################################################################
###############################################################################


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'    

def deprecated(func: Callable) -> Callable:
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used."""
    @functools.wraps(func)
    def new_func(*args, **kwargs):
        warnings.simplefilter('always', DeprecationWarning)  # turn off filter
        warnings.warn("Call to deprecated function {}.".format(func.__name__),
                      category=DeprecationWarning,
                      stacklevel=2)
        warnings.simplefilter('default', DeprecationWarning)  # reset filter
        return func(*args, **kwargs)
    return new_func

def DEPRECATED(t: type) -> type:
    """Declares a type as deprecated (e. g. a class)"""
    
    func = t.__init__
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        warnings.simplefilter('always', DeprecationWarning)  # turn off filter
        warnings.warn(
            f'Initialization of deprecated type {t}.',
            category   = DeprecationWarning,
            stacklevel = 2
        )
        warnings.simplefilter('default', DeprecationWarning)  # reset filter
        
        return func(*args, **kwargs)
    
    t.__init__ = wrapper
    
    return t

def globalize(func):
    def result(*args, **kwargs):
        return func(*args, **kwargs)
    result.__name__ = result.__qualname__ = uuid.uuid4().hex
    setattr(sys.modules[result.__module__], result.__name__, result)
    return result