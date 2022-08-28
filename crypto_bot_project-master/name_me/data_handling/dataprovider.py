###############################################################################
############################### IMPORTS #######################################
###############################################################################

# external imports

###############################################################################

# internal imports

###############################################################################

# typing (external | internal)
from pandas import DataFrame

from ..data_handling.data_lambdas import DataLambda
from ..data_handling.transformers import DataTransformer

###############################################################################
############################### PROVIDER ######################################
###############################################################################

class DataProvider(object):
    """wraps a data lambda and a transformer into one functionality"""
    def __init__(self, data_lambda: DataLambda, transformer: DataTransformer):
        self.data_lambda  = data_lambda
        self.transformer  = transformer
        self.data_fetched = False

    def get_data(self) -> DataFrame:
        """return transformed, fetched data"""
        if not self.data_fetched:
            self.orig_data = self.data_lambda()
            self.final     = self.transformer.transform(self.orig_data)
            
            self.data_fetched ^= True
            
        return self.final.copy()

###############################################################################
###############################################################################
###############################################################################