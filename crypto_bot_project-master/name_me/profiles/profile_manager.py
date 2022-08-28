###############################################################################
############################### IMPORTS #######################################
###############################################################################

# external imports

###############################################################################

# internal imports
from . import config_manager as conf

###############################################################################
###############################################################################
###############################################################################


def get_profile_by_file(path: str) -> dict:

    profile_file = conf.read_config(path)

    profile = next(
        (e for e in profile_file['profiles'] if e['id']
         == profile_file['selected_profile']),
        None
    )

    if profile == None:
        raise RuntimeError(
            f"Profile with id '{profile_file['selected_profile']}' not found!"
        )

    return profile
