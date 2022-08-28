###############################################################################
############################### IMPORTS #######################################
###############################################################################

#external imports
import numpy as np

import matplotlib.pyplot as plt

###############################################################################

# internal imports

###############################################################################
###############################################################################
###############################################################################

def hist(A, bins=50):
    plt.figure()
    plt.hist(A, bins=bins)

def mean(A):
    return np.mean(A)

def var(A):
    return np.var(A)

def std(A):
    return np.std(A)
    
def skew(A):
    return 1/std(A)**3 * mean((A - mean(A))**3)

def demean(A):
    return A - mean(A)

def standard_score(A):
    return (A - np.mean(A))/np.std(A)

def min_max_scaling(A):
    return (A - np.min(A))/(np.max(A) - np.min(A))

def corr_coeff(A, B):
    return np.mean((A - mean(A)) * (B - mean(B)))/ (var(A) * var(B))**(1/2)

# def test_model(agent, env, num_eps, max_steps, profile):
    
#     assert env.train_phase == True
    
#     # on train set
#     train_ie_rews, train_eoe_rews = ddqnt.run_environment(
#         num_eps,
#         max_steps,
#         env,
#         agent,
#         profile
#     )
    
#     env.toggle_training()
    
#     assert env.train_phase == False
    
#     valid_ie_rews, valid_eoe_rews = ddqnt.run_environment(
#         num_eps,
#         max_steps,
#         env,
#         agent,
#         profile
#     )
    
#     env.toggle_training()
    
#     assert env.train_phase == True
    
#     return train_ie_rews, train_eoe_rews, valid_ie_rews, valid_eoe_rews

# def analyze(tie, toe, vie, voe):
    
#     # list<list<float>> -> np.ndarray<np.float64>
#     tie = np.concatenate(tie)
#     vie = np.concatenate(vie)
    
#     mean_tie = mean(tie)
#     mean_toe = mean(toe)
#     mean_vie = mean(vie)
#     mean_voe = mean(voe)
    
#     var_tie = var(tie)
#     var_toe = var(toe)
#     var_vie = var(vie)
#     var_voe = var(voe)
    
#     skew_tie = skew(tie)
#     skew_toe = skew(toe)
#     skew_vie = skew(vie)
#     skew_voe = skew(voe)
    
#     print(f"Training   : EOE mean : {mean_toe:.2f}")
#     print(f"             EOE var  : {var_toe:.2f}")
#     print(f"             EOE skew : {skew_toe:.2f}")
#     print(f"             IE mean  : {mean_tie:.4f}")
#     print(f"             IE var   : {var_tie:.4f}")
#     print(f"             IE skew  : {skew_tie:.4f}")
    
#     plt.figure()
#     plt.hist(tie, label="TIE", bins=128)
#     plt.legend()
#     plt.show()
    
#     plt.figure()
#     plt.hist(toe, label="TOE", bins=64)
#     plt.legend()
#     plt.show()
    
    
#     print(f"Validation : EOE mean : {mean_voe:.2f}")
#     print(f"             EOE var  : {var_voe:.2f}")
#     print(f"             EOE skew : {skew_voe:.2f}")
#     print(f"             IE mean  : {mean_vie:.4f}")
#     print(f"             IE var   : {var_vie:.4f}")
#     print(f"             IE skew  : {skew_vie:.4f}")
    
    
#     plt.figure()
#     plt.hist(vie, label="VIE", bins=128)
#     plt.legend()
#     plt.show()
    
#     plt.figure()
#     plt.hist(voe, label="VOE", bins=64)
#     plt.legend()
#     plt.show()
    
#     corr_eoe = corr_coeff(toe, voe)
#     corr_ie  = corr_coeff(tie, vie)
    
#     print(f"correlation coefficient: EOE: {corr_eoe:.4f}, IE: {corr_ie:.4f}")
    