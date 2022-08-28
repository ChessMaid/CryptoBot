# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 22:08:59 2020

@author: Cedric
"""

import numpy as np
import matplotlib.pyplot as plt


#modules that might help:
    #pandas,cython,Numba
#methods that might help:
    #generator

#approximate cdf
def approx_cdf(X,T,plot=False):
    '''
    Take sample of random variable and return values of approxiamted cdf at
    times specified. If "plot" is set to "True" the function will plot the
    approximated cdf.

    Parameters
    ----------
    X : list.
    T : list.
    plot : bool.

    Returns
    -------
    cdf_values : numpy.matrix.

    '''
    #transpose time vector
    T = np.matrix(T)
    
    #broadcast X and T, compare elementwise and sum over each row,
    #take relation to sample number
    cdf_values = 1/len(X) * np.sum(np.less(X,T.T),-1) #PROBLEMATISCH
    
    #plot
    if plot == True:
        #plot cdf
        plt.figure()
        plt.suptitle("cdf")
        plt.plot(T.T,cdf_values)

    return cdf_values.T



#generate a specified number of instances modelling the random variable given
#through samples
def inversion(X,instances,fineness,plot=False):
    '''
    Take samples of a random variable and return samples of a new random
    variable qX modelling the original using the inversion method. "instances"
    specifies the number of returned samples, "fineness" regulates the fineness
    of time steps at which the approximated cdf of the original random variable
    is evaluated. If "plot" is set to "True" the function will plot the
    approximated cdf, aswell as generate a histogram to visualize the old and
    new samples of the random variable respectively.

    Parameters
    ----------
    X : list.
    instances : int.
    fineness : int.
    plot : bool.

    Returns
    -------
    qX : numpy.matrix.
    
    '''
    
    #linearly space between min and max values of X according to fineness
    T = np.linspace(np.min(X)-1,np.max(X)+1,10**fineness)
    
    #get approximated cdf
    F = approx_cdf(X,T,plot)
    
    #simulating uniformly distributed random variable
    U = np.random.uniform(0,1,instances)
    U = np.matrix(U)


    #stack T for each samples of uniform random variable
    Tstack = np.tile(T,(U.shape[1],1)) #PROBLEMATISCH
    
    #get mask where F(t)>=u
    mask = np.less(F,U.T) #PROBLEMATISCH
    
    #set all entries where F(t)<u to +inf    
    Tstack[mask] = np.inf
    
    #get minimum of each row (aka infimum)
    qX = np.min(Tstack,-1)
    
    if plot == True:
        #plot orignal data
        plt.figure()
        plt.title("original samples")
        plt.hist(X)
        
        #plot generated data
        plt.figure()
        plt.title("q^F(U)")
        plt.hist(qX)
    
    return qX

#simulate curve with the same time step as data samples given
def simulate(X,time,time_interval,fineness=4,plot=False):
    '''
    Takes samples of a random variable and simulates a random walk. This random
    walk is based on samples of the random vairable you get from applying the 
    inversion method to the original data. The function allows to adjust the 
    overall length and time intervall between each step of the walk. It is 
    advised to use the time interval in which the orignal data was recorded. 
    Returns list with distances of random walk after each time step. 
    
    Parameters
    ----------
    X : list.
    time : int.
    time_interval : int.
    fineness : int, optional
        adjust number of evaluation points of approximated cdf. The default is 
        4.
    plot : bool, optional
        plot cdf, histograms of old ans new data, plot random walk versus time 
        elapsed. The default is False.

    Returns
    -------
    distance : np.ndarray

    '''
    
    #overall number of steps
    instances = time // time_interval
    
    #get "instances" amount of simulated data points
    qX = inversion(X,instances,fineness,plot)
    
    #generate heigth of datapoint
    distance = np.cumsum(qX)
    
    if plot == True:
        #updated time (round to multiple of time intervall):
        time = instances * time_interval
        
        #plot:
        plt.figure()
        plt.title("relative price")
        plt.plot(np.linspace(0,time,instances),distance)
        
    return distance
    

# from scipy.interpolate import interp1d
# import numpy as np
# import matplotlib.pyplot as plt


# class Inverse(object):
#     def __init__(self, X):
#         self.X = np.sort(X)
        
#         self.min = self.X[0]
#         self.max = self.X[-1]
        
#         self.Y = (self.X + np.pad(self.X[1:], (1,0)))/2
        
#         self.cdf = interp1d(
#             self.X, np.linspace(0, 1, len(self.X))
#         )
        
#         self.scdf = interp1d(
#             self.Y, np.linspace(0, 1, len(self.X))
#         )
        
#     def cdf(x):
#         low = x <= self.min
#         high = x >= selfmax
    
#     def sample(N: int):
#         U = np.random.uniform(0, 1, (N,))
    
# if __name__ == "__main__":
    
#     # X = np.random.normal(loc=2, scale=1, size=(50000,))
    
#     # plt.hist(X, bins=100)
    
#     # y = inversion(X, 5000, 4)

#     # plt.figure()
#     # plt.hist(y, bins=100)    
    
#     N = 30
    
#     facs = np.random.normal(0,1,(N,))
    
#     I = Inverse(facs)
    
#     x = np.linspace(np.min(facs), np.max(facs), 500)
    
#     plt.figure()
#     plt.plot(x,I.cdf(x))
#     plt.plot(x,I.scdf(x))        
    
    
    
    
    