'''
MODULE: utilities.py

@Author: 
    G. D'Alessio [1,2]
    [1]: Université Libre de Bruxelles, Aero-Thermo-Mechanics Laboratory, Bruxelles, Belgium
    [2]: CRECK Modeling Lab, Department of Chemistry, Materials and Chemical Engineering, Politecnico di Milano

@Contacts:
    giuseppe.dalessio@ulb.ac.be

@Details:
    This module contains a set of functions which are useful for reduced-order modelling with PCA.
    A detailed description is available under the definition of each function.

@Additional notes:
    This cose is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; 
    Please report any bug to: giuseppe.dalessio@ulb.ac.be

'''


import numpy as np
from numpy import linalg as LA
from reduced_order_modelling import *

import matplotlib
import matplotlib.pyplot as plt
__all__ = ["check_sanity_int", "check_sanity_NaN", "unscale", "uncenter", "center", "scale", "center_scale", "PHC_index", "check_dummy"]


# ------------------------------
# Functions (alphabetical order)
# ------------------------------


def center(X, method, return_centered_matrix= False):
    '''
    Computes the centering factor (the mean/min value [mu]) of each variable of all data-set observations and
    (eventually) return the centered matrix.
    - Input:
    X = original data matrix -- dim: (observations x variables)
    method = "string", it is the method which has to be used. Two choices are available: MEAN or MIN
    return_centered_matrix = boolean, choose if the script must return the centered matrix (optional)
    - Output:
    mu = centering factor for the data matrix X
    X0 = centered data matrix (optional)
    '''
    # Main
    if not return_centered_matrix:
        if method == 'MEAN' or method == 'mean' or method == 'Mean':
            mu = np.mean(X, axis = 0)
        elif method == 'MIN' or method == 'min' or method == 'Min':
            mu = np.min(X, axis = 0)
        else:
            raise Exception("Unsupported centering option. Please choose: MEAN or MIN.")
        return mu
    else:
        if method == 'MEAN' or method == 'mean' or method == 'Mean':
            mu = np.mean(X, axis = 0)
            X0 = X - mu
        elif method == 'MIN' or method == 'min' or method == 'Min':
            mu = np.min(X, axis = 0)
            X0 = X - mu
        else:
            raise Exception("Unsupported centering option. Please choose: MEAN or MIN.")
        return mu, X0


def center_scale(X, mu, sig):
    '''
    Center and scale a given multivariate data-set X.
    Centering consists of subtracting the mean/min value of each variable to all data-set
    observations. Scaling is achieved by dividing each variable by a given scaling factor. Therefore, the
    i-th observation of the j-th variable, x_{i,j} can be
    centered and scaled by means of:

    \tilde{x_{i,j}} = (x_{i,j} - mu_{j}) / (sig_{j}),

    where mu_{j} and sig_{j} are the centering and scaling factor for the considered j-th variable, respectively.

    AUTO: the standard deviation of each variable is used as a scaling factor.
    PARETO: the squared root of the standard deviation is used as a scaling f.
    RANGE: the difference between the minimum and the maximum value is adopted as a scaling f.
    VAST: the ratio between the variance and the mean of each variable is used as a scaling f.
    '''
    TOL = 1E-16
    if X.shape[1] == mu.shape[0] and X.shape[1] == sig.shape[0]:
        X0 = X - mu
        X0 = X0 / (sig + TOL)
        return X0
    else:
        raise Exception("The matrix to be centered & scaled the centering/scaling vectors must have the same dimensionality.")


def check_sanity_int(kappa):
    '''
    Check if the input is an integer.
    '''
    if isinstance(kappa, int) == True:
        return kappa
    else:
        raise Exception("The number of cluster and/or eigenvectors input must be integers. Please provide a valid input.")
        exit()


def check_sanity_NaN(X):
    '''
    Check if a matrix contains NaNs.
    '''
    if X.isna().values.any() == False:
        return X
    else:
        raise Exception("The input matrix contains NaN values. Please double-check your input.")
        exit()


def check_dummy(X, k, n_eigs):
    if X.shape[0] < X.shape[1]:
        raise Exception("It is not possible to apply PCA or LPCA to a matrix with less observations than variables.")
    elif k > X.shape[0]:
        raise Exception("It is not possible to have more cluster than observations. Please consider to use a lower number of clusters.")
    elif n_eigs > X.shape[1]:
        raise Exception("It is not possible to have more Principal Components than variables. Please consider to use a lower number of PCs.")


def PHC_index(X, idx):
    '''
    Computes the PHC (Physical Homogeneity of the Cluster) index.
    For many applications, more than a pure mathematical tool to assess the quality of the clustering solution, 
    such as the Silhouette Coefficient, a measure of the variables variation is more suitable. This coefficient 
    assess the quality of the clustering solution measuring the variables variation in each cluster. The more the PHC 
    approaches to zero, the better the clustering.
    - Input:
    X = UNCENTERED/UNSCALED data matrix -- dim: (observations x variables)
    idx = class membership vector -- dim: (obs x 1)
    - Output:
    PHC_coeff = vector with the PHC scores for each cluster -- dim: (number_of_cluster)
    '''

    k = max(idx) +1
    TOL = 1E-16
    PHC_coeff=[None] *k
    PHC_deviations=[None] *k

    for ii in range (0,k):
        cluster_ = get_cluster(X, idx, ii)

        maxima = np.max(cluster_, axis = 0)
        minima = np.min(cluster_, axis = 0)
        media = np.mean(cluster_, axis=0) 

        dev = np.std(cluster_, axis=0)

        PHC_coeff[ii] = np.mean((maxima-minima)/(media +TOL))
        PHC_deviations[ii] = np.mean(dev)
        
    return PHC_coeff, PHC_deviations


def scale(X, method, return_scaled_matrix=False):
    '''
    Computes the scaling factor [sigma] of each variable of all data-set observations and
    (eventually) return the scaled matrix.
    - Input:
    X = original data matrix -- dim: (observations x variables)
    method = "string", it is the method which has to be used. Four choices are available: AUTO, PARETO, VAST or RANGE≥
    return_scaled_matrix = boolean, choose if the script must return the scaled matrix (optional)
    - Output:
    sig = scaling factor for the data matrix X
    X0 = centered data matrix (optional)
    '''
    # Main
    if not return_scaled_matrix:
        if method == 'AUTO' or method == 'auto' or method == 'Auto':
            sig = np.std(X, axis = 0)
        elif method == 'PARETO' or method == 'pareto' or method == 'Pareto':
            sig = np.sqrt(np.std(X, axis = 0))
        elif method == 'VAST' or method == 'vast' or method == 'Vast':
            variances = np.var(X, axis = 0)
            means = np.mean(X, axis = 0)
            sig = variances / means
        elif method == 'RANGE' or method == 'range' or method == 'Range':
            maxima = np.max(X, axis = 0)
            minima = np.min(X, axis = 0)
            sig = maxima - minima
        else:
            raise Exception("Unsupported scaling option. Please choose: AUTO, PARETO, VAST or RANGE.")
        return sig
    else:
        if method == 'AUTO' or method == 'auto' or method == 'Auto':
            sig = np.std(X, axis = 0)
            X0 = X / (sig + TOL)
        elif method == 'PARETO' or method == 'pareto' or method == 'Pareto':
            sig = np.sqrt(np.std(X, axis = 0))
            X0 = X / (sig + TOL)
        elif method == 'VAST' or method == 'vast' or method == 'Vast':
            variances = np.var(X, axis = 0)
            means = np.mean(X, axis = 0)
            sig = variances / means
            X0 = X / (sig + TOL)
        elif method == 'RANGE' or method == 'range' or method == 'Range':
            maxima = np.max(X, axis = 0)
            minima = np.min(X, axis = 0)
            sig = maxima - minima
            X0 = X / (sig + TOL)
        else:
            raise Exception("Unsupported scaling option. Please choose: AUTO, PARETO, VAST or RANGE.")
        return sig, X0


def uncenter(X_tilde, mu):
    '''
    Uncenter a standardized matrix.
    - Input:
    X_tilde: centered matrix -- dim: (observations x variables)
    mu: centering factor -- dim: (1 x variables)
    - Output:
    X0 = uncentered matrix -- dim: (observations x variables)
    '''
    if X_tilde.shape[1] == mu.shape[0]:
        X0 = np.zeros_like(X_tilde, dtype=float)
        for i in range(0, len(mu)):
            X0[:,i] = X_tilde[:,i] + mu[i]
        return X0
    else:
        raise Exception("The matrix to be uncentered and the centering vector must have the same dimensionality.")
        exit()


def unscale(X_tilde, sigma):
    '''
    Unscale a standardized matrix.
    - Input:
    X_tilde = scaled matrix -- dim: (observations x variables)
    sigma = scaling factor -- dim: (1 x variables)
    - Output:
    X0 = unscaled matrix -- dim: (observations x variables)
    '''
    TOL = 1E-16
    if X_tilde.shape[1] == sigma.shape[0]:
        X0 = np.zeros_like(X_tilde, dtype=float)
        for i in range(0, len(sigma)):
            X0[:,i] = X_tilde[:,i] * (sigma[i] + TOL)
        return X0
    else:
        raise Exception("The matrix to be unscaled and the scaling vector must have the same dimensionality.")
        exit()