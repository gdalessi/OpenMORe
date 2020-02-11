'''
MODULE: operations.py

@Authors: 
    G. D'Alessio [1,2], G. Aversano [1], A. Parente[1]
    [1]: Université Libre de Bruxelles, Aero-Thermo-Mechanics Laboratory, Bruxelles, Belgium
    [2]: CRECK Modeling Lab, Department of Chemistry, Materials and Chemical Engineering, Politecnico di Milano

@Contacts:
    giuseppe.dalessio@ulb.ac.be

@Additional notes:
    This cose is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; 
    Please report any bug to: giuseppe.dalessio@ulb.ac.be

'''

__all__ = ["get_cluster", "get_centroids", "fitPCA", "check_sanity_int", "check_sanity_NaN", "plot_residuals", "unscale", "uncenter", "center", "scale", "center_scale"]

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

import matplotlib
import matplotlib.pyplot as plt

# ------------------------------
# Functions (alphabetical order)
# ------------------------------


def center(X, method):
    '''
    Compute the mean/min value (mu) of each variable of all data-set observations.
    '''
    # Main
    if method == 'MEAN' or method == 'mean' or method == 'Mean':
        mu = np.mean(X, axis = 0)
    elif method == 'MIN' or method == 'min' or method == 'Min':
        mu = np.min(X, axis = 0)
    else:
        raise Exception("Unsupported centering option. Please choose: MEAN or MIN.")
    return mu


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
    TOL = 1E-10
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
    Check if in a matrix NaNs are contained.
    '''
    if X.isna().values.any() == False:
        return X
    else:
        raise Exception("The input matrix contains NaN values. Please double-check your input.")
        exit()


def fitPCA(X, n_eig):
    '''
    Perform PCA using sklearn.
    - Input:
    X = data matrix -- dim: (observations x variables)
    n_eig = number of PCs to retain -- dim: (scalar)
    - Output:
    eigenvec: Principal Components -- dim: (n_eig x variables)
    '''
    pca = PCA(n_components = n_eig)
    pca.fit(X)
    eigenvec = pca.components_
    return np.array(eigenvec)


def get_centroids(X):
    '''
    Given a matrix (or a cluster), calculate its
    centroid.
    - Input:
    X = data matrix -- dim: (observations x variables)
    - Output:
    centroid = centroid vector -- dim: (1 x variables)
    '''
    centroid = np.mean(X, axis = 0)
    return centroid


def get_cluster(X, idx, index):
    ''' 
    Given an index, group all the observations
    of the matrix X given their membership vector idx.
    - Input:
    X = data matrix -- dim: (observations x variables)
    idx = class membership vector -- dim: (obs x 1)
    index = index of the requested group -- dim (scalar)
    - Output:
    cluster: matrix with the grouped observations -- dim: (obs x var)
    '''
    positions = np.where(idx == index)
    cluster = X[positions]
    return cluster


def plot_residuals(iterations, error):
    '''
    Plot the reconstruction error behavior for the LPCA iterative
    algorithm vs the iterations.
    - Input:
    iterations = linspace vector from 1 to the total number of iterations
    error = reconstruction error story
    '''
    matplotlib.rcParams.update({'font.size' : 18, 'text.usetex' : True})
    itr = np.linspace(1,iterations, iterations)
    fig = plt.figure()
    axes = fig.add_axes([0.15,0.15,0.7,0.7], frameon=True)
    axes.plot(itr,error[1:], color='b', marker='s', linestyle='-', linewidth=2, markersize=4, markerfacecolor='b')
    axes.set_xlabel('Iterations [-]')
    axes.set_ylabel('Reconstruction error [-]')
    axes.set_title('Convergence residuals')
    plt.show()


def scale(X, method):
    '''
    Compute the scaling factor (sig) for each variable of the data-set
    '''
    # Main
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
        X0 = X_tilde + mu
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
    if X_tilde.shape[1] == sigma.shape[0]:
        X0 = X_tilde * sigma
        return X0
    else:
        raise Exception("The matrix to be unscaled and the scaling vector must have the same dimensionality.")
        exit()