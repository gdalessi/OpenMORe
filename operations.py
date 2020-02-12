'''
MODULE: operations.py

@Authors: 
    G. D'Alessio [1,2], G. Aversano [1], A. Parente[1]
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

__all__ = ["get_cluster", "get_centroids", "check_sanity_int", "check_sanity_NaN", "plot_residuals", "unscale", "uncenter", "center", "scale", "center_scale", "recover_from_PCA", "PCA_fit", "explained_variance", "plot_parity"]

import numpy as np
import pandas as pd
from numpy import linalg as LA

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


def explained_variance(X, n_eigs, plot=False):
    '''
    Assess the variance explained by the first 'n_eigs' retained
    Principal Components.
    - Input:
    X = data matrix -- dim: (observations x variables)
    n_eigs = number of components to retain -- dim: (scalar)
    plot = choose if you want to plot the cumulative variance --dim: (boolean), false is default
    - Output:
    explained: percentage of explained variance -- dim: (scalar)
    '''
    PCs, eigens = PCA_fit(X, n_eigs)
    #n_PCs = np.linspace(1, X.shape[1]+1, X.shape[1])
    explained_variance = np.cumsum(eigens)/sum(eigens)

    explained = explained_variance[n_eigs]

    if plot:
        matplotlib.rcParams.update({'font.size' : 18, 'text.usetex' : True})
        fig = plt.figure()
        axes = fig.add_axes([0.15,0.15,0.7,0.7], frameon=True)
        axes.plot(np.linspace(1, X.shape[1]+1, X.shape[1]), explained_variance, color='b', marker='s', linestyle='-', linewidth=2, markersize=4, markerfacecolor='b', label='Cumulative explained')
        axes.plot([n_eigs, n_eigs], [explained_variance[0], explained_variance[n_eigs]], color='r', marker='s', linestyle='-', linewidth=2, markersize=4, markerfacecolor='r', label='Explained by {} PCs'.format(n_eigs))
        axes.set_xlabel('Number of PCs [-]')
        axes.set_ylabel('Explained variance [-]')
        axes.set_title('Variance explained by {} PCs: {}'.format(n_eigs, round(explained,3)))
        axes.legend()
    plt.show()

    return explained


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


def get_cluster(X, idx, index, write=False):
    ''' 
    Given an index, group all the observations
    of the matrix X given their membership vector idx.
    - Input:
    X = data matrix -- dim: (observations x variables)
    idx = class membership vector -- dim: (obs x 1)
    index = index of the requested group -- dim (scalar)
    - Output:
    cluster: matrix with the grouped observations -- dim: (p x var)
    '''
    positions = np.where(idx == index)
    cluster = X[positions]

    if write:
        np.savetxt("Observations in cluster number{}.txt".format(index))

    return cluster


def PCA_fit(X, n_eig):
    '''
    Perform Principal Component Analysis on the dataset X, 
    and retain 'n_eig' Principal Components.
    The covariance matrix is firstly calculated, then it is
    decomposed in eigenvalues and eigenvectors.
    Lastly, the eigenvalues are ordered depending on their 
    magnitude and the associated eigenvectors are retained.
    - Input:
    X = centered/scaled data matrix -- dim: (observations x variables)
    n_eig = number of principal components to retain -- dim: (scalar)
    - Output:
    evecs: eigenvectors from the covariance matrix decomposition (PCs)
    evals: eigenvalues from the covariance matrix decomposition (lambda)
    !!! WARNING !!! the PCs are already ordered (decreasing, for importance)
    because the eigenvalues are also ordered in terms of magnitude.
    '''
    if n_eig < X.shape[1]:
        C = np.cov(X, rowvar=False) 

        evals, evecs = LA.eig(C)
        mask = np.argsort(evals)[::-1]
        evecs = evecs[:,mask]
        evals = evals[mask]

        evecs = evecs[:, 0:n_eig]

        return evecs, evals
    
    else:
        raise Exception("The number of PCs exceeds the number of variables in the data-set.")


def plot_parity(Y,f, name_of_variable= ''):
    '''
    Produce the parity plot of the reconstructed profile from the PCA 
    manifold. The more the scatter plot (black dots) is in line with the
    red line, the better it is the reconstruction.
    - Input:
    Y = Original vector, to be put on x axis.
    f = Reconstructed vector, to be put on the y axis.
    '''
    if name_of_variable == '':
        matplotlib.rcParams.update({'font.size' : 18, 'text.usetex' : True})
        fig = plt.figure()
        axes = fig.add_axes([0.15,0.15,0.7,0.7], frameon=True)
        axes.plot(Y,Y, color='r', linestyle='-', linewidth=2, markerfacecolor='b')
        axes.scatter(Y,f, 1, color= 'k')
        axes.set_xlabel('Y, original [-]')
        axes.set_ylabel('Y, reconstructed [-]')
        plt.xlim(min(Y), max(Y))
        plt.ylim(min(Y), max(Y))
        #axes.set_title('Parity plot')
        plt.show()
    else:
        matplotlib.rcParams.update({'font.size' : 18, 'text.usetex' : True})
        fig = plt.figure()
        axes = fig.add_axes([0.15,0.15,0.7,0.7], frameon=True)
        axes.plot(Y,Y, color='r', linestyle='-', linewidth=2, markerfacecolor='b')
        axes.scatter(Y,f, 1, color= 'k')
        axes.set_xlabel('{}, real [-]'.format(name_of_variable))
        axes.set_ylabel('{}, predicted [-]'.format(name_of_variable))
        plt.xlim(min(Y), max(Y))
        plt.ylim(min(Y), max(Y))
        #axes.set_title('Parity plot')
        plt.show()


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


def recover_from_PCA(X, modes, mu, sigma):
    '''
    Reconstruct the original matrix from the reduced PCA-manifold.
    - Input:
    X = centered/scaled data matrix -- dim: (observations x variables)
    modes = set of eigenvectors (PCs) -- dim: (num_PCs x variables)
    - Output:
    X0 = centered and scaled reconstructed matrix -- dim: (observations x variables)
    X_rec = uncentered and unscaled reconstructed matrix -- dim: (observations x variables)
    '''
    if X.shape[1] == mu.shape[0] and X.shape[1] == sigma.shape[0]:
        X0 = X @ modes[0] @ modes[0].T
        X_unsc = unscale(X0, sigma)
        X_rec = uncenter(X_unsc, mu)
        return X_rec
    else:
        raise Exception("The dimensionality of X does not match the dimensionality of the scaling factors.")


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