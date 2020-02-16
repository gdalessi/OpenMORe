__all__ = ["recover_from_PCA", "PCA_fit", "explained_variance", "plot_parity", "lpca", "recover_from_LPCA", "plot_PCs", "get_centroids", "get_cluster"]

import numpy as np
from numpy import linalg as LA
from operations import *

import matplotlib
import matplotlib.pyplot as plt

# ------------------------------
# Functions (alphabetical order)
# ------------------------------

def explained_variance(X, n_eigs, plot=False):
    '''
    Assess the variance explained by the first 'n_eigs' retained
    Principal Components. This is important to know if the percentage
    of explained variance is enough, or additional PCs must be retained.
    Usually, it is considered accepted a percentage of explained variable
    above 95%.
    - Input:
    X = CENTERED/SCALED data matrix -- dim: (observations x variables)
    n_eigs = number of components to retain -- dim: (scalar)
    plot = choose if you want to plot the cumulative variance --dim: (boolean), false is default
    - Output:
    explained: percentage of explained variance -- dim: (scalar)
    '''
    PCs, eigens = PCA_fit(X, n_eigs)
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


def lpca(X, idx, n_eigs, cent_crit, scal_crit=False):
    '''
    This function computes the LPCs (Local Principal Components), the u_scores
    (the projection of the points on the local manifold), and the eigenvalues
    in each cluster found by the lpca iterative algorithm, given a previous clustering
    solution.
    - Input:
    X = CENTERED/SCALED data matrix -- dim: (observations x variables)
    idx = class membership vector -- dim: (obs x 1)
    n_eigs = number of LPCs and eigenvalues to retain -- dim: (scalar)
    cent_crit = local centering criterion (MEAN or MIN)
    scal_crit = local scaling criterion, optional (AUTO)
    - Output:
    LPCs = list with the LPCs in each cluster -- dim: [k]
    u_scores = list with the scores in each cluster -- dim: [k]
    Leigen = list with the eigenvalues in each cluster -- dim: [k]
    centroids = list with the centroids in each cluster -- dim: [k]
    '''
    k = max(idx)+1
    n_var = X.shape[1]

    centroids = [None] *k
    LPCs = [None] *k
    u_scores = [None] *k
    Leigen = [None] *k

    for ii in range (0,k):
        cluster = get_cluster(X, idx, ii)
        centroids[ii], cluster_ = center(cluster, cent_crit, True)
        if scal_crit:
            scaling = "AUTO"
            sig, cluster_ = scale(cluster, scaling)
        LPCs[ii], Leigen[ii] = PCA_fit(cluster_, n_eigs)
        u_scores[ii] = cluster_ @ LPCs[ii]

    return LPCs, u_scores, Leigen, centroids


def PCA_fit(X, n_eig):
    '''
    Perform Principal Component Analysis on the dataset X, 
    and retain 'n_eig' Principal Components.
    The covariance matrix is firstly calculated, then it is
    decomposed in eigenvalues and eigenvectors.
    Lastly, the eigenvalues are ordered depending on their 
    magnitude and the associated eigenvectors (the PCs) are retained.
    - Input:
    X = CENTERED/SCALED data matrix -- dim: (observations x variables)
    n_eig = number of principal components to retain -- dim: (scalar)
    - Output:
    evecs: eigenvectors from the covariance matrix decomposition (PCs)
    evals: eigenvalues from the covariance matrix decomposition (lambda)

    !!! WARNING !!! the PCs are already ordered (decreasing, for importance)
    because the eigenvalues are also ordered in terms of magnitude.
    '''
    if n_eig < X.shape[1]:
        C = np.cov(X, rowvar=False) #rowvar=False because the X matrix is (observations x variables)

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
        axes.set_xlabel('{}, original [-]'.format(name_of_variable))
        axes.set_ylabel('{}, reconstructed [-]'.format(name_of_variable))
        plt.xlim(min(Y), max(Y))
        plt.ylim(min(Y), max(Y))
        #axes.set_title('Parity plot')
        plt.show()


def plot_PCs(modes, n):
    '''
    Plot the variables' weights on a selected Principal Component (PC). 
    The PCs are linear combination of the original variables: examining the 
    weights, especially for the PCs associated with the largest eigenvalues,
    can be important for data-analysis purposes.
    - Input:
    modes: Set of PCs -- dim: (variables x n_PCs)
    n : number of the selected PC to examine
    '''
    matplotlib.rcParams.update({'font.size' : 18, 'text.usetex' : True})
    fig = plt.figure()
    axes = fig.add_axes([0.15,0.15,0.7,0.7], frameon=True)
    n_var = modes[0].shape[0]
    x = np.linspace(1, n_var, n_var)
    axes.bar(x, modes[0][:,n])
    axes.set_xlabel('Variable [-]')
    axes.set_ylabel('Weight on the PC number: {} [-]'.format(n))
    plt.show()


def recover_from_PCA(X, modes, cent_crit, scal_crit):
    '''
    Reconstruct the original matrix from the reduced PCA-manifold.
    - Input:
    X = UNCENTERED/UNSCALED data matrix -- dim: (observations x variables)
    modes = set of eigenvectors (PCs) -- dim: (num_PCs x variables)
    cent_crit = centering criterion for the X matrix (MEAN or MIN)
    scal_crit = scaling criterion for the X matrix (AUTO, RANGE, VAST, PARETO)
    - Output:
    X_rec = uncentered and unscaled reconstructed matrix with PCA modes -- dim: (observations x variables)
    '''
    mu = center(X, cent_crit)
    sigma = scale(X, scal_crit)
    X_tilde = center_scale(X, mu, sigma)

    X0 = X_tilde @ modes[0] @ modes[0].T
    X_unsc = unscale(X0, sigma)
    X_rec = uncenter(X_unsc, mu)
    return X_rec
   

def recover_from_LPCA(X, idx, n_eigs, cent_crit, scal_crit):
    '''
    Reconstruct the original matrix from the 'k' local reduced PCA-manifolds.
    Given the idx vector, for each cluster the points are reconstructed from the 
    local manifolds spanned by the local PCs.
    - Input:
    X = UNCENTERED/UNSCALED data matrix -- dim: (observations x variables)
    idx = class membership vector -- dim: (obs x 1)
    modes = set of eigenvectors (PCs) -- dim: (num_PCs x variables)
    cent_crit = centering criterion for the X matrix (MEAN or MIN)
    scal_crit = scaling criterion for the X matrix (AUTO, RANGE, VAST, PARETO)
    - Output:
    X_rec = uncentered and unscaled reconstructed matrix with LPCA modes -- dim: (observations x variables)
    '''

    k = max(idx)+1
    X_rec = np.zeros(X.shape)

    mu_global = center(X, cent_crit)
    sigma_global = scale(X, scal_crit)
    X_tilde = center_scale(X, mu_global, sigma_global)

    LPCs, u_scores, Leigen, centroids = lpca(X_tilde, idx, n_eigs, cent_crit)


    for ii in range (0,k):
        cluster_ = get_cluster(X_tilde, idx, ii)
        centroid_ = get_centroids(cluster_)
        C = np.zeros(cluster_.shape) 
        C = (cluster_ - centroid_) @ LPCs[ii] @ LPCs[ii].T
        C_ = uncenter(C, centroid_)
        positions = np.where(idx == ii)
        X_rec[positions] = C_
    
    X_rec = unscale(X_rec, sigma_global)
    X_rec = uncenter(X_rec, mu_global)

    return X_rec