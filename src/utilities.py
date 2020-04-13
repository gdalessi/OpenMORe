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
    This code is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
    Please report any bug to: giuseppe.dalessio@ulb.ac.be
'''


import numpy as np
from numpy import linalg as LA
import functools
import time


import matplotlib
import matplotlib.pyplot as plt
__all__ = ["unscale", "uncenter", "center", "scale", "center_scale", "PHC_index", "get_centroids", "get_cluster", "get_all_clusters", "explained_variance", "evaluate_clustering_DB", "NRMSE", "PCA_fit", "accepts", "readCSV", "allowed_centering","allowed_scaling", "PHC_robustTrim", "PHC_median", "varimax_rotation", "get_medianoids", "get_medoids"]


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
        if method.lower() == 'mean':
            mu = np.mean(X, axis = 0)
        elif method.lower() == 'min':
            mu = np.min(X, axis = 0)
        else:
            raise Exception("Unsupported centering option. Please choose: MEAN or MIN.")
        return mu
    else:
        if method.lower() == 'mean':
            mu = np.mean(X, axis = 0)
            X0 = X - mu
        elif method.lower() == 'min':
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
        raise Exception("The matrix to be centered & scaled and the centering/scaling vectors must have the same dimensionality.")


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


def evaluate_clustering_DB(X, idx):
    """
    Davies-Bouldin index a coefficient to evaluate the goodness of a
    clustering solution. The more it approaches to zero, the better
    the clustering solution is. -- Tested OK with comparison Matlab
    """
    from scipy.spatial.distance import euclidean, cdist
    #Initialize matrix and other quantitites
    k = int(np.max(idx) +1)
    centroids_list = [None] *k
    S_i = [None] *k
    M_ij = np.zeros((k,k), dtype=float)
    TOL = 1E-16

    #For each cluster, compute the mean distance between the points and their centroids
    for ii in range(0,k):
        cluster_ = get_cluster(X, idx, ii)
        centroids_list[ii] = get_centroids(cluster_)
        S_i[ii] = np.mean(cdist(cluster_, centroids_list[ii].reshape((1,-1))))  #reshape centroids_list[ii] from (n,) to (1,n)

    #Compute the distance between once centroid and all the others:
    for ii in range(0,k):
        for jj in range(0,k):
            if ii != jj:
                M_ij[ii,jj] = euclidean(centroids_list[ii], centroids_list[jj])
            else:
                M_ij[ii,jj] = 1

    R_ij = np.empty((k,k),dtype=float)

    #Compute the R_ij coefficient for each couple of clusters, using the
    #two coefficients S_ij and M_ij
    for ii in range(0,k):
        for jj in range(0,k):
            if ii != jj:
                R_ij[ii,jj] = (S_i[ii] + S_i[jj])/M_ij[ii,jj] +TOL
            else:
                R_ij[ii,jj] = 0

    D_i = [None] *k

    #Compute the Davies-Bouldin index as the mean of the maximum R_ij value
    for ii in range(0,k):
        D_i[ii] = np.max(R_ij[ii], axis=0)

    DB = np.mean(D_i)

    return DB


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


def get_medianoids(X):
    '''
    Given a matrix (or a cluster), calculate its
    medianoid (same as centroid, but with median).
    - Input:
    X = data matrix -- dim: (observations x variables)
    - Output:
    medianoid = medianoid vector -- dim: (1 x variables)
    '''
    medianoid = np.median(X, axis = 0)
    return medianoid

def get_medoids(X):
    '''
    Given a matrix (or a cluster), calculate its
    medoid: the point which minimize the sum of distances
    with respect to the other observations.
    - Input:
    X = data matrix -- dim: (observations x variables)
    - Output:
    medoid = medoid vector -- dim: (1 x variables)
    '''
    from scipy.spatial.distance import euclidean, cdist
    #Compute the distances between each point of the matrix
    dist = cdist(X, X)
    #Sum all the distances along the columns, to have the 
    #cumulative sum of distances.
    cumSum = np.sum(dist, axis=1)
    #Pick up the point which minimize the distances from
    #all the other points as a medoid
    mask = np.argmin(cumSum)
    medoid = X[mask,:]

    return medoid



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
        np.savetxt("Observations in cluster number{}.txt".format(index), cluster)

    return cluster


def get_all_clusters(X, idx):
    '''
    Group all the observations of the matrix X given their membership vector idx,
    and collect the different groups into a list.
    - Input:
    X = data matrix -- dim: (observations x variables)
    idx = class membership vector -- dim: (obs x 1)
    - Output:
    clusters: list with the clusters from 0 to k -- dim: (k)
    '''
    k = max(idx) +1
    clusters = [None] *k

    for ii in range (0,k):
        clusters[ii] = get_cluster(X, idx, ii)

    return clusters


def NRMSE (X_true, X_pred):
    n_obs, n_var = X_true.shape
    NRMSE = [None] *n_var

    for ii in range(0, n_var):
        NRMSE[ii] = np.sqrt(np.mean((X_true[:,ii] - X_pred[:,ii])**2)) / np.sqrt(np.mean(X_true[:,ii]**2))

    return NRMSE


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


def PHC_median(X, idx):
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
        PHC_deviations[ii] = np.median(dev)

    return PHC_coeff, PHC_deviations


def PHC_robustTrim(X,idx):

    import model_order_reduction

    k = np.max(idx) +1
    TOL = 1E-16
    PHC_coeff=[None] *k
    PHC_deviations=[None] *k

    for ii in range(0,k):
        cluster_ = get_cluster(X, idx, ii)

        model = model_order_reduction.PCA(cluster_)
        model.centering = 'mean'
        model.scaling = 'auto'
        model.eigens = cluster_.shape[1] -1
        PCs, eigval = model.fit()
        scores = model.get_scores()
        mahalanobis_ = np.empty((cluster_.shape[0],),dtype=float)

        for jj in range(0,cluster_.shape[0]):
            t_sq = 0
            lam_j = 0
            for jj in range(0, cluster_.shape[1]-1):
                t_sq += scores[ii,jj]**2
                lam_j += eigval[jj]
            mahalanobis_[ii] = t_sq/(lam_j + TOL)
            
        #A fraction alpha (typically 0.01%-0.1%) of the data points characterized by the largest 
        #value of DM are classified as outliers and removed.
        
        alpha = 0.000007
        
        
        #compute the new number of observations after the trim factor:
        trim = int((1-alpha)*cluster_.shape[0])
        to_trim = np.argsort(mahalanobis_)
    
        new_mask = to_trim[:trim]
        cluster_ = cluster_[new_mask,:]

        maxima = np.max(cluster_, axis = 0)
        minima = np.min(cluster_, axis = 0)
        media = np.mean(cluster_, axis=0)

        dev = np.std(cluster_, axis=0)

        PHC_coeff[ii] = np.mean((maxima-minima)/(media +TOL))
        PHC_deviations[ii] = np.mean(dev)

    return PHC_coeff, PHC_deviations


def readCSV(path, name):
    try:
        print("Reading training matrix..")
        X = np.genfromtxt(path + "/" + name, delimiter= ',')
    except OSError:
        print("Could not open/read the selected file: " + name)
        exit()

    return X


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
    TOL = 1E-16
    if not return_scaled_matrix:
        if method.lower() == 'auto':
            sig = np.std(X, axis = 0)
        elif method.lower() == 'pareto':
            sig = np.sqrt(np.std(X, axis = 0))
        elif method.lower() == 'vast':
            variances = np.var(X, axis = 0)
            means = np.mean(X, axis = 0)
            sig = variances / means
        elif method.lower() == 'range':
            maxima = np.max(X, axis = 0)
            minima = np.min(X, axis = 0)
            sig = maxima - minima
        else:
            raise Exception("Unsupported scaling option. Please choose: AUTO, PARETO, VAST or RANGE.")
        return sig
    else:
        if method.lower() == 'auto':
            sig = np.std(X, axis = 0)
            X0 = X / (sig + TOL)
        elif method.lower() == 'pareto':
            sig = np.sqrt(np.std(X, axis = 0))
            X0 = X / (sig + TOL)
        elif method.lower() == 'vast':
            variances = np.var(X, axis = 0)
            means = np.mean(X, axis = 0)
            sig = variances / means
            X0 = X / (sig + TOL)
        elif method.lower() == 'range':
            maxima = np.max(X, axis = 0)
            minima = np.min(X, axis = 0)
            sig = maxima - minima
            X0 = X / (sig + TOL)
        else:
            raise Exception("Unsupported scaling option. Please choose: AUTO, PARETO, VAST or RANGE.")
        return sig, X0

def split_for_validation(X, validation_quota):
    '''
    Split the data into two matrices, one to train the model (X_train) and the
    other to validate it.
    - Input:
    X = matrix to be split -- dim: (observations x variables)
    validation_quota = percentage of observations to take as validation
    - Output:
    X_train = matrix to be used to train the reduced model 
    X_test = matrix to be used to test the reduced model
    '''

    nObs = X.shape[0]
    nVar = X.shape[1]

    nTest = int(nObs * validation_quota)

    np.random.shuffle(X)

    X_test = X[:nTest,:]
    X_train = X[nTest+1:,:]

    return X_train, X_test


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

def varimax_rotation(X, b, normalize=True):
    '''
    STILL UNDER CONSTRUCTION
    Warning: has not been tested, yet -- there is still something to be checked.
    '''
    eigens = b.shape[1]
    norm_factor = np.std(b)
    loadings = np.empty((b.shape[0], b.shape[1]), dtype=float)
    rot_loadings = np.empty((b.shape[0], b.shape[1]), dtype=float)

    for ii in range(0, eigens):
        loadings[:,ii] = b[:,ii]/norm_factor

    C = np.cov(loadings, rowvar=False) 

    iter_max = 1000
    convergence_tolerance = 1E-16
    iter = 1
    convergence = False

    variance_explained = np.sum(loadings**2)

    while not convergence:
        for ii in range(0,eigens):
            for jj in range(ii+1, eigens):
                x_j = loadings[:,ii]
                y_j = loadings[:,jj]

                u_j = x_j**2 - y_j**2
                v_j = 2*x_j * y_j

                A = np.sum(u_j)
                B = np.sum(v_j)
                C = u_j.T * u_j - v_j.T *v_j
                D = 2 * u_j.T * v_j 

                num = D - 3*A*B/X.shape[0]
                den = C - (A**2 - B**2)/X.shape[0]

                phi = np.arctan2(num, den)/4
                angle = phi *180/3.141592653589793238462643383279502

                if np.mean(np.abs(phi)) > 0.00001:
                    X_j = np.cos(phi) *x_j + np.sin(phi) *y_j
                    Y_j = -np.sin(phi) *x_j + np.cos(phi) *y_j

                    loadings[:,ii] = X_j
                    loadings[:,jj] = Y_j

        var_old = variance_explained
        variance_explained = np.sum(loadings**2)
        check = np.abs((variance_explained - var_old)/(variance_explained + 1E-16))

        if check < convergence_tolerance or iter > iter_max:
            convergence = True 
        else:
            iter += 1
        
        print("Iteration number: {}".format(iter))
        print("Convergence residuals: {}".format(check))

    
    for ii in range(0, eigens):
        rot_loadings[:,ii] = loadings[:,ii] * norm_factor

    return rot_loadings


                    
# ------------------------------
# Decorators (alphabetical order)
# ------------------------------

def accepts(*types):
    """
    Checks argument types.
    """
    def decorator(f):
        assert len(types) == f.__code__.co_argcount
        @functools.wraps(f)
        def wrapper(*args, **kwds):
            for (a, t) in zip(args, types):
                assert isinstance(a, t), "The input argument %r must be of type <%s>" % (a,t)
            return f(*args, **kwds)
        wrapper.__name__ = f.__name__
        return wrapper
    return decorator

def allowed_centering(func):
    '''
    Checks the user input for centering criterion.
    Exit with error if the centering is not allowed.
    '''
    def func_check(dummy, x):
        if x.lower() != 'mean' and x.lower() != 'min':
            raise Exception("Centering criterion not allowed. Supported options: 'mean', 'min'. Exiting with error..")
            exit()
        res = func(dummy, x)
        return res
    return func_check


def allowed_scaling(func):
    '''
    Checks the user input for scaling criterion.
    Exit with error if the scaling is not allowed.
    '''
    def func_check(dummy, x):
        if x.lower() != 'auto' and x.lower() != 'pareto' and x.lower() != 'range' and x.lower() != 'vast':
            raise Exception("Scaling criterion not allowed. Supported options: 'auto', 'vast', 'pareto' or 'range'. Exiting with error..")
            exit()
        res = func(dummy, x)
        return res
    return func_check