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


import matplotlib
import matplotlib.pyplot as plt
__all__ = ["unscale", "uncenter", "center", "scale", "center_scale", "evaluate_clustering_PHC", "fastSVD", "get_centroids", "get_cluster", "get_all_clusters", "explained_variance", "evaluate_clustering_DB", "NRMSE", "PCA_fit", "readCSV", "varimax_rotation", "get_medianoids", "split_for_validation", "get_medoids"]


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
            #compute the vector containing the mean for each variable
            mu = np.mean(X, axis = 0)
        elif method.lower() == 'min':
            #compute the vector containing the minimum for each variable
            mu = np.min(X, axis = 0)
        else:
            raise Exception("Unsupported centering option. Please choose: MEAN or MIN.")
        return mu
    else:
        if method.lower() == 'mean':
            #compute the vector containing the mean for each variable
            mu = np.mean(X, axis = 0)
            #subtract the mean to each observation of the matrix X
            X0 = X - mu
        elif method.lower() == 'min':
            #compute the vector containing the minimum for each variable
            mu = np.min(X, axis = 0)
            #subtract the minimum to each observation of the matrix X
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
        #centering step: subtract the centering factor to each observation
        X0 = X - mu

        #scaling step: divide each observation for the scaling factor. TOL is added to avoid dividing by 0 
        X0 = X0 / (sig + TOL)
        return X0
    else:
        raise Exception("The matrix to be centered & scaled and the centering/scaling vectors must have the same dimensionality.")


def explained_variance(X, n_eigs, plot=False):
    '''
    Assess the variance explained by the first 'n_eigs' retained
    Principal Components. This is important to know if the percentage
    of explained variance is enough, or additional PCs must be retained.
    Usually, it is considered acceptable a percentage of explained variable
    above 95%.
    - Input:
    X = CENTERED/SCALED data matrix -- dim: (observations x variables)
    n_eigs = number of components to retain -- dim: (scalar)
    plot = choose if you want to plot the cumulative variance --dim: (boolean), false is default
    - Output:
    explained: percentage of explained variance -- dim: (scalar)
    '''
    #compute PCA
    PCs, eigens = PCA_fit(X, n_eigs)
    
    #the explained variance is defined as the sum of the 'q' eigenvalues to retain
    #divided by the total sum of the eigenvalues
    explained_variance = np.cumsum(eigens)/sum(eigens)
    explained = explained_variance[n_eigs]

    #plot the curve of the cumulative variance, if the option is activated
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

    #The final DB index is the mean value of the DBs for each cluster
    DB = np.mean(D_i)

    return DB

def evaluate_clustering_PHC(X, idx):
    '''
    Computes the PHC (Physical Homogeneity of the Cluster) index.
    For many applications, more than a pure mathematical tool to assess the quality of the clustering solution,
    such as the Silhouette Coefficient, a measure of the variables variation is more suitable. This coefficient
    assess the quality of the clustering solution measuring the variables variation in each cluster. The more the PHC
    approaches to zero, the better the clustering.
    - Input:
    X = UNCENTERED/UNSCALED data matrix -- dim: (observations x variables)
    idx = class membership vector -- dim: (obs x 1)
    method = method to be used for the PHC computation:'PHC_standard', 'PHC_median', 'PHC_robust' are available.
    - Output:
    PHC_coeff = vector with the PHC scores for each cluster -- dim: (number_of_cluster)
    '''

    k = max(idx) +1
    TOL = 1E-16
    PHC_coeff=[None] *k
    PHC_deviations=[None] *k

    #The standard PHC for one variable is computed as: (max - min)/ mean
    #If the training matrix has more than 1 variable, the PHCs are stored in a list PHC_coeff.

    for ii in range (0,k):
        try:
            #take the clusters' observations
            cluster_ = get_cluster(X, idx, ii)

            #compute max, min and mean
            maxima = np.max(cluster_, axis = 0)
            minima = np.min(cluster_, axis = 0)
            media = np.mean(cluster_, axis=0)

            #compute the standard deviation of each cluster, because PHC can be sensitive to outliers
            #so if dev it's high PHC could not be completely reliable
            dev = np.std(cluster_, axis=0)

            #compute PHCs. TOL is added to avoid dividing by 0
            PHC_coeff[ii] = np.mean((maxima-minima)/(media +TOL))
            
            #compute the average standard deviations
            PHC_deviations[ii] = np.mean(dev)
        except ValueError:
            print("An exception was thrown by Python during the PHC computation. Probably the considered cluster was found empty.")
            print("Passing..")
            pass

    return PHC_coeff, PHC_deviations


def fastSVD(X_tilde, n_eigs):
    '''
    Perform a fast Singular Value Decomposition with the algorithm described in:
    [1] Halko, Nathan, et al. SIAM Journal on Scientific computing 33.5 (2011): 2580-2594.
    - Input:
    X = CENTERED/SCALED data matrix -- dim: (observations x variables)
    n_eigs = number of eigenvectors to retain -- dim: (scalar)
   
    - Output:
    U, V, Sigma: matrices such that ||X_tilde - U @ Sigma @ V.T || < eps
    
    '''
    rows, cols = X_tilde.shape 
    
    l = n_eigs+2
    i = 2
    
    #Step 1: construct a random matrix G, with 0 mean and variance equal to 1.
    G = np.random.rand(cols, l)
    ____, X_ = center(G, "mean", True)
    ____, G = scale(X_, "auto", True)   
    
    #Step 2: Compute the H matrix of eq. (2.2) from [1]. 
    #To do that, an iterative procedure must be adopted (in the for loop):
    Hmatr = [None] * i
    Hmatr[0] = X_tilde @ G

    for ii in range(1,i):
        Hmatr[ii] = X_tilde @ (X_tilde.T @ Hmatr[ii-1])

            
    H = Hmatr[0]
    for ii in range(1, i):
        H = np.concatenate((H,Hmatr[ii]),axis=1)
            
            
    #Step 3: decompose H with a QR decomposition, the column of the matrix
    #Q must be orthonormal
    Q, ____ = np.linalg.qr(H, mode='reduced')


    #Step 4: compute the T matrix as described in eq. (2.4) of [1]
    T = X_tilde.T @ Q 

    #Step 5: decompose the T matrix, and retrieve the scores matrix (Z) and
    #the modes matrix (A)
    V_tilde, Sig_tilde, Wt = np.linalg.svd(T, full_matrices=True)
    U_tilde = Q @ Wt.T 

    U = U_tilde[:, :n_eigs]         #modes
    V = V_tilde[:, :n_eigs]
    Sigma = Sig_tilde[:n_eigs]
    
    return U, V, Sigma



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
    try:
        #identify the observations which belong to a certain cluster
        positions = np.where(idx == index)
        cluster = X[positions]
        
        if write:
            np.savetxt("Observations in cluster number{}.txt".format(index), cluster)

        return cluster
    
    except:
        print("No observations in cluster number: {}. Passing by.".format(index))
        pass

    
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
        #identify the observations which belong to a certain cluster
        #and store them in the clusters' list
        clusters[ii] = get_cluster(X, idx, ii)

    return clusters


def NRMSE (X_true, X_pred):
    n_obs, n_var = X_true.shape
    #initialize the NRMSE 
    NRMSE = [None] *n_var

    #compute the NRMSE for each observation of the two matrices
    for ii in range(0, n_var):
        NRMSE[ii] = np.sqrt(np.mean((X_true[:,ii] - X_pred[:,ii])**2)) / np.mean(X_true[:,ii])

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
        #compute the covariance matrix from the original data matrix
        C = np.cov(X, rowvar=False) #rowvar=False because the X matrix is (observations x variables)

        #perform an eigendecomposition
        evals, evecs = LA.eig(C)

        #sort the eigenvalues in descending order
        mask = np.argsort(evals)[::-1]
        evecs = evecs[:,mask]
        evals = evals[mask]

        #retain only the selected number 'q' of eigenvalues
        evecs = evecs[:, :n_eig]

        return evecs, evals

    else:
        raise Exception("The number of PCs exceeds the number of variables in the data-set.")


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
            #compute the auto scaling factor: standard deviation of each variable
            sig = np.std(X, axis = 0)
        elif method.lower() == 'pareto':
            #compute the pareto scaling factor: sqrt of the standard deviation of each variable
            sig = np.sqrt(np.std(X, axis = 0))
        elif method.lower() == 'vast':
            #compute the vast scaling factor: variance/mean of each variable
            variances = np.var(X, axis = 0)
            means = np.mean(X, axis = 0)
            sig = variances / means
        elif method.lower() == 'range':
            #compute the range scaling factor: max-min for each variable
            maxima = np.max(X, axis = 0)
            minima = np.min(X, axis = 0)
            sig = maxima - minima
        else:
            raise Exception("Unsupported scaling option. Please choose: AUTO, PARETO, VAST or RANGE.")
        return sig
    else:
        if method.lower() == 'auto':
            #compute the auto scaling factor: standard deviation of each variable
            sig = np.std(X, axis = 0)
            X0 = X / (sig + TOL)
        elif method.lower() == 'pareto':
            #compute the pareto scaling factor: sqrt of the standard deviation of each variable
            sig = np.sqrt(np.std(X, axis = 0))
            X0 = X / (sig + TOL)
        elif method.lower() == 'vast':
            #compute the vast scaling factor: variance/mean of each variable
            variances = np.var(X, axis = 0)
            means = np.mean(X, axis = 0)
            sig = variances / means
            X0 = X / (sig + TOL)
        elif method.lower() == 'range':
            #compute the range scaling factor: max-min for each variable
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

    #compute the number of variables and observations of the training matrix
    nObs = X.shape[0]
    nVar = X.shape[1]

    #compute how many observations must be included in the test
    nTest = int(nObs * validation_quota)

    #shuffle the training matrix to maximize the randomness of the subset
    np.random.shuffle(X)

    #split
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
        #initialize the uncentered matrix with all zeros
        X0 = np.zeros_like(X_tilde, dtype=float)
        for i in range(0, len(mu)):
            #add the centering factor to unscale
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
        #initialize the uncentered matrix with all zeros
        X0 = np.zeros_like(X_tilde, dtype=float)
        for i in range(0, len(sigma)):
            #multiply for the scaling factor to unscale
            X0[:,i] = X_tilde[:,i] * (sigma[i] + TOL)
        return X0
    else:
        raise Exception("The matrix to be unscaled and the scaling vector must have the same dimensionality.")
        exit()


def varimax_rotation(X, b):
    '''
    Rotate the factors by means of the varimax rotation.
    - Input:
    X = training data matrix, SCALED/UNSCALED it makes no difference
    b = factors/modes to be rotated

    - Output:
    rot_loadings = rotated modes or factors, returned after the algorithm convergenceß
    '''
    import math
    #compute the dimensionality of the problem
    eigens = b.shape[1]
    
    #compute the normalization factor for the modes/factors before the rotation
    norm_factor = np.std(b, axis=0)
    
    #initialize an empty matrix for the rotated loadings
    loadings = np.empty((b.shape[0], b.shape[1]), dtype=float)
    rot_loadings = np.empty((b.shape[0], b.shape[1]), dtype=float)

    #standardize the loadings dividing for the normalization factor
    for ii in range(0, eigens):
        loadings[:,ii] = b[:,ii]/norm_factor[ii]

    #compute the loadings covariance matrix
    C = np.cov(loadings, rowvar=False)

    #initialize the algorithm's parameter to reach convergence
    iter_max = 1000
    convergence_tolerance = 1E-16
    iter = 1
    convergence = False

    #compute the percentage of explained variance
    variance_explained = np.sum(loadings**2)

    while not convergence:
        for ii in range(0,eigens):
            for jj in range(ii+1, eigens):
                x_j = loadings[:,ii]
                y_j = loadings[:,jj]

                u_j = np.reshape(np.array(np.multiply(x_j,x_j) - np.multiply(y_j,y_j)), (b.shape[0],1)) #nvarx1
                v_j = np.reshape(np.array(2*np.multiply(x_j, y_j)), (b.shape[0],1)) #nvarx1

                A = np.sum(u_j)
                B = np.sum(v_j)
                C = u_j.T @ u_j - v_j.T @ v_j
                D = 2 * u_j.T @ v_j

                num = D - 3*A*B/X.shape[0]
                den = C - (A**2 - B**2)/X.shape[0]

                phi = np.arctan2(num, den)/4
                angle = phi *180/math.pi 

                if np.abs(phi) > 0.00001:
                    Z_j = np.cos(phi) *x_j + np.sin(phi) *y_j
                    Y_j = -np.sin(phi) *x_j + np.cos(phi) *y_j

                    loadings[:,ii] = Z_j
                    loadings[:,jj] = Y_j

        var_old = variance_explained
        variance_explained = np.sum(loadings**2)

        #convergence criterion
        check = np.abs((variance_explained - var_old)/(variance_explained + 1E-16))

        if check < convergence_tolerance or iter > iter_max:
            iter += 1
            convergence = True
        else:
            iter += 1

        print("Iteration number: {}".format(iter))
        print("Convergence residuals: {}".format(check))


    for ii in range(0, eigens):
        rot_loadings[:,ii] = loadings[:,ii] * norm_factor[ii]

    return rot_loadings