'''
MODULE: clustering.py

@Authors: 
    G. D'Alessio [1,2], G. Aversano [1], A. Parente[1]
    [1]: Université Libre de Bruxelles, Aero-Thermo-Mechanics Laboratory, Bruxelles, Belgium
    [2]: CRECK Modeling Lab, Department of Chemistry, Materials and Chemical Engineering, Politecnico di Milano

@Contacts:
    giuseppe.dalessio@ulb.ac.be

@Brief: 
    Class lpca: Clustering via Local Principal Component Analysis (LPCA).
    Class VQclassifier: Classify new observations via LPCA on the basis of a previous clustering solution.

@Details: 
    ***** CLUSTERING *****
    The iterative Local Principal Component Analysis clustering algorithm is based on the following steps:
    0. Preprocessing: The training matrix X is centered and scaled, after being loaded. Four scaling are available,
    AUTO, VAST, PARETO, RANGE - Two centering are available, MEAN and MIN;
    1. Initialization: The cluster centroids are initializated: a random allocation (RANDOM)
    or a previous clustering solution (KMEANS) can be chosen to compute the centroids initial values; 
    2. Partition: Each observation is assigned to a cluster k such that the local reconstruction
    error is minimized;
    3. PCA: The Principal Component Analysis is performed in each of the clusters found
    in the previous step. A new set of centroids is computed after the new partitioning
    step, their coordinates are calculated as the mean of all the observations in each
    cluster;
    4. Iteration: All the previous steps are iterated until convergence is reached. The convergence
    criterion is that the variation of the global mean reconstruction error between two consecutive
    iterations must be below a fixed threshold.

    ***** CLASSIFICATION *****
    For the classification task, instead, the following steps are accomplished:
    0. Preprocessing: The set of new observations Y is centered and scaled with the centering and scaling factors
    computed from the training dataset, X.
    1. For each cluster of the training matrix, computes the Principal Components.
    2. Assign each observation y \in Y to the cluster which minimizes the local reconstruction error.

@Cite:
    - Local algorithm for dimensionality reduction:
    [a] Kambhatla, Nandakishore, and Todd K. Leen. "Dimension reduction by local principal component analysis.", Neural computation 9.7 (1997): 1493-1516.

    - Clustering applications:
    [b] D’Alessio, Giuseppe, et al. "Adaptive chemistry via pre-partitioning of composition space and mechanism reduction.", Combustion and Flame 211 (2020): 68-82.

    - Data analysis applications:
    [c] Parente, Alessandro, et al. "Investigation of the MILD combustion regime via principal component analysis." Proceedings of the Combustion Institute 33.2 (2011): 3333-3341.
    [d] D'Alessio, Giuseppe, et al. "Analysis of turbulent reacting jets via Principal Component Analysis", Data Analysis in Direct Numerical Simulation of Turbulent Combustion, Springer (2020).
    [e] Bellemans, Aurélie, et al. "Feature extraction and reduced-order modelling of nitrogen plasma models using principal component analysis." Computers & chemical engineering 115 (2018): 504-514.

    - Preprocessing effects on PCA:
    [f] Parente, Alessandro, and James C. Sutherland. "Principal component analysis of turbulent combustion data: Data pre-processing and manifold sensitivity." Combustion and flame 160.2 (2013): 340-350.

    - Model order reduction:
    [g] Parente, Alessandro, et al. "Identification of low-dimensional manifolds in turbulent flames." Proceedings of the Combustion Institute. 2009 Jan 1;32(1):1579-86.
    [h] Aversano, Gianmarco, et al. "Application of reduced-order models based on PCA & Kriging for the development of digital twins of reacting flow applications." Computers & chemical engineering 121 (2019): 422-441.

@Additional notes:
    This cose is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; 
    Please report any bug to: giuseppe.dalessio@ulb.ac.be

'''


from operations import *
import numpy as np
from sklearn.cluster import KMeans
import matplotlib
import matplotlib.pyplot as plt

class lpca:
    def __init__(self, X, k, n_eigs, method):
        self.X = np.array(X)
        self.k = k
        self.nPCs = n_eigs
        self.method = method
    

    @staticmethod
    def initialize_clusters(X, k, method):
        '''
        The clustering solution must be initialized. Two methods are available,
        a random allocation (RANDOM) or a previous clustering solution (KMEANS).
        '''
        if method == 'RANDOM' or method == 'random' or method == 'Random':
            idx = np.random.random_integers(1, k, size=(X.shape[0], 1))
        elif method == 'KMEANS' or method == 'kmeans' or method == 'Kmeans':
            kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
            idx = kmeans.labels_
        else:
            raise Exception("Initialization option not supported. Please choose one between RANDOM or KMEANS.")
        return idx
    

    @staticmethod
    def initialize_parameters():
        iteration = 0
        eps_rec = 1.0
        residuals = np.array(0)
        iter_max = 500
        eps_tol = 1E-16
        return iteration, eps_rec, residuals, iter_max, eps_tol
    

    @staticmethod
    def merge_clusters(X, idx):
        '''
        Remove a cluster if it is empty, or not statistically meaningful.
        '''
        for jj in range(1, max(idx)):
            cluster_ = get_cluster(X, idx, jj)
            if cluster_.shape[0] < 2:
                pos = np.where(idx != 0)
                idx[pos] -= 1
                print("WARNING:")
                print("\tAn empty cluster was found:")
                print("\tThe number of cluster was lowered to ensure statistically meaningful results.")
                print("\tThe current number of clusters is equal to: {}".format(max(idx)))
                break
        return idx


    @staticmethod
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


    def fit(self):
        '''
        Group the observations depending on the PCA reconstruction error.
        '''
        print("Fitting Local PCA model...")
        # Initialization
        iteration, eps_rec, residuals, iter_max, eps_tol = lpca.initialize_parameters()
        rows, cols = np.shape(self.X)
        # Initialize solution
        idx = lpca.initialize_clusters(self.X, self.k, self.method)
        residuals = np.array(0)
        # Iterate
        while(iteration < iter_max):
            sq_rec_oss = np.zeros((rows, cols), dtype=float)
            sq_rec_err = np.zeros((rows, self.k), dtype=float)
            for ii in range(0, self.k):
                cluster = get_cluster(self.X, idx, ii)
                centroids = get_centroids(cluster)
                modes = PCA_fit(cluster, self.nPCs)
                C_mat = np.matlib.repmat(centroids, rows, 1)
                rec_err_os = (self.X - C_mat) - (self.X - C_mat) @ modes[0] @ modes[0].T
                sq_rec_oss = np.power(rec_err_os, 2)
                sq_rec_err[:,ii] = sq_rec_oss.sum(axis=1)
            # Update idx
            idx = np.argmin(sq_rec_err, axis = 1)
            # Update convergence
            rec_err_min = np.min(sq_rec_err, axis = 1)
            eps_rec_new = np.mean(rec_err_min, axis = 0)
            eps_rec_var = np.abs((eps_rec_new - eps_rec) / (eps_rec_new) + eps_tol)
            eps_rec = eps_rec_new
            # Print info
            print("- Iteration number: {}".format(iteration+1))
            print("\tReconstruction error: {}".format(eps_rec_new))
            print("\tReconstruction error variance: {}".format(eps_rec_var))
            # Check condition
            if (eps_rec_var <= eps_tol):
                break
            else:
                residuals = np.append(residuals, eps_rec_new)
            # Update counter
            iteration += 1
            # Consider only statistical meaningful groups of points
            idx = lpca.merge_clusters(self.X, idx)
            self.k = max(idx)+1
        print("Convergence reached in {} iterations.".format(iteration))
        lpca.plot_residuals(iteration, residuals)
        return idx


class VQclassifier:
    def __init__(self, X, cent_crit, scal_crit, idx, Y):
        self.X = X
        self.cent_crit = cent_crit
        self.scal_crit = scal_crit
        self.idx = idx
        self.k = max(self.idx)
        self.Y = Y
        self.nPCs = round(self.Y.shape[1] - (self.Y.shape[1]) /5) #Use the max number of PCs, remove only the last 20% which contains noise
    
    def fit(self):
        '''
        Classify a new set of observations on the basis of a previous
        LPCA partitioning.
        '''
        print("Classifying the new observations...")
        # Compute the centering/scaling factors of the training matrix
        mu = center(self.X, self.cent_crit)
        sigma = scale(self.X, self.scal_crit)
        # Scale the new matrix with these factors
        Y_tilde = center_scale(self.Y, mu, sigma)
        # Initialize arrays
        rows, cols = np.shape(self.Y)
        sq_rec_oss = np.zeros((rows, cols), dtype=float)
        sq_rec_err = np.zeros((rows, self.k), dtype=float)
        # Compute the reconstruction errors
        for ii in range (0, self.k):
            cluster = get_cluster(self.X, self.idx, ii)
            centroids = get_centroids(cluster)
            modes = PCA_fit(cluster, self.nPCs)
            C_mat = np.matlib.repmat(centroids, rows, 1)
            rec_err_os = (self.Y - C_mat) - (self.Y - C_mat) @ modes[0] @ modes[0].T
            sq_rec_oss = np.power(rec_err_os, 2)
            sq_rec_err[:,ii] = sq_rec_oss.sum(axis=1)
        
        # Assign the label
        idx_classification = np.argmin(sq_rec_err, axis = 1)

        return idx_classification