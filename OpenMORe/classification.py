'''
MODULE: classification.py

@Authors:
    G. D'Alessio [1,2]
    [1]: Université Libre de Bruxelles, Aero-Thermo-Mechanics Laboratory, Bruxelles, Belgium
    [2]: CRECK Modeling Lab, Department of Chemistry, Materials and Chemical Engineering, Politecnico di Milano

@Contacts:
    giuseppe.dalessio@ulb.ac.be


@Additional notes:
    This code is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
    Please report any bug to: giuseppe.dalessio@ulb.ac.be

'''
from .utilities import *
from . import clustering


import numpy as np
import numpy.matlib


class VQPCA(clustering.lpca):
    '''
    For the classification task the following steps are accomplished:
    0. Preprocessing: The set of new observations Y is centered and scaled with the centering and scaling factors
    computed from the training dataset, X. Warning: center = mean and scaling = auto is default.
    1. For each cluster of the training matrix, compute the Principal Components.
    2. Assign each observation y \in Y to the cluster which minimizes the local reconstruction error.
    '''
    def __init__(self, X, idx, Y):
        self.X = X
        self.idx = idx
        self.Y = Y

        super().__init__(X)
        self.k = int(max(self.idx) +1)
        self.nPCs = 2 #round(self.Y.shape[1] - 1)#(self.Y.shape[1]) /10) Use a very high number of PCs to classify,removing only the last 20% which contains noise

    def check_sanity_input(self):
        if self.X.shape[0] > len(self.idx) or self.X.shape[0] < len(self.idx):
            raise Exception("The first dimension of the matrix X and the length of idx must agree.")
            print("Exiting with error..")
            exit()

        if self.X.shape[1] > self.Y.shape[1] or self.X.shape[1] < self.Y.shape[1]:
            raise Exception("The second dimension of the matrix X and the second dimension of the matrix Y must agree.")
            print("Exiting with error..")
            exit()


    def fit(self):
        '''
        Classify a new set of observations on the basis of a previous
        LPCA partitioning.
        '''
        self.check_sanity_input()
        print("Classifying the new observations...")
        # Compute the centering/scaling factors of the training matrix
        mu = center(self.X, self._centering)
        sigma = scale(self.X, self._scaling)
        
        # Scale the new matrix with these factors
        Y_tilde = center_scale(self.Y, mu, sigma)
        
        # Initialize arrays
        rows, cols = np.shape(self.Y)
        sq_rec_oss = np.empty((rows, cols), dtype=float)
        sq_rec_err = np.empty((rows, self.k), dtype=float)
        
        # Compute the reconstruction errors
        for ii in range (0, self.k):
            cluster = get_cluster(self.X, self.idx, ii)
            centroids = get_centroids(cluster)
            modes = PCA_fit(cluster, self.nPCs)
            C_mat = np.matlib.repmat(centroids, rows, 1)
            rec_err_os = (Y_tilde - C_mat) - (Y_tilde - C_mat) @ modes[0] @ modes[0].T
            sq_rec_oss = np.power(rec_err_os, 2)
            sq_rec_err[:,ii] = sq_rec_oss.sum(axis=1)

        # Assign the label
        idx_classification = np.argmin(sq_rec_err, axis = 1)

        return idx_classification  