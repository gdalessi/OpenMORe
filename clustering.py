from initialization import *
from operations import *
import numpy as np

class lpca:
    def __init__(self, X, k, n_eigs, initialization):
        self.X = np.array(X)
        self.k = k
        self.nPCs = n_eigs
        self.method = initialization

    def fit(self):
        print("Fitting Local PCA model...")
        # Initialization
        convergence = 0
        iteration = 0
        eps_rec = 1.0
        rows, cols = np.shape(self.X)
        residuals = np.array(0)
        iter_max = 500
        eps_tol = 1E-8
        # Data pre-processing
        
        # Initialize solution
        idx = initialize_clusters(self.X, self.method, self.k)
        # Iterate
        while(iteration < iter_max):
            sq_rec_oss = np.zeros((rows, cols), dtype=float)
            sq_rec_err = np.zeros((rows, self.k), dtype=float)
            for ii in range(0, self.k):
                cluster = get_cluster(self.X, idx, ii)
                centroids = get_centroids(cluster)
                modes = fitPCA(cluster, self.nPCs)
                C_mat = np.matlib.repmat(centroids, rows, 1)
                rec_err_os = (self.X - C_mat) - (self.X - C_mat) @ np.transpose(modes) @ modes 
                sq_rec_oss = np.power(rec_err_os, 2)
                sq_rec_err[:,ii] = sq_rec_oss.sum(axis=1)
            # Update idx
            idx = np.argmin(sq_rec_err, axis = 1)
            # Update convergence
            rec_err_min = np.min(sq_rec_err, axis = 1)
            eps_rec_new = np.mean(rec_err_min, axis = 0)
            eps_rec_var = np.abs((eps_rec_new - eps_rec) / (eps_rec_new))
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
        print("Convergence reached in {} iterations.".format(iteration))
        return idx
        
    
