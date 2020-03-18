'''
MODULE: model_order_reduction.py

@Authors: 
    G. D'Alessio [1,2]
    [1]: Université Libre de Bruxelles, Aero-Thermo-Mechanics Laboratory, Bruxelles, Belgium
    [2]: CRECK Modeling Lab, Department of Chemistry, Materials and Chemical Engineering, Politecnico di Milano

@Contacts:
    giuseppe.dalessio@ulb.ac.be

@Brief: 
    Class PCA: PCA-based (linear method) functions to get reduced-order models.
    Class LPCA: LPCA-based (piecewise-linear method) functions to get ROMs. 
    Class KPCA: Kernel-based PCA (non-linear method) function to get ROM.
    
    More detailed descriptions are available under each function's declaration.

@Additional notes:
    This code is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; 
    Please report any bug to: giuseppe.dalessio@ulb.ac.be

'''

import numpy as np
from numpy import linalg as LA
import matplotlib
import matplotlib.pyplot as plt

from utilities import *

class PCA:
    def __init__(self, X):
        #Useful variables from training dataset
        self.X = X
        self.n_obs = X.shape[0]
        self.n_var = X.shape[1]

        #Decide if the input matrix must be centered:
        self._center = True
        #Set the centering method:
        self._centering = 'mean'                                                                    #'mean' or 'min' are available
        #Decide if the input matrix must be scaled:
        self._scale = True          
        #Set the scaling method:
        self._scaling = 'auto'                                                                      #'auto';'vast';'range';'pareto' are available
        #Enable the plotting property to show the explained variance
        self._plot_explained_variance = True
        #Automatically assess the number of PCs to retain with one of the available methods
        self._assessPCs = 'var'                                                                     #False (bool) = skip; 'var' = explained variance criterion; 'nrmse' = average reconstruction error
        self._threshold_var = 0.99                                                                  # threshold in case of explained variance assessment criterion
        self._threshold_nrmse = 0.1                                                                 # treshold in case of reconstruction error criterion
        #Set the PC number to plot or the variable's number to plot
        self._num_to_plot = 1
        #Initialize the number of PCs
        self._nPCs = X.shape[1]

    @property
    def eigens(self):
        return self._nPCs
    
    @eigens.setter
    def eigens(self, new_value):
        self._nPCs = new_value

        if self._nPCs <= 0:
            raise Exception("The number of Principal Components must be a positive integer. Exiting..")
            exit()
        elif isinstance(self._nPCs, int) != True: 
            raise Exception("The number of Principal Components must be an integer. Exiting..")
            exit()
        elif self._nPCs >= self.n_var:
            raise Exception("The number of PCs exceeds (or is equal to) the number of variables in the data-set. Exiting..")
            exit()

    @property
    def to_center(self):
        return self._center
    
    @to_center.setter
    def to_center(self, new_bool):
        self._center = new_bool

        if isinstance(self._center, bool) != True: 
            raise Exception("The instruction to center (or not) must be a boolean . Exiting..")
            exit()

    @property
    def centering(self):
        return self._centering
    
    @centering.setter
    def centering(self, new_string):
        self._centering = new_string

        if self._centering != 'mean' and self._centering != 'min':
            raise Exception("Unsupported centering option. Supported options: 'mean' or 'min'. ")

    
    @property
    def to_scale(self):
        return self._scale
    
    @to_scale.setter
    def to_scale(self, new_bool):
        self._scale = new_bool

        if isinstance(self._scale, bool) != True: 
            raise Exception("The instruction to center (or not) must be a boolean . Exiting..")
            exit()

    @property
    def scaling(self):
        return self._scaling
    
    @scaling.setter
    def scaling(self, new_string):
        self._scaling = new_string

        if self._scaling != 'auto' and self.scaling != 'pareto' and self.scaling != 'range' and self.scaling != 'vast':
            raise Exception("Unsupported scaling option. Supported options: 'auto', 'vast', 'pareto' or 'range'. ")

    @property
    def plot_explained_variance(self):
        return self._plot_explained_variance
    
    @plot_explained_variance.setter
    def plot_explained_variance(self, new_bool):
        self._plot_explained_variance = new_bool

        if isinstance(self._scale, bool) != True: 
            raise Exception("The instruction to plot (or not) must be a boolean . Exiting..")
            exit()
    
    @property
    def set_explained_variance_perc(self):
        return self._explained_variance
    
    @set_explained_variance_perc.setter
    def set_explained_variance_perc(self, new_value):
        self._explained_variance = new_value

        if isinstance(self._nPCs, int) != True and isinstance(self._nPCs, float) != True: 
            raise Exception("The percentage of variance to be explained must be a number. Exiting..")
            exit()

    @property
    def set_PCs_method(self):
        return self._assessPCs
    
    @set_PCs_method.setter
    def set_PCs_method(self, new_method):
        self._assessPCs = new_method

    @property
    def set_num_to_plot(self):
        return self._assessPCs
    
    @set_num_to_plot.setter
    def set_num_to_plot(self, new_number):
        self._assessPCs = new_number

        if isinstance(self._assessPCs, int) != True: 
            raise Exception("The number of PC or the variable to plot must be an integer. Exiting..")
            exit()

    
    @staticmethod
    def preprocess_training(X, centering_decision, scaling_decision, centering_method, scaling_method):

        if centering_decision and scaling_decision:
            mu, X_ = center(X, centering_method, True)
            sigma, X_tilde = scale(X_, scaling_method, True)
        elif centering_decision and not scaling_decision:
            mu, X_tilde = center(X, centering_method, True)
        elif scaling_decision and not centering_decision:
            sigma, X_tilde = scale(X, scaling_method, True)
        else:
            X_tilde = X

        return X_tilde

    
    def fit(self):
        '''
        Perform Principal Component Analysis on the dataset X, 
        and retain 'n' Principal Components. The covariance matrix 
        is firstly calculated, then it is decomposed in eigenvalues 
        and eigenvectors. Lastly, the eigenvalues are ordered depending 
        on their magnitude and the associated eigenvectors (the PCs) 
        are retained.
       
        - Output:
        evecs: eigenvectors from the covariance matrix decomposition (PCs)
        evals: eigenvalues from the covariance matrix decomposition (lambda)

        !!! WARNING !!! the PCs are already ordered (decreasing, for importance)
        because the eigenvalues are also ordered in terms of magnitude.
        '''
        #Center and scale the original training dataset
        self.X_tilde = PCA.preprocess_training(self.X, self._center, self._scale, self._centering, self._scaling)

        #Compute the covariance matrix
        C = np.cov(self.X_tilde, rowvar=False) #rowvar=False because the X matrix is (observations x variables)

        evals, evecs = LA.eig(C)
        mask = np.argsort(evals)[::-1]
        self.evecs = evecs[:,mask]
        self.evals = evals[mask]

        self.evecs = self.evecs[:, 0:self._nPCs]

        return self.evecs, self.evals

    
    def recover(self):
        '''
        Reconstruct the original matrix from the reduced PCA-manifold.

        - Output:
        X_rec = uncentered and unscaled reconstructed matrix with PCA modes -- dim: (observations x variables)
        '''
       
        #Compute the centering and the scaling factors. Later they will be useful.
        self.mu = center(self.X, self.centering)
        self.sigma = scale(self.X, self.scaling)
        #Reconstruct the original matrix
        self.X_r = self.X_tilde @ self.evecs @ self.evecs.T 
        # Uncenter and unscale to get the reconstructed original matrix
        X_unsc = unscale(self.X_r, self.sigma)
        self.X_rec = uncenter(X_unsc, self.mu)
        
        return self.X_rec

    
    def get_explained(self):
        '''
        Assess the variance explained by the first 'n_eigs' retained
        Principal Components. This is important to know if the percentage
        of explained variance is enough, or additional PCs must be retained.
        Usually, it is considered accepted a percentage of explained variable
        above 95%.
        
        - Output:
        explained: percentage of explained variance -- dim: (scalar)
        '''
        explained_variance = np.cumsum(self.evals)/sum(self.evals)
        explained = explained_variance[self.eigens]
        #If the plot boolean is True, produce an image to show the explained variance curve.
        if self._plot_explained_variance:
            matplotlib.rcParams.update({'font.size' : 18, 'text.usetex' : True})
            fig = plt.figure()
            axes = fig.add_axes([0.15,0.15,0.7,0.7], frameon=True)
            axes.plot(np.linspace(1, self.X_tilde.shape[1]+1, self.X_tilde.shape[1]), explained_variance, color='b', marker='s', linestyle='-', linewidth=2, markersize=4, markerfacecolor='b', label='Cumulative explained')
            axes.plot([self.eigens, self.eigens], [explained_variance[0], explained_variance[self.eigens]], color='r', marker='s', linestyle='-', linewidth=2, markersize=4, markerfacecolor='r', label='Explained by {} PCs'.format(self.eigens))
            axes.set_xlabel('Number of PCs [-]')
            axes.set_ylabel('Explained variance [-]')
            axes.set_title('Variance explained by {} PCs: {}'.format(self.eigens, round(explained,3)))
            axes.legend()
        plt.show()

        return explained

    
    def get_scores(self):
        '''
        Project the original data matrix (cent and scaled) X_tilde
        on the low dimensional manifold spanned by the first 'n'
        Principal Components and obtain the scores matrix (Z).
        '''
        self.scores = self.X_tilde @ self.evecs

        return self.scores

    
    def set_PCs(self):
        if self._assessPCs != False:
            self.plot_explained_variance = False
            for ii in range(1, self.n_var):
                print("Assessing the optimal number of PCs by means of: " + self._assessPCs + " criterion. Now using: {} Principal Component(s).".format(ii))
                
                self.eigens = ii
                self.fit()
                #Assess the optimal number of PCs by means of the explained variance threshold (>= 99% explained is the default setting)
                if self._assessPCs == 'var':
                    explained = self.get_explained()

                    if explained >= self._threshold_var:
                        print("With {} PCs, the following percentage of variance is explained: {}".format(self.eigens, explained))
                        print("The variance is larger than the given fixed threshold: {}. Thus, {} PCs will be retained.".format(self._threshold_var, self.eigens))
                        self.plot_explained_variance = True
                        break
                #Otherwise with the nrmse of the reconstructed matrix, which has to be below a specific threshold (<= 10% error is the default setting)
                elif self._assessPCs == 'nrmse':
                    reconstructed_ = self.recover()
                    variables_reconstruction = NRMSE(self.X, reconstructed_)
                    
                    if np.mean(variables_reconstruction) <= self._threshold_nrmse:
                        print("With {} PCs, the following average error (NRMSE) for the variables reconstruction (from the PCA manifold) is obtained: {}".format(self.eigens, np.mean(variables_reconstruction)))
                        print("The error is lower than the given fixed threshold: {}. Thus, {} PCs will be retained.".format(self._threshold_nrmse, self.eigens))
                        break

    
    def plot_PCs(self):
        '''
        Plot the variables' weights on a selected Principal Component (PC). 
        The PCs are linear combination of the original variables: examining the 
        weights, especially for the PCs associated with the largest eigenvalues,
        can be important for data-analysis purposes.

        '''

        matplotlib.rcParams.update({'font.size' : 18, 'text.usetex' : True})
        fig = plt.figure()
        axes = fig.add_axes([0.15,0.15,0.7,0.7], frameon=True)
        
        x = np.linspace(1, self.n_var, self.n_var)
        axes.bar(x, self.evecs[:,self._num_to_plot])
        axes.set_xlabel('Variables [-]')
        axes.set_ylabel('Weights on the PC number: {} [-]'.format(self._num_to_plot))
        plt.show()


    def plot_parity(self):
        '''
        Print the parity plot of the reconstructed profile from the PCA 
        manifold. The more the scatter plot (black dots) is in line with the
        red line, the better it is the reconstruction.

        '''

        self.fit()
        reconstructed_ = self.recover()

        matplotlib.rcParams.update({'font.size' : 18, 'text.usetex' : True})
        fig = plt.figure()
        axes = fig.add_axes([0.15,0.15,0.7,0.7], frameon=True)
        axes.plot(self.X[:,self._num_to_plot], self.X[:,self._num_to_plot], color='r', linestyle='-', linewidth=2, markerfacecolor='b')
        axes.scatter(self.X[:,self._num_to_plot], reconstructed_[:,self._num_to_plot], 1, color= 'k')
        axes.set_xlabel('Original variable')
        axes.set_ylabel('Reconstructed from PCA manifold')
        plt.xlim(min(self.X[:,self._num_to_plot]), max(self.X[:,self._num_to_plot]))
        plt.ylim(min(self.X[:,self._num_to_plot]), max(self.X[:,self._num_to_plot]))
        #axes.set_title('Parity plot')
        plt.show()


class LPCA:
    def __init__(self, X):
        self.X = X

        self.n_obs = X.shape[0]
        self.n_var = X.shape[1]
        #Set the number of PCs - to be modified later, obviously
        self._nPCs = X.shape[1]

        #Decide if the input matrix must be centered:
        self._center = True
        #Set the centering method:
        self._centering = 'mean'                                                                    #'mean' or 'min' are available
        #Decide if the input matrix must be scaled:
        self._scale = True          
        #Set the scaling method:
        self._scaling = 'auto'                                                                      #'auto';'vast';'range';'pareto' are available
        #Set the path where the file 'idx.txt' (containing the partitioning solution) is located
        self._path_to_idx = 'path'
        #Set the PC number to plot or the variable's number to plot
        self._num_to_plot = 1
        #Set the cluster number where plot the PC
        self._clust_to_plot = 1

    
    @property
    def eigens(self):
        return self._nPCs
    
    @eigens.setter
    def eigens(self, new_value):
        self._nPCs = new_value

        if self._nPCs <= 0:
            raise Exception("The number of Principal Components must be a positive integer. Exiting..")
            exit()
        elif isinstance(self._nPCs, int) != True: 
            raise Exception("The number of Principal Components must be an integer. Exiting..")
            exit()
        elif self._nPCs >= self.n_var:
            raise Exception("The number of PCs exceeds (or is equal to) the number of variables in the data-set. Exiting..")
            exit()

    @property
    def to_center(self):
        return self._center
    
    @to_center.setter
    def to_center(self, new_bool):
        self._center = new_bool

        if isinstance(self._center, bool) != True: 
            raise Exception("The instruction to center (or not) must be a boolean . Exiting..")
            exit()

    @property
    def centering(self):
        return self._centering
    
    @centering.setter
    def centering(self, new_string):
        self._centering = new_string

        if self._centering != 'mean' and self._centering != 'min':
            raise Exception("Unsupported centering option. Supported options: 'mean' or 'min'. ")

    
    @property
    def to_scale(self):
        return self._scale
    
    @to_scale.setter
    def to_scale(self, new_bool):
        self._scale = new_bool

        if isinstance(self._scale, bool) != True: 
            raise Exception("The instruction to center (or not) must be a boolean . Exiting..")
            exit()

    @property
    def scaling(self):
        return self._scaling
    
    @scaling.setter
    def scaling(self, new_string):
        self._scaling = new_string

        if self._scaling != 'auto' and self.scaling != 'pareto' and self.scaling != 'range' and self.scaling != 'vast':
            raise Exception("Unsupported scaling option. Supported options: 'auto', 'vast', 'pareto' or 'range'. ")

    
    @property
    def path_to_idx(self):
        return self._path_to_idx
    
    @path_to_idx.setter
    def path_to_idx(self, new_string):
        self._path_to_idx = new_string     


    @property
    def set_num_to_plot(self):
        return self._num_to_plot
    
    @set_num_to_plot.setter
    def set_num_to_plot(self, new_number):
        self._num_to_plot = new_number

        if isinstance(self._num_to_plot, int) != True: 
            raise Exception("The number of PC or the variable to plot must be an integer. Exiting..")
            exit()

    
    @property
    def clust_to_plot(self):
        return self._clust_to_plot
    
    @clust_to_plot.setter
    def clust_to_plot(self, new_number):
        self._clust_to_plot = new_number

        if isinstance(self._clust_to_plot, int) != True: 
            raise Exception("The number of PC or the variable to plot must be an integer. Exiting..")
            exit()


    @staticmethod
    def get_idx(path):

        try:
            print("Reading idx..")
            idx = np.genfromtxt(path + '/idx.txt', delimiter= ',')
        except OSError:
            print("Could not open/read the selected file: " + path + 'idx.txt')
            exit()
        
        return idx

    
    @staticmethod
    def preprocess_training(X, centering_decision, scaling_decision, centering_method, scaling_method):

        if centering_decision and scaling_decision:
            mu, X_ = center(X, centering_method, True)
            sigma, X_tilde = scale(X_, scaling_method, True)
        elif centering_decision and not scaling_decision:
            mu, X_tilde = center(X, centering_method, True)
        elif scaling_decision and not centering_decision:
            sigma, X_tilde = scale(X, scaling_method, True)
        else:
            X_tilde = X

        return X_tilde


    def fit(self):
        '''
        This function computes the LPCs (Local Principal Components), the u_scores
        (the projection of the points on the local manifold), and the eigenvalues
        in each cluster found by the lpca iterative algorithm, given a previous clustering
        solution.
        
        - Output:
        LPCs = list with the LPCs in each cluster -- dim: [k]
        u_scores = list with the scores in each cluster -- dim: [k]
        Leigen = list with the eigenvalues in each cluster -- dim: [k]
        centroids = list with the centroids in each cluster -- dim: [k]
        '''
        self.idx = LPCA.get_idx(self.path_to_idx)
        self.k = int(max(self.idx) +1)
        self.X_tilde = LPCA.preprocess_training(self.X, self.to_center, self.to_scale, self.centering, self.scaling)
        
        
        self.centroids = [None] *self.k
        self.LPCs = [None] *self.k
        self.u_scores = [None] *self.k
        self.Leigen = [None] *self.k

        for ii in range (0,self.k):
            cluster = get_cluster(self.X_tilde, self.idx, ii)
            self.centroids[ii], cluster_ = center(cluster, self._centering, True)
            self.LPCs[ii], self.Leigen[ii] = PCA_fit(cluster_, self._nPCs)
            self.u_scores[ii] = cluster_ @ self.LPCs[ii]

        return self.LPCs, self.u_scores, self.Leigen, self.centroids


    def recover(self):
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

        
        self.X_rec = np.empty(self.X.shape, dtype=float)
        self.X_tilde = LPCA.preprocess_training(self.X, self.to_center, self.to_scale, self.centering, self.scaling)

        mu_global = center(self.X, self.centering)
        sigma_global = scale(self.X, self.scaling)
        

        self.LPCs, self.u_scores, self.Leigen, self.centroids = self.fit()


        for ii in range (0,self.k):
            cluster_ = get_cluster(self.X_tilde, self.idx, ii)
            centroid_ =self.centroids[ii]
            C = np.empty(cluster_.shape, dtype=float) 
            C = (cluster_ - centroid_) @ self.LPCs[ii] @ self.LPCs[ii].T
            C_ = uncenter(C, centroid_)
            positions = np.where(self.idx == ii)
            self.X_rec[positions] = C_
        
        self.X_rec = unscale(self.X_rec, sigma_global)
        self.X_rec = uncenter(self.X_rec, mu_global)

        return self.X_rec


    def plot_parity(self):
        '''
        Print the parity plot of the reconstructed profile from the PCA 
        manifold. The more the scatter plot (black dots) is in line with the
        red line, the better it is the reconstruction.

        '''


        reconstructed_ = self.recover()

        matplotlib.rcParams.update({'font.size' : 18, 'text.usetex' : True})
        fig = plt.figure()
        axes = fig.add_axes([0.15,0.15,0.7,0.7], frameon=True)
        axes.plot(self.X[:,self._num_to_plot], self.X[:,self._num_to_plot], color='r', linestyle='-', linewidth=2, markerfacecolor='b')
        axes.scatter(self.X[:,self._num_to_plot], reconstructed_[:,self._num_to_plot], 1, color= 'k')
        axes.set_xlabel('Original variable')
        axes.set_ylabel('Reconstructed from LPCA manifold')
        plt.xlim(min(self.X[:,self._num_to_plot]), max(self.X[:,self._num_to_plot]))
        plt.ylim(min(self.X[:,self._num_to_plot]), max(self.X[:,self._num_to_plot]))
        #axes.set_title('Parity plot')
        plt.show()


    def plot_PCs(self):
        '''
        Plot the variables' weights on a selected Principal Component (PC). 
        The PCs are linear combination of the original variables: examining the 
        weights, especially for the PCs associated with the largest eigenvalues,
        can be important for data-analysis purposes.
        '''
        local_eigen = self.LPCs[self.clust_to_plot]

        matplotlib.rcParams.update({'font.size' : 18, 'text.usetex' : True})
        fig = plt.figure()
        axes = fig.add_axes([0.15,0.15,0.7,0.7], frameon=True)
        
        x = np.linspace(1, self.n_var, self.n_var)
        axes.bar(x, local_eigen[:,self.set_num_to_plot])
        axes.set_xlabel('Variables [-]')
        axes.set_ylabel('Weights on the PC number: {}, k = {} [-]'.format(self.set_num_to_plot, self.clust_to_plot))
        plt.show()


class KPCA:
    def __init__(self, X):
        self.X = X
        self.n_obs = X.shape[0]
        self.n_var = X.shape[1]

        #Decide if the input matrix must be centered:
        self._center = True
        #Set the centering method:
        self._centering = 'mean'                                                                    #'mean' or 'min' are available
        #Decide if the input matrix must be scaled:
        self._scale = True          
        #Set the scaling method:
        self._scaling = 'auto'      
        #Set the sigma - the coefficient for the kernel construction
        self.sigma = 10
        #Initialize a number of PCs to retain
        self._nPCs = self.n_var

     
    @property
    def eigens(self):
        return self._nPCs
    
    @eigens.setter
    def eigens(self, new_value):
        self._nPCs = new_value

        if self._nPCs <= 0:
            raise Exception("The number of Principal Components must be a positive integer. Exiting..")
            exit()
        elif isinstance(self._nPCs, int) != True: 
            raise Exception("The number of Principal Components must be an integer. Exiting..")
            exit()
        elif self._nPCs >= self.n_var:
            raise Exception("The number of PCs exceeds (or is equal to) the number of variables in the data-set. Exiting..")
            exit()

    @property
    def to_center(self):
        return self._center
    
    @to_center.setter
    def to_center(self, new_bool):
        self._center = new_bool

        if isinstance(self._center, bool) != True: 
            raise Exception("The instruction to center (or not) must be a boolean . Exiting..")
            exit()

    @property
    def centering(self):
        return self._centering
    
    @centering.setter
    def centering(self, new_string):
        self._centering = new_string

        if self._centering != 'mean' and self._centering != 'min':
            raise Exception("Unsupported centering option. Supported options: 'mean' or 'min'. ")

    
    @property
    def to_scale(self):
        return self._scale
    
    @to_scale.setter
    def to_scale(self, new_bool):
        self._scale = new_bool

        if isinstance(self._scale, bool) != True: 
            raise Exception("The instruction to center (or not) must be a boolean . Exiting..")
            exit()

    @property
    def scaling(self):
        return self._scaling
    
    @scaling.setter
    def scaling(self, new_string):
        self._scaling = new_string

        if self._scaling != 'auto' and self.scaling != 'pareto' and self.scaling != 'range' and self.scaling != 'vast':
            raise Exception("Unsupported scaling option. Supported options: 'auto', 'vast', 'pareto' or 'range'. ")

    
    @staticmethod
    def preprocess_training(X, centering_decision, scaling_decision, centering_method, scaling_method):

        if centering_decision and scaling_decision:
            mu, X_ = center(X, centering_method, True)
            sigma, X_tilde = scale(X_, scaling_method, True)
        elif centering_decision and not scaling_decision:
            mu, X_tilde = center(X, centering_method, True)
        elif scaling_decision and not centering_decision:
            sigma, X_tilde = scale(X, scaling_method, True)
        else:
            X_tilde = X

        return X_tilde


    def fit(self):
        '''
        Compute the Kernel Principal Component Analysis for a dataset X, using
        a gaussian Radial Basis Function (RBF).
        - Input:
        X = CENTERED/SCALED data matrix -- dim: (observations x variables)
        n_eigs = number of principal components to retain -- dim: (scalar)
        sigma = free parameter for the kernel construction, optional -- dim: (scalar)

        WARNING: This method can be extremely expensive for large matrices.
        '''
        from scipy.spatial.distance import pdist, squareform

        self.X_tilde = KPCA.preprocess_training(self.X, self.to_center, self.to_scale, self.centering, self.scaling)

        print("Starting kernel computation..")
        # compute the distances between all the observations:
        distances = pdist(self.X_tilde, 'sqeuclidean')
        square_distances = squareform(distances)
        
        # build the first kernel, center it with the 1n matrix, then compute the centered kernel K:
        Kernel = np.exp(-sigma * square_distances)
        centering_k = np.ones((n_obs, n_obs))/n_obs
        K = Kernel - centering_k @ Kernel - Kernel @ centering_k + centering_k @ Kernel @ centering_k
        
        print("Starting kernel decomposition..")
        # Compute the PCs from the kernel matrix, and retain the first 'n_eigs' PCs:
        evals, evecs = LA.eigh(K)
        mask = np.argsort(evals)[::-1]
        evecs = evecs[:,mask]
        evals = evals[mask]
        evecs = evecs[:,:self.eigens]

        return evecs


class variables_selection:
    '''
    In many applications, rather than reducing the dimensionality considering a new set of coordinates 
    which are linear combination of the original ones, the main interest is to achieve a dimensionality 
    reduction selecting a subset of m variables from the original set of p variables. One of the possible 
    ways to accomplish this task is to couple the PCA dimensionality reduction with a Procustes Analysis.

    The iterative variable algorithm introduced by Krzanovski [a] is based on the following steps (1-3):
    0. Preprocessing: The training matrix X is centered and scaled, after being loaded. 
    1. The dimensionality of m is initially set equal to p.
    2. Each variable is deleted from the matrix X, obtaining p ~X matrices. The
    corresponding scores matrices are computed by means of PCA. For each of them, a Procustes Analysis 
    is performed with respect to the scores of the original matrix X, and the corresponding M2 coeffcient is computed.
    3. The variable which, once excluded, leads to the smallest M2 coefficient is deleted from the matrix.
    4. Steps 2 and 3 are repeated until m variables are left.

    @Cite:
    [a] Wojtek J Krzanowski. Journal of the Royal Statistical Society: Series C (Applied Statistics), 36(1):22{33, 1987.
    [b] Ian Jolliffe. Principal component analysis. Springer, 2011.
    
    '''
    def __init__(self, X, labels):
        self.X = X
        self.labels = labels
        #Initialize the number of variables to select and the PCs to retain
        self._n_ret = 1
        self._n_eig = 1


    @property
    def retained(self):
        return self._n_neurons
    
    @retained.setter
    def retained(self, new_number):
        self._n_ret = new_number

        if self._n_ret <= 0:
            raise Exception("The number of retained variables must be a positive integer. Exiting..")
            exit()
        elif isinstance(self._n_ret, int) != True: 
            raise Exception("The number of retained variables must be an integer. Exiting..")
            exit()

    @property
    def eigens(self):
        return self._n_eig
    
    @eigens.setter
    def eigens(self, new_number):
        self._n_eig = new_number

        if self._n_eig <= 0:
            raise Exception("The number of eigenvectors must be a positive integer. Exiting..")
            exit()
        elif isinstance(self._n_eig, int) != True: 
            raise Exception("The number of eigenvectors must be an integer. Exiting..")
            exit()

    @staticmethod
    def check_sanity_input(X, labels, retained):
        if X.shape[1] != labels.shape[1]:
            print("Variables number: {}, Labels length: {}".format(X.shape[1], labels.shape[1]))
            raise Exception("The number of variables does not match the labels.")
            exit()
        elif retained >= X.shape[1]:
            raise Exception("The number of retained variables must be lower than the number of original variables.")
            exit()

    
    def fit(self):
        print("Selecting global variables via PCA and Procustes Analysis...")
        variables_selection.check_sanity_input(self.X, self.labels, self._n_ret)

        eigenvec = PCA_fit(self.X, self._n_eig)
        Z = self.X @ eigenvec[0]
        #Start the backward elimination
        while self.X.shape[1] > self._n_ret:
            M2 = 1E12
            M2_tmp = 0
            var_tmp = 0

            for ii in range(0,self.X.shape[1]):
                X_cut = np.delete(self.X, ii, axis=1)
                eigenvec = PCA_fit(X_cut, self._n_eig)
                Z_tilde = X_cut @ eigenvec[0]

                covZZ = np.transpose(Z_tilde) @ Z

                u, s, vh = np.linalg.svd(covZZ, full_matrices=True)
                M2_tmp = np.trace((np.transpose(Z) @ Z) + (np.transpose(Z_tilde) @ Z_tilde) - 2*s)
                #If the Silhouette score is lower, store the variable 'ii' to remove it
                if M2_tmp < M2:
                    M2 = M2_tmp
                    var_tmp = ii

            self.X = np.delete(self.X, var_tmp, axis=1)
            self.labels = np.delete(self.labels, var_tmp, axis=1)
            print("Current number of variables: {}".format(self.X.shape[1]))

        return self.labels