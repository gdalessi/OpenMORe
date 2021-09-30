'''
MODULE: model_order_reduction.py

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

import numpy as np
from numpy import linalg as LA
import matplotlib
import matplotlib.pyplot as plt
import warnings
import random
import scipy.special as sp
import math

from .utilities import *
from . import clustering

class PCA:
    '''
    Perfom model order reduction via Principal Component Analysis (PCA).
    PCA is a statistical technique for the dimensionality reduction of a matrix
    X (n_observations x n_variables), by means of an eigendecomposition of 
    its covariance matrix, C. 

    The eigenvectors obtained from the decomposition are called Principal
    Components (PCs), while the eigenvalues represent the percentage of information
    they account for.


    --- PARAMETERS ---
    X:          RAW data matrix, uncentered and unscaled. It must be organized
                with the structure: (observations x variables).
    type X :    numpy array


    --- SETTERS ---
    _center:            Enable the centering function
    type   _center:     boolean True/False

    _centering:             set the centering method. Available choices for scaling
                            are 'mean' or 'min'.
    type   _centering:      string

    _scale:                Enable the scaling function
    type   _scale:         boolean 

    _scaling:               set the scaling method. Available choices for scaling
                            are 'auto' or 'vast' or 'range' or 'pareto'.
    type _scaling:          string

    _plot_explained_variance:          enable plotting for the explained variance function
    type   _plot_explained_variance:   boolean

    _assessPCs:             automatically set the number of PCs to retain. Available choices 
                            are: False (skip this option), 'var': explained variance criterion (> 95%)
                            'nrmse' set the num of PCs considering the reconstruction error, which
                            has to be < 10% on average
    type _assessPCs:        boolean or string
    '''
    def __init__(self, X, *dictionary):
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
        self._threshold_var = 0.95                                                                  # threshold in case of explained variance assessment criterion
        self._threshold_nrmse = 0.1                                                                 # treshold in case of reconstruction error criterion
        #Set the PC number to plot or the variable's number to plot
        self._num_to_plot = 1
        #Initialize the number of PCs
        self._nPCs = X.shape[1] -1

        if dictionary:
            settings = dictionary[0] 
            try:
                self._nPCs = settings["number_of_eigenvectors"]
                if self._nPCs < 0 or self._nPCs >= self.X.shape[1]:
                    raise Exception
            except:
                self._nPCs = self.X.shape[1]-1
                warnings.warn("An exception occured with regard to the input value for the number of PCs. It could be not acceptable, or not given to the dictionary.")
                print("\tIt will be automatically set equal to: X.shape[1]-1.")
                print("\tYou can ignore this warning if the number of PCs has been assigned later via setter.")
                print("\tOtherwise, please check the conditions which must be satisfied by the input in the detailed documentation.")
            try:
                self._center = settings["center"]
                if not isinstance(self._center, bool):
                    raise Exception
            except:
                self._center = True
                warnings.warn("An exception occured with regard to the input value for the centering decision. It could be not acceptable, or not given to the dictionary.")
                print("\tIt will be automatically set equal to: true.")
                print("\tYou can ignore this warning if the centering decision has been assigned later via setter.")
                print("\tOtherwise, please check the conditions which must be satisfied by the input in the detailed documentation.")
            try:
                self._centering = settings["centering_method"]
                if not isinstance(self._centering, str):
                    raise Exception
                elif self._centering.lower() != "mean" and self._centering.lower() != "min":
                    raise Exception
            except:
                self._centering = "mean"
                warnings.warn("An exception occured with regard to the input value for the centering criterion . It could be not acceptable, or not given to the dictionary.")
                print("\tIt will be automatically set equal to: mean.")
                print("\tYou can ignore this warning if the centering criterion has been assigned later via setter.")
                print("\tOtherwise, please check the conditions which must be satisfied by the input in the detailed documentation.")
            try:
                self._scale = settings["scale"]
                if not isinstance(self._scale, bool):
                    raise Exception
            except:
                self._scale = True 
                warnings.warn("An exception occured with regard to the input value for the scaling decision. It could be not acceptable, or not given to the dictionary.")
                print("\tIt will be automatically set equal to: true.")
                print("\tYou can ignore this warning if the scaling decision has been assigned later via setter.")
                print("\tOtherwise, please check the conditions which must be satisfied by the input in the detailed documentation.")
            try: 
                self._scaling = settings["scaling_method"]
                if not isinstance(self._scaling, str):
                    raise Exception
                elif self._scaling.lower() != "auto" and self._scaling.lower() != "vast" and self._scaling.lower() != "pareto" and self._scaling.lower() != "range":
                    raise Exception
            except:
                self._scaling = "auto"
                warnings.warn("An exception occured with regard to the input value for the scaling criterion. It could be not acceptable, or not given to the dictionary.")
                print("\tIt will be automatically set equal to: auto.")
                print("\tYou can ignore this warning if the scaling criterion has been assigned later via setter.")
                print("\tOtherwise, please check the conditions which must be satisfied by the input in the detailed documentation.")
            try:
                self._plot_explained_variance = settings["enable_plot_variance"]
                if not isinstance(self._plot_explained_variance, bool):
                    raise Exception
            except:
                self._plot_explained_variance = True 
                warnings.warn("An exception occured with regard to the input value for decision to plot the variance. It could be not acceptable, or not given to the dictionary.")
                print("\tIt will be automatically set equal to: True.")
                print("\tYou can ignore this warning if the decision has been assigned later via setter.")
                print("\tOtherwise, please check the conditions which must be satisfied by the input in the detailed documentation.")
            try:
                self._assessPCs = settings["set_criterion_autoPCs"]
                if not isinstance(self._assessPCs, str):
                    raise Exception
                elif self._assessPCs.lower() != "var" or self._assessPCs.lower() != "nrmse":
                    raise Exception
            except:
                self._assessPCs = "var"
            try:
                self._threshold_var = settings["variance_to_explain"]
                if not isinstance(self._threshold_var, float) and not isinstance(self._threshold_var, int):
                    raise Exception
                elif self._threshold_var < 0 or self._threshold_var > 1:
                    raise Exception
            except:
                self._threshold_var = 0.95
            try:
                self._num_to_plot = settings["variable_to_plot"]
                if not isinstance(self._num_to_plot, int) or self._num_to_plot > self.X.shape[1]:
                    raise Exception
            except:
                self._num_to_plot = 0
                warnings.warn("An exception occured with regard to the input value for the number of the variable/PC to plot. It could be not acceptable, or not given to the dictionary.")
                print("\tIt will be automatically set equal to: 0.")
                print("\tYou can ignore this warning if it has been assigned later via setter.")
                print("\tOtherwise, please check the conditions which must be satisfied by the input in the detailed documentation.")

    @property
    def eigens(self):
        return self._nPCs

    @eigens.setter
    def eigens(self, new_value):
        self._nPCs = new_value

        if self._nPCs <= 0 or self._nPCs >= self.X.shape[1]:
            self._nPCs = int(self.X.shape[1]/2)
            warnings.warn("An exception occured with regard to the input value for the number of PCs. It could be not acceptable, or not given to the dictionary.")
            print("\tIt will be automatically set equal to: X.shape[1]/2.")
            print("\tPlease check the conditions which must be satisfied by the input in the detailed documentation.")

    @property
    def to_center(self):
        return self._center

    @to_center.setter
    def to_center(self, new_bool):
        self._center = new_bool

        if not isinstance(self._center, bool):
            warnings.warn("An exception occured with regard to the input value for the centering decision. It could be not acceptable, or not given to the dictionary.")
            print("\tIt will be automatically set equal to: true.")
            print("\tPlease check the conditions which must be satisfied by the input in the detailed documentation.")


    @property
    def centering(self):
        return self._centering

    @centering.setter
    def centering(self, new_string):
        self._centering = new_string

        if not isinstance(self._centering, str):
            self._centering = "mean"
            warnings.warn("An exception occured with regard to the input value for the centering criterion . It could be not acceptable, or not given to the dictionary.")
            print("\tIt will be automatically set equal to: mean.")
            print("\tPlease check the conditions which must be satisfied by the input in the detailed documentation.")
        elif self._centering.lower() != "mean" and self._centering.lower() != "min":
            self._centering = "mean"
            warnings.warn("An exception occured with regard to the input value for the centering criterion . It could be not acceptable, or not given to the dictionary.")
            print("\tIt will be automatically set equal to: mean.")
            print("\tPlease check the conditions which must be satisfied by the input in the detailed documentation.")


    @property
    def to_scale(self):
        return self._scale

    @to_scale.setter
    def to_scale(self, new_bool):
        self._scale = new_bool

        if not isinstance(self._scale, bool):
            warnings.warn("An exception occured with regard to the input value for the scaling decision. It could be not acceptable, or not given to the dictionary.")
            print("\tIt will be automatically set equal to: true.")
            print("\tPlease check the conditions which must be satisfied by the input in the detailed documentation.")


    @property
    def scaling(self):
        return self._scaling

    @scaling.setter
    def scaling(self, new_string):
        self._scaling = new_string

        if not isinstance(self._scaling, str):
            self._scaling = "auto"
            warnings.warn("An exception occured with regard to the input value for the scaling criterion. It could be not acceptable, or not given to the dictionary.")
            print("\tIt will be automatically set equal to: auto.")
            print("\tYou can ignore this warning if the scaling criterion has been assigned later via setter.")
            print("\tPlease check the conditions which must be satisfied by the input in the detailed documentation.")
        elif self._scaling.lower() != "auto" and self._scaling.lower() != "vast" and self._scaling.lower() != "pareto" and self._scaling.lower() != "range":
            self._scaling = "auto"
            warnings.warn("An exception occured with regard to the input value for the scaling criterion. It could be not acceptable, or not given to the dictionary.")
            print("\tIt will be automatically set equal to: auto.")
            print("\tYou can ignore this warning if the scaling criterion has been assigned later via setter.")
            print("\tPlease check the conditions which must be satisfied by the input in the detailed documentation.")

    @property
    def plot_explained_variance(self):
        return self._plot_explained_variance

    @plot_explained_variance.setter
    def plot_explained_variance(self, new_bool):
        self._plot_explained_variance = new_bool

        if not isinstance(self._plot_explained_variance, bool):
            self._plot_explained_variance = True

    @property
    def set_explained_variance_perc(self):
        return self._explained_variance

    @set_explained_variance_perc.setter
    def set_explained_variance_perc(self, new_value):
        self._explained_variance = new_value

        if not isinstance(self._plot_explained_variance, float):
            self._plot_explained_variance = 0.95

    @property
    def set_PCs_method(self):
        return self._assessPCs

    @set_PCs_method.setter
    def set_PCs_method(self, new_method):
        self._assessPCs = new_method

        if not isinstance(self._assessPCs, str):
            self._assessPCs = "var"

    @property
    def set_num_to_plot(self):
        return self._num_to_plot

    @set_num_to_plot.setter
    def set_num_to_plot(self, new_number):
        if not isinstance(self._num_to_plot, int) or self._num_to_plot > self.X.shape[1]:
            self._num_to_plot = 0


    @staticmethod
    def preprocess_training(X, centering_decision, scaling_decision, centering_method, scaling_method):
        '''
        Center and scale the matrix X, depending on the bool values
        centering_decision and scaling_decision
        '''
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
        and retain 'n' Principal Components. 

        The eigenvectors are returned already ordered in decreasing
        order of importance.


        --- RETURNS ---
        evecs:      eigenvectors from the covariance matrix decomposition (PCs)
        type evecs: list

        evals:      eigenvalues from the covariance matrix decomposition (lambda)
        type evecs: list

        '''
        #Center and scale the original training dataset
        self.X_tilde = self.preprocess_training(self.X, self._center, self._scale, self._centering, self._scaling)

        #Compute the covariance matrix
        C = np.cov(self.X_tilde, rowvar=False) #rowvar=False because the X matrix is (observations x variables)

        #All the eigenvectors and eigenvalues are firstly calculated
        self.ALLevals, self.ALLevecs = LA.eig(C)
        
        #After that, they are ordered in decreasing order of magnitude
        mask = np.argsort(self.ALLevals)[::-1]
        self.evecs = self.ALLevecs[:,mask]
        self.evals = self.ALLevals[mask]

        self.evals_uncut = self.evals

        #Cut the last PCs and consider only the prescribed number of eigenvectors
        self.evecs = self.evecs[:,:self._nPCs]
        self.evals = self.evals[:self._nPCs]

        return self.evecs, self.evals


    def recover(self):
        '''
        Reconstruct the original matrix from the reduced PCA-manifold.


        --- RETURNS ---
        X_rec:      uncentered and unscaled reconstructed matrix with PCA modes 
        type X_rec: numpy array    
        '''

        try:
            self.evecs
        except:
            self.evecs, ____ = self.fit()
            

        #Compute the centering and the scaling factors. Later they will be useful.
        self.mu = center(self.X, self.centering)
        self.sigma = scale(self.X, self.scaling)
        #Reconstruct the original matrix
        self.X_r = self.X_tilde @ self.evecs @ self.evecs.T
        # Uncenter and unscale to get the reconstructed original matrix depending on
        #the centering and scaling decisions:

        #If the matrix has been centered and scaled, unscale and uncenter:
        if self._center and self._scale:
            X_unsc = unscale(self.X_r, self.sigma)
            self.X_rec = uncenter(X_unsc, self.mu)
        #If it has only been centered, uncenter:
        elif self._center and not self._scale:
            self.X_rec = uncenter(self.X_r, self.mu)
        #If it has only been scaled, unscale:
        elif self._scale and not self._center:
            self.X_rec = unscale(self.X_r, self.sigma)
        #Otherwise no additional operation is needed
        else:
            self.X_rec = self.X_r


        return self.X_rec


    def get_explained(self):
        '''
        Assess the variance explained by the first 'n_eigs' retained
        Principal Components. This is important to know if the percentage
        of explained variance is enough, or additional PCs must be retained.
        Usually, it is considered accepted a percentage of explained variable
        above 95%.


        --- RETURNS ---
        explained:      percentage of explained variance 
        type explained: scalar  
        '''
        #compute the explained variance by means of the cumulative sum of the
        #considered "q" eigenvalues
        try: 
            self.evals
        except:
            self.evecs, self.evals = self.fit()
        
        explained_variance = np.cumsum(self.evals)/sum(self.ALLevals)
        explained = explained_variance[-1]
        #If the plot boolean is True, produce an image to show the explained variance curve.
        if self._plot_explained_variance:
            matplotlib.rcParams.update({'font.size' : 18, 'text.usetex' : True})
            fig = plt.figure()
            axes = fig.add_axes([0.15,0.15,0.7,0.7], frameon=True)
            axes.plot(np.linspace(1, len(explained_variance), len(explained_variance)), explained_variance, color='mediumblue', marker='s', linestyle='-', linewidth=2, markersize=4, markerfacecolor='mediumblue', label='$Cumulative$')
            axes.plot([self._nPCs, self._nPCs], [explained_variance[0], explained_variance[-1]], color='peru', marker='s', linestyle='-.', linewidth=2, markersize=4, markerfacecolor='peru', label='$Explained\ by\ {}\ PCs$'.format(self._nPCs))
            axes.set_xlabel('$Number\ of\ PCs\ [-]$')
            axes.set_ylabel('$Explained\ variance\ [-]$')
            axes.set_title('$Variance\ explained\ by\ {}\ PCs:\ {}$'.format(self.eigens, round(explained,3)))
            axes.legend(fontsize =14)
            plt.savefig("explainedIncrease.eps")
        #plt.show()


        if self._plot_explained_variance:
            matplotlib.rcParams.update({'font.size' : 18, 'text.usetex' : True})
            fig = plt.figure()
            axes = fig.add_axes([0.15,0.15,0.7,0.7], frameon=True)
            axes.plot(np.linspace(1, len(self.evals_uncut), len(self.evals_uncut)), self.evals_uncut, color='mediumblue', marker='s', linestyle='-', linewidth=2, markersize=4)
            axes.set_xlabel('$PC\ number\ [-]$')
            axes.set_ylabel('$Eigenvalues\ magnitude\ [-]$')
            plt.savefig("eigenDecay.eps")
        #plt.show()

        return explained


    def get_scores(self):
        '''
        Project the original data matrix (cent and scaled) X_tilde
        on the low dimensional manifold spanned by the first 'n'
        Principal Components and obtain the scores matrix (Z).


        --- RETURNS ---
        scores:      matrix of the scores 
        type scores: numpy matrix  
        '''
        #Project the full matrix on the reduced basis
        #Z = XA --> (nxq) = (nxp) x (pxq)
        try:
            self.evecs
        except:
            self.evecs, ____ = self.fit()
        

        self.scores = self.X_tilde @ self.evecs

        return self.scores


    def set_PCs(self):
        '''
        Automatically assess the number of PCs to be retained. This can be done in two different ways, the first one is
        the explained variance criterion: retain nPCs such that the 95% of the original data variance is explained; the 
        second one is the nrmse criterion: retain nPCs such that the difference between the original and the reconstructed
        matrix is lower than the 10%.


        --- RETURNS ---
        optimalPCs:         number of PCs required to satisfy the chosen criterion
        type optimalPCs:    scalar  
        '''
        optimalPCs = None
        if self._assessPCs != False:
            self.plot_explained_variance = False
            for ii in range(1, self.n_var):
                #print("Assessing the optimal number of PCs by means of: " + self._assessPCs + " criterion. Now using: {} Principal Component(s).".format(ii))

                self.eigens = ii
                self.fit()
                #Assess the optimal number of PCs by means of the explained variance threshold (>= 99% explained is the default setting)
                if self._assessPCs == 'var':
                    explained = self.get_explained()

                    if explained >= self._threshold_var:
                        #print("With {} PCs, the following percentage of variance is explained: {}".format(self.eigens, explained))
                        #print("The variance is larger than the given fixed threshold: {}. Thus, {} PCs will be retained.".format(self._threshold_var, self.eigens))
                        self.plot_explained_variance = True
                        optimalPCs = ii
                        break
                #Otherwise with the nrmse of the reconstructed matrix, which has to be below a specific threshold (<= 10% error is the default setting)
                elif self._assessPCs == 'nrmse':
                    reconstructed_ = self.recover()
                    variables_reconstruction = NRMSE(self.X, reconstructed_)

                    if np.mean(variables_reconstruction) <= self._threshold_nrmse:
                        print("With {} PCs, the following average error (NRMSE) for the variables reconstruction (from the PCA manifold) is obtained: {}".format(self.eigens, np.mean(variables_reconstruction)))
                        print("The error is lower than the given fixed threshold: {}. Thus, {} PCs will be retained.".format(self._threshold_nrmse, self.eigens))
                        optimalPCs = ii
                        break
        return optimalPCs


    def plot_PCs(self):
        '''
        Plot the variables' weights on a selected Principal Component (PC).
        The PCs are linear combination of the original variables: examining the
        weights, especially for the PCs associated with the largest eigenvalues,
        can be important for data-analysis purposes.

        '''

        try:
            self.evecs
        except:
            self.evecs, ____ = self.fit()

        #plot
        matplotlib.rcParams.update({'font.size' : 18, 'text.usetex' : True})
        fig = plt.figure()
        axes = fig.add_axes([0.15,0.15,0.7,0.7], frameon=True)

        x = np.linspace(1, self.n_var, self.n_var)
        axes.bar(x, self.evecs[:,self._num_to_plot])
        axes.set_xlabel('$Variables\ [-]$')
        axes.set_ylabel('$Associated\ weight\ on\ PC:\ {}\ [-]$'.format(self._num_to_plot +1))
        plt.show()


    def plot_parity(self):
        '''
        Print the parity plot of the reconstructed profile from the PCA
        manifold. The more the scatter plot (black dots) is in line with the
        red line, the better it is the reconstruction.

        '''
        try:
            self.X_rec
        except:
            #perform PCA
            #self.fit()
            #recontruct the original matrix from the reduced manifold:
            #X_rec = Z*A^{T}
            #it is possible to do this because the matrix A is orthonormal, i.e., A^{-1} = A^{T}
            self.X_rec = self.recover()

        #plot
        matplotlib.rcParams.update({'font.size' : 18, 'text.usetex' : True})
        fig = plt.figure()
        axes = fig.add_axes([0.15,0.15,0.7,0.7], frameon=True)
        axes.plot(self.X[:,self._num_to_plot], self.X[:,self._num_to_plot], color='darkblue', linestyle='-.', linewidth=2)
        axes.scatter(self.X[:,self._num_to_plot], self.X_rec[:,self._num_to_plot], 1, color= 'slategrey', alpha=0.4)
        axes.set_xlabel('$Original\ variable$')
        axes.set_ylabel('$Reconstructed\ via\ PCA\ $')
        plt.xlim(min(self.X[:,self._num_to_plot]), max(self.X[:,self._num_to_plot]))
        plt.ylim(min(self.X[:,self._num_to_plot]), max(self.X[:,self._num_to_plot]))
        plt.savefig("parityReconstruction.eps")
        plt.show()


    def outlier_removal_leverage(self):
        '''
        This function removes the multivariate outliers (leverage) eventually contained
        in the training dataset, via PCA. In fact, examining the data projection
        on the PCA manifold (i.e., the scores), and measuring the score distance
        from the manifold center, it is possible to identify the so-called
        leverage points. They are characterized by very high distance from the
        center of mass, once detected they can easily be removed.
        Additional info on outlier identification and removal can be found here:

        Jolliffe pag 237 --- formula (10.1.2):

        dist^{2}_{2,i} = sum_{k=p-q+1}^{p}(z^{2}_{ik}/l_{k})
        where:
        p = number of variables
        q = number of required PCs
        i = index to count the observations
        k = index to count the PCs


        --- RETURNS ---
        X:         the matrix without leverage outliers
        type X:    numpy matrix

        bin:        the 'clusters' of the Cumulative Density Function
        new_mask:   the vector containing the ID of the non-outlier observations 

        '''
        
        #Leverage points removal:
        #Compute the PCA scores. Override the eventual number of PCs: ALL the
        #PCs are needed, as the outliers are given by the last PCs examination
        input_eigens = self._nPCs
        self.eigens = self.X.shape[1]-1
        PCs, eigval = self.fit()
        scores = self.get_scores()
        TOL = 1E-16
        #put again the user-defined number of PCs
        self.eigens = input_eigens

        scores_dist = np.empty((self.X.shape[0],), dtype=float)
        #For each observation, compute the distance from the center of the manifold
        for ii in range(0,self.X.shape[0]):
            t_sq = 0
            lam_j = 0
            cumsum_tmp = 0
            for jj in range(input_eigens, scores.shape[1]):
                t_sq = scores[ii,jj]**2
                lam_j = eigval[jj]
                cumsum_tmp += t_sq/(lam_j + TOL)
            scores_dist[ii] = cumsum_tmp

        #Now compute the distance distribution, and delete the observations in the
        #upper 2% (done in the while loop) to get the outlier-free matrix.

        #Divide the distance vector in 100 bins
        n_bins = 100
        min_interval = np.min(scores_dist)
        max_interval = np.max(scores_dist)
        
        #compute the step of each bin
        delta_step = (max_interval - min_interval) / n_bins

        counter = 0
        bin = np.empty((len(scores_dist),))
        var_left = min_interval

        #Find the observations in each bin (find the idx, where the classes are
        #the different bins number)
        while counter <= n_bins:
            var_right = var_left + delta_step
            mask = np.logical_and(scores_dist >= var_left, scores_dist < var_right)
            bin[np.where(mask)] = counter
            counter += 1
            var_left += delta_step

        unique,counts=np.unique(bin,return_counts=True)
        cumulativeDensity = 0
        new_counter = 0

        #remove the upper 2% of the CDF, i.e., x_{i} \in [0.98; 1]
        while cumulativeDensity < 0.98:
            cumulative_ = counts[new_counter]/self.X.shape[0]
            cumulativeDensity += cumulative_
            new_counter += 1

        new_mask = np.where(bin > new_counter)
        self.X = np.delete(self.X, new_mask, axis=0)

        return self.X , bin, new_mask



    def outlier_removal_orthogonal(self):
        '''
        This function removes the multivariate outliers (orthogonal out) eventually contained
        in the training dataset, via PCA. In fact, examining the reconstruction error
        it is possible to identify the so-called orthogonal outliers. They are characterized
        by very high distance from the manifold (large rec error), once detected they can easily
        be removed.
        Additional info on outlier identification and removal can be found here:

        Hubert, Mia, Peter Rousseeuw, and Tim Verdonck. Computational Statistics & Data Analysis 53.6 (2009): 2264-2274.


        --- RETURNS ---
        X:         the matrix without leverage outliers
        type X:    numpy matrix

        bin:        the 'clusters' of the Cumulative Density Function
        new_mask:   the vector containing the ID of the non-outlier observations
        '''
        #Orthogonal outliers removal:

        #Compute the PCs
        PCs, eigval = self.fit()
        #eventually preprocess the training matrix
        X_cs = self.preprocess_training(self.X, self._center, self._scale, self._centering, self._scaling)

        #compute the squared reconstruction error
        epsilon_rec = X_cs - X_cs @ PCs @ PCs.T
        sq_rec_oss = np.power(epsilon_rec, 2)

        #Now compute the distance distribution, and delete the observations in the
        #upper 2% (done in the while loop) to get the outlier-free matrix.

        #Divide the distance vector in 100 bins
        n_bins = 100
        min_interval = np.min(sq_rec_oss)
        max_interval = np.max(sq_rec_oss)
        
        #compute the step of each bin
        delta_step = (max_interval - min_interval) / n_bins

        counter = 0
        bin_id = np.empty((len(epsilon_rec),))
        var_left = min_interval

        #Find the observations in each bin (find the idx, where the classes are
        #the different bins number)
        while counter <= n_bins:
            var_right = var_left + delta_step
            mask = np.logical_and(sq_rec_oss >= var_left, sq_rec_oss < var_right)
            bin_id[np.where(mask)[0]] = counter
            counter += 1
            var_left += delta_step

        #Compute the classes (unique) and the number of elements per class (counts)
        unique,counts=np.unique(bin_id,return_counts=True)
        #Declare the variables to build the CDF to select the observations belonging to the
        #98% of the total
        cumulativeDensity = 0
        new_counter = 0

        while cumulativeDensity < 0.98:
            cumulative_ = counts[new_counter]/self.X.shape[0]
            cumulativeDensity += cumulative_
            new_counter += 1

        #delete the observations in the upper 2%, i.e., x_{i} \in [0.98; 1]
        new_mask = np.where(bin_id > new_counter)
        self.X = np.delete(self.X, new_mask, axis=0)

        return self.X, bin_id, new_mask


class LPCA(PCA):
    '''
    Perfom model order reduction via Local Principal Component Analysis (LPCA).
    LPCA is a statistical technique for the dimensionality reduction and reduced
    order modelling which was introduced to extend the applicability of PCA also
    to non-linear applications. In fact, LPCA is a piecewise linear: different
    manifold are found for different groups of points. 
    The partitioning of the data matrix can be calculated via LPCA clustering
    algorithm (implemented in the clustering module) or can be given in input.

    This function direcly inherits all the properties from PCA, so for many
    of the parameters and setters a detailed description can be found in PCA.


    --- PARAMETERS ---
    X:          RAW data matrix, uncentered and unscaled. It must be organized
                with the structure: (observations x variables).
    type X :    numpy array


    --- SETTERS ---
    _path_to_idx:           path to the .txt file containing the class assignment for the matrix
    type _path_to_idx:      string

    _clust_to_plot:          number of cluster where the chosen LPCs must be plotted
    type   _clust_to_plot:   scalar

    '''
    def __init__(self,X, *dictionary):
        #Set the path where the file 'idx.txt' (containing the partitioning solution) is located
        self._path_to_idx = 'path'
        #Set the PC number to plot or the variable's number to plot
        self._num_to_plot = 1
        #Set the cluster number where plot the PC
        self._clust_to_plot = 1

        super().__init__(X)

        if dictionary:
                settings = dictionary[0]
                try:
                    self._nPCs = settings["number_of_eigenvectors"]
                    if self._nPCs < 0 or self._nPCs >= self.X.shape[1]:
                        raise Exception
                except:
                    self._nPCs = self.X.shape[1]-1
                    warnings.warn("An exception occured with regard to the input value for the number of PCs. It could be not acceptable, or not given to the dictionary.")
                    print("\tIt will be automatically set equal to: X.shape[1]-1.")
                    print("\tYou can ignore this warning if the number of PCs has been assigned later via setter.")
                    print("\tOtherwise, please check the conditions which must be satisfied by the input in the detailed documentation.")
                try:
                    self._center = settings["center"]
                    if not isinstance(self._center, bool):
                        raise Exception
                except:
                    self._center = True
                    warnings.warn("An exception occured with regard to the input value for the centering decision. It could be not acceptable, or not given to the dictionary.")
                    print("\tIt will be automatically set equal to: true.")
                    print("\tYou can ignore this warning if the centering decision has been assigned later via setter.")
                    print("\tOtherwise, please check the conditions which must be satisfied by the input in the detailed documentation.")
                try:
                    self._centering = settings["centering_method"]
                    if not isinstance(self._centering, str):
                        raise Exception
                    elif self._centering.lower() != "mean" and self._centering.lower() != "min":
                        raise Exception
                except:
                    self._centering = "mean"
                    warnings.warn("An exception occured with regard to the input value for the centering criterion . It could be not acceptable, or not given to the dictionary.")
                    print("\tIt will be automatically set equal to: mean.")
                    print("\tYou can ignore this warning if the centering criterion has been assigned later via setter.")
                    print("\tOtherwise, please check the conditions which must be satisfied by the input in the detailed documentation.")
                try:
                    self._scale = settings["scale"]
                    if not isinstance(self._scale, bool):
                        raise Exception
                except:
                    self._scale = True 
                    warnings.warn("An exception occured with regard to the input value for the scaling decision. It could be not acceptable, or not given to the dictionary.")
                    print("\tIt will be automatically set equal to: true.")
                    print("\tYou can ignore this warning if the scaling decision has been assigned later via setter.")
                    print("\tOtherwise, please check the conditions which must be satisfied by the input in the detailed documentation.")
                try: 
                    self._scaling = settings["scaling_method"]
                    if not isinstance(self._scaling, str):
                        raise Exception
                    elif self._scaling.lower() != "auto" and self._scaling.lower() != "vast" and self._scaling.lower() != "pareto" and self._scaling.lower() != "range":
                        raise Exception
                except:
                    self._scaling = "auto"
                    warnings.warn("An exception occured with regard to the input value for the scaling criterion. It could be not acceptable, or not given to the dictionary.")
                    print("\tIt will be automatically set equal to: auto.")
                    print("\tYou can ignore this warning if the scaling criterion has been assigned later via setter.")
                    print("\tOtherwise, please check the conditions which must be satisfied by the input in the detailed documentation.")
                try:
                    self._path_to_idx = settings["path_to_idx"]

                    if not isinstance(self._path_to_idx, str):
                        raise Exception
                except:
                    self._path_to_idx = ' '
                try:
                    self._clust_to_plot = settings["cluster_to_plot"]

                    if not isinstance(self._clust_to_plot, int) or self._clust_to_plot <0:
                        raise Exception
                except:
                    self._clust_to_plot = 0
                try:
                    self._num_to_plot = settings["PC_to_plot"]

                    if not isinstance(self._num_to_plot, int) or self._num_to_plot > self._nPCs:
                        raise Exception
                except:
                    self._num_to_plot = 0
            


    @property
    def path_to_idx(self):
        return self._path_to_idx

    @path_to_idx.setter
    def path_to_idx(self, new_string):
        self._path_to_idx = new_string

        if not isinstance(self._path_to_idx, str):
            self._path_to_idx = ' '

    @property
    def clust_to_plot(self):
        return self._clust_to_plot

    @clust_to_plot.setter
    def clust_to_plot(self, new_number):
        self._clust_to_plot = new_number

        if not isinstance(self._clust_to_plot, int) or self._clust_to_plot < 0:
            raise Exception

    @staticmethod
    def get_idx(path):
        '''
        try to load the solution obtained from a previous partitioned given a path
        '''

        try:
            print("Reading idx..")
            idx = np.genfromtxt(path + '/idx.txt', delimiter= ',')
        except OSError:
            print("Could not open/read the selected file: " + path + 'idx.txt')
            exit()

        return idx

    def check_sanity_input(self):
        if self.X.shape[0] > len(self.idx) or self.X.shape[0] < len(self.idx):
            raise Exception("The first dimension of the matrix X and the length of idx must agree.")
            print("Exiting with error..")
            exit()


    def fit(self):
        '''
        This function computes the LPCs (Local Principal Components), the u_scores
        (the projection of the points on the local manifold), and the eigenvalues
        in each cluster found by the lpca iterative algorithm, given a previous clustering
        solution.


        --- RETURNS ---
        LPCs:       LPCs in each cluster 
        type LPCs:  list of k elements

        u_scores:       scores in each cluster 
        type u_scores:  list of k elements

        Leigen:         eigenvalues in each cluster
        type Leigen:    list of k elements

        centroids:      centroids in each cluster 
        type centroids:  list of k elements
        '''
        #Load the idx (from a previous clustering partitioning) from the given path.
        #after that, compute the number of clusters and preprocess the training matrix
        #with the given settings
        self.idx = self.get_idx(self.path_to_idx)
        self.check_sanity_input()
        self.k = int(max(self.idx) +1)
        self.X_tilde = self.preprocess_training(self.X, self.to_center, self.to_scale, self.centering, self.scaling)

        #Initialize the lists containing the variables of interest
        self.centroids = [None] *self.k
        self.LPCs = [None] *self.k
        self.u_scores = [None] *self.k
        self.Leigen = [None] *self.k

        #In each cluster, compute the centroid. After that, center inside the cluster (with the centroid)
        #and perform PCA to obtain the LocalPCs and the LocalEigens.
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


        --- RETURNS ---
        X_rec:      matrix reconstructed by means of the LPCs 
        type X_rec: numpy matrix
        '''

        #Initialize the reconstructed matrix, and center the input matrix
        self.X_rec = np.empty(self.X.shape, dtype=float)
        self.X_tilde = self.preprocess_training(self.X, self.to_center, self.to_scale, self.centering, self.scaling)

        #The centering and scaling factors are computed from the training matrix
        #because they will be used in the future to unscale/uncenter
        mu_global = center(self.X, self.centering)
        sigma_global = scale(self.X, self.scaling)

        #Perform LPCA
        self.LPCs, self.u_scores, self.Leigen, self.centroids = self.fit()

        #Considering the idx of that set of partition
        for ii in range (0,self.k):
            cluster_ = get_cluster(self.X_tilde, self.idx, ii)
            centroid_ =self.centroids[ii]
            C = np.empty(cluster_.shape, dtype=float)
            C = (cluster_ - centroid_) @ self.LPCs[ii] @ self.LPCs[ii].T
            C_ = uncenter(C, centroid_)
            positions = np.where(self.idx == ii)
            self.X_rec[positions] = C_

        #unscale and uncenter to get back the original values

        if self._center and self._scale:
            X_unsc = unscale(self.X_rec, sigma_global)
            self.X_rec = uncenter(X_unsc, mu_global)
        #If it has only been centered, uncenter:
        elif self._center and not self._scale:
            self.X_rec = uncenter(self.X_rec, mu_global)
        #If it has only been scaled, unscale:
        elif self._scale and not self._center:
            self.X_rec = unscale(self.X_rec, sigma_global)
        #Otherwise no additional operation is needed
        else:
            self.X_rec = self.X_rec

        return self.X_rec


    def plot_parity(self):
        '''
        Print the parity plot of the reconstructed profile from the PCA
        manifold. The more the scatter plot (black dots) is in line with the
        red line, the better it is the reconstruction.
        '''
        #reconstruct the matrix from the low-dimensional manifold
        reconstructed_ = self.recover()

        #plot
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
        #consider the PCs in a given cluster (LPCs)
        local_eigen = self.LPCs[self.clust_to_plot]

        #barplot
        matplotlib.rcParams.update({'font.size' : 18, 'text.usetex' : True})
        fig = plt.figure()
        axes = fig.add_axes([0.15,0.15,0.7,0.7], frameon=True)

        x = np.linspace(1, self.n_var, self.n_var)
        axes.bar(x, local_eigen[:,self.set_num_to_plot])
        axes.set_xlabel('Variables [-]')
        axes.set_ylabel('Weights on the PC number: {}, k = {} [-]'.format(self.set_num_to_plot, self.clust_to_plot))
        plt.show()


class KPCA(PCA):
    def __init__(self, X, *dictionary):

        #Set the sigma - the coefficient for the kernel construction
        #self.sigma = 10
        self._kernel = 'rbf'

        super().__init__(X)

        if dictionary:
            settings = dictionary[0]
            try:
                self._nPCs = settings["number_of_eigenvectors"]
                if self._nPCs < 0 or self._nPCs > self.X.shape[1]:
                    raise Exception
            except:
                self._nPCs = self.X.shape[1]
                warnings.warn("An exception occured with regard to the input value for the number of PCs. It could be not acceptable, or not given to the dictionary.")
                print("\tIt will be automatically set equal to: X.shape[1]-1.")
                print("\tYou can ignore this warning if the number of PCs has been assigned later via setter.")
                print("\tOtherwise, please check the conditions which must be satisfied by the input in the detailed documentation.")
            try:
                self._center = settings["center"]
                if not isinstance(self._center, bool):
                    raise Exception
            except:
                self._center = True
                warnings.warn("An exception occured with regard to the input value for the centering decision. It could be not acceptable, or not given to the dictionary.")
                print("\tIt will be automatically set equal to: true.")
                print("\tYou can ignore this warning if the centering decision has been assigned later via setter.")
                print("\tOtherwise, please check the conditions which must be satisfied by the input in the detailed documentation.")
            try:
                self._centering = settings["centering_method"]
                if not isinstance(self._centering, str):
                    raise Exception
                elif self._centering.lower() != "mean" and self._centering.lower() != "min":
                    raise Exception
            except:
                self._centering = "mean"
                warnings.warn("An exception occured with regard to the input value for the centering criterion . It could be not acceptable, or not given to the dictionary.")
                print("\tIt will be automatically set equal to: mean.")
                print("\tYou can ignore this warning if the centering criterion has been assigned later via setter.")
                print("\tOtherwise, please check the conditions which must be satisfied by the input in the detailed documentation.")
            try:
                self._scale = settings["scale"]
                if not isinstance(self._scale, bool):
                    raise Exception
            except:
                self._scale = True 
                warnings.warn("An exception occured with regard to the input value for the scaling decision. It could be not acceptable, or not given to the dictionary.")
                print("\tIt will be automatically set equal to: true.")
                print("\tYou can ignore this warning if the scaling decision has been assigned later via setter.")
                print("\tOtherwise, please check the conditions which must be satisfied by the input in the detailed documentation.")
            try: 
                self._scaling = settings["scaling_method"]
                if not isinstance(self._scaling, str):
                    raise Exception
                elif self._scaling.lower() != "auto" and self._scaling.lower() != "vast" and self._scaling.lower() != "pareto" and self._scaling.lower() != "range":
                    raise Exception
            except:
                self._scaling = "auto"
                warnings.warn("An exception occured with regard to the input value for the scaling criterion. It could be not acceptable, or not given to the dictionary.")
                print("\tIt will be automatically set equal to: auto.")
                print("\tYou can ignore this warning if the scaling criterion has been assigned later via setter.")
                print("\tOtherwise, please check the conditions which must be satisfied by the input in the detailed documentation.")
            try:
                self._kernel = settings["selected_kernel"]
                if not isinstance(self._kernel, str):
                    raise Exception
                elif self._kernel != 'rbf' and self._kernel != "polynomial":
                    raise Exception
            except:
                self._kernel = 'rbf'
                warnings.warn("An exception occured with regard to the input value for the kernel type. It could be not acceptable, or not given to the dictionary.")
                print("\tIt will be automatically set equal to: rbf (radial basis function).")
                print("\tYou can ignore this warning if the kernel type has been assigned later via setter.")
                print("\tOtherwise, please check the conditions which must be satisfied by the input in the detailed documentation.")
            
            try:
                self._sigma = settings["sigma"]
                if not isinstance(self._sigma, float) and not isinstance(self._sigma, int):
                    raise Exception
            except:
                self._sigma = 1
                warnings.warn("An exception occured with regard to the input value for the sigma. It could be not acceptable, or not given to the dictionary.")
                print("\tIt will be automatically set equal to: rbf (radial basis function).")
                print("\tYou can ignore this warning if the sigma has been assigned later via setter.")
                print("\tOtherwise, please check the conditions which must be satisfied by the input in the detailed documentation.")

            try:
                self._Nystrom = settings["use_Nystrom"]
                if not isinstance(self._Nystrom, bool):
                    raise Exception
            except:
                self._Nystrom = False
                warnings.warn("An exception occured with regard to the input value for the use Nystrom approximation. It could be not acceptable, or not given to the dictionary.")
                print("\tIt will be automatically set equal to: False --> the full algorithm will be employed.")
                print("\tPlease check the conditions which must be satisfied by the input in the detailed documentation.")
            
            try:
                self._fastSVD = settings["fast_SVD"]
                if not isinstance(self._fastSVD, bool):
                    raise Exception
            except:
                self._fastSVD = True
                warnings.warn("An exception occured with regard to the input value for the use of the fast SVD algorithm. It could be not acceptable, or not given to the dictionary.")
                print("\tIt will be automatically set equal to: True.")
                print("\tPlease check the conditions which must be satisfied by the input in the detailed documentation.")

            try:
                self._eigensFast = settings["eigensFast"]
                if not isinstance(self._eigensFast, int) and self._fastSVD:
                    raise Exception
            except:
                self._eigensFast = 100
                warnings.warn("An exception occured with regard to the input value for the number of eigens to be used in the fast SVD algorithm. It could be not acceptable, or not given to the dictionary.")
                print("\tIt will be automatically set equal to: 100.")
                print("\tPlease check the conditions which must be satisfied by the input in the detailed documentation.")


    @property
    def kernel_type(self):
        return self._kernel

    @kernel_type.setter
    def kernel_type(self, new_string):
        self._kernel = new_string

        if not isinstance(self._kernel, str):
            self._kernel = 'rbf'
            warnings.warn("An exception occured with regard to the input value for the kernel type. It could be not acceptable, or not given to the dictionary.")
            print("\tIt will be automatically set equal to: rbf (radial basis function).")
            print("\tOtherwise, please check the conditions which must be satisfied by the input in the detailed documentation.")
        elif self._kernel != 'rbf' and self._kernel != "polynomial":
            self._kernel = 'rbf'
            warnings.warn("An exception occured with regard to the input value for the kernel type. It could be not acceptable, or not given to the dictionary.")
            print("\tIt will be automatically set equal to: rbf (radial basis function).")
            print("\tOtherwise, please check the conditions which must be satisfied by the input in the detailed documentation.")


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
        #As usual, preprocess the matrix
        self.X_tilde = KPCA.preprocess_training(self.X, self.to_center, self.to_scale, self.centering, self.scaling)

        import time 
        
        

        if self.X_tilde.shape[0] > 20000:
            rowsToPick = 200
        else:
            rowsToPick = 100
        
        t = time.time()
        #compute the Kernel 
        if self._kernel == 'polynomial':
            print("Computing the Kernel (poly)..")
            model = Kernel_approximation(self.X_tilde, kernelType="polynomial", toCenter=False, toScale=False, centerCrit="mean", scalCrit="auto", numToPick=rowsToPick, sigma=self._sigma, rank=50, d=2, c=1)
            Kernel = model.Nystrom_standard()
            Kernel = Kernel.real
        elif self._kernel == 'rbf':
            print("Computing the Kernel (rbf)..")
            if self._Nystrom:
                model = Kernel_approximation(self.X_tilde, kernelType="rbf", toCenter=False, toScale=False, centerCrit="mean", scalCrit="auto", numToPick=rowsToPick, sigma=self._sigma, rank=50, p=1)
                Kernel = model.Nystrom_standard()
                Kernel = Kernel.real
            else:
                Kernel = Kernel_approximation.RBFkernel(self.X_tilde, self.X_tilde, self._sigma, selfKernel="yes")
        else:
            raise Exception("The selected Kernel is not supported. Exiting with error..")
            exit()

        elapsed_kernel = time.time() - t   
        print("Kernel computed in {} s.".format(elapsed_kernel)) 


        #Compute the centering matrix;
        N1 = np.ones((Kernel.shape[0], Kernel.shape[1]), dtype=float)
        N1 = N1/Kernel.shape[0]

        print("Centering the Kernel matrix..")
        #Compute the Gram matrix
        K_tilde = Kernel - N1@Kernel - Kernel@N1 + N1@Kernel@N1

        t = time.time()
        #now perform fast SVD to decompose the matrix
        if self._fastSVD:
            print("Decomposing Kernel with fast SVD algorithm..")
            U, V, Sigma = fastSVD(K_tilde, self._eigensFast)
        else:
            print("Decomposing Kernel with standard SVD algorithm..")
            from scipy import linalg
            U, Sigma, V = linalg.svd(K_tilde)
        elapsed_SVD = time.time() - t

        print("Elapsed SVD: {}".format(elapsed_SVD))


        U = U.real
        V = V.real
        Sigma = Sigma.real

        U = U[:,:self._nPCs]
        V = V[:,:self._nPCs]
        Sigma = Sigma[:self._nPCs]

        return U, V, Sigma


class variables_selection(PCA):
    '''
    In many applications, rather than reducing the dimensionality considering a new set of coordinates
    which are linear combination of the original ones, the main interest is to achieve a dimensionality
    reduction selecting a subset of m variables from the original set of p variables.
    
    Three methods for variables selection via PCA are implemented in this class:
    i) Method B2 backward;
    ii) Method B4 forward;
    iii) Variables selection via PCA and procrustes Analysis, by means of the Krzanovski iterative algorithm [b]

    Additional info about the methods are available in the fit method.

    This function direcly inherits all the properties from PCA, so for many
    of the parameters and setters a detailed description can be found in PCA.


    --- PARAMETERS ---
    X:          RAW data matrix, uncentered and unscaled. It must be organized
                with the structure: (observations x variables).
    type X :    numpy array

    *dictionary:    instruction for the PVs algorithm. (Optional)
    type *dic:      Python dictionary


    --- SETTERS ---
    _n_ret:                 number of variables to retain
    type _n_ret:            scalar

    _path:                  path to the file containing the variables' labels
    type   _path:           string

    _labels_name:           variables' labels
    type   _labels_name:    csv file

    _method:                method to be used for variables selection
    type   _method:         string


    --- METHODS ---

    i)      B2:
    The variable with the largest weight is found on the last PC, and then it is deleted.
    This is accomplished via iterative backward elimination algorithm, until the number of
    variables is equal to the selected number of variables.

    ii)     B4:
    The variables associated with the largest weights on each of the 'm' first PCs are
    selected. This is not an iterative algorithm.

    iii)    Variables selection via PCA and procrustes Analysis:
    The iterative variable selection algorithm introduced by Krzanovski is based on the following steps (1-3):
    1.  The dimensionality of m is initially set equal to p.
    2.  Each variable is deleted from the matrix X, obtaining p ~X matrices. The
        corresponding scores matrices are computed by means of PCA. For each of them, a procrustes Analysis
        is performed with respect to the scores of the original matrix X, and the corresponding M2 coeffcient is computed.
    3.  The variable which, once excluded, leads to the smallest M2 coefficient is deleted from the matrix.
    4.  Steps 2 and 3 are repeated until m variables are left.

    iv)  Variables selection via PCA, procrustes Analysis and Varimax rotation:
    It follows the same steps of the method iii), but Varimax rotation is performed before procrustes analysis.


    '''
    def __init__(self, X, *dictionary):
        self.X = X


        #Initialize the number of variables to select and the PCs to retain

        self._n_ret = 1
        self._path = ' '
        self._labels_name = ' '

        super().__init__(self.X)

        self._method = 'b2' #'B2', 'B4', "procrustes", "procrustes_rotation"

        if dictionary:
            settings = dictionary[0]

            try:
                self._nPCs = settings["number_of_eigenvectors"]
                if self._nPCs < 0 or self._nPCs >= self.X.shape[1]:
                    raise Exception
            except:
                self._nPCs = self.X.shape[1]-1
                warnings.warn("An exception occured with regard to the input value for the number of PCs. It could be not acceptable, or not given to the dictionary.")
                print("\tIt will be automatically set equal to: X.shape[1]-1.")
                print("\tYou can ignore this warning if the number of PCs has been assigned later via setter.")
                print("\tOtherwise, please check the conditions which must be satisfied by the input in the detailed documentation.")
            try:
                self._method = settings["method"]
                if not isinstance(self._method, str):
                    raise Exception
                elif self._method.lower() != "procrustes" and self._method.lower() != "b2" and self._method.lower() != "b4" and self._method.lower() != "procrustes_rotation" and self._method.lower() != "b2_rotation" and self._method.lower() != "b4_rotation" and self._method.lower() != "mccabe" and self._method.lower() != "mccabe_rotation":
                    raise Exception

                if self._method.lower() == "mccabe" or self._method.lower() == "mccabe_rotation":
                    try:
                        self._McCriterion = settings["McCabe_criterion"]
                    except:
                        self._McCriterion = 1
                        warnings.warn("An exception occured with regard to the input value for the McCabe criterion. It could be not acceptable, or not given to the dictionary.")
                        print("\tIt will be automatically set equal to: 1.")
        

            except:
                self._method = 'procrustes'
                warnings.warn("An exception occured with regard to the input value for the variables selection method. It could be not acceptable, or not given to the dictionary.")
                print("\tIt will be automatically set equal to: procrustes.")
                print("\tYou can ignore this warning if the variables selection method has been assigned later via setter.")
                print("\tOtherwise, please check the conditions which must be satisfied by the input in the detailed documentation.")
            try:
                self._center = settings["center"]
                if not isinstance(self._center, bool):
                    raise Exception
            except:
                self._center = True
                warnings.warn("An exception occured with regard to the input value for the centering decision. It could be not acceptable, or not given to the dictionary.")
                print("\tIt will be automatically set equal to: true.")
                print("\tYou can ignore this warning if the centering decision has been assigned later via setter.")
                print("\tOtherwise, please check the conditions which must be satisfied by the input in the detailed documentation.")
            try:
                self._centering = settings["centering_method"]
                if not isinstance(self._centering, str):
                    raise Exception
                elif self._centering.lower() != "mean" and self._centering.lower() != "min":
                    raise Exception
            except:
                self._centering = "mean"
                warnings.warn("An exception occured with regard to the input value for the centering criterion . It could be not acceptable, or not given to the dictionary.")
                print("\tIt will be automatically set equal to: mean.")
                print("\tYou can ignore this warning if the centering criterion has been assigned later via setter.")
                print("\tOtherwise, please check the conditions which must be satisfied by the input in the detailed documentation.")
            try:
                self._scale = settings["scale"]
                if not isinstance(self._scale, bool):
                    raise Exception
            except:
                self._scale = True 
                warnings.warn("An exception occured with regard to the input value for the scaling decision. It could be not acceptable, or not given to the dictionary.")
                print("\tIt will be automatically set equal to: true.")
                print("\tYou can ignore this warning if the scaling decision has been assigned later via setter.")
                print("\tOtherwise, please check the conditions which must be satisfied by the input in the detailed documentation.")
            try: 
                self._scaling = settings["scaling_method"]
                if not isinstance(self._scaling, str):
                    raise Exception
                elif self._scaling.lower() != "auto" and self._scaling.lower() != "vast" and self._scaling.lower() != "pareto" and self._scaling.lower() != "range":
                    raise Exception
            except:
                self._scaling = "auto"
                warnings.warn("An exception occured with regard to the input value for the scaling criterion. It could be not acceptable, or not given to the dictionary.")
                print("\tIt will be automatically set equal to: auto.")
                print("\tYou can ignore this warning if the scaling criterion has been assigned later via setter.")
                print("\tOtherwise, please check the conditions which must be satisfied by the input in the detailed documentation.")
            try:
                self._n_ret = settings["number_of_variables"]
                if not isinstance(self._n_ret,int) or self._n_ret >= self.X.shape[1] or self._n_ret < 0:
                    raise Exception
            except:
                warnings.warn("An exception occured with regard to the input value for the number of variables to retain. It could be not acceptable, or not given to the dictionary.")
                print("\tIt will be automatically set equal to: 1.")
                print("\tYou can ignore this warning if the scaling criterion has been assigned later via setter.")
                print("\tOtherwise, please check the conditions which must be satisfied by the input in the detailed documentation.")
                self._n_ret = 1
            try:
                self._path = settings["path_to_labels"]

                if not isinstance(self._path, str):
                    raise Exception 
            except:
                #the class will work with the variables' number, instead of their names (managed in load labels method)
                self._path = "not given"
            try:
                self._labels_name = settings["labels_name"]

                if not isinstance(self._labels_name, str):
                    raise Exception
            except:
                #the class will work with the variables' number, instead of their names (managed in load labels method)
                self._labels_name = "not given"


    @property
    def retained(self):
        return self._n_ret

    @retained.setter
    def retained(self, new_number):
        self._n_ret = new_number

        if not isinstance(self._n_ret,int) or self._n_ret >= self.X.shape[1] or self._n_ret < 0:
            warnings.warn("An exception occured with regard to the input value for the number of variables to retain. It could be not acceptable, or not given to the dictionary.")
            print("\tIt will be automatically set equal to: 1.")
            print("\tPlease check the conditions which must be satisfied by the input in the detailed documentation.")
            self._n_ret = 1

    @property
    def path_to_labels(self):
        return self._path

    @path_to_labels.setter
    def path_to_labels(self, new_string):
        self._path = new_string

        if not isinstance(self._path, str):
            self._path = "not given"

    @property
    def labels_file_name(self):
        return self._labels_name

    @labels_file_name.setter
    def labels_file_name(self, new_string):
        self._labels_name = new_string

        if not isinstance(self._labels_name, str):
            self._labels_name = "not given"

    @property
    def method(self):
        return self._method

    @method.setter
    def method(self, new_string):
        self._method = new_string

        if not isinstance(self._method, str):
            self._method = 'procrustes'
            warnings.warn("An exception occured with regard to the input value for the variables selection method. It could be not acceptable, or not given to the dictionary.")
            print("\tIt will be automatically set equal to: procrustes.")
            print("\tPlease check the conditions which must be satisfied by the input in the detailed documentation.")
        elif self._method.lower() != "procrustes" and self._method.lower() != "b2" and self._method.lower() != "b4":
            self._method = 'procrustes'
            warnings.warn("An exception occured with regard to the input value for the variables selection method. It could be not acceptable, or not given to the dictionary.")
            print("\tIt will be automatically set equal to: procrustes.")
            print("\tPlease check the conditions which must be satisfied by the input in the detailed documentation.")


    def load_labels(self):
        import pandas as pd
        try:
            self.labels= np.array(pd.read_csv(self._path + '/' + self._labels_name, sep = ',', header = None))
        except OSError:
            print("Could not open/read the selected file: " + self._labels_name)
            print("Using variables numbers instead of names.")
            self.labels = np.linspace(0, self.X.shape[1]-1, self.X.shape[1], dtype=int)


    @staticmethod
    def check_sanity_input(X, labels, retained):
        #print(labels)
        if X.shape[1] != len(labels):
            print("Variables number: {}, Labels length: {}".format(X.shape[1], len(labels)))#.shape[1]))
            print(labels)
            raise Exception("The number of variables does not match the labels.")
            exit()
        elif retained >= X.shape[1]:
            raise Exception("The number of retained variables must be lower than the number of original variables.")
            exit()


    def fit(self):
        '''
        --- RETURNS ---
        labels:      labels for the retained species
        type labels: array
        '''

        #load the variables' labels (or the numbers, if the path is not given)
        self.load_labels()
        variables_selection.check_sanity_input(self.X, self.labels, self._n_ret)
        
        #preprocess the training matrix
        self.X_tilde = PCA.preprocess_training(self.X, self.to_center, self.to_scale, self.centering, self.scaling)
        self.var_num = np.linspace(0, self.X.shape[1]-1, self.X.shape[1], dtype=int)
        if self._method.lower() == 'procrustes':
            print("Selecting global variables via PCA and procrustes Analysis...")
            
            #Start with PCA, and compute the scores (Z)
            eigenvec = PCA_fit(self.X_tilde, self._nPCs)
            Z = self.X_tilde @ eigenvec[0]
            #Start the backward elimination:
            while self.X_tilde.shape[1] > self._n_ret:
                M2 = 1E12
                M2_tmp = 0
                var_tmp = 0

                for ii in range(0,self.X_tilde.shape[1]):
                    #Delete each variable from the original matrix (one at the time)
                    X_cut = np.delete(self.X_tilde, ii, axis=1)
                    #Compute the scores of the reduced matrix, after PCA
                    eigenvec = PCA_fit(X_cut, self._nPCs)
                    Z_tilde = X_cut @ eigenvec[0]
                    #Compute the reduced scores covariance matrix, and then apply SVD
                    covZZ = np.transpose(Z_tilde) @ Z
                    u, s, vh = np.linalg.svd(covZZ, full_matrices=True)
                    #Compute the procrustes Analysis score M2 for the matrix without the 'ii' variable
                    M2_tmp = np.trace((np.transpose(Z) @ Z) + (np.transpose(Z_tilde) @ Z_tilde) - 2*s)
                    #If the Silhouette score M2 is lower than the previous one in M2_tmp, store the
                    #variable 'ii' to remove it after the for loop
                    if M2_tmp < M2:
                        M2 = M2_tmp
                        var_tmp = ii
                #Remove the variable from the matrix and the labels list
                self.X_tilde = np.delete(self.X_tilde, var_tmp, axis=1)
                self.labels = np.delete(self.labels, var_tmp, axis=0)
                self.var_num = np.delete(self.var_num, var_tmp, axis=0)
                print("Current number of variables: {}".format(self.X_tilde.shape[1]))

            return self.labels, self.var_num

        elif self._method.lower() == 'b2':
            print("Selecting global variables via B2 method..")
            #Number of variables before the elimination starts:
            max_var = self.X.shape[1]
            counter = 1
            #While the number of variables is larger than the number you want to retain ('m'), go on
            #with the elimination process:
            self._nPCs = self.X.shape[1] -1
            PCs, eigenValues__ = PCA_fit(self.X, self._nPCs)
            
            while max_var > self._n_ret:#self.retained:
                #Check which variable has the max weight on the last PC. Python starts to count from
                #zero, that's why the last number is "self._nPCs -1" and not "self._nPCs"
                max_on_last = np.max(np.abs(PCs[:,-counter]))
                argmax_on_last = np.argmax(np.abs(PCs[:,-counter]))
                #Delete the corresponding label
                self.labels = np.delete(self.labels, argmax_on_last, axis=0)
                #same for the numbers list
                self.var_num = np.delete(self.var_num, argmax_on_last, axis=0)
                #same for the corresponding PC rows
                PCs = np.delete(PCs, argmax_on_last, axis =0)
                print("Current number of variables: {}".format(len(self.labels)))
                #Get the current number of variables for the while loop
                counter += 1
                max_var = len(self.labels)


            return self.labels, self.var_num

        elif self._method.lower() == 'b4':
            print("Selecting global variables via B4 method..")


            if self._nPCs < self._n_ret:
                print("For the B4 it is not possible to choose a number of PCs lower than the required number of variables.")
                print("The number of PCs will be set equal to the required number of variables.")
                print(" ")
                self._nPCs = self._n_ret

            PCs, eigenValues__ = PCA_fit(self.X, self._nPCs)
            PVs = []
            self.var_num = []

            for ii in range(0, self._n_ret):
                #Check the largest weight on the first 'm' PCs and add it to the PVs list.
                argmax_= np.argmax(np.abs(PCs[:,ii]))
                PVs.append(self.labels[argmax_])
                #self.var_num = argmax_
                self.var_num.append(argmax_)
                #Set the variable weight to zero on all the PCs, to avoid repetition in the PVs list.
                #PCs = np.delete(PCs, argmax_, axis=0)
                PCs[argmax_,:] = 0

            return PVs, self.var_num

        elif self._method.lower() == 'procrustes_rotation':

            #Start with PCA, and compute the scores (Z)
            eigenvec = PCA_fit(self.X_tilde, self._nPCs)
            Z = self.X_tilde @ eigenvec[0]
            #Start the backward elimination:
            while self.X_tilde.shape[1] > self._n_ret:
                M2 = 1E12
                M2_tmp = 0
                var_tmp = 0

                for ii in range(0,self.X_tilde.shape[1]):
                    #Delete each variable from the original matrix (one at the time)
                    X_cut = np.delete(self.X_tilde, ii, axis=1)
                    #Compute the scores of the reduced matrix, after PCA
                    eigenvec = PCA_fit(X_cut, self._nPCs)
                    #ROTATE THE MODES VIA VARIMAX
                    rotated_PCs = varimax_rotation(self.X_tilde, eigenvec[0])
                    Z_tilde = X_cut @ rotated_PCs
                    #Compute the reduced scores covariance matrix, and then apply SVD
                    covZZ = np.transpose(Z_tilde) @ Z
                    u, s, vh = np.linalg.svd(covZZ, full_matrices=True)
                    #Compute the procrustes Analysis score M2 for the matrix without the 'ii' variable
                    M2_tmp = np.trace((np.transpose(Z) @ Z) + (np.transpose(Z_tilde) @ Z_tilde) - 2*s)
                    #If the Silhouette score M2 is lower than the previous one in M2_tmp, store the
                    #variable 'ii' to remove it after the for loop
                    if M2_tmp < M2:
                        M2 = M2_tmp
                        var_tmp = ii
                #Remove the variable from the matrix and the labels list
                self.X_tilde = np.delete(self.X_tilde, var_tmp, axis=1)
                self.labels = np.delete(self.labels, var_tmp, axis=0)
                self.var_num = np.delete(self.var_num, var_tmp, axis=0)
                print("Current number of variables: {}".format(self.X_tilde.shape[1]))

            return self.labels, self.var_num

        
        elif self._method.lower() == 'b2_rotation':
            print("Selecting global variables via B2 method..")
            #Number of variables before the elimination starts:
            max_var = self.X.shape[1]
            counter = 1
            #While the number of variables is larger than the number you want to retain ('m'), go on
            #with the elimination process:

            #Perform PCA:
            PCs, eigenValues__ = PCA_fit(self.X, self._nPCs)
            PCs = varimax_rotation(self.X_tilde, PCs)
            while max_var > self.retained:
                #Check which variable has the max weight on the last PC. Python starts to count from
                #zero, that's why the last number is "self._nPCs -1" and not "self._nPCs"
                max_on_last = np.max(np.abs(PCs[:,-counter]))
                argmax_on_last = np.argmax(np.abs(PCs[:,-counter]))
                #Delete the corresponding label
                self.labels = np.delete(self.labels, argmax_on_last, axis=0)
                #same for the numbers list
                self.var_num = np.delete(self.var_num, argmax_on_last, axis=0)
                #same for the corresponding PC rows
                PCs = np.delete(PCs, argmax_on_last, axis =0)
                print("Current number of variables: {}".format(len(self.labels)))
                #Get the current number of variables for the while loop
                counter += 1
                max_var = len(self.labels)
            
            return self.labels, self.var_num

        
        elif self._method.lower() == 'b4_rotation':
            print("Selecting global variables via B4 method..")

            if self._nPCs < self._n_ret:
                print("For the B4 it is not possible to choose a number of PCs lower than the required number of variables.")
                print("The number of PCs will be set equal to the required number of variables.")
                print(" ")
                self._nPCs = self._n_ret

            PCs, eigenValues__ = PCA_fit(self.X, self._nPCs)
            PCs = varimax_rotation(self.X_tilde, PCs)
            PVs = []
            self.var_num = []

            for ii in range(0, self._n_ret):
                #Check the largest weight on the first 'm' PCs and add it to the PVs list.
                argmax_= np.argmax(np.abs(PCs[:,ii]))
                PVs.append(self.labels[argmax_])
                self.var_num.append(argmax_)
                #Set the variable weight to zero on all the PCs, to avoid repetition in the PVs list.
                #PCs = np.delete(PCs, argmax_, axis=0)
                PCs[argmax_,:] = 0

            return PVs, self.var_num


        elif self.method.lower() == 'mccabe':
            '''
            The following lines for the McCabe PVs extraction were written by Nadia Bernair, from Université Libre de Bruxelles, Faculty of Electromechanical engineering (mechatronics), in the 
            context of her MA2 project: "Application of reduced order models and machine learning techniques for reacting flows".
            '''
            import itertools

            ret_index = list(itertools.combinations(range(self.X.shape[1]), self._n_ret))
            n_comb = len(ret_index)

            model = PCA(self.X)
            eigvec, eigval = model.fit()

            MC_1_min = 1.0e+16
            MC_2_min = 1.0e+16
            MC_3_min = 1.0e+16

            ret_name= np.linspace(0, self.X.shape[1]-1, self.X.shape[1], dtype=int) 
            disc_index = [0] * self.X.shape[1]    

            sumcov=0

            for i in range(0,n_comb-1):     
                print("Combination number: {} out of: {}".format(i, n_comb-1))
                p=0

                for j in range(0,self.X.shape[1]-1):

                    a= True 
                    for k in range(0,self._n_ret): 
                        if j == ret_index[i][k]:
                            a = False 
                            
                    if a == True:
                        disc_index[p]=j
                        p=p+1

                  
                sub_scal_data_1 = np.zeros((self.X.shape[0], self._n_ret))
                sub_name_1= np.zeros((1, self._n_ret))
                    
                
                sub_scal_data_2 = np.zeros((self.X.shape[0], p-1))
                sub_name_2= np.zeros((1, p-1))


                for j in range(self._n_ret):  
                    sub_scal_data_1[:,j] = self.X_tilde[:,ret_index[i][j]]
                    sub_name_1[0][j] = ret_index[i][j]
        
        
                for j in range(p-1):
                        sub_scal_data_2[:,j] = self.X_tilde[:,disc_index[j]]
                        sub_name_2[0][j] = disc_index[j]
    
                # compute the covariance matrix : variance en chaque dim (sym)

                cov_data_11 = (1/(self.X.shape[0] -1))*(np.transpose(sub_scal_data_1).dot(sub_scal_data_1))
                cov_data_22 = (1/(self.X.shape[0] -1))*(np.transpose(sub_scal_data_2).dot(sub_scal_data_2))
                cov_data_12 = (1/(self.X.shape[0] -1))*(np.transpose(sub_scal_data_1).dot(sub_scal_data_2))
                cov_data_21 = (1/(self.X.shape[0] -1))*(np.transpose(sub_scal_data_2).dot(sub_scal_data_1))

                cov_data_22_1 = np.empty(cov_data_22.shape, dtype=float)

                try:
                    cov_data_22_1 = cov_data_22 - cov_data_21.dot(np.linalg.inv(cov_data_11).dot(cov_data_12))
                    sumcov += cov_data_22_1
                except:                                         #if singular use as cov the average cov obtain with the previous comb
                    if i>0:
                        cov_data_22_1=sumcov/i
                    else: 
                        voc_data_22_1=sumcov
                
                
                eig_cov_data_22_1, ____ = LA.eig(cov_data_22_1)
                
                if self._McCriterion == 1:
                    
                    MC_1 = np.prod(eig_cov_data_22_1)       #the criterion : product of the eigenvalue = min
        
                    if MC_1 < MC_1_min:
                        MC_1_min = MC_1
                        ret_name = sub_name_1

                elif self._McCriterion == 2:
                    MC_2 = np.sum(eig_cov_data_22_1)
        
                    if MC_2 < MC_2_min:
                        MC_2_min = MC_2
                        ret_name = sub_name_1

                elif self._McCriterion == 3:
                    MC_3 = np.sum(eig_cov_data_22_1**2)

                    if MC_3 < MC_3_min:
                        MC_3_min = MC_3
                        ret_name = sub_name_1

                else:
                    raise Exception("The selected McCabe criterion of choice is not available. Please choose an integer between 1 and 3.")
                    exit()

                
            ret_names = [int(x) for x in ret_name[0]]
            Y = self.X[:, ret_names]

            return ret_names


        
        elif self.method.lower() == 'mccabe_rotation':
            '''
            The following lines for the McCabe PVs extraction (+ rotation) were written by Nadia Bernair, from Université Libre de Bruxelles, Faculty of Electromechanical engineering (mechatronics), in the 
            context of her MA2 project: "Application of reduced order models and machine learning techniques for reacting flows".
            '''

            import itertools

            ret_index = list(itertools.combinations(range(self.X.shape[1]), self._n_ret))
            n_comb = len(ret_index)

            model = PCA(self.X)
            eigvec, eigval = model.fit()

            #rotate the PCs
            eigvec = varimax_rotation(self.X_tilde, eigvec)

            MC_1_min = 1.0e+16
            MC_2_min = 1.0e+16
            MC_3_min = 1.0e+16

            ret_name= np.linspace(0, self.X.shape[1]-1, self.X.shape[1], dtype=int) 
            disc_index = [0] * self.X.shape[1]    

            sumcov=0

            for i in range(0,n_comb-1):     
                print("Combination number: {} out of: {}".format(i, n_comb-1))
                p=0

                for j in range(0,self.X.shape[1]-1):

                    a= True 
                    for k in range(0,self._n_ret): 
                        if j == ret_index[i][k]:
                            a = False 
                            
                    if a == True:
                        disc_index[p]=j
                        p=p+1

                  
                sub_scal_data_1 = np.zeros((self.X.shape[0], self._n_ret))
                sub_name_1= np.zeros((1, self._n_ret))
                    
                
                sub_scal_data_2 = np.zeros((self.X.shape[0], p-1))
                sub_name_2= np.zeros((1, p-1))


                for j in range(self._n_ret):  
                    sub_scal_data_1[:,j] = self.X_tilde[:,ret_index[i][j]]
                    sub_name_1[0][j] = ret_index[i][j]
        
        
                for j in range(p-1):
                        sub_scal_data_2[:,j] = self.X_tilde[:,disc_index[j]]
                        sub_name_2[0][j] = disc_index[j]
    
                # compute the covariance matrix : variance en chaque dim (sym)

                cov_data_11 = (1/(self.X.shape[0] -1))*(np.transpose(sub_scal_data_1).dot(sub_scal_data_1))
                cov_data_22 = (1/(self.X.shape[0] -1))*(np.transpose(sub_scal_data_2).dot(sub_scal_data_2))
                cov_data_12 = (1/(self.X.shape[0] -1))*(np.transpose(sub_scal_data_1).dot(sub_scal_data_2))
                cov_data_21 = (1/(self.X.shape[0] -1))*(np.transpose(sub_scal_data_2).dot(sub_scal_data_1))

                cov_data_22_1 = np.empty(cov_data_22.shape, dtype=float)

                try:
                    cov_data_22_1 = cov_data_22 - cov_data_21.dot(np.linalg.inv(cov_data_11).dot(cov_data_12))
                    sumcov += cov_data_22_1
                except:                                         #if singular use as cov the average cov obtain with the previous comb
                    if i>0:
                        cov_data_22_1=sumcov/i
                    else: 
                        voc_data_22_1=sumcov
                
                
                eig_cov_data_22_1, ____ = LA.eig(cov_data_22_1)
                
                if self._McCriterion == 1:
                    
                    MC_1 = np.prod(eig_cov_data_22_1)       #the criterion : product of the eigenvalue = min
        
                    if MC_1 < MC_1_min:
                        MC_1_min = MC_1
                        ret_name = sub_name_1

                elif self._McCriterion == 2:
                    MC_2 = np.sum(eig_cov_data_22_1)
        
                    if MC_2 < MC_2_min:
                        MC_2_min = MC_2
                        ret_name = sub_name_1

                elif self._McCriterion == 3:
                    MC_3 = np.sum(eig_cov_data_22_1**2)

                    if MC_3 < MC_3_min:
                        MC_3_min = MC_3
                        ret_name = sub_name_1

                else:
                    raise Exception("The selected McCabe criterion of choice is not available. Please choose an integer between 1 and 3.")
                    exit()

                
            ret_names = [int(x) for x in ret_name[0]]
            

            return ret_names




class SamplePopulation():
    '''
    This class contains a set of methods to consider only a subset of the original training
    matrix X. These methods should be used when the size of the training dataset is large, and
    becomes impractical to train the model on the whole training population.
    Three classes of sampling strategies are available:
    i) Simple random sampling; ii) Clustered sampling; iii) Conditioned sampling.
    The 'clustered' sampling can be accomplished either via KMeans, or LPCA.

    The simple random sampling just shuffle the data and takes the required amount of observations.
    The clustered approach groups the observations in different clusters, and from each of them
    a certain amount of samples are taken.


    --- PARAMETERS ---
    X:          RAW data matrix, uncentered and unscaled. It must be organized
                with the structure: (observations x variables).
    type X :    numpy array


    --- SETTERS ---
    _dimensions:            final dimensions (number of observations) of the sampled matrix
    type   _dimensions:     scalar

    _method:                method to be used for sampling
    type   _method:         string

    '''
    def __init__(self, X, *dictionary):
        #Initial training dataset (raw data).
        self.X = X
        self.__nObs = self.X.shape[0]
        self.__nVar = self.X.shape[1]
        
        #Choose the sampling strategy: random, clustered (KMeans, LPCA) or conditioned.
        self._method = 'KMeans'
        self.__condVec = False
        
        #Choose the dimensions of the sampled dataset.
        self._dimensions = 1
        
        #Predefined number of clusters, if 'cluster' (kmeans or lpca) or 'conditioned' are chosen
        self.__k = 32
        
        #Variable to use for the matrix conditioning in case of conditioned sampling
        self._conditioning =0

        if dictionary:
            settings = dictionary[0]
        
            try:
                self._method = settings["method"]

                if not isinstance(self._method, str):
                    raise Exception
                elif self._method.lower() != "random" and self._method.lower() != "lpca" and self._method.lower() != 'kmeans' and self._method.lower() != "conditioned":
                    raise Exception
            except:
                self._method = "random"
                warnings.warn("An exception occured with regard to the input value for the sampling method. It could be not acceptable, or not given to the dictionary.")
                print("\tIt will be automatically set equal to: random.")
                print("\tYou can ignore this warning if the sampling method has been assigned later via setter.")
                print("\tOtherwise, please check the conditions which must be satisfied by the input in the detailed documentation.")
            try:
                self._dimensions = settings["final_size"]

                if not isinstance(self._dimensions, int) or self._dimensions > self.X.shape[0] or self._dimensions < 0:
                    raise Exception
            except:
                self._dimensions = int(self.X.shape[0]/2)
                warnings.warn("An exception occured with regard to the input value for the sampled dataset dimensions. It could be not acceptable, or not given to the dictionary.")
                print("\tIt will be automatically set equal to: X.shape[0] / 2.")
                print("\tYou can ignore this warning if the sampled dataset dimensions has been assigned later via setter.")
                print("\tOtherwise, please check the conditions which must be satisfied by the input in the detailed documentation.")

    @property
    def sampling_strategy(self):
        return self._method

    @sampling_strategy.setter
    def sampling_strategy(self, new_string):
        self._method = new_string

        if not isinstance(self._method, str):
            self._method = "random"
            warnings.warn("An exception occured with regard to the input value for the sampling method. It could be not acceptable, or not given to the dictionary.")
            print("\tIt will be automatically set equal to: random.")
            print("\tPlease check the conditions which must be satisfied by the input in the detailed documentation.")
        elif self._method.lower() != "random" and self._method.lower() != "lpca" and self._method.lower() != 'kmeans' and self._method.lower() != "conditioned":
            self._method = "random"
            warnings.warn("An exception occured with regard to the input value for the sampling method. It could be not acceptable, or not given to the dictionary.")
            print("\tIt will be automatically set equal to: random.")
            print("\tPlease check the conditions which must be satisfied by the input in the detailed documentation.")

    @property
    def set_size(self):
        return self._dimensions

    @set_size.setter
    def set_size(self, new_value):
        self._dimensions = new_value

        if not isinstance(self._dimensions, int) or self._dimensions > self.X.shape[0] or self._dimensions < 0:
            self._dimensions = int(self.X.shape[0]/2)
            warnings.warn("An exception occured with regard to the input value for the sampled dataset dimensions. It could be not acceptable, or not given to the dictionary.")
            print("\tIt will be automatically set equal to: X.shape[0]/2.")
            print("\tPlease check the conditions which must be satisfied by the input in the detailed documentation.")

    @property
    def set_conditioning(self):
        return self._conditioning

    @set_conditioning.setter
    def set_conditioning(self, new_value):
        self._conditioning = new_value
        if not isinstance(self._conditioning, int):
            self._conditioning = new_value
            self.__condVec = True

    def fit(self):
            '''
            --- RETURNS ---
            miniX:      sampled matrix with the prescribed number of observations 
            type miniX: numpy matrix
            '''
            #Center and scale the matrix (auto preset). There's no harm in doing this even if the 
            #data are univariate, so the code does it by default just in case the input consists of multivariate data.
            self.X_tilde = center_scale(self.X, center(self.X,'mean'), scale(self.X, 'auto'))
            self.__batchSize = int(self._dimensions / self.__k)

            if self._method.lower() == 'random':
                #Randomly shuffle the observations and take the prescribed number
                np.random.shuffle(self.X)
                miniX = self.X[:self._dimensions,:]
            
            elif self._method.lower() == 'kmeans':
                #Perform KMeans and take (randomly) from each cluster a certain
                #batch to form the sampled matrix.
                model_KM = clustering.KMeans(self.X_tilde)
                model_KM.clusters = self.__k
                model_KM.initMode = True
                id = model_KM.fit()
                miniX = self.X[1:3,:]
                for ii in range(0, max(id)+1):
                    cluster_ = get_cluster(self.X, id, ii)
                    #If the cluster contains less observations than the batch
                    #size, then take all the cluster's observations.
                    if cluster_.shape[0] < self.__batchSize:
                        miniX = np.concatenate((miniX, cluster_), axis=0)
                    else:
                        #Otherwise, shuffle the cluster and take the first
                        #batchsize observations.
                        np.random.shuffle(cluster_)
                        miniX = np.concatenate((miniX, cluster_[:self.__batchSize,:]), axis=0)
            
            elif self._method.lower() == 'lpca':
                #Do the exact same thing done in KMeans, but using the LPCA
                #clustering algorithm. The number of PCs is automatically
                #assessed via explained variance. The number of LPCs is chosen
                #as LPCs = PCs/2
                global_model = PCA(self.X)
                optimalPCs = global_model.set_PCs()
                model = clustering.lpca(self.X_tilde)
                model.clusters = self.__k
                model.eigens = int(optimalPCs / 2)
                model.writeFolder = False
                id = model.fit()
                miniX = self.X[1:3,:]
                for ii in range(0, max(id)+1):
                    cluster_ = get_cluster(self.X, id, ii)
                    #If the cluster contains less observations than the batch
                    #size, then take all the cluster's observations.
                    if cluster_.shape[0] < self.__batchSize:
                        miniX = np.concatenate((miniX, cluster_), axis=0)
                    else:
                        np.random.shuffle(cluster_)
                        miniX = np.concatenate((miniX, cluster_[:self.__batchSize,:]), axis=0)

            elif self._method.lower() == 'conditioned':
                #Condition the dataset dividing the interval of one variable in
                #'k' bins and sample from each bin. The default variable to
                #condition with is '0'. This setting must be eventually modified
                #via setter.
                if not self.__condVec:
                    min_con = np.min(self.X[:,self._conditioning])
                    max_con = np.max(self.X[:,self._conditioning])
                else:
                    min_con = np.min(self._conditioning)
                    max_con = np.max(self._conditioning)
                #Compute the extension of each bin (delta_step). 100 bins are imposed for the partitioning of the vector:
                self.__kHardConditioning = 100
                local_batchSize = int(self._dimensions/self.__kHardConditioning)
                #compute the extension of each bin
                delta_step = ((max_con - min_con) / self.__kHardConditioning)
                counter = 0
                #var left is the minimum value of the conditioning variable:
                # min con --> | ....... conditioning vector ....... | <-- max con
                var_left = min_con
                #initialize the sampled matrix miniX
                miniX = self.X[1:3,:]
                while counter <= self.__kHardConditioning:
                    #Compute the two extremes for each bin, and take all the observations in
                    #the dataset which lie in the interval.
                    # var_left --> | ... bin_{i} ... | <-- var_right = var_left + delta_step
                    var_right = var_left + delta_step
                    #take all the observations in this interval: [var_left; var_right] and form a cluster
                    if not self.__condVec:
                        mask = np.logical_and(self.X[:,self._conditioning] >= var_left, self.X[:,self._conditioning] < var_right)
                    else:
                        mask = np.logical_and(self._conditioning >= var_left, self._conditioning < var_right)
                    cluster_ = self.X[mask,:]

                    #Also in this case, if the cluster size is lower than the
                    #batch size, take all the cluster.
                    if cluster_.shape[0] <= local_batchSize:
                        miniX = np.concatenate((miniX, cluster_), axis=0)
                        var_left += delta_step
                        counter+=1
                    else:
                        #otherwise, shuffle and take a number of observations equal to the bin's dimension:
                        np.random.shuffle(cluster_)
                        #add the new observations to the sampled matrix
                        miniX = np.concatenate((miniX, cluster_[:local_batchSize,:]), axis=0)
                        var_left += delta_step
                        counter+=1
            else:
                raise Exception("The selected sampling method is not valid. Please choose between: 'random', 'kmeans', 'lpca', 'conditioned'.")

            #it can happen that few observations are missing to reach the prescribed number of observations of
            #the sampled matrix. It is due to the fact that it can happen that the number of observations in each cluster
            #is lower than the number required for the bin, i.e., it is verified the condition: cluster_.shape[0] < self.__batchSize.
            
            #In this case, fill the gap with random observations from the training matrix to get the required number of observations
            if miniX.shape[0] < self._dimensions:
                np.random.shuffle(self.X)
                delta = self._dimensions - miniX.shape[0]

                miniX = np.concatenate((miniX, self.X[:delta,:]), axis=0)
            elif miniX.shape[0] > self._dimensions:
                miniX = miniX[:self._dimensions,:]

            return miniX



class Kernel_approximation:
    '''
    The aim of the following class is to provide the possibility to extend the use of kernel-based
    learning to large matrices, by means of algorithms for the approximation of the Kernel matrix.

    You can find additional information regarding the theoretical aspects behind the algorithms in:
    [1] P. Indreas, M.W. Mahoney. Journal of machine learning research, pages 2153-2175, 2005.
    [2] M. Li, J. T. Kwok, B-L Lu. ICML 2010 Proceedings, 27th International Conference on Machine Learning, pages 631-638, 2010.
    [3] Kumar, Sanjiv, Mehryar Mohri, and Ameet Talwalkar. "Ensemble nystrom method." (2009).
    [4] T. Hofmann, B. Scholkopf, A. Smola. Annals of statistics, pages 1171-1220, 2008.
    [5] F. Pourkamali, S. Becker. Neurocomputing, pages 261-272, 2019.

    The core of the following Python code for the Kernel matrix approximation has been developed by Grégoire Stéphane 
    Corluy (Gregoire.Stephane.Corluy@vub.be) in the context of his MA1 thesis at Université Libre de Bruxelles and 
    Vrije Universiteit Brussels, Faculty of Electromechanical engineering.
    '''

    def __init__(self, X, *args, **kwargs):

        self.X = X
        self._number_of_rows = self.X.shape[0]
        self._number_of_columns = self.X.shape[1]

        #predefined
        if kwargs.get('kernelType'):
            self._kernelType = kwargs['kernelType']
        if kwargs.get('toCenter'):
            self._center = True
        else:
            self._center = False
        if kwargs.get('toScale'):
            self._scale = kwargs['toScale']
        else:
            self._scale = False
        if kwargs.get('centerCrit'):
            self._centering = kwargs['centerCrit']
        if kwargs.get('scalCrit'):
            self._scaling = kwargs['scalCrit']
        if kwargs.get('numToPick'):
            self._numberToPick = kwargs['numToPick']
        if kwargs.get('sigma'):
            self._sigma = kwargs['sigma']
        if kwargs.get('rank'):
            self._rank = kwargs['rank'] 
        if kwargs.get('p'):
            self._p = kwargs['p']

        #optional
        if kwargs.get('d'):
            self._d = kwargs['d']
        else:
            self._d = 2
        if kwargs.get('c'):
            self._c = kwargs['c']
        else:
            self._c = 1
        if kwargs.get('nu'):
            self._nu = kwargs['nu']
        else:
            self._nu = 1
        if kwargs.get('rho'):
            self._rho = kwargs['rho']
        else:
            self._rho = 1
        if kwargs.get('sigmaMatern'):
            self._sigmaMatern = kwargs['sigmaMatern']
        else:
            self._sigmaMatern = 1


        if args:
            settings = args[0]
        
            try:
                self._numberToPick = settings["number_to_pick"]

                if not isinstance(self._numberToPick, int):
                    raise Exception(" ")
                    exit()
                    
            except:
                print("The number of columns to pick has not been given in input to the dictionary via the predefined entry: dictionary['number_to_pick']")
                print("\tIt will be automatically set equal to: 2000.")
            
            try:
                self._sigma = settings["sigma"]

                if not isinstance(self._sigma, int) and not isinstance(self._sigma, float):
                    raise Exception(" ")
                    exit()
            except:
                print("The parameter sigma has not been given in input to the dictionary via the predefined entry: dictionary['sigma']")
                print("\tIt will be automatically set equal to: 1.")

            try:
                self._rank = settings["rank"]

                if not isinstance(self._rank, int):
                    raise Exception(" ")
                    exit()

            except:
                print("The parameter rank has not been given in input to the dictionary via the predefined entry: dictionary['rank']")
                print("\tIt will be automatically set equal to: 200.")

            try:
                self._p = settings["number_of_matrices"]

            except:
                print("The number of matrices has not been specified. This could be a problem only if the Nystrom ensemble algorith is chosen.")
                print("\tIt will be automatically set equal to: 1.")

            
            try:
                self._center = settings["center"]
                if not isinstance(self._center, bool):
                    raise Exception
            except:
                warnings.warn("An exception occured with regard to the input value for the centering decision. It could be not acceptable, or not given to the dictionary.")
                print("\tIt will be automatically set equal to: true.")
                print("\tYou can ignore this warning if the centering decision has been assigned later via setter.")
                print("\tOtherwise, please check the conditions which must be satisfied by the input in the detailed documentation.")
            try:
                self._centering = settings["centering_method"]
                if not isinstance(self._centering, str):
                    raise Exception
                elif self._centering.lower() != "mean" and self._centering.lower() != "min":
                    raise Exception
            except:
                warnings.warn("An exception occured with regard to the input value for the centering criterion . It could be not acceptable, or not given to the dictionary.")
                print("\tIt will be automatically set equal to: mean.")
                print("\tYou can ignore this warning if the centering criterion has been assigned later via setter.")
                print("\tOtherwise, please check the conditions which must be satisfied by the input in the detailed documentation.")
            try:
                self._scale = settings["scale"]
                if not isinstance(self._scale, bool):
                    raise Exception
            except:
                warnings.warn("An exception occured with regard to the input value for the scaling decision. It could be not acceptable, or not given to the dictionary.")
                print("\tIt will be automatically set equal to: true.")
                print("\tYou can ignore this warning if the scaling decision has been assigned later via setter.")
                print("\tOtherwise, please check the conditions which must be satisfied by the input in the detailed documentation.")
            try: 
                self._scaling = settings["scaling_method"]
                if not isinstance(self._scaling, str):
                    raise Exception
                elif self._scaling.lower() != "auto" and self._scaling.lower() != "vast" and self._scaling.lower() != "pareto" and self._scaling.lower() != "range":
                    raise Exception
            except:
                warnings.warn("An exception occured with regard to the input value for the scaling criterion. It could be not acceptable, or not given to the dictionary.")
                print("\tIt will be automatically set equal to: auto.")
                print("\tYou can ignore this warning if the scaling criterion has been assigned later via setter.")
                print("\tOtherwise, please check the conditions which must be satisfied by the input in the detailed documentation.")

            #Optional 
            self._kernelType = settings["kernel_type"]
            self._d = settings["polynomial_degree"]
            self._c = settings["polynomial_freeParameter"]
            self._nu = settings["nu_matern"]
            self._rho = settings["rho_matern"]
            self._sigmaMatern = settings["sigma_matern"]

    
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

        
    @staticmethod
    def uniformRandomSamp(number_of_samples_possible_to_pick, number_of_samples_to_pick):
        #make a list of the possible indices that can be picked
        indices = np.linspace(0,number_of_samples_possible_to_pick-1,number_of_samples_possible_to_pick)
        #pick a number of indices
        selected_indices = random.sample(list(indices),number_of_samples_to_pick)

        return selected_indices


    @staticmethod
    def randomSamplingWeightsDiagonal(Data, number_of_samples_to_pick):
        n_obs = Data.shape[0]
        probabilities = np.zeros(n_obs) #set all the probablities to zero
        for i in range(n_obs):
            probabilities[i] = RBFkernel(Data[i],Data[i])**2  #calculate the probablity of each component; equal to the square of the diagonalelements of the kernelmatrix
        
        total_probabilities = sum(probabilities)   #total probablity
        probabilities = np.divide(probabilities,total_probabilities)  #scale to have in total a probability of 100%
        
        indices = np.linspace(0,n_obs-1,n_obs)  #make an array of indices from 0 to the index of the last element
        selected_indices = np.random.choice(list(indices),number_of_samples_to_pick,probabilities.all()) #take randomly a number of samples with weights and with replacement
        
        return selected_indices


    @staticmethod
    def RBFkernel(x1,x2, sigma, **kwargs):
        #x1 and x2 are the vectors that we are "comparing"
        
        if not kwargs.get('selfKernel'):
            
            x1 = np.array(x1)
            x2 = np.array(x2)
            xtot = np.subtract(x1,x2)
            xtot_norm_square = np.dot(xtot,xtot.T)
            kernel = np.exp(-xtot_norm_square/(2*sigma**2))
        else:
            from scipy.spatial.distance import pdist, squareform
            pairwise_dists = squareform(pdist(x1, 'euclidean'))
            kernel = np.exp(-pairwise_dists**2 / (2*sigma ** 2))

        
        return kernel


    @staticmethod
    def PolynomialKernel(x1, x2, d, c):
        #x1 and x2 are the vectors that we are "comparing"
        #d is the degree of the polynomial
        #c is a free parameter trading off the influence of higher-order versus lower-order terms in the polynomial
        x1 = np.array(x1)
        x2 = np.array(x2)
        dot_product = np.dot(x1,x2)

        kernel = (dot_product + c)**d
        
        return kernel

    @staticmethod
    def Maternkernel(x1, x2 , nu, rho, sigma):
        #x1 and x2 are the vectors that we are "comparing"
        #nu is typically 1/2, 3/2 or 5/2
        
        #calculate the distance between the two vectors
        x1 = np.array(x1)
        x2 = np.array(x2)
        xtot = np.subtract(x1,x2)
        distance = np.dot(xtot,xtot)**0.5

        if(distance==0):    #if distance is equal to zero, the function return NaN, so make the distance very small but not zero
            distance = 1e-16
        
        #calculate in parts and multiply at the end each part
        part_1 = (sigma**2)*(2**(1-nu))/math.gamma(nu)
        part_2 = (np.sqrt(2*nu)*distance/rho)**nu
        part_3 = sp.kv(nu, np.sqrt(2*nu)*distance/rho)

        kernel = part_1*part_2*part_3
        
        return kernel


    def Nystrom_computeWC(self):
        #indices that were picked uniform randomly
        #these are the indices that will be used to make the C and W matrix
        if self._numberToPick > self._number_of_rows:
            print("The number of samples cannot be lower than the number of the observations of X.")
            print("Exiting with error..")
            exit()

        indices = self.uniformRandomSamp(self._number_of_rows, self._numberToPick)
        
        #sort the indices from small to big so that the for loops for W below can work
        indices = sorted(indices)

        #initialize the C and W matrix
        C = np.zeros((self._number_of_rows, self._numberToPick))
        W = np.zeros((self._numberToPick, self._numberToPick))

        #make some counters that will be used in the next loop
        counter_row = 0
        counter_column_W = 0
        counter_row_C = 0

        #fill in the W and C matrix with the sampled columns
        for i in indices:
            counter_column_W = 0
            counter_row_C = 0
            for m in range(self._number_of_rows):
                #calculate the matrix C and if the element matches with an element from W, add the element also to W
                                
                #could speed up more by using the symmetry of the matrices, but it is complicated to implement
                #as the symmetric matrix W is not clearly at one place in the matrix C, but the elements of matrix W are spread in the whole matrix C
                if self._kernelType.lower() == "rbf":
                    kernel_calculated = self.RBFkernel(self.X[int(i)],self.X[int(m)], self._sigma) #calculate the kernel only once for the common elements of C and W
                elif self._kernelType.lower() == "matern":
                    kernel_calculated = self.Maternkernel(self.X[int(i)],self.X[int(m)], self._nu, self._rho, self._sigmaMatern) #calculate the kernel only once for the common elements of C and W
                elif self._kernelType.lower() == "polynomial" or self._kernelType.lower() == "poly":
                    kernel_calculated = self.PolynomialKernel(self.X[int(i)],self.X[int(m)], self._d, self._c) #calculate the kernel only once for the common elements of C and W
                else:
                    print("The chosen Kernel is not available. Please check the spelling in your dictionary, or the available kernels in the documentation.")
                    print("exiting with error.")
                    exit()
                C[counter_row_C][counter_row] = kernel_calculated
                
                #see if the calculated kernel is also an element of W
                if(counter_column_W<=self._numberToPick-1): #avoid to be out the bounds of the vector
                    if(m==indices[counter_column_W]): #W is a submatrix of C, only sometimes add the kernel to W
                        W[counter_row][counter_column_W] = kernel_calculated
                        counter_column_W+=1
                counter_row_C +=1
            counter_row +=1
        return W, C  


    def Nystrom_standard(self):
        self.X = self.preprocess_training(self.X, self._center, self._scale, self._centering, self._scaling)

        W, C = self.Nystrom_computeWC()

        #eigenvalue decomposition of W
        W_eig_decomp = np.linalg.eig(W)
        W_diagonals = W_eig_decomp[0]
        W_eigvect = W_eig_decomp[1]
        
        indices_max_eigval = np.zeros(self._rank)
        copy_eigval = list(W_diagonals)

        #select the k biggest eigenvalues to make afterwards the low rank k kernelmatrix
        for i in range(self._rank):
            index_max = copy_eigval.index(max(copy_eigval)) #index of the biggest element
            indices_max_eigval[i] = index_max
            copy_eigval[index_max] = float('-inf')  #eigenvalue cannnot be selected again, because it is set to the minimal value possible

        #make diagonalmatrix containing the biggest eigenvalues
        #and take the equivalent eigenvectors
        max_eigval = np.zeros(self._rank)
        max_eigvect = np.zeros((self._numberToPick,self._rank))
        counter = 0
        for i in indices_max_eigval:
            max_eigval[counter] = W_diagonals[int(i)]
            eigvect_column = [W_eigvect[int(j)][int(i)] for j in range(self._numberToPick)]
            counter_row = 0
            for col_element in eigvect_column:
                max_eigvect[counter_row][counter] = col_element
                counter_row+=1
            counter+=1

        #make a diagonal matrix with the biggest eigenvalues
        sigma_k = np.diag(max_eigval)

        #make L_nys
        L_nys = np.matmul(np.matmul(C,max_eigvect),np.linalg.inv(sigma_k)**0.5)

        #L_nys_transpose*L_nys to make a (k x k) matrix
        matrix_rank_k = np.matmul(np.transpose(L_nys),L_nys)

        #eigenvalue decomposition of this rank k matrix
        matrix_rank_k_eig_decomp = np.linalg.eig(matrix_rank_k)
        matrix_rank_k_diagonals = matrix_rank_k_eig_decomp[0]
        matrix_rank_k_eigvect = matrix_rank_k_eig_decomp[1]

        #diagonal matrix of the eigenvalues
        sigma_tilde = np.diag(matrix_rank_k_diagonals)

        #eigenvectors
        U_nys_rank_k = np.matmul(np.matmul(L_nys,matrix_rank_k_eigvect),np.linalg.inv(sigma_tilde)**0.5)

        lambda_nys_rank_k = sigma_tilde

        #K_approximation = U*diagonal matrix of the eigenvalues*U_transpose
        K_approximation = np.matmul(np.matmul(U_nys_rank_k,lambda_nys_rank_k),np.transpose(U_nys_rank_k))

        return K_approximation


    def Nystrom_ensemble(self):
        #all the calculated kernelmatrices are stored in the matrix K_temporary
        K_temporary = np.zeros(self._number_of_rows)  
        #use p times the standard Nyström algorithm and combine the approximated kernelmatrices together afterwards
        for iteration in range(self._p):
            print("Ensemble Nyström Method  --> iteration n.: {}".format(iteration))

            W, C = self.Nystrom_computeWC()

            #calculate the pseudo inverse of W
            W_pinv = np.linalg.pinv(W)
            
            #K_approximation = C*W_pinv*C_transpose
            Temporary_result = np.matmul(C,W_pinv)
            K_approximation = np.matmul(Temporary_result,np.transpose(C))
            
            #K_ensemble = sum(1/p*K_approximated) with 1/p the weigth of each kernelmatrix
            #in this case it is a uniform weigth
            K_temporary = np.array(K_temporary) + 1/self._p*np.array(K_approximation)  #uniform coefficients

        #after the loop K_temporary becomes the final approximation of the kernelmatrix
        K_approximation = K_temporary
        
        return K_approximation


    def QRdecomposition(self):
        W, C = self.Nystrom_computeWC()

        #calculate the pseudoinverse of W    
        W_pinv = np.linalg.pinv(W)

        #calculate the QR decomposition of C
        Q, R= np.linalg.qr(C)

        #eigenvalue decomposition of the matrix R*W_pseudoinverse*R_transpose
        eig_decomp = np.linalg.eig(np.matmul(R,np.matmul(W_pinv,np.transpose(R))))
        sigma = eig_decomp[0]  #eigenvalues
        V = eig_decomp[1]  #eigenvectors
        
        indices_max_eigval = np.zeros(self._rank)
        copy_eigval = list(sigma)

        #select the k biggest eigenvalues to make afterwards the low rank k kernelmatrix
        for i in range(self._rank):
            index_max = copy_eigval.index(max(copy_eigval)) #index of the biggest element
            indices_max_eigval[i] = index_max
            copy_eigval[index_max] = float('-inf')  #eigenvalue cannnot be selected again, because it is set to the minimal value possible

        #make diagonalmatrix containing the biggest eigenvalues
        #and take the equivalent eigenvectors
        max_eigval = np.zeros(self._rank)
        max_eigvect = np.zeros((self._numberToPick,self._rank))
        counter = 0
        for i in indices_max_eigval:
            max_eigval[counter] = sigma[int(i)]
            eigvect_column = [V[int(j)][int(i)] for j in range(self._numberToPick)]
            counter_row = 0
            for col_element in eigvect_column:
                max_eigvect[counter_row][counter] = col_element
                counter_row+=1
            counter+=1

        #make a diagonal matrix with the biggest eigenvalues
        sigma_k = np.diag(max_eigval)

        #eigenvectors of the k biggest eigenvalues
        U = np.matmul(Q,max_eigvect)  #eigenvectors of the low rank k Kernelmatrix

        #rebuild the approximated kernelmatrix with the eigenvectors and eigenvalues of the biggest eigenvalues
        #K_rank_k = U*sigma_k*U_transpose
        K_approximation = np.matmul(U,np.matmul(sigma_k,np.transpose(U)))

        return K_approximation