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
            except:
                self._nPCs = self.X.shape[1]-1
                print("Number of PCs to retain not given to dictionary. It will be automatically set equal to X.shape[1]-1.")
                print("You can ignore this warning if the number of PCs has been assigned later via setter.")
            try:
                self._center = settings["center"]
            except:
                self._center = True
            try:
                self._centering = settings["centering_method"]
            except:
                self._centering = "mean"
            try:
                self._scale = settings["scale"]
            except:
                self._scale = True 
            try: 
                self._scaling = settings["scaling_method"]
            except:
                self._scaling = "auto"
            try:
                self._plot_explained_variance = settings["enable_plot_variance"]
            except:
                self._plot_explained_variance = True 
            try:
                self._assessPCs = settings["set_criterion_autoPCs"]
            except:
                self._assessPCs = "var"
            try:
                self._threshold_var = settings["variance_to_explain"]
            except:
                self._threshold_var = 0.95
            try:
                self._num_to_plot = settings["variable_to_plot"]
            except:
                self._num_to_plot = 0

    @property
    def eigens(self):
        return self._nPCs

    @eigens.setter
    @accepts(object, int)
    def eigens(self, new_value):
        self._nPCs = new_value

        if self._nPCs <= 0:
            raise Exception("The number of Principal Components must be a positive integer. Exiting..")
            exit()
        elif self._nPCs >= self.n_var:
            raise Exception("The number of PCs exceeds (or is equal to) the number of variables in the data-set. Exiting..")
            exit()

    @property
    def to_center(self):
        return self._center

    @to_center.setter
    @accepts(object, bool)
    def to_center(self, new_bool):
        self._center = new_bool


    @property
    def centering(self):
        return self._centering

    @centering.setter
    @allowed_centering
    def centering(self, new_string):
        self._centering = new_string


    @property
    def to_scale(self):
        return self._scale

    @to_scale.setter
    @accepts(object, bool)
    def to_scale(self, new_bool):
        self._scale = new_bool


    @property
    def scaling(self):
        return self._scaling

    @scaling.setter
    @allowed_scaling
    def scaling(self, new_string):
        self._scaling = new_string

    @property
    def plot_explained_variance(self):
        return self._plot_explained_variance

    @plot_explained_variance.setter
    @accepts(object, bool)
    def plot_explained_variance(self, new_bool):
        self._plot_explained_variance = new_bool

    @property
    def set_explained_variance_perc(self):
        return self._explained_variance

    @set_explained_variance_perc.setter
    @accepts(object, float)
    def set_explained_variance_perc(self, new_value):
        self._explained_variance = new_value

    @property
    def set_PCs_method(self):
        return self._assessPCs

    @set_PCs_method.setter
    def set_PCs_method(self, new_method):
        self._assessPCs = new_method

    @property
    def set_num_to_plot(self):
        return self._num_to_plot

    @set_num_to_plot.setter
    @accepts(object, int)
    def set_num_to_plot(self, new_number):
        self._num_to_plot = new_number


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


        --- RETURNS ---
        explained:      percentage of explained variance 
        type explained: scalar  
        '''
        explained_variance = np.cumsum(self.evals)/sum(self.ALLevals)
        explained = explained_variance[-1]
        #If the plot boolean is True, produce an image to show the explained variance curve.
        if self._plot_explained_variance:
            matplotlib.rcParams.update({'font.size' : 18, 'text.usetex' : True})
            fig = plt.figure()
            axes = fig.add_axes([0.15,0.15,0.7,0.7], frameon=True)
            axes.plot(np.linspace(1, len(explained_variance), len(explained_variance)), explained_variance, color='b', marker='s', linestyle='-', linewidth=2, markersize=4, markerfacecolor='b', label='Cumulative explained')
            axes.plot([self.eigens, self.eigens], [explained_variance[0], explained_variance[-1]], color='r', marker='s', linestyle='-', linewidth=2, markersize=4, markerfacecolor='r', label='Explained by {} PCs'.format(self.eigens))
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


        --- RETURNS ---
        scores:      matrix of the scores 
        type scores: numpy matrix  
        '''
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
        axes = fig.add_axes([0.25,0.15,0.7,0.7], frameon=True)
        axes.plot(self.X[:,self._num_to_plot], self.X[:,self._num_to_plot], color='r', linestyle='-', linewidth=2, markerfacecolor='b')
        axes.scatter(self.X[:,self._num_to_plot], reconstructed_[:,self._num_to_plot], 1, color= 'k')
        axes.set_xlabel('Original variable')
        axes.set_ylabel('Reconstructed from PCA manifold')
        plt.xlim(min(self.X[:,self._num_to_plot]), max(self.X[:,self._num_to_plot]))
        plt.ylim(min(self.X[:,self._num_to_plot]), max(self.X[:,self._num_to_plot]))
        #axes.set_title('Parity plot')
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
        input_eigens = self.eigens
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
            for jj in range(input_eigens, scores.shape[1]):
                t_sq += scores[ii,jj]**2
                lam_j += eigval[jj]
            scores_dist[ii] = t_sq/(lam_j + TOL)

        #Now compute the distance distribution, and delete the observations in the
        #upper 3% (done in the while loop) to get the outlier-free matrix.

        #Divide the distance vector in 100 bins
        n_bins = 100
        min_interval = np.min(scores_dist)
        max_interval = np.max(scores_dist)

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
        PCs, eigval = self.fit()

        mu_X = center(self.X, self.centering)
        sigma_X = scale(self.X, self.scaling)

        X_cs = center_scale(self.X, mu_X, sigma_X)


        epsilon_rec = X_cs - X_cs @ PCs @ PCs.T
        sq_rec_oss = np.power(epsilon_rec, 2)

        #Now compute the distance distribution, and delete the observations in the
        #upper 3% (done in the while loop) to get the outlier-free matrix.

        #Divide the distance vector in 100 bins
        n_bins = 100
        min_interval = np.min(sq_rec_oss)
        max_interval = np.max(sq_rec_oss)

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

        new_mask = np.where(bin_id > new_counter)
        self.X = np.delete(self.X, new_mask, axis=0)

        return self.X, bin_id, new_mask


    def outlier_removal_multistep(self):
        '''
        Parente, Alessandro, and James C. Sutherland. Combustion and flame 160.2 (2013): 340-350.


        This function removes the outlier via PCA. Firstly, the input matrix is trimmed by means of
        the calculation of the Mahalanobis distance: a very small percentage (0.01 - 0.1%) of the 
        observations are discarded.
        After that, an iterative algorithm is started for outlier removal, and the convergence is reached
        when the data kurtosis' change is below a fixed threshold between two consecutive iterations.


        --- RETURNS ---
        X:         the matrix without outliers
        type X:    numpy matrix
        
        '''

        from scipy.stats import kurtosis

        convergence = False
        iterations = 0
        kurtosis_ = 1
        conv_tol = 1E-6
        iterMax = 150
        TOL = 1E-16

        while not convergence:
            #Trim the input data to have a robust Mahalanobis distance, after PCA
            input_eigens = self.eigens
            self.eigens = self.X.shape[1]-1

            PCs, eigval = self.fit()
            scores = self.get_scores()
            mahalanobis_ = np.empty((self.X.shape[0],), dtype=float)

            #The observations associated with large values of DM (mahal) are classified as outliers
            #and then discarded
            for ii in range(0,self.X.shape[0]):
                t_sq = 0
                lam_j = 0
                for jj in range(0, self.X.shape[1]-1):
                    t_sq += scores[ii,jj]**2
                    lam_j += eigval[jj]
                mahalanobis_[ii] = t_sq/(lam_j + TOL)

            #A fraction alpha (typically 0.01%-0.1%) of the data points characterized by the largest
            #value of DM are classified as outliers and removed.
            if iterations < 20:
                alpha = 0.000007
            else:
                alpha = 0

            #compute the new number of observations after the trim factor:
            trim = int((1-alpha)*self.X.shape[0])
            to_trim = np.argsort(mahalanobis_)

            new_mask = to_trim[:trim]
            self.X = self.X[new_mask,:]

            print("X trimmed dim: {}".format(self.X.shape))

            #Compute the PCA scores. Override the eventual number of PCs: ALL the
            #PCs are needed, as the outliers are also given by the last PCs examination
            PCs, eigval = self.fit()
            scores = self.get_scores()
            w_scores = scores/np.sqrt(eigval)
            #Select as "important" the PCs which explain the 20% of the average eigenvalue
            self.eigens = input_eigens


            #compute the curtosis of the new w_scores = u_scores/sqrt(eigenvalues)
            new_kurt = np.mean(kurtosis(w_scores[:,:self.eigens]))
            #check if the kurtosis variation is below the fixed threshold, to activate convergence
            check_conv = (new_kurt - np.mean(kurtosis_))/(new_kurt + TOL)
            print("Delta Kurtosis: {}".format(check_conv))

            if check_conv <= conv_tol or iterations >= iterMax:
                convergence = True
                break

            mahalanobis_ = np.empty((self.X.shape[0],), dtype=float)
            scores_dist = np.empty((self.X.shape[0],), dtype=float)
            #Compute again the Mahalanobis distance for the robust set of scores, evaluating the
            #important PCs
            for ii in range(0,self.X.shape[0]):
                t_sq = 0
                lam_j = 0
                for jj in range(0, self.eigens):
                    t_sq += scores[ii,jj]**2
                    lam_j += eigval[jj]
                mahalanobis_[ii] = t_sq/(lam_j + TOL)
            #remove the points in the 99th quantile
            to_remove = np.quantile(mahalanobis_, 0.9995)
            new_mask = np.where(mahalanobis_ > to_remove)

            #Do the same, but on the very last PCs.
            r = len(np.where(eigval < 0.2*np.mean(eigval))[0])

            for ii in range(0,self.X.shape[0]):
                t_sq = 0
                lam_j = 0
                for jj in range(scores.shape[1]-r+1, scores.shape[1]):
                    t_sq += scores[ii,jj]**2
                    lam_j += eigval[jj]
                scores_dist[ii] = t_sq/(lam_j + TOL)
            #remove also here the scores in the 99th quantile
            to_remove2 = np.quantile(scores_dist, 0.9995)
            new_mask2 = np.where(scores_dist >= to_remove2)
            #merge the idx of the points which were selected in the two steps, and delete
            #the repetitions (unique)
            temp = np.concatenate((new_mask, new_mask2), axis=1)
            to_delete = np.unique(temp)

            print('Observations to delete: {}'.format(len(to_delete)))
            #Delete the points from the matrix
            self.X = np.delete(self.X, to_delete, axis=0)

            print("Iteration number: {}".format(iterations))
            iterations +=1
            #store the "previous" Kurtosis
            kurtosis_ = kurtosis(w_scores[:,:self.eigens])


            print("The training matrix dimensions without outliers are: {}x{}".format(self.X.shape[0], self.X.shape[1]))


        return self.X


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
                except:
                    self._nPCs = self.X.shape[1]-1
                    print("Number of PCs to retain not given to dictionary. It will be automatically set equal to X.shape[1]-1.")
                    print("You can ignore this warning if the number of PCs has been assigned later via setter.")
                try:
                    self._center = settings["center"]
                except:
                    self._center = True
                try:
                    self._centering = settings["centering_method"]
                except:
                    self._centering = "mean"
                try:
                    self._scale = settings["scale"]
                except:
                    self._scale = True 
                try: 
                    self._scaling = settings["scaling_method"]
                except:
                    self._scaling = "auto"
                try:
                    self._path_to_idx = settings["path_to_idx"]
                except:
                    self._path_to_idx = ' '
                try:
                    self._clust_to_plot = settings["cluster_to_plot"]
                except:
                    self._clust_to_plot = 0
                try:
                    self._num_to_plot = settings["PC_to_plot"]
                except:
                    self._num_to_plot = 0
            


    @property
    def path_to_idx(self):
        return self._path_to_idx

    @path_to_idx.setter
    def path_to_idx(self, new_string):
        self._path_to_idx = new_string

    @property
    def clust_to_plot(self):
        return self._clust_to_plot

    @clust_to_plot.setter
    @accepts(object, int)
    def clust_to_plot(self, new_number):
        self._clust_to_plot = new_number

    @staticmethod
    def get_idx(path):

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
        #because they will be used in the future.
        mu_global = center(self.X, self.centering)
        sigma_global = scale(self.X, self.scaling)

        #Perform LPCA
        self.LPCs, self.u_scores, self.Leigen, self.centroids = self.fit()

        #Considering the idx of that set of partitio
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
            except:
                self._nPCs = self.X.shape[1]-1
                print("Number of PCs to retain not given to dictionary. It will be automatically set equal to X.shape[1]-1.")
                print("You can ignore this warning if the number of PCs has been assigned later via setter.")
            try:
                self._center = settings["center"]
            except:
                self._center = True
            try:
                self._centering = settings["centering_method"]
            except:
                self._centering = "mean"
            try:
                self._scale = settings["scale"]
            except:
                self._scale = True 
            try: 
                self._scaling = settings["scaling_method"]
            except:
                self._scaling = "auto"
            try:
                self._kernel = settings["selected_kernel"]
            except:
                self._kernel = 'rbf'

    @property
    def retained(self):
        return self._n_ret

    @property
    def kernel_type(self):
        return self._kernel

    @kernel_type.setter
    def kernel_type(self, new_string):
        self._kernel = new_string



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
        #from scipy.spatial.distance import pdist, squareform

        self.X_tilde = KPCA.preprocess_training(self.X, self.to_center, self.to_scale, self.centering, self.scaling)

        from sklearn.metrics.pairwise import rbf_kernel
        from sklearn.metrics.pairwise import polynomial_kernel
        from sklearn.preprocessing import KernelCenterer

        import time 
        
        t = time.time()
        

        if self._kernel == 'polynomial':
            print("Computing the Kernel (poly)..")
            Kernel = polynomial_kernel(self.X_tilde, degree=3, gamma=1/self._n_ret)
        elif self._kernel == 'rbf':
            print("Computing the Kernel (rbf)..")
            Kernel = rbf_kernel(self.X_tilde, gamma=1/self.X_tilde.shape[0])
        else:
            raise Exception("The selected Kernel is not supported. Exiting with error..")
            exit()

        elapsed_kernel = time.time() - t   
        print("Kernel computed in {} s.".format(elapsed_kernel)) 
        
        print("Centering the Kernel..")
        tk2 = time.time()
        transformer = KernelCenterer().fit(Kernel)
        K3 = transformer.transform(Kernel)
        elapsed_k2 = time.time() - tk2 
        print("Kernel centered in {} s.".format(elapsed_k2))
       
        print("Decomposing Kernel with fast SVD algorithm..")
        tSVD = time.time()
        self.Zt, self.A, singularVal = fastSVD(K3, self._nPCs)
        elapsed_SVD = time.time() - tSVD
        print("Decomposition accomplished in {} s.".format(elapsed_SVD))

        #return modes (n x k)
        return self.Zt, self.A, singularVal



class variables_selection(PCA):
    '''
    In many applications, rather than reducing the dimensionality considering a new set of coordinates
    which are linear combination of the original ones, the main interest is to achieve a dimensionality
    reduction selecting a subset of m variables from the original set of p variables.
    
    Three methods for variables selection via PCA are implemented in this class:
    i) Method B2 backward;
    ii) Method B4 forward;
    iii) Variables selection via PCA and Procustes Analysis, by means of the Krzanovski iterative algorithm [b]

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

    iii)    Variables selection via PCA and Procustes Analysis:
    The iterative variable selection algorithm introduced by Krzanovski is based on the following steps (1-3):
    1.  The dimensionality of m is initially set equal to p.
    2.  Each variable is deleted from the matrix X, obtaining p ~X matrices. The
        corresponding scores matrices are computed by means of PCA. For each of them, a Procustes Analysis
        is performed with respect to the scores of the original matrix X, and the corresponding M2 coeffcient is computed.
    3.  The variable which, once excluded, leads to the smallest M2 coefficient is deleted from the matrix.
    4.  Steps 2 and 3 are repeated until m variables are left.

    iv)  Variables selection via PCA, Procustes Analysis and Varimax rotation:
    It follows the same steps of the method iii), but Varimax rotation is performed before Procustes analysis.


    '''
    def __init__(self, X, *dictionary):
        self.X = X


        #Initialize the number of variables to select and the PCs to retain

        self._n_ret = 1
        self._path = ' '
        self._labels_name = ' '

        super().__init__(self.X)

        self._method = 'B2' #'B2', 'B4', "Procustes", "procustes_rotation"

        if dictionary:
            settings = dictionary[0]

            try:
                self._nPCs = settings["number_of_eigenvectors"]
            except:
                self._nPCs = self.X.shape[1]-1
                print("Number of PCs to retain not given to dictionary. It will be automatically set equal to X.shape[1]-1.")
                print("You can ignore this warning if the number of PCs has been assigned later via setter.")
            try:
                self._method = settings["method"]
            except:
                self.method = 'procustes'
                print("Selection method not given to dictionary. It will be automatically set equal to 'procrustes'.")
                print("You can ignore this warning if the selection method has been assigned later via setter.")
            try:
                self._center = settings["center"]
            except:
                self._center = True
            try:
                self._centering = settings["centering_method"]
            except:
                self._centering = "mean"
            try:
                self._scale = settings["scale"]
            except:
                self._scale = True 
            try: 
                self._scaling = settings["scaling_method"]
            except:
                self._scaling = "auto"
            try:
                self._n_ret = settings["number_of_variables"]
            except:
                raise Exception("Number of variables to retain (settings[number_of_variables]) not set in the dictionary.")
                print("Exiting with error..")
                exit()
            try:
                self._path = settings["path_to_labels"]
            except:
                #the class will work with the variables' number, instead of their names (managed in load labels method)
                self._path = "not given"
            try:
                self._labels_name = settings["labels_name"]
            except:
                #the class will work with the variables' number, instead of their names (managed in load labels method)
                self._labels_name = "not given"


    @property
    def retained(self):
        return self._n_ret

    @retained.setter
    @accepts(object, int)
    def retained(self, new_number):
        self._n_ret = new_number

        if self._n_ret <= 0:
            raise Exception("The number of retained variables must be a positive integer. Exiting..")
            exit()

    @property
    def path_to_labels(self):
        return self._path

    @path_to_labels.setter
    def path_to_labels(self, new_string):
        self._path = new_string


    @property
    def labels_file_name(self):
        return self._labels_name

    @labels_file_name.setter
    def labels_file_name(self, new_string):
        self._labels_name = new_string

    @property
    def method(self):
        return self._method

    @method.setter
    def method(self, new_string):
        self._method = new_string


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
            print("Variables number: {}, Labels length: {}".format(X.shape[1], labels.shape[1]))
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

        print("Selecting global variables via PCA and Procustes Analysis...")
        self.load_labels()
        variables_selection.check_sanity_input(self.X, self.labels, self._n_ret)

        self.X_tilde = PCA.preprocess_training(self.X, self.to_center, self.to_scale, self.centering, self.scaling)
        self.var_num = np.linspace(0, self.X.shape[1]-1, self.X.shape[1], dtype=int)
        if self._method.lower() == 'procustes':

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
                    #Compute the Procustes Analysis score M2 for the matrix without the 'ii' variable
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

            #Number of variables before the elimination starts:
            max_var = self.X.shape[1]
            #While the number of variables is larger than the number you want to retain 'm', go on
            #with the elimination process:
            while max_var > self.retained:
                #Perform PCA:
                model = PCA(self.X)
                model.eigens = self._nPCs
                PCs,eigvals = model.fit()
                #Check which variable has the max weight on the last PC. Python starts to count from
                #zero, that's why the last number is "self._nPCs -1" and not "self._nPCs"
                max_on_last = np.max(np.abs(PCs[:,self._nPCs-1]))
                argmax_on_last = np.max(np.abs(PCs[:,self._nPCs-1]))
                #Delete the selected variable
                self.X_tilde = np.delete(self.X_tilde, argmax_on_last, axis=1)
                self.labels = np.delete(self.labels, argmax_on_last, axis=0)
                print("Current number of variables: {}".format(self.X_tilde.shape[1]))
                #Get the current number of variables for the while loop
                max_var = self.X_tilde.shape[1]


            return self.labels

        if self._method.lower() == 'procustes_rotation':

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
                    #ROTATE THE SCORES VIA VARIMAX
                    Z_tilde = varimax_rotation(self.X_tilde, Z_tilde, normalize=True)
                    #Compute the reduced scores covariance matrix, and then apply SVD
                    covZZ = np.transpose(Z_tilde) @ Z
                    u, s, vh = np.linalg.svd(covZZ, full_matrices=True)
                    #Compute the Procustes Analysis score M2 for the matrix without the 'ii' variable
                    M2_tmp = np.trace((np.transpose(Z) @ Z) + (np.transpose(Z_tilde) @ Z_tilde) - 2*s)
                    #If the Silhouette score M2 is lower than the previous one in M2_tmp, store the
                    #variable 'ii' to remove it after the for loop
                    if M2_tmp < M2:
                        M2 = M2_tmp
                        var_tmp = ii
                #Remove the variable from the matrix and the labels list
                self.X_tilde = np.delete(self.X_tilde, var_tmp, axis=1)
                self.labels = np.delete(self.labels, var_tmp, axis=0)
                print("Current number of variables: {}".format(self.X_tilde.shape[1]))

            return self.labels

        elif self._method.lower() == 'b4':

            model = PCA(self.X)
            if self._nPCs < self._n_ret:
                print("For the B4 it is not possible to choose a number of PCs lower than the required number of variables.")
                print("The number of PCs will be set equal to the required number of variables.")
                print(" ")
                self._nPCs = self._n_ret

            model.eigens = self._nPCs
            PCs,eigvals = model.fit()
            PVs = []

            for ii in range(0, self._n_ret):
                #Check the largest weight on the first 'm' PCs and add it to the PVs list.
                argmax_= np.argmax(np.abs(PCs[:,ii]))
                PVs.append(self.labels[argmax_])
                #Set the variable weight to zero on all the PCs, to avoid repetition in the PVs list.
                PCs[argmax_,:]= 0

            return PVs

        else:
            raise Exception("Variables selection method not supported. Please choose one between 'B2', 'Procustes', 'Procustes_rotation'.")
            print("Exiting with error..")
            exit()



class SamplePopulation():
    '''
    This class contains a set of methods to consider only a subset of the original training
    matrix X. These methods should be used when the size of the training dataset is large, and
    becomes impractical to train the model on the whole training population.
    Three classes of sampling strategies are available:
    i) Simple random sampling; ii) Clustered sampling; iii) Stratified sampling; iv) multistage.
    The 'clustered' sampling can be accomplished either via KMeans, or LPCA.

    The simple random sampling just shuffle the data and takes the required amount of observations.
    The clustered approach groups the observations in different clusters, and from each of them
    a certain amount of samples are taken.
    The multistage couples stratified (first stage) and clustered (second stage) sampling.


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
        
        #Choose the sampling strategy: random, cluster (KMeans, LPCA), stratifed or multistage.
        self._method = 'KMeans'
        self.__condVec = False
        
        #Choose the dimensions of the sampled dataset.
        self._dimensions = 1
        
        #Predefined number of clusters, if 'cluster', 'stratified' or 'multistage' are chosen
        self.__k = 32
        
        #Variable to use for the matrix conditioning in case of stratified sampling
        self._conditioning =0

        if dictionary:
            settings = dictionary[0]
        
            try:
                self._method = settings["method"]
            except:
                self._method = "random"
            try:
                self._dimensions = settings["final_size"]
            except:
                self._dimensions = int(self.X.shape[0]/2)

    @property
    def sampling_strategy(self):
        return self._method

    @sampling_strategy.setter
    def sampling_strategy(self, new_string):
        self._method = new_string

    @property
    def set_size(self):
        return self._dimensions

    @set_size.setter
    def set_size(self, new_value):
        self._dimensions = new_value

    @property
    def set_conditioning(self):
        return self._conditioning

    @set_conditioning.setter
    def set_conditioning(self, new_value):
        self._conditioning = new_value
        if not isinstance(self._conditioning, int) and  not isinstance(self._conditioning, float):
            self._conditioning = new_value
            self.__condVec = True

    def fit(self):
            '''
            --- RETURNS ---
            miniX:      sampled matrix with the prescribed number of observations 
            type miniX: numpy matrix
            '''
            #Center and scale the matrix (auto preset)
            self.X_tilde = center_scale(self.X, center(self.X,'mean'), scale(self.X, 'auto'))
            self.__batchSize = int(self._dimensions / self.__k)

            if self._method.lower() == 'random':
                #Randomly shuffle the observations and take the prescribed number
                np.random.shuffle(self.X)
                miniX = self.X[:self.__batchSize,:]
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
                    #If the cluster contains less observation than the batch
                    #size, then take all the clusters' observations.
                    if cluster_.shape[0] < self.__batchSize:
                        miniX = np.concatenate((miniX, cluster_), axis=0)
                    else:
                        #Otherwise, shuffle the cluster and take the first
                        #batchsize observations.
                        np.random.shuffle(cluster_)
                        miniX = np.concatenate((miniX, cluster_[:self.__batchSize,:]), axis=0)
                        if miniX.shape[0] < self._dimensions and ii == max(id):
                            delta = self._dimensions - miniX.shape[0]
                            miniX= np.concatenate((miniX, cluster_[(self.__batchSize+1):(self.__batchSize+1+delta),:]), axis=0)
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
                    if cluster_.shape[0] < self.__batchSize:
                        miniX = np.concatenate((miniX, cluster_), axis=0)
                    else:
                        np.random.shuffle(cluster_)
                        miniX = np.concatenate((miniX, cluster_[:self.__batchSize,:]), axis=0)
                        if miniX.shape[0] < self._dimensions and ii == max(id):
                            delta = self._dimensions - miniX.shape[0]
                            miniX= np.concatenate((miniX, cluster_[(self.__batchSize+1):(self.__batchSize+1+delta),:]), axis=0)
            elif self._method.lower() == 'stratified':
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
                #Compute the extension of each bin (delta_step)
                self.__kHardConditioning = 100
                local_batchSize = int(self._dimensions/self.__kHardConditioning)
                delta_step = ((max_con - min_con) / self.__kHardConditioning)
                counter = 0
                var_left = min_con
                miniX = self.X[1:3,:]
                while counter <= self.__kHardConditioning:
                    #Compute the two extremes, and take all the observations in
                    #the dataset which lie in the interval.
                    var_right = var_left + delta_step
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
                        np.random.shuffle(cluster_)
                        miniX = np.concatenate((miniX, cluster_[:local_batchSize,:]), axis=0)
                        '''if miniX.shape[0] < self._dimensions and counter == self.__kHardConditioning:
                            delta = self._dimensions - miniX.shape[0]
                            miniX= np.concatenate((miniX, cluster_[(self.__batchSize+1):(self.__batchSize+1+delta),:]), axis=0)'''
                        var_left += delta_step
                        counter+=1
            elif self._method.lower() == 'multistage':
                #Stratified sampling step: build multiMiniX from X conditioning,
                #and after that cluster to have a further reduction in the
                #dataset' size. The conditioning is done with k = 32, while the
                #clustering takes k = 16
                self.__multiBatchSize = 2*self.__batchSize
                self.__kHardConditioning = 100
                if not self.__condVec:
                    min_con = np.min(self.X[:,self._conditioning])
                    max_con = np.max(self.X[:,self._conditioning])
                else:
                    min_con = np.min(self._conditioning)
                    max_con = np.max(self._conditioning)
                delta_step = ((max_con - min_con) / self.__kHardConditioning)
                counter = 0
                var_left = min_con
                multiMiniX = self.X[1:3,:]
                while counter <= self.__kHardConditioning:
                    var_right = var_left + delta_step
                    if not self.__condVec:
                        mask = np.logical_and(self.X[:,self._conditioning] >= var_left, self.X[:,self._conditioning] < var_right)
                    else:
                        mask = np.logical_and(self._conditioning >= var_left, self._conditioning < var_right)
                    cluster_ = self.X[mask]
                    if cluster_.shape[0] < self.__multiBatchSize:
                        multiMiniX = np.concatenate((multiMiniX, cluster_), axis=0)
                        var_left += delta_step
                        counter+=1
                    else:
                        np.random.shuffle(cluster_)
                        multiMiniX = np.concatenate((multiMiniX, cluster_[:self.__multiBatchSize,:]), axis=0)
                        if multiMiniX.shape[0] < self._dimensions and counter == self.__kHardConditioning:
                            delta = self._dimensions - multiMiniX.shape[0]
                            multiMiniX= np.concatenate((multiMiniX, cluster_[(self.__multiBatchSize+1):(self.__multiBatchSize+1+delta),:]), axis=0)
                        var_left += delta_step
                        counter+=1
                #Clustering sampling step: build miniX from multiMiniX via KMeans
                multiMiniX_tilde = center_scale(multiMiniX, center(multiMiniX,'mean'), scale(multiMiniX, 'auto'))
                multiK = int(self.__k / 2)
                miniX = self.X[1:3,:]
                model_KM = clustering.KMeans(multiMiniX_tilde)
                model_KM.clusters = self.__k
                model_KM.initMode = True
                id = model_KM.fit()
                for ii in range(0, max(id)+1):
                    cluster_ = get_cluster(multiMiniX, id, ii)
                    if cluster_.shape[0] < self.__batchSize:
                        miniX = np.concatenate((miniX, cluster_), axis=0)
                    else:
                        np.random.shuffle(cluster_)
                        miniX = np.concatenate((miniX, cluster_[:self.__batchSize,:]), axis=0)
                        if miniX.shape[0] < self._dimensions and ii == max(id):
                            delta = self._dimensions - miniX.shape[0]
                            miniX= np.concatenate((miniX, cluster_[(self.__batchSize+1):(self.__batchSize+1+delta),:]), axis=0)
            
            else:
                raise Exception("The selected sampling method is not valid. Please choose between: 'random', 'kmeans', 'lpca', 'stratified' or 'multistage'.")

            #it can happen that few observations are missing to reach the prescribed number of observations of
            #the sampled matrix. In this case, fill the gap with random observations from the training matrix
            if miniX.shape[0] < self._dimensions:
                np.random.shuffle(self.X)
                delta = self._dimensions - miniX.shape[0]

                miniX = np.concatenate((miniX, self.X[:delta,:]), axis=0)
            elif miniX.shape[0] > self._dimensions:
                miniX = miniX[:self._dimensions,:]

            return miniX

class NMF():
    '''
    Perform model order reduction via Non-negative Matrix Factorization (NMF).
    NMF is a technique for low-rank approximation of a matrix X, as the product
    of two non-negative matrices: W and H.
    The matrix H, whose dimensions are (observations x k, with the condition: k < var), 
    is representative for the compressed data (NMF scores). The W matrix, instead,
    contains the basis for the representation of X in the reduced space. The dimension
    of the W matrix are, naturally, (k x variables).

    In order to find the matrices W and H which are capable to minimize the difference
    between the original and the approximated matrix, i.e., which are capable to minimize
    the quantity ||X - VH||, a biconvex problem has to be solved.
    In this library for model order reduction, the alternating least squares algorithm [1,2]
    has been implemented to solve this task, as well as the multiplicative update rule [6].
    The multiplicative update rules in case of Kullback-Leibler Divergence (KLD) have been 
    implemented from the detailed derivation, which can be found in [7].

    Moreover, NMF can also be used for clustering purposes, given the additivity of
    the modes which are found. The cluster() function assigns a label to each observation
    examining the scores values: if an observation has the maximum value on the  j-th scores,
    this means that its class it 'j'.

    With regard to the metric to be used (Frob or KLD), in [5] it is mentioned that for
    data which are more or less centered around a region, i.e., without outliers, Frob 
    or MSE could be better. If the input data have a skewed distribution or outliers, instead,
    it is better to use KLD.


    Additional details regarding NMF and the implemented algorithm can be found in 
    refs [1-5].
    
    1. https://www.mpi-inf.mpg.de/fileadmin/inf/d5/teaching/ss15_dmm/lectures/2015-05-26-intro-to-nmf.pdf
    2. Türkmen, Ali Caner. "A review of nonnegative matrix factorization methods for clustering." arXiv preprint arXiv:1507.03194 (2015).
    3. Kim, Jingu, and Haesun Park. Sparse nonnegative matrix factorization for clustering. Georgia Institute of Technology, 2008.
    4. Blondel, Vincent D., Ngoc-Diep Ho, and Paul Dooren. In Image and Vision Computing. 2008.
    5. Lin, Xihui, and Paul C. Boutros. BMC bioinformatics 21.1 (2020): 1-10.
    6. Lee, Daniel D., and H. Sebastian Seung. Advances in neural information processing systems. 2001.
    7. https://www.jjburred.com/research/pdf/jjburred_nmf_updates.pdf


    --- PARAMETERS ---
    X:          RAW data matrix, uncentered and unscaled. It must be organized
                with the structure: (observations x variables).
    type X :    numpy array
    


    --- SETTERS ---
    _dim:                   set the reduced space dimensionality
    type   _dim:            scalar

    _center:                Enable the centering function
    type   _center:         boolean 
    
    _centering:             set the centering method. Available choices for scaling
                            are 'mean' or 'min'.
    type   _centering:      string

    _scale:                 Enable the scaling function
    type   _scale:          boolean 

    _scaling:               set the scaling method. Available choices for scaling
                            are 'auto' or 'vast' or 'range' or 'pareto'.
    type _scaling:          string

    _method:                set the method for ALS: standard or sparse
    type   _method:         string

    _beta:                  it's a parameter to control the degree of sparsity. In [3], they suggest
                            Beta = 0.1 for high-dimensional data-sets, but also in [0.3-0.5] gave good
                            experimental results.
    type _beta:             scalar

    _metric:                the metric which has to be used to measure the convergence criteria. Two metrics
                            are available: 'frobenius' and 'kld'. If 'frobenius' is chosen, the frobenius distance
                            between the "real" data matrix (A) and the reconstructed (X_rec) is used as a metric.
                            Otherwise, if 'kld' is chosen, the Kullback-Leibler divergence is used as a metric [4].
    type _metric:           string

    _algorithm:             algorithm to be used for NMF updates. Two are available: the alternating least
                            squares ('ALS') or the multiplicative update rule ('mur').
    type _algorithm:        string

    __iterMax:              maximum number of iterations for the iterative algorithm (PRIVATE)
    type   __iterMax:       scalar
    
    '''

    def __init__(self, X, *dictionary):

        #standard construction - initialize the number of reduced dim
        self.X = X 
        self.rows, self.cols = self.X.shape
        self._dim = self.cols -1

        #tell to the algorithm if the matrix must be centered/scaled.
        #only Range scaling is possible, so no setter for the method 
        #is prescribed.
        self._center = True
        self._centering = 'min'
        self._scale = True
        self._scaling = 'auto'

        #info about the method: standard ALS or sparse ALS are implemented:
        self._method = 'sparse'
        self._eta = 0.01
        self._beta = 0.01

        #metric to be used to measure convergence criterion. Two settings are
        #available: "frobenius" and "kld" (Kullback-Leibler)
        self._metric = 'frobenius'

        #algorithm to be used for NMF updates. Two are available: the 
        #alternating least squares ('ALS') or the multiplicative update rule ('mur')
        self._algorithm = 'mur'

        #private properties for iterative algorithm
        self.__iterMax = 1000
        self.__convergence = False

        if dictionary:
            settings = dictionary[0]
            try:
                self._center = settings["center"]
            except:
                self._center = True
            try:
                self._centering = settings["centering_method"]
            except:
                self._centering = "mean"
            try:
                self._scale = settings["scale"]
            except:
                self._scale = True 
            try: 
                self._scaling = settings["scaling_method"]
            except:
                self._scaling = "auto"
            try:
                self._dim = settings["number_of_features"]
            except:
                self._dim = self.X.shape[1]
                print("Number of features to retain not given to dictionary. It will be automatically set equal to X.shape[1].")
                print("You can ignore this warning if the number of PCs has been assigned later via setter.")
            try:
                self._algorithm = settings["optimization_algorithm"]
            except:
                self._algorithm = 'als'
                print("NMF algorithm not given to dictionary. It will be automatically set equal to Alternating Least Squares (als).")
                print("You can ignore this warning if the algorithm has been assigned later via setter.")
            try:
                self._method = settings["als_method"]
            except:
                self._method = "standard"
            try:
                self._eta = settings["sparsity_eta"]
            except:
                self._eta = 0.01
            try:
                self._beta = settings["sparsity_beta"]
            except:
                self._beta = 0.01
            try:
                self._metric = settings["optimization_metric"]
            except:
                self._metric = 'frobenius'


    @property
    def to_center(self):
        return self._center

    @to_center.setter
    @accepts(object, bool)
    def to_center(self, new_bool):
        self._center = new_bool
    
    @property
    def centering(self):
        return self._centering

    @centering.setter
    def centering(self, new_string):
        self._centering = new_string

        if self._centering != 'min':
            raise Exception("No other centering than 'min' is allowed for NMF. Data will be scaled with their minimum.")
            self._centering = 'min'

    @property
    def to_scale(self):
        return self._scale

    @to_scale.setter
    @accepts(object, bool)
    def to_scale(self, new_bool):
        self._scale = new_bool

    @property
    def scaling(self):
        return self._scaling

    @scaling.setter
    @allowed_scaling
    def scaling(self, new_string):
        self._scaling = new_string

    @property
    def encoding(self):
        return self._dim

    @encoding.setter
    @accepts(object, int)
    def encoding(self, new_int):
        self._dim = new_int

    @property
    def method(self):
        return self._method

    @method.setter
    def method(self, new_string):
        self._method = new_string

    @property
    def eta(self):
        return self._eta

    @eta.setter
    def eta(self, new_value):
        self._eta = new_value

    @property
    def beta(self):
        return self._beta

    @beta.setter
    def beta(self, new_value):
        self._beta = new_value

    @property
    def metric(self):
        return self._metric

    @metric.setter
    def metric(self, new_string):
        self._metric = new_string

    @property
    def algorithm(self):
        return self._algorithm

    @algorithm.setter
    def algorithm(self, new_string):
        self._algorithm = new_string

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
    def KL_divergence(X, Y):
        '''
        Compute the generalized Kullback-Leibler Divergence between two
        matrices, X and Y.
        KLD is a measure which is often used to quantify the differences 
        between two probability distributions.

        Two formulations are available, one is for continuous functions
        (compute the integral) and the other is for discrete functions
        (compute the sum). In this function, the generalized form is implemented,
        which is described in [i], formula (2) - pag. 3.

        [i]: Blondel, Vincent D., Ngoc-Diep Ho, and Paul Dooren. In Image and Vision Computing. 2008.
        

        --- PARAMETERS ---
        X:          Original data matrix (observations x variables). 
        type X :    numpy array

        Y:          Reconstructed data matrix (observations x variables). 
        type Y :    numpy array


        --- RETURNS ---
        KLD:        Kullback-Leibler divergence between the two input matrices, X and Y.
        type KLD:   scalar

        '''
        
        #cleaning to remove any value <= 0 (it would be a problem for the log):
        mask_X = np.where(X <= 0)
        X[mask_X] = 1E-16
        mask_Y = np.where(Y <= 0)
        Y[mask_Y] = 1E-16

        #compute the KLD as described in [i]
        tmp = np.log(X/Y) 
        temp = np.multiply(X,tmp)
        
        KL = temp - X + Y
        KLD = np.sum(KL)

        return KLD


    def fit(self):
        '''
        --- RETURNS ---
        W:          matrix containing the NMF reduced basis
        type W:     numpy matrix

        H:          matrix containing the scores 
        type H:     numpy matrix
        '''
        from numpy.linalg import lstsq
        from numpy.linalg import norm


        #Center and scale the matrix. 
        self.X_tilde = self.preprocess_training(self.X, self._center, self._scale, self._centering, self._scaling)

        #To follow the notation in [3], we consider the A matrix with shape: (n_features x n_observations)
        A = self.X_tilde.T
        features, observations = A.shape 

        #Initialize weights matrix for low-rank approximation
        self.W = np.random.rand(features, self._dim)
        #Remove any negative value as prescribed in [3]
        mask = np.where(self.W < 0)
        self.W[mask] = 0


        if self._algorithm.lower() == 'als':

            #scale the W matrix to have unit L2-norm as prescribed in [3]
            for jj in range(0,self.W.shape[1]):
                tmp = norm(self.W[:,jj])
                self.W[:,jj] = self.W[:,jj]/tmp 


            if self._method == 'sparse':
                
                #Initialize sparsity factors as described in [3], Par. 3.2
                sparsityW = np.sqrt(self._beta) * np.ones((1,self.W.shape[1]), dtype=float)         #1xk
                sparsityH = np.sqrt(self._eta) * np.eye(self._dim)                                  #k x k
                sparsityX1 = np.zeros((1,A.shape[1]), dtype=float)
                sparsityX2 = np.zeros((self._dim, A.shape[0]), dtype=float)

                modX1 = np.r_[A, sparsityX1]                                                        #(mxn) + (1xn) = (m+1) x n
                modX2 = np.r_[A.T, sparsityX2]                                                      #(nxm) + (kxm) =(n+k) x m 
            

            #Initialize the parameters for the iterative algorithm
            iteration = 0
            eps_rec = 1.0
            convTol = 1E-5
            eps_tol = 1E-16

            while not self.__convergence and iteration < self.__iterMax:

                if self._method.lower() == 'standard':
                    #optimize H, and after that delete negative coefficients
                    self.H = lstsq(self.W, A, rcond=None)[0]
                    mask = np.where(self.H < 0)
                    self.H[mask] = 0

                    #optimize W, and after that delete negative coefficients
                    self.W = lstsq(self.H.T, A.T, rcond=None)[0]
                    mask = np.where(self.W < 0)
                    self.W[mask] = 0
                    self.W = self.W.T


                elif self._method.lower() == 'sparse':
                    modW = np.r_[self.W, sparsityW]                                                 #(m x k) + (1 x k) = (m+1) x k 
                    self.H = lstsq(modW, modX1, rcond=None)[0]                                      #(m+1)x k ++ (m+1)x n == (k x n) == H
                    mask = np.where(self.H < 0)
                    self.H[mask] = 0
                    
                    modH = np.r_[self.H.T, sparsityH]                                               #(nxk) + (kxk) = (n+k) x k

                    self.W = lstsq(modH, modX2, rcond=None)[0]                                      #(n+k) x k ++ (n+k) x m == (k x m)
                    mask = np.where(self.W < 0)
                    self.W[mask] = 0
                    self.W = self.W.T

                else:
                    raise Exception("NMF method not implemented. Please choose one between 'standard' or 'sparse'. Exiting with error..")
                    exit()


                #Compute the reconstructed matrix from the reduced-order basis
                X_rec = self.W @ self.H 


                #Compute the error between the original and the reconstructed
                #via Frobenius norm or with the Kullback-Leibler divergence
                if self._metric.lower() == 'frobenius':
                    eps_rec_new = np.linalg.norm((A.T - X_rec.T), 'fro')
                elif self._metric.lower() == 'kld':
                    eps_rec_new = self.KL_divergence(A.T, X_rec.T)


                #Check the reconstruction error variation to see if the convergence
                #has been reached
                eps_rec_var = np.abs((eps_rec_new - eps_rec) / (eps_rec_new) + eps_tol)
                eps_rec = eps_rec_new

                #Check if the convergence conditions have been satisfied
                if eps_rec_var > convTol and iteration < self.__iterMax:
                    print("Iteration number: {}".format(iteration))
                    print("\tReconstruction error: {}".format(eps_rec_new))
                    print("\tReconstruction error variance: {}".format(eps_rec_var))
                    iteration +=1
                else:
                    #scale again the W matrix to have unit L2-norm as prescribed in [3]
                    for jj in range(0,self.W.shape[1]):
                        tmp = norm(self.W[:,jj])
                        self.W[:,jj] = self.W[:,jj]/tmp 
                        tmp2 = norm(self.H[jj,:])
                        self.H[jj,:] = self.H[jj,:]/tmp
                    print("Convergence has been reached after {} iterations.".format(iteration))
                    print("\tFinal reconstruction error variance: {}".format(eps_rec_var))
                    
                    break
        
        elif self._algorithm.lower() == 'mur':

            #Initialize the parameters for the iterative algorithm
            iteration = 0
            eps_rec = 1.0
            convTol = 1E-5
            eps_tol = 1E-16

            #Initialize weights matrix for low-rank approximation
            self.H = np.random.rand(self._dim, observations)                                    #dim: k x n
            #Remove any negative value as prescribed in [3]
            mask = np.where(self.H < 0)
            self.H[mask] = 0

            while not self.__convergence and iteration < self.__iterMax:
                if self._metric.lower() == 'frobenius':
                    phi = self.W.T @ A                                                          #(k x m) @ (m x n) = (k x n)
                    chi = (self.W.T @ self.W @ self.H) + +1E-16                                 #(k x m) @ (m x k) @ (k x n) = (k x n)

                    alpha = np.divide(phi, chi)
                    self.H *= alpha
                    
                    
                    zeta = A @ self.H.T                                                         #(m x n) @ (n x k) = (m x k)
                    theta = (self.W @ self.H @ self.H.T) +1E-16                                 #(m x k) @ (k x n) @ (n x k) = (m x k)

                    gamma = np.divide(zeta, theta)
                    self.W *= gamma
                
                elif self._metric.lower() == 'kld':
                    psi = np.divide(A, (self.W @ self.H) +1E-16)                                #(m x n) / [(m x k) @ (k x n)]
                    phi = self.W.T @ psi                                                        #(k x m) @ (m x n) = (k x n)
                    chi = self.W.T @ np.ones((A.shape), dtype=float) +1E-16                     #(k x m) @ (m x n) = (k x n)

                    alpha = np.divide(phi, chi) 
                    self.H *= alpha

                    omicron = (self.W @ self.H) + 1E-16                                         #(m x k) @ (k x n) = (m x n)
                    omicron_star = np.divide(A, omicron)                                        #(m x n) / (m x n) = (m x n)
                    zeta = omicron_star @ self.H.T                                              #(m x n) @ (n x k) = (m x k)
                    theta = (np.ones((omicron_star.shape), dtype=float) @ self.H.T) +1E-16      #(m x n) @ (n x k) = (m x k)

                    gamma = np.divide(zeta, theta)
                    self.W *= gamma



                X_rec = self.W @ self.H 


                #Compute the error between the original and the reconstructed
                #via Frobenius norm or with the Kullback-Leibler divergence
                if self._metric.lower() == 'frobenius':
                    eps_rec_new = np.linalg.norm((A.T - X_rec.T), 'fro')
                elif self._metric.lower() == 'kld':
                    eps_rec_new = self.KL_divergence(A.T, X_rec.T)


                #Check the reconstruction error variation to see if the convergence
                #has been reached
                eps_rec_var = np.abs((eps_rec_new - eps_rec) / (eps_rec_new) + eps_tol)
                eps_rec = eps_rec_new

                #Check if the convergence conditions have been satisfied
                if eps_rec_var > convTol and iteration < self.__iterMax:
                    print("Iteration number: {}".format(iteration))
                    print("\tReconstruction error: {}".format(eps_rec_new))
                    print("\tReconstruction error variance: {}".format(eps_rec_var))
                    iteration +=1
                else:
                    #scale again the W matrix to have unit L2-norm as prescribed in [3]
                    for jj in range(0,self.W.shape[1]):
                        tmp = norm(self.W[:,jj]) + 1E-16
                        self.W[:,jj] = self.W[:,jj]/tmp 
                        #tmp2 = norm(self.H[jj,:])
                        self.H[jj,:] = self.H[jj,:]/tmp
                    print("Convergence has been reached after {} iterations.".format(iteration))
                    print("\tFinal reconstruction error variance: {}".format(eps_rec_var))
                    
                    break



        return self.W, self.H

    def cluster(self):
        '''
        --- RETURNS ---
        idx:            vector containing the cluster assignment, whose shape is (observations x 1)
        type idx:       numpy vector
        '''
        idx = np.empty((self.rows,), dtype=int)
        idx = np.argmax(self.H.T, axis=1)

        idx = clustering.lpca.merge_clusters(self.X_tilde, idx)
        print("The final number of clusters is equal to: {}".format(np.max(idx+1)))

        return idx

