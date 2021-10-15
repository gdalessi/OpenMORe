'''
MODULE: clustering.py

@Authors:
    G. D'Alessio [1,2], G. Aversano [1], A. Parente[1]
    [1]: Université Libre de Bruxelles, Aero-Thermo-Mechanics Laboratory, Bruxelles, Belgium
    [2]: CRECK Modeling Lab, Department of Chemistry, Materials and Chemical Engineering, Politecnico di Milano

@Contacts:
    giuseppe.dalessio@ulb.ac.be


@Additional notes:
    This code is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
    Please report any bug to: giuseppe.dalessio@ulb.ac.be

'''

from .utilities import *
from . import model_order_reduction
import warnings
import time

import numpy as np
from numpy import linalg as LA
import numpy.matlib
import matplotlib
import matplotlib.pyplot as plt


class lpca:
    '''
    The iterative Local Principal Component Analysis clustering algorithm is based on the following steps:
    
    0.  Preprocessing: The training matrix X is centered and scaled.
    
    1.  Initialization: The cluster centroids are initializated, several options are available.
        The first is a random allocation ('random'), assigning random values to the class membership
        vector, idx. The second option, is the initialization by means of a previous clustering solution
        obtained by the Kmeans algorithm ('kmeans'). The third option is 'observations': a number 'k'
        (where k = selected number of clusters) is randomly selected from the data-set and chosen as cluster
        centroids. The idx is calculated via euclidean distance minimization between the observations and
        the random centroids. The last available initialization is 'pkcia', compute the positive definite 
        matrix Y = XX.T and assign the initial idx value on the basis of the first eigenvector obtained
        from Y.   
    
    2.  Partition: Each observation is assigned to a cluster k such that the local reconstruction
        error is minimized;
    
    3.  PCA: The Principal Component Analysis is performed in each of the clusters found
        in the previous step. A new set of centroids is computed after the new partitioning
        step, their coordinates are calculated as the mean of all the observations in each
        cluster;
    
    4.  Iteration: All the previous steps are iterated until convergence is reached. The convergence
        criterion is that the variation of the global mean reconstruction error between two consecutive
        iterations must be below a fixed threshold.

    
    --- PARAMETERS ---
    X:          RAW data matrix, uncentered and unscaled. It must be organized
                with the structure: (observations x variables).
    type X :    numpy array

    dictionary:         Dictionary containing all the instruction for the setters
    type dictionary:    dictionary

    
    --- SETTERS ---
    clusters:               number of clusters to be used for the partitioning
    type   k:               scalar

    to_center:              Enable the centering function
    type   _center:         boolean 
    
    centering:              set the centering method. Available choices for scaling
                            are 'mean' or 'min'.
    type   _centering:      string

    to_scale:               Enable the scaling function
    type   _scale:          boolean 

    scaling:                set the scaling method. Available choices for scaling
                            are 'auto' or 'vast' or 'range' or 'pareto'.
    type _scaling:          string

    initialization:         initialization method: 'random', 'kmeans', 'observations', 'pkcia' are available.
    type   _method:         string

    correction:             multiplicative or additive correction factor to be used for the lpca algorithm
    type _beta:             string

    eigens:                 number of Principal Components which have to be used locally for the dimensionality reduction task.
    type _nPCs:             scalar

    
    '''
    def __init__(self, X, *dictionary):
        self.X = np.array(X)
        #Initialize the number of clusters:
        self._k = 2
        #Initialize the number of PCs to retain in each cluster:
        self._nPCs = 2
        #Set the initialization method:
        self._method = 'uniform'                                             #Available options: 'KMEANS' or 'RANDOM'
        #Set the (eventual) corrector for the rec error computation:
        self._correction = "off"                                            #Available options: 'off', 'mean', 'max', 'std', 'var'
        self.__activateCorrection = False
        #Adaptive PCs per cluster:
        self._adaptive = False                                              #Available options: True or False (boolean)

        #Decide if the input matrix must be centered:
        self._center = True
        #Set the centering method:
        self._centering = 'mean'                                                                    #'mean' or 'min' are available
        #Decide if the input matrix must be scaled:
        self._scale = True
        #Set the scaling method:
        self._scaling = 'auto'

        self._writeFolder = True

        self._postKNN = False
        self._neighborsNum = 0

        if dictionary:
            settings = dictionary[0]
            try:
                self._k = settings["number_of_clusters"]
                if not isinstance(self._k, int) or self._k <= 1:
                    raise Exception
            except:
                self._k = 2
                warnings.warn("An exception occured with regard to the input value for the number of clusters (k). It could be not acceptable, or not given to the dictionary.")
                print("\tIt will be automatically set equal to: 2.")
                print("\tYou can ignore this warning if the number of clusters (k) has been assigned later via setter.")
                print("\tOtherwise, please check the conditions which must be satisfied by the input in the detailed documentation.")
            
            try:
                self._nPCs = settings["number_of_eigenvectors"]
                if self._nPCs <= 0 or self._nPCs >= self.X.shape[1]:
                    raise Exception
            except:
                self._nPCs = int(self.X.shape[1]/2)
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
                self._method = settings["initialization_method"]
                if not isinstance(self._method, str):
                    raise Exception
                elif self._method.lower() != "uniform" and self._method.lower() != "kmeans" and self._method.lower() != "pkcia" and self._method.lower() != "observations" and self._method.lower() != "random":
                    raise Exception
            except:
                self._method = 'uniform'
                warnings.warn("An exception occured with regard to the input value for the initialization criterion. It could be not acceptable, or not given to the dictionary.")
                print("\tIt will be automatically set equal to: uniform.")
                print("\tYou can ignore this warning if the initialization criterion has been assigned later via setter.")
                print("\tOtherwise, please check the conditions which must be satisfied by the input in the detailed documentation.")
            try:
                self._correction = settings["correction_factor"]   
                if not isinstance(self._correction, str):
                    raise Exception
                elif self._correction != "off" and self._correction != "phc_multi" and self._correction != "c_range" and self._correction != "uncorrelation" and self._correction != "local_variance" and self._correction !=  "local_skewness":
                    raise Exception    
            except:
                self._correction = "off"
                print("\tCorrection factor automatically set equal to 'off'.")
                print("\tYou can ignore this warning if the correction factor has been assigned later via setter.")
            try:
                self._adaptive = settings["adaptive_PCs"]
                if not isinstance(self._adaptive, bool):
                    raise Exception
            except:
                self._adaptive = False
            try:
                self._writeFolder = settings["write_stats"]
                if not isinstance(self._writeFolder, bool):
                    raise Exception
            except:
                self._writeFolder = True
            try:
                self._postKNN = settings["kNN_post"]
                if not isinstance(self._postKNN, bool):
                    raise Exception
            except:
                self._postKNN = True
            try:
                self._postKNN = settings["kNN_post"]
                if not isinstance(self._postKNN, bool):
                    raise Exception
            except:
                self._postKNN = False
            try:
                self._neighborsNum = settings["neighbors_number"]
                if not isinstance(self._neighborsNum, int) or self._neighborsNum < 0:
                    raise Exception
            except:
                self._postKNN = False
                print("WARNING: number of neighbors for the LPCA clustering algorithm not given. This option will be automatically set to: OFF.")
                
              


    @property
    def clusters(self):
        return self._k

    @clusters.setter
    def clusters(self, new_number):
        self._k = new_number

        if not isinstance(self._k, int) or self._k <= 1:
            warnings.warn("An exception occured with regard to the input value for the number of clusters (k). It could be not acceptable, or not given to the dictionary.")
            print("\tIt will be automatically set equal to: 2.")
            print("\tYou can ignore this warning if the number of clusters (k) has been assigned later via setter.")
            print("\tOtherwise, please check the conditions which must be satisfied by the input in the detailed documentation.")
            self._k = 2

    @property
    def eigens(self):
        return self._nPCs

    @eigens.setter
    def eigens(self, new_number):
        self._nPCs = new_number

        if self._nPCs <= 0 or self._nPCs >= self.X.shape[1]:
            self._nPCs = int(self.X.shape[1]/2)
            warnings.warn("An exception occured with regard to the input value for the number of PCs. It could be not acceptable, or not given to the dictionary.")
            print("\tIt will be automatically set equal to: X.shape[1]/2.")
            print("\tPlease check the conditions which must be satisfied by the input in the detailed documentation.")

    @property
    def initialization(self):
        return self._method

    @initialization.setter
    def initialization(self, new_method):
        self._method = new_method

        if not isinstance(self._method, str):
            self._method = 'uniform'
            warnings.warn("An exception occured with regard to the input value for the initialization criterion. It could be not acceptable, or not given to the dictionary.")
            print("\tIt will be automatically set equal to: uniform.")
            print("\tPlease check the conditions which must be satisfied by the input in the detailed documentation.")
        elif self._method.lower() != "uniform" and self._method.lower() != "kmeans" and self._method.lower() != "pkcia" and self._method.lower() != "observations" and self._method.lower() != "random":
            self._method = 'uniform'
            warnings.warn("An exception occured with regard to the input value for the initialization criterion. It could be not acceptable, or not given to the dictionary.")
            print("\tIt will be automatically set equal to: uniform.")
            print("\tPlease check the conditions which must be satisfied by the input in the detailed documentation.")


    @property
    def correction(self):
        return self._correction

    @correction.setter
    def correction(self, new_method):
        self._correction = new_method

        if not isinstance(self._correction, str):
            self._correction = "off"
            warnings.warn("An exception occured with regard to the input value for the correction factor to use . It could be not acceptable, or not given to the dictionary.")
            print("\tCorrection factor automatically set equal to 'off'.")
        elif self._correction != "off" and self._correction != "phc_multi" and self._correction != "c_range" and self._correction != "uncorrelation" and self._correction != "local_variance" and self._correction !=  "local_skewness":
            self._correction = "off" 
            warnings.warn("An exception occured with regard to the input value for the correction factor to use . It could be not acceptable, or not given to the dictionary.")
            print("\tCorrection factor automatically set equal to 'off'.")

    @property
    def adaptivePCs(self):
        return self._adaptive

    @adaptivePCs.setter
    def adaptivePCs(self, new_bool):
        self._adaptive = new_bool
        if not isinstance(self._adaptive, bool):
            self._adaptive = False

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
            print("\tPlease check the conditions which must be satisfied by the input in the detailed documentation.")
        elif self._scaling.lower() != "auto" and self._scaling.lower() != "vast" and self._scaling.lower() != "pareto" and self._scaling.lower() != "range":
            self._scaling = "auto"
            warnings.warn("An exception occured with regard to the input value for the scaling criterion. It could be not acceptable, or not given to the dictionary.")
            print("\tIt will be automatically set equal to: auto.")
            print("\tPlease check the conditions which must be satisfied by the input in the detailed documentation.")

    @property
    def writeFolder(self):
        return self._writeFolder

    @writeFolder.setter
    def writeFolder(self, new_string):
        self._writeFolder = new_string

        if not isinstance(self._writeFolder, bool):
            self._writeFolder = False 


    @staticmethod
    def initialize_clusters(X, k, method):
        '''
        The clustering solution must be initialized to start the lpca iterative algorithm.
        Several initialization are available, and they can lead to different clustering solutions.

        
        --- PARAMETERS ---
        X:          Original data matrix (observations x variables). 
        type X :    numpy array

        k:          numbers of clusters. 
        type k :    scalar


        --- RETURNS ---
        idx:        vector whose dimensions are (n,) containing the cluster assignment.
        type idx:   numpy array 
        '''
        if method.lower() == 'random':
            #Assign randomly an integer between 0 and k to each observation.
            idx = np.random.random_integers(0, k, size=(X.shape[0],))

        elif method.lower() == 'kmeans':
            #call the KMeans class from the very same module. Set the number of clusters and
            #choose 'initMode' to use a lower tolerance with respect to the normal algorithm.
            init = KMeans(X)
            init.clusters = k
            init.initMode =True
            idx = init.fit()

        elif method.lower() == 'observations':
            from scipy.spatial.distance import euclidean, cdist
            #Initialize the centroids using 'k' random observations taken from the
            #dataset.
            C_mat = np.empty((k, X.shape[1]), dtype=float)
            idx = np.empty((X.shape[0],), dtype=int)
            for ii in range(0,k):
                C_mat[ii,:] = X[np.random.randint(0,X.shape[0]),:]

            #Compute the euclidean distances between the matrix and all the random vectors
            #chosen as centroids. The function cdist returns a matrix 'dist' = (nObs x k)
            dist = cdist(X, C_mat)**2

            #For each observation, choose the nearest centroid (cdist --> Euclidean dist).
            #and compute the idx for the initialization.
            for ii in range(0, X.shape[0]):
                idx[ii] = np.argmin(dist[ii,:])

        elif method.lower() == 'pkcia':
            #Initialize the centroids with the method described in:
            #Manochandar, S., M. Punniyamoorthy, and R. K. Jeyachitra. Computers & Industrial Engineering (2020): 106290.
            from numpy import linalg as LA
            from scipy.spatial.distance import euclidean, cdist

            #compute the positive definite matrix Y (n x n) from the training data matrix (n x p),
            #with n = observations and p = variables
            Y = X @ X.T

            #compute the eigenvectors and the eigenvalues associated to the new matrix
            evals, evecs = LA.eig(Y)

            #order the eigens in descending order, as done in PCA
            mask = np.argsort(evals)[::-1]
            evecs = evecs[:,mask]
            evals = evals[mask]

            #consider only the eigenvector associated to the largest eigenvalue, V = (n x 1)
            V = evecs[:,0]

            #the min and the max of V squared will be useful later
            v_min = np.min(V**2)
            v_max = np.max(V**2)

            G = np.empty((len(V),), dtype=float)
            idx = np.empty((X.shape[0],), dtype=int)

            #computation of G is the first step to initialize the centroids:
            for ii in range(0, len(G)):
                G[ii] = 1 + ((V[ii]**2-v_min)/(v_max - v_min) + 1E-16) *k

            #compute the range of G and the delta step:
            RG = np.max(G) - np.min(G)
            CPC = RG/k

            counter = 0
            left_bound = 0
            C_mat = np.empty((k, X.shape[1]), dtype=float)

            #Partition the observations on the basis of their G value. Basically the G vector is
            #partitioned in k bins, and the observations are assigned to each bin to form a cluster.
            #The bin width is chosen on the basis of the CPC coefficient.
            #After that, in each cluster the centroid is computed.
            while counter < k:
                right_bound = (left_bound + CPC) + 0.01* (left_bound + CPC)
                try:
                    mask = np.logical_and(G >= left_bound, G < right_bound)
                    cluster_ = X[mask,:]
                    C_mat[counter,:] = np.mean(cluster_, axis=0)
                    left_bound = right_bound
                    counter += 1
                except:
                    left_bound = right_bound
                    counter += 1

            #Compute the squared euclidean distances between the matrix and all the random vectors
            #chosen as centroids. The function cdist returns a matrix 'dist' = (nObs x k)
            dist = cdist(X, C_mat)**2

            #For each observation, choose the nearest centroid and compute the idx for the initialization.
            for ii in range(0, X.shape[0]):
                idx[ii] = np.argmin(dist[ii,:])
        
        elif method.lower() == 'uniform':
            idx = np.zeros(X.shape[0], dtype=int)
            spacing = np.round(X.shape[0]/k) +1
            for ii in range(1, k):
                if ii != (k -1):
                    start = int(ii*spacing+1)
                    endID = int((ii+1)*spacing)
                    idx[start:endID] = ii 
                else:
                    start = int(ii*spacing+1)
                    idx[start:] = ii 

        else:
            raise Exception("Initialization option not supported. Please choose one between RANDOM or KMEANS.")

        return idx


    @staticmethod
    def initialize_parameters():
        '''
        Set some private parameters for the algorithm convergence.
        '''
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

        --- PARAMETERS ---
        X:          Original data matrix (observations x variables). 
        type X :    numpy array

        idx:        vector whose dimensions are (n,) containing the cluster assignment. 
        type idx :  numpy array


        --- RETURNS ---
        idx:        vector whose dimensions are (n,) containing the cluster assignment, WITHOUT EMPTY CLASSES.
        type idx:   numpy array 
        '''
        k = np.max(idx) +1
        jj = 0
        while jj < k:
            cluster_ = get_cluster(X, idx, jj)
            if cluster_.shape[0] < 2: #2 or cluster_.shape[1]:
                if jj > 0:
                    mask = np.where(idx >=jj)
                    idx[mask] -= 1
                else:
                    mask = np.where(idx >jj)
                    idx[mask] -= 1
                print("WARNING:")
                print("\tAn empty cluster was found:")
                print("\tThe number of cluster was lowered to ensure statistically meaningful results.")
                print("\tThe current number of clusters is equal to: {}".format(np.max(idx) +1))
                k = np.max(idx) +1
                jj = 0
            else:
                jj += 1

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
        plt.savefig('Residual_history.eps')
        plt.show()


    @staticmethod
    def set_environment():
        '''
        This function creates a new folder where all the produced files
        will be saved.
        '''
        import datetime
        import sys
        import os

        now = datetime.datetime.now()
        newDirName = "Clustering LPCA - " + now.strftime("%Y_%m_%d-%H%M%S")

        try:
            os.mkdir(newDirName)
            os.chdir(newDirName)
        except FileExistsError:
            print("Folder already existing. Skipping folder creation step.")
            pass


    @staticmethod
    def write_recap_text(k_input, retained_PCs, correction_yn, initialization_type):
        '''
        This function writes a txt with all the hyperparameters
        recaped, to not forget the settings if several trainings are
        launched all together.
        '''
        text_file = open("recap_training.txt", "wt")
        k_number = text_file.write("The number of clusters in input is equal to: {} \n".format(k_input))
        PCs_number = text_file.write("The number of retained PCs is equal to: {} \n".format(retained_PCs))
        init_used = text_file.write("The adopted inizialization method is: "+ initialization_type + ". \n")
        scores_corr = text_file.write("The scores correction is: "+ correction_yn + ". \n")
        text_file.close()


    @staticmethod
    def write_final_stats(iterations_conv, final_error):
        '''
        This function writes a txt with all the hyperparameters
        recaped, to not forget the settings if several trainings are
        launched all together.
        '''
        text_stats = open("convergence_stats.txt", "wt")
        iter_numb = text_stats.write("The number of the total iterations is equal to: {} \n".format(iterations_conv))
        rec_err_final = text_stats.write("The final reconstruction error is equal to: {} \n".format(final_error))
        text_stats.close()


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

    @staticmethod
    def kNNpost(X, idx, neighborsNumber):
        from collections import Counter

        id1 = idx
        yo1 = np.zeros((len(idx)),dtype=int)

        for ii in range(X.shape[0]):
            print("Observation number: {}".format(ii))
            dist = np.exp(np.linalg.norm(X - X[ii,:], axis=1))**2
            Nearest = dist.argsort()[:neighborsNumber+1]
            nn_id = idx[Nearest]
            #print("Nearest idx: {}".format(nn_id))
            c = Counter(nn_id)
            #print("Attributed value by LPCA: {}".format(idx[ii]))
            #print(c)
            id_num = 0
            for jj in range(np.max(idx)+1):
                if c[jj] > id_num:
                    yo1[ii] = jj 
                    id_num = c[ii]
            id_num = 0

        yo = id1 - yo1

        print("Changed {} elements".format(np.count_nonzero(yo)))


        return yo1     


    def fit(self):
        '''
        Group the observations depending on the PCA reconstruction error.

        --- RETURNS ---
        idx:        vector whose dimensions are (n,) containing the cluster assignment for each observation.
        type idx:   numpy array 
        
        '''
        #Center and scale the original training dataset
        print("Preprocessing training matrix..")
        self.X_tilde = self.preprocess_training(self.X, self._center, self._scale, self._centering, self._scaling)
        print("Fitting Local PCA model...")
        if self._writeFolder:
            lpca.set_environment()
            lpca.write_recap_text(self._k, self._nPCs, self._correction, self._method)
        # Initialization
        iteration, eps_rec, residuals, iter_max, eps_tol = lpca.initialize_parameters()
        rows, cols = np.shape(self.X_tilde)
        # Initialize the solution vector
        idx = lpca.initialize_clusters(self.X_tilde, self._k, self._method)
        residuals = np.array(0)
        if self._correction != "off":
            correction_ = np.zeros((rows, self._k), dtype=float)
            scores_factor = np.zeros((rows, self._k), dtype=float)
        # Iterate
        while(iteration < iter_max):
            sq_rec_oss = np.zeros((rows, cols), dtype=float)
            sq_rec_err = np.zeros((rows, self._k), dtype=float)

            if self._correction == 'phc_multi':
                PHC_coefficients, PHC_std = evaluate_clustering_PHC(self.X, idx)   #PHC_index(self.X, idx) or PHC_robustTrim(self.X, idx)
                PHC_coefficients = PHC_coefficients/np.max(PHC_coefficients)

            for ii in range(0, self._k):
                #group the observations of a certain cluster
                cluster = get_cluster(self.X_tilde, idx, ii)
                #compute the centroids, or the medianoids or the medoids, depending on the 
                #selected choice
                if self._correction.lower() != 'medianoids' and self._correction.lower() != 'medoids':
                    centroids = get_centroids(cluster)
                elif self.correction.lower() == 'medianoids':
                    centroids = get_medianoids(cluster)
                elif self.correction.lower() == 'medoids':
                    centroids = get_medoids(cluster)
                #perform PCA in the cluster, centering and scaling can be avoided
                #because the observations are already standardized
                local_model = model_order_reduction.PCA(cluster)
                local_model.to_center = False
                local_model.to_scale = False
                if not self._adaptive:
                    local_model.eigens = self._nPCs
                else:
                    local_model.set_PCs()
                modes = local_model.fit()
                #create the centroids (medoids or medianoids, respectively) matrix
                C_mat = np.matlib.repmat(centroids, rows, 1)
                #compute the rec error for the considered cluster
                rec_err_os = (self.X_tilde - C_mat) - (self.X_tilde - C_mat) @ modes[0] @ modes[0].T
                sq_rec_oss = np.power(rec_err_os, 2)
                sq_rec_err[:,ii] = sq_rec_oss.sum(axis=1)
                
                #use a penalty to eventually enhance the clustering performances
                if self.correction.lower() == "c_range":
                    #add a penalty if the observations are not in the centroids neighbourhood

                    #compute the cluster considering the raw data, and compute
                    #the cluster's centroids
                    cluster2 = get_cluster(self.X, idx, ii)
                    centroids2 = get_centroids(cluster2)  
                    #compute the range for the centroid values: /2 = +-50%, /3 = +- 33% etc.
                    C_mStar = centroids2/2      
                    #lower bound: centroid - interval
                    check1 = centroids2 - C_mStar
                    #upper boundL centroid + interval
                    check2 = centroids2 + C_mStar       
                    #boolean matrix initialization as matrix of ones    
                    boolean_mat = np.ones(self.X_tilde.shape)
                    count = 0

                    #for each element of the raw data matrix, check if it's in the interval.
                    #If yes, put 0 in the boolean matrix
                    for mm in range(0, self.X.shape[0]):
                        for nn in range(0, self.X.shape[1]):
                            if self.X[mm,nn] >= check1[nn] and self.X[mm,nn] <= check2[nn]:
                                boolean_mat[mm,nn] = 0                                               
                                count +=1         
                    #For each row, sum up all the columns to obtain the multiplicative correction coefficient                                                  
                    yo = np.sum(boolean_mat, axis=1)
                    scores_factor[:,ii] = sq_rec_err[:,ii] * yo
                    #activate the option to take into account the penalty in the error
                    self.__activateCorrection = True
                
                elif self._correction.lower() == "uncorrelation":
                    #the clusters where the observations maximize the uncorrelation are favoured
                    maxF = np.max(np.var((self.X_tilde - C_mat) @ modes[0], axis=0))
                    minF = np.min(np.var((self.X_tilde - C_mat) @ modes[0], axis=0))
                    yo = 1-minF/maxF
                    
                    scores_factor[:,ii] = sq_rec_err[:,ii] * yo
                    self.__activateCorrection = True

                elif self._correction.lower() == "local_variance":
                    #try to assign the observations to each cluster such that the
                    #variance in that cluster is minimized, i.e., the variables are
                    #more homogeneous
                    cluster2 = get_cluster(self.X, idx, ii)
                    yo = np.mean(np.var(cluster2))
                    scores_factor[:,ii] = sq_rec_err[:,ii] * yo
                    self.__activateCorrection = True

                elif self._correction.lower() == "phc_multi":
                    #assign the clusters to minimize the PHC 
                    local_homogeneity = PHC_coefficients[ii]
                    scores_factor[:,ii] = sq_rec_err[:,ii] * local_homogeneity
                    self.__activateCorrection = True
                
                elif self._correction.lower() == "local_skewness":
                    #assign the clusters to minimize the variables' skewness
                    from scipy.stats import skew

                    yo = np.mean(skew(cluster, axis=0))
                    scores_factor[:,ii] = sq_rec_err[:,ii] * yo
                    self.__activateCorrection = True
                
                else:
                    pass                            
            # Update idx --> choose the cluster where the rec err is minimized
            if self.__activateCorrection:
                idx = np.argmin(scores_factor, axis = 1)
            else:
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
            # Check convergence condition
            if (eps_rec_var <= eps_tol):
                lpca.write_final_stats(iteration, eps_rec)
                idx = self.merge_clusters(self.X_tilde, idx)
                break
            else:
                residuals = np.append(residuals, eps_rec_new)
            # Update counter
            iteration += 1
            # Consider only statistical meaningful groups of points: if there are <2 points
            #in a cluster, delete it because it's not statistically meaningful
            idx = self.merge_clusters(self.X_tilde, idx)
            self._k = max(idx) +1
        print("Convergence reached in {} iterations.".format(iteration))
        #lpca.plot_residuals(iteration, residuals)
        lpca.write_final_stats(iteration, eps_rec)
        idx = self.merge_clusters(self.X_tilde, idx)
        if self._postKNN == True:
            print("Moving observations via kNN..")
            idx = self.kNNpost(self.X_tilde, idx, self._neighborsNum)
            # Consider only statistical meaningful groups of points: if there are <2 points
            #in a cluster, delete it because it's not statistically meaningful
            idx = self.merge_clusters(self.X_tilde, idx)
        return idx


class fpca(lpca):
    '''
    Supervised partitioning based on an a-priori conditioning (and subsequent dim reduction), by means
    of a selected variable which is known to be important for the process. As it
    is not an iterative algorithm, it allows for a faster clustering in comparison with
    lpca via Vector Quantization, even if the choice of the optimal variable could constitute a
    difficult task for some applications, as it requires prior knowledge on the process, and the choice must
    be assessed case-by-case. For non-premixed, turbulent combustion applications, the
    mixture fraction Z is an optimal variable for the data conditioning, leading to excellent
    results both for data compression and interpretation tasks.

    Input:
    X = raw data matrix (observations x variables)
    condVec = the vector to be used in the partitioning phase

    '''
    def __init__(self, X, condVec, *dictionary):
        self.X = X
        self.condVec = condVec

        super().__init__(X)

        self._nPCs = self.X.shape[1]-1

        #Decide if the input matrix must be centered:
        self._center = True
        #Set the centering method:
        self._centering = 'mean'                                                                    #'mean' or 'min' are available
        #Decide if the input matrix must be scaled:
        self._scale = True
        #Set the scaling method:
        self._scaling = 'auto'

        if dictionary:
            settings = dictionary[0]

            try:
                self._k = settings["number_of_clusters"]
                if not isinstance(self._k, int) or self._k <= 1:
                    raise Exception
            except:
                self._k = 2
                warnings.warn("An exception occured with regard to the input value for the number of clusters (k). It could be not acceptable, or not given to the dictionary.")
                print("\tIt will be automatically set equal to: 2.")
                print("\tYou can ignore this warning if the number of clusters (k) has been assigned later via setter.")
                print("\tOtherwise, please check the conditions which must be satisfied by the input in the detailed documentation.")
            
            try:
                self._nPCs = settings["number_of_eigenvectors"]
                if self._nPCs < 0 or self._nPCs >= self.X.shape[1]:
                    raise Exception
            except:
                self._nPCs = int(self.X.shape[1]/2)
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


    def condition(self):
        '''
        This function is used to partition the data matrix 'X' in 'k' different
        bins, depending on the conditioning vector interval.
        '''
        #preprocess the training matrix
        self.X_tilde = self.preprocess_training(self.X, self._center, self._scale, self._centering, self._scaling)

        #compute the interval of the conditioning variable
        min_interval = np.min(self.condVec)
        max_interval = np.max(self.condVec)

        #depending on the final number of bins, the extension of each bin (delta_step)
        #is computed
        delta_step = (max_interval - min_interval) / self._k

        #assign each observation depending on its conditioning variable's value
        counter = 0
        self.idx = np.empty((len(self.condVec),),dtype=int)
        var_left = min_interval

        #Find the observations in each bin (find the idx, where the classes are
        #the different bins number)
        while counter < self._k:
            var_right = var_left + delta_step
            mask = np.logical_and(self.condVec >= var_left, self.condVec < var_right)
            self.idx[np.where(mask)] = counter
            counter += 1
            var_left += delta_step


        return self.idx


    def fit(self):
        '''
        This function performs PCA in each bin, and then it returns the LPCs,
        the local eigenvalues, the local scores and the centroids .
        '''

        for ii in range(0, self._k):
            #initialize the lists
            self.centroids = [None] *self._k
            self.LPCs = [None] *self._k
            self.u_scores = [None] *self._k
            self.Leigen = [None] *self._k

            for ii in range (0,self._k):
                #compute the cluster
                cluster = get_cluster(self.X_tilde, self.idx, ii)
                #the centroid is computed via function in the module: utility.py
                self.centroids[ii], cluster_ = center(cluster, self._centering, True)
                #solve the eigendecomposition problem for the centered cluster
                self.LPCs[ii], self.Leigen[ii] = PCA_fit(cluster_, self._nPCs)
                self.u_scores[ii] = cluster_ @ self.LPCs[ii]

            return self.LPCs, self.u_scores, self.Leigen, self.centroids


class KMeans(lpca):
    '''
    The K-Means clustering is an iterative algorithm to partition a matrix X, composed
    by 'n' observations and 'p' variables, into 'k' groups of similar points (clusters).
    The number of clusters is a-priori defined by the user.
    Initially, the clusters are assigned randomly, then, the algorithm shift the center 
    of mass of each cluster by means of the minimization of their squared euclidean 
    distances and the observations.

    --- PARAMETERS ---
    X:          RAW data matrix, uncentered and unscaled. It must be organized
                with the structure: (observations x variables).
    type X :    numpy array

    dictionary:         Dictionary containing all the instruction for the setters
    type dictionary:    dictionary



    --- SETTERS --- (inherited from LPCA)
    clusters:               number of clusters to be used for the partitioning
    type   k:               scalar

    to_center:              Enable the centering function
    type   _center:         boolean 
    
    centering:              set the centering method. Available choices for scaling
                            are 'mean' or 'min'.
    type   _centering:      string

    to_scale:               Enable the scaling function
    type   _scale:          boolean 

    scaling:                set the scaling method. Available choices for scaling
                            are 'auto' or 'vast' or 'range' or 'pareto'.
    type _scaling:          string

    initMode:               to activate in case Kmeans is used to initialize LPCA (has a lower tol for convergence)
    type   _method:         boolean

    '''
    
    def __init__(self,X, *dictionary):
        #Initialize matrix and number of clusters.
        self.X = X
        self._k = 2
        super().__init__(X)
        #This option must be set to 'True' if the kMeans is used only to
        #initialize other clustering algorithms, therefore a lower tolerance
        #is required for convergence.
        self._initMode = False
        #Set hard parameters (private, not meant to be modified).
        self.__convergence = False
        self.__iterMax = 100
        self.__numericTol = 1e-16
        self.__convergeTol = 1E-16

        #Decide if the input matrix must be centered:
        self._center = True
        #Set the centering method:
        self._centering = 'mean'                                                                    #'mean' or 'min' are available
        #Decide if the input matrix must be scaled:
        self._scale = True
        #Set the scaling method:
        self._scaling = 'auto'

        if dictionary:
            settings = dictionary[0]
            try:
                self._k = settings["number_of_clusters"]
                if not isinstance(self._k, int) or self._k <= 1:
                    raise Exception
            except:
                self._k = 2
                warnings.warn("An exception occured with regard to the input value for the number of clusters (k). It could be not acceptable, or not given to the dictionary.")
                print("\tIt will be automatically set equal to: 2.")
                print("\tYou can ignore this warning if the number of clusters (k) has been assigned later via setter.")
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
            

    @property
    def initMode(self):
        return self._initMode

    @initMode.setter
    def initMode(self, new_bool):
        self._initMode = new_bool

    @staticmethod
    def remove_empty(X, idx):
        '''
        Remove a cluster if it is empty, or not statistically meaningful.

        --- PARAMETERS ---
        X:          Original data matrix (observations x variables). 
        type X :    numpy array

        idx:        vector whose dimensions are (n,) containing the cluster assignment. 
        type idx :  numpy array


        --- RETURNS ---
        idx:        vector whose dimensions are (n,) containing the cluster assignment, WITHOUT EMPTY CLASSES.
        type idx:   numpy array 
        '''

        k = np.max(idx) +1
        jj = 0
        while jj < k:
            cluster_ = get_cluster(X, idx, jj)
            if cluster_.shape[0] < 2: 
                if jj > 0:
                    mask = np.where(idx >=jj)
                    idx[mask] -= 1
                else:
                    mask = np.where(idx >jj)
                    idx[mask] -= 1
                print("WARNING:")
                print("\tAn empty cluster was found:")
                print("\tThe number of cluster was lowered to ensure statistically meaningful results.")
                print("\tThe current number of clusters is equal to: {}".format(np.max(idx) +1))
                k = np.max(idx) +1
                jj = 0
            else:
                jj += 1

        return idx


    def fit(self):
        '''
        Group the observations depending on the sum of squared Euclidean distances.

        --- RETURNS ---
        idx:        vector whose dimensions are (n,) containing the cluster assignment for each observation.
        type idx:   numpy array 
        '''
        from scipy.spatial.distance import euclidean, cdist
        if not self._initMode:
            print("Fitting kmeans model..")
            self.X = self.preprocess_training(self.X, self._center, self._scale, self._centering, self._scaling)
        else:
            print("Initializing clusters via KMeans algorithm..")
            #pass the centering/scaling if the kMeans is used for the initialization, if
            #explicitely asked.
        #Declare matrix and variables to be used:
        C_mat = np.empty((self._k, self.X.shape[1]), dtype=float)
        C_old = np.empty((self._k, self.X.shape[1]), dtype=float)
        dist = np.empty((self.X.shape[0], self._k), dtype=float)
        idx = np.empty((self.X.shape[0],), dtype=int)
        minDist_ = np.empty((self.X.shape[0],), dtype=float)
        minDist_OLD = 1E15
        iter = 0

        #Initialize the centroids using 'k' random observations taken from the
        #dataset.
        for ii in range(0,self._k):
            C_mat[ii,:] = self.X[np.random.randint(0,self.X.shape[0]),:]

        #Start with the iterative algorithm:
        while iter < self.__iterMax:
            #Compute the euclidean distances between the matrix and all the
            #centroids. The function cdist returns a matrix 'dist' = (nObs x k)
            dist = cdist(self.X, C_mat)**2
            #For each observation, choose the nearest centroid.
            #The vector idx contains the corresponding class, while the minDist_
            #vector contains the numerical value of the distance, which will
            #be useful later, for the convergence check.
            for ii in range(0, self.X.shape[0]):
                idx[ii] = np.argmin(dist[ii,:])
                minDist_[ii] = np.min(dist[ii,:])
            #Compute the new clusters and the sum of the distances.
            clusters = get_all_clusters(self.X, idx)
            C_old = C_mat
            minDist_sum = np.sum(minDist_)
            #Compute the new centroids, and build the new C_mat matrix.
            for ii in range(0, self._k):
                centroid = get_centroids(clusters[ii])
                C_mat[ii,:] = centroid
            #Check the convergence measuring how much the centroids have changed
            varDist = np.abs((minDist_sum - minDist_OLD) / (minDist_sum + 1E-16))
            minDist_OLD = minDist_sum

            #If the variation between the new and the old position is below the
            #convergence tolerance, then stop the iterative algorithm and return
            #the current idx. Otherwise, keep iterating.
            if varDist < self.__convergeTol:
                print("The kMeans algorithm has reached convergence.")
                break

            iter += 1
            if not self._initMode:
                print("Iteration number: {}".format(iter))
                print("The SSE over all cluster is equal to: {}".format(minDist_sum))
                print("The SSE variance is equal to: {}".format(varDist))
            
            #Consider only statistical meaningful groups of points: if there
            #are empty cluster, delete them. The algorithm's iterations will
            #be re-initialized each time a cluster is deleted.
            idx = self.remove_empty(self.X, idx)
            check_ = np.max(idx)
            if check_+1 != self._k:
                self._k = max(idx) +1
                C_mat = np.empty((self._k, self.X.shape[1]), dtype=float)
                C_old = np.empty((self._k, self.X.shape[1]), dtype=float)
                dist = np.empty((self.X.shape[0], self._k), dtype=float)
                idx = np.empty((self.X.shape[0],), dtype=int)
                minDist_ = np.empty((self.X.shape[0],), dtype=float)
                minDist_OLD = 1E15
                iter = 0
                for ii in range(0,self._k):
                    C_mat[ii,:] = self.X[np.random.randint(0,self.X.shape[0]),:]
            

        return idx


class spectralClustering():
    '''
    [1] Von Luxburg, Ulrike. "A tutorial on spectral clustering." Statistics and computing 17.4 (2007): 395-416.

    Spectral clustering is an unsupervised algorithm based on the eigenvectors decomposition
    of a graph Laplacian matrix L to partition a (n x p) data-set X in 'k' different groups.

    The implemented algorithm is based on the computation of the unnormalized Laplacian, and
    it is based on the following steps:

    0.  Preprocessing: The training matrix X is centered and scaled.

    1.  Computation of S: a similarity matrix S, whose dimensions are (n x n), is computed from
        the centered/scaled matrix X_tilde, by means of a rb function.

    2.  Construction of the Laplacian: the unnormalized laplacian is computed by means of the weight
        matrix and the degree matrix.

    3.  Decomposition of the Laplacian: the eigendecomposition of the laplacian matrix is performed,
        and the first 'k' eigenvectors, corresponding to the 'k' smallest eigenvalues, are retained.

    4.  Clustering: The matrix obtained @ step number 3 is clustered with KMeans, and a vector with
        the labels for each observations is obtained.



    --- PARAMETERS ---
    X:          RAW data matrix, uncentered and unscaled. It must be organized
                with the structure: (observations x variables).
    type X :    numpy array


    
    --- SETTERS ---
    clusters:               number of clusters to be used for the partitioning
    type   k:               scalar

    to_center:              Enable the centering function
    type   _center:         boolean 
    
    centering:              set the centering method. Available choices for scaling
                            are 'mean' or 'min'.
    type   _centering:      string

    to_scale:               Enable the scaling function
    type   _scale:          boolean 

    scaling:                set the scaling method. Available choices for scaling
                            are 'auto' or 'vast' or 'range' or 'pareto'.
    type _scaling:          string

    affinity:               Which function to use to compute the affinity matrix. RBF is selected
    type   _affinity:       string

    sigma:                  value of sigma to be used in the affinity matrix computation formula
    type _sigma:            float

    '''
    def __init__(self,X, *dictionary):
        self.X = X
        self._k = 2
        self._affinity = 'rbf'
        self._sigma = 1.0

        self._center = True
        self._centering = 'mean'
        self._scale = True
        self._scaling = 'auto'

        self._n_obs = self.X.shape[0]

        if dictionary:
            settings = dictionary[0]
            try:
                self._k = settings["number_of_clusters"]
                if not isinstance(self._k, int) or self._k <= 1:
                    raise Exception
            except:
                self._k = 2
                warnings.warn("An exception occured with regard to the input value for the number of clusters (k). It could be not acceptable, or not given to the dictionary.")
                print("\tIt will be automatically set equal to: 2.")
                print("\tYou can ignore this warning if the number of clusters (k) has been assigned later via setter.")
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
                self._sigma = settings["sigma"]
                if not isinstance(self._sigma, float) and not isinstance(self._sigma, int):
                    raise Exception
                elif self._sigma < 0:
                    raise Exception
            except:
                self._sigma = 1.0
                warnings.warn("An exception occured with regard to the input value for sigma. It could be not acceptable, or not given to the dictionary.")
                print("\tIt will be automatically set equal to: 1.0.")
                print("\tYou can ignore this warning if sigma has been assigned later via setter.")
                print("\tOtherwise, please check the conditions which must be satisfied by the input in the detailed documentation.")


    @property
    def clusters(self):
        return self._k

    @clusters.setter
    def clusters(self, new_number):
        self._k = new_number

        if not isinstance(self._k, int) or self._k <= 1:
            self._k = 2
            warnings.warn("An exception occured with regard to the input value for the number of clusters (k). It could be not acceptable, or not given to the dictionary.")
            print("\tIt will be automatically set equal to: 2.")
            print("\tPlease check the conditions which must be satisfied by the input in the detailed documentation.")

    @property
    def sigma(self):
        return self._sigma

    @sigma.setter
    def sigma(self, new_value):
        self._sigma = new_value

        if not isinstance(self._sigma, float) and not isinstance(self._sigma, int):
            self._sigma = 1.0
            warnings.warn("An exception occured with regard to the input value for sigma. It could be not acceptable, or not given to the dictionary.")
            print("\tIt will be automatically set equal to: 1.0.")
            print("\tYou can ignore this warning if sigma has been assigned later via setter.")
            print("\tOtherwise, please check the conditions which must be satisfied by the input in the detailed documentation.")
        elif self._sigma < 0:
            self._sigma = 1.0
            warnings.warn("An exception occured with regard to the input value for sigma. It could be not acceptable, or not given to the dictionary.")
            print("\tIt will be automatically set equal to: 1.0.")
            print("\tYou can ignore this warning if sigma has been assigned later via setter.")
            print("\tOtherwise, please check the conditions which must be satisfied by the input in the detailed documentation.")

    
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
            print("\tOtherwise, please check the conditions which must be satisfied by the input in the detailed documentation.")
        elif self._scaling.lower() != "auto" and self._scaling.lower() != "vast" and self._scaling.lower() != "pareto" and self._scaling.lower() != "range":
            self._scaling = "auto"
            warnings.warn("An exception occured with regard to the input value for the scaling criterion. It could be not acceptable, or not given to the dictionary.")
            print("\tIt will be automatically set equal to: auto.")
            print("\tYou can ignore this warning if the scaling criterion has been assigned later via setter.")
            print("\tOtherwise, please check the conditions which must be satisfied by the input in the detailed documentation.")

    
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
        Group the observations with Spectral clustering.

        --- RETURNS ---
        idx:        vector whose dimensions are (n,) containing the cluster assignment for each observation.
        type idx:   numpy array 
        
        '''
        
        print("Preprocessing training matrix..")
        self.X_tilde = np.array(self.preprocess_training(self.X, self._center, self._scale, self._centering, self._scaling))
        #initialize the similarity matrix, whose dimensions are (nxn) --> WARNING: IT'S EXPENSIVE FOR LARGE MATRICES  
        W = np.zeros([self._n_obs, self._n_obs], dtype=float)
        print("Building weighted adjacency matrix..")
        for ii in range(0, self._n_obs):
            for jj in range(0, self._n_obs):
                W[ii,jj] = np.exp(-LA.norm(self.X_tilde[ii,:]-self.X_tilde[jj,:])**2/(2*self._sigma**2))

        D= np.zeros([self._n_obs, self._n_obs],dtype=float)
        print("Building degree matrix..")
        #build the diagonal degree matrix
        for ii in range(0, self._n_obs):
            D[ii,ii] = np.sum(W[ii,:])

        #Now build Laplacian matrix and do an eigendecomposition
        L = D-W 
        eigval, eigvec = LA.eigh(L)

        #Consider only the first 'k' columns of the eigenvector matrix
        #it is ok to consider the firsts and not the lasts because the eigh function orders them
        #in ascending order of magnitude, so the firsts will be the smallest ones,
        # as prescribed by the algorithm. 
        eigvec = eigvec[:,:self._k]

        #Now perform K-means on it, to partition in 'k' different clusters
        modelK = KMeans(eigvec)
        modelK.to_center = False
        modelK.to_scale = False
        modelK.initMode = False 
        modelK.clusters = self._k
        
        index = modelK.fit()

        return index


    def fitApprox(self):

        '''
        Group the observations with Spectral clustering, but compute the W matrix by means of
        the Nyström algorithm.

        --- RETURNS ---
        idx:        vector whose dimensions are (n,) containing the cluster assignment for each observation.
        type idx:   numpy array 
        
        '''
        
        
        self.X_tilde = self.preprocess_training(self.X, self._center, self._scale, self._centering, self._scaling)

        if self.X_tilde.shape[0] > 20000:
            rowsToPick = 100
        else:
            rowsToPick = 50

        print("Computing the W matrix via Nyström approximation (std Nyström algorithm)..")

        model = model_order_reduction.Kernel_approximation(self.X_tilde, kernelType="rbf", toCenter=False, toScale=False, centerCrit="mean", scalCrit="auto", numToPick=rowsToPick, sigma=self._sigma, rank=50, p=1)
        W = model.Nystrom_standard()
        W = W.real

       
        D= np.zeros([self.X_tilde.shape[0], self.X_tilde.shape[0]],dtype=float)
        print("Building degree matrix..")
        #build the diagonal degree matrix
        for ii in range(0, self.X_tilde.shape[0]):
            D[ii,ii] = np.sum(W[ii,:])

        
        #Now build Laplacian matrix and do an eigendecomposition
        L = D-W 

        print("Eigendecomposition step..")
        eigval, eigvec = LA.eigh(L)
        eigvec = eigvec[:,:self._k]
        
        print("K-means step")
        #Now perform K-means on it, to partition in 'k' different clusters
        modelK = KMeans(eigvec)
        modelK.to_center = False
        modelK.to_scale = False
        modelK.initMode = False 
        modelK.clusters = self._k
        
        index = modelK.fit()

        return index