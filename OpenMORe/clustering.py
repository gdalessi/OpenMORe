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

        if dictionary:
            settings = dictionary[0]
            try:
                self._k = settings["number_of_clusters"]
            except:
                self._k = 2
                print("Number of clusters not given to dictionary. Data will be automatically partitioned with k = 2..")
                print("You can ignore this warning if 'k' has been assigned later via setter.")
                
            try:
                self._nPCs = settings["number_of_eigenvectors"]
            except:
                self._nPCs = self.X.shape[1]-1
                print("Number of PCs to retain not given to dictionary. It will be automatically set equal to X.shape[1]-1.")
                print("You can ignore this warning if the number of PCs has been assigned later via setter.")
            try:
                self._method = settings["initialization_method"]
            except:
                self._method = 'uniform'
                print("Initialization method not given to dictionary. It will be automatically set equal to 'uniform'.")
                print("You can ignore this warning if the initialization method has been assigned later via setter.")
            try:
                self._correction = settings["correction_factor"]         
            except:
                self._correction = "off"
                print("Correction factor not given to dictionary. It will be automatically set equal to 'off'.")
                print("You can ignore this warning if the correction factor has been assigned later via setter.")
            try:
                self._adaptive = settings["adaptive_PCs"]
            except:
                self._adaptive = False
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
                self._writeFolder = settings["write_stats"]
            except:
                self._writeFolder = True


    @property
    def clusters(self):
        return self._k

    @clusters.setter
    @accepts(object, int)
    def clusters(self, new_number):
        self._k = new_number

        if self._k <= 0:
            raise Exception("The number of clusters in input must be a positive integer. Exiting..")
            exit()

    @property
    def eigens(self):
        return self._nPCs

    @eigens.setter
    @accepts(object, int)
    def eigens(self, new_number):
        self._nPCs = new_number

        if self._nPCs <= 0:
            raise Exception("The number of eigenvectors in input must be a positive integer. Exiting..")
            exit()

    @property
    def initialization(self):
        return self._method

    @initialization.setter
    def initialization(self, new_method):
        self._method = new_method


    @property
    def correction(self):
        return self._correction

    @correction.setter
    def correction(self, new_method):
        self._correction = new_method

    @property
    def adaptivePCs(self):
        return self._adaptive

    @adaptivePCs.setter
    @accepts(object, bool)
    def adaptivePCs(self, new_bool):
        self._adaptive = new_bool

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
    def writeFolder(self):
        return self._writeFolder

    @writeFolder.setter
    @accepts(object, bool)
    def writeFolder(self, new_string):
        self._writeFolder = new_string


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
        iter_max = 250
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
        # Initialize solution
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
                PHC_coefficients, PHC_std = evaluate_clustering_PHC(self.X, idx, method='phc_standard')   #PHC_index(self.X, idx) or PHC_robustTrim(self.X, idx)
                PHC_coefficients = PHC_coefficients/np.max(PHC_coefficients)

            for ii in range(0, self._k):
                cluster = get_cluster(self.X_tilde, idx, ii)
                if self._correction.lower() != 'medianoids' and self._correction.lower() != 'medoids':
                    centroids = get_centroids(cluster)
                elif self.correction.lower() == 'medianoids':
                    centroids = get_medianoids(cluster)
                elif self.correction.lower() == 'medoids':
                    centroids = get_medoids(cluster)
                local_model = model_order_reduction.PCA(cluster)
                local_model.to_center = False
                local_model.to_scale = False
                if not self._adaptive:
                    local_model.eigens = self._nPCs
                else:
                    local_model.set_PCs()
                modes = local_model.fit()
                C_mat = np.matlib.repmat(centroids, rows, 1)
                rec_err_os = (self.X_tilde - C_mat) - (self.X_tilde - C_mat) @ modes[0] @ modes[0].T
                sq_rec_oss = np.power(rec_err_os, 2)
                sq_rec_err[:,ii] = sq_rec_oss.sum(axis=1)
                
                if self.correction.lower() == "c_range":
                    
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
                    
                    self.__activateCorrection = True
                
                elif self._correction.lower() == "uncorrelation":
                    maxF = np.max(np.var((self.X_tilde - C_mat) @ modes[0], axis=0))
                    minF = np.min(np.var((self.X_tilde - C_mat) @ modes[0], axis=0))
                    yo = 1-minF/maxF
                    
                    scores_factor[:,ii] = sq_rec_err[:,ii] * yo
                    self.__activateCorrection = True

                elif self._correction.lower() == "local_variance":
                    yo = np.mean(np.var(cluster2))
                    scores_factor[:,ii] = sq_rec_err[:,ii] * yo
                    self.__activateCorrection = True

                elif self._correction.lower() == "phc_multi":
                    local_homogeneity = PHC_coefficients[ii]
                    scores_factor[:,ii] = sq_rec_err[:,ii] * local_homogeneity
                    self.__activateCorrection = True
                
                elif self._correction.lower() == "local_skewness":
                    from scipy.stats import skew

                    yo = np.mean(skew(cluster, axis=0))
                    scores_factor[:,ii] = sq_rec_err[:,ii] * yo
                    self.__activateCorrection = True
                
                else:
                    pass                            
            # Update idx
            if self.__activateCorrection == True:
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
            # Check condition
            if (eps_rec_var <= eps_tol):
                lpca.write_final_stats(iteration, eps_rec)
                break
            else:
                residuals = np.append(residuals, eps_rec_new)
            # Update counter
            iteration += 1
            # Consider only statistical meaningful groups of points
            idx = lpca.merge_clusters(self.X_tilde, idx)
            self._k = max(idx) +1
        print("Convergence reached in {} iterations.".format(iteration))
        #lpca.plot_residuals(iteration, residuals)
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
            except:
                self._k = 2
                print("Number of clusters not given to dictionary. Data will be automatically partitioned with k = 2..")
                print("You can ignore this warning if 'k' has been assigned later via setter.")
            try:
                self._nPCs = settings["number_of_eigenvectors"]
            except:
                self._nPCs = self.X.shape[1]-1
                print("Number of PCs to retain not given to dictionary. It will be automatically set equal to X.shape[1]-1.")
                print("You can ignore this warning if the number of PCs has been assigned later via setter.")
            try:
                self._method = settings["initialization_method"]
            except:
                self._method = 'uniform'
                print("Initialization method not given to dictionary. It will be automatically set equal to 'uniform'.")
                print("You can ignore this warning if the initialization method has been assigned later via setter.")
            try:
                self._correction = settings["correction_factor"]
            except:
                self._correction = "off"
                print("Correction factor not given to dictionary. It will be automatically set equal to 'off'.")
                print("You can ignore this warning if the correction factor has been assigned later via setter.")
            try:
                self._adaptive = settings["adaptive_PCs"]
            except:
                self._adaptive = False
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
                self._writeFolder = settings["write_stats"]
            except:
                self._writeFolder = True



    def condition(self):
        '''
        This function is used to partition the data matrix 'X' in 'k' different
        bins, depending on the conditioning vector interval.
        '''

        self.X_tilde = self.preprocess_training(self.X, self._center, self._scale, self._centering, self._scaling)


        min_interval = np.min(self.condVec)
        max_interval = np.max(self.condVec)

        delta_step = (max_interval - min_interval) / self._k

        counter = 0
        self.idx = np.empty((len(self.condVec),),dtype=int)
        var_left = min_interval

        #Find the observations in each bin (find the idx, where the classes are
        #the different bins number)
        while counter <= self._k:
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
            self.centroids = [None] *self._k
            self.LPCs = [None] *self._k
            self.u_scores = [None] *self._k
            self.Leigen = [None] *self._k

            for ii in range (0,self._k):
                cluster = get_cluster(self.X_tilde, self.idx, ii)
                self.centroids[ii], cluster_ = center(cluster, self._centering, True)
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
            except:
                self._k = 2
                print("Number of clusters not given to dictionary. Data will be automatically partitioned with k = 2..")
                print("You can ignore this warning if 'k' has been assigned later via setter.")
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


class multistageLPCA(lpca):
    '''
    Perform both local-PCA algorithm in two steps: first, with a supervised partitioning by means of a
    conditioning variable (fpca), a set of clusters is found. After that, in each of these groups, the
    vqpca algorithm is perform with a bisection logic, i.e., in each of the conditioned clusters, vqpca
    is performed with k = 2. 

    --- PARAMETERS ---
    X:                      RAW data matrix, uncentered and unscaled. It must be organized
                            with the structure: (observations x variables).
    type X :                numpy array

    conditioninig:          Column vector (observations x 1) containing the variable which must be 
                            used to condition the dataset.
    type conditioninig:     numpy array

    
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
    def __init__(self,X, conditioning):
        self.X = X
        self.condVec = conditioning

        super().__init__(X)

    def partition(self):
        '''
        Group the observations depending coupling fpca/vqpca.

        --- RETURNS ---
        idx:        vector whose dimensions are (n,) containing the cluster assignment for each observation.
        type idx:   numpy array 
        
        '''
        #First of all, center and scale the data matrix
        self.X_tilde = self.preprocess_training(self.X, self._center, self._scale, self._centering, self._scaling)

        #Compute the bin width (delta_step), after calculating the max and the min of the
        #conditioning vector, condVec. An additional 1% is subtracted (resp. added) to the
        #interval, because in this process few points at the interval limits could be lost otherwise
        min_interval = np.min(self.condVec) -0.01*np.min(self.condVec)
        max_interval = np.max(self.condVec) +0.01*np.max(self.condVec)
        delta_step = (max_interval - min_interval) / self._k

        #Declare an id vector, to keep track of the observations which will be split into
        #the clusters.
        id = np.linspace(1,self.X.shape[0], self.X.shape[0])

        #Start the conditioning by means of condVec
        counter = 0
        self.idxF = np.empty((len(self.condVec),), dtype=int)
        var_left = min_interval

        #Put the observations in the bin where they belong, until the max number of
        #bins is reached:
        while counter <= self._k:
            var_right = var_left + delta_step
            mask = np.logical_and(self.condVec >= var_left, self.condVec < var_right)
            self.idxF[np.where(mask)] = counter
            counter += 1
            var_left += delta_step

        #In each bin, compute the iterative VQPCA algorithm. In this way, each bin of
        #condVec is split in two:
        for ii in range(0,self._k):
            #Get the cluster's observations
            cluster_ = get_cluster(self.X_tilde, self.idxF, ii)
            #Get the number of the observation in the original matrix
            id_ = get_cluster(id, self.idxF, ii)

            #Perform VQPCA in the cluster, using k = 2.
            modelVQ = lpca(cluster_)
            modelVQ.to_center = False
            modelVQ.to_scale = False
            modelVQ.clusters = 2
            modelVQ.eigens = self.eigens
            modelVQ.initialization = 'observations'
            idxLoc = modelVQ.fit()
            #There is need to have 1 as k_min in each cluster, so increase the python
            #enumeration
            idxLoc = idxLoc +1 #now the indeces are [1,2]

            #The following code makes sure that all the integers between 1 and k, with k
            #equal to the selected number of clusters, are covered. So that:
            #Bin1 --> k = 1,2
            #Bin2 --> k = 3,4
            #Bin3 --> k = 4,5
            #etc..
            ii+=1
            if ii > 1:
                mask1 = np.where(idxLoc == 1)
                idxLoc[mask1] = ii + (ii-1)
                mask2 = np.where(idxLoc == 2)
                idxLoc[mask2] = 2*ii
            else:
                mask1 = np.where(idxLoc == 1)
                idxLoc[mask1] = ii
                mask2 = np.where(idxLoc == 2)
                idxLoc[mask2] = ii+1
            ii-=1

            #Now we temporarely merge the observation id vector with the cluster, because
            #later we'll have to reassemble the original matrix in the exact same order, so
            #in the end the 'tracking' matrix is obtained, whose dimensions are (n x (p+1)).
            tracking = np.hstack([id_.reshape((-1,1)), cluster_])

            #As soon as the matrix cluster_ is partitioned via VQPCA, rebuild the training
            #matrix. For now, the training matrix will still be not ordered as the original
            #one.
            if ii == 0:
                X_yo = tracking
                idx_yo = idxLoc
            else:
                X_yo = np.concatenate((X_yo, tracking), axis=0)
                idx_yo = np.concatenate((idx_yo, idxLoc), axis=0)


        #To order again the observations of the clustered matrix in the exact same way as
        #the original one, isolate and sort in ascending order the previous id vector which
        #was put in the matrix first column.
        id_back = X_yo[:,0]
        mask = np.argsort(id_back)

        #Sort the matrix and the idx:
        X_yo = X_yo[mask,:]
        idx_yo = idx_yo[mask]

        idx_yo = idx_yo-1


        return idx_yo

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
    def __init__(self,X):
        self.X = X
        self._k = 2
        self._affinity = 'rbf'
        self._sigma = 1.0

        self._center = True
        self._centering = 'mean'
        self._scale = True
        self._scaling = 'auto'

        self._n_obs = self.X.shape[0]

    @property
    def clusters(self):
        return self._k

    @clusters.setter
    @accepts(object, int)
    def clusters(self, new_number):
        self._k = new_number

        if self._k <= 0:
            raise Exception("The number of clusters in input must be a positive integer. Exiting..")
            exit()

    @property
    def affinity(self):
        return self._affinity

    @affinity.setter
    @accepts(object, str)
    def affinity(self, new_string):
        self._affinity = new_string


    @property
    def sigma(self):
        return self._sigma

    @sigma.setter
    @accepts(object, float)
    def sigma(self, new_value):
        self._sigma = new_value

    
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
        self.X_tilde = self.preprocess_training(self.X, self._center, self._scale, self._centering, self._scaling)

        W = np.zeros([self._n_obs, self._n_obs], dtype=float)
        
        print("Building weighted adjacency matrix..")
        for ii in range(0, self._n_obs):
            for jj in range(0, self._n_obs):
                W[ii,jj] = np.exp(-LA.norm(self.X_tilde[ii,:]-self.X_tilde[jj,:])**2/(2*self._sigma**2))

        D= np.zeros([self._n_obs, self._n_obs],dtype=float)
        print("Building degree matrix..")
        
        for ii in range(0, self._n_obs):
            D[ii,ii] = np.sum(W[ii,:])

        #Now build Laplacian matrix and do an eigendecomposition
        L = D-W 
        eigval, eigvec = LA.eigh(L)

        #Consider only the first 'k' columns of the eigenvector matrix
        eigvec = eigvec[:,:self._k]

        #Now perform K-means on it, to partition in 'k' different clusters
        modelK = KMeans(eigvec)
        modelK.to_center = False
        modelK.to_scale = False
        modelK.initMode = False 
        modelK.clusters = self._k

        index = modelK.fit()

        return index