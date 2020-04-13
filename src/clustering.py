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
    Class spectral: Clustering via unnormalized spectral clustering.

@Additional notes:
    This code is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
    Please report any bug to: giuseppe.dalessio@ulb.ac.be

'''


from utilities import *
import model_order_reduction

import numpy as np
import numpy.matlib
import matplotlib
import matplotlib.pyplot as plt


class lpca:
    '''
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
    '''
    def __init__(self, X):
        self.X = np.array(X)
        #Initialize the number of clusters:
        self._k = 2
        #Initialize the number of PCs to retain in each cluster:
        self._nPCs = 2
        #Set the initialization method:
        self._method = 'KMEANS'                                             #Available options: 'KMEANS' or 'RANDOM'
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


    @staticmethod
    def initialize_clusters(X, k, method):
        '''
        The clustering solution must be initialized. Two methods are available,
        a random allocation (RANDOM) or a previous clustering solution (KMEANS).
        '''
        if method.lower() == 'random':
            #Assign randomly an integer between 0 and k to each observation.
            idx = np.random.random_integers(0, k, size=(X.shape[0], 1))
        
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
            dist = cdist(X, C_mat)
        
            #For each observation, choose the nearest centroid (cdist --> Euclidean dist).
            #and compute the idx for the initialization.
            for ii in range(0, X.shape[0]):
                idx[ii] = np.argmin(dist[ii,:])
        
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
        for jj in range(0, max(idx)+1):
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
        newDirName = "Clustering LPCA - " + now.strftime("%Y_%m_%d-%H%M")
        
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
        '''
        #Center and scale the original training dataset
        print("Preprocessing training matrix..")
        self.X_tilde = self.preprocess_training(self.X, self._center, self._scale, self._centering, self._scaling)
        print("Fitting Local PCA model...")
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
            if self._correction == 'phc_standard':
                PHC_coefficients, PHC_std = PHC_index(self.X, idx)   #PHC_index(self.X, idx) or PHC_robustTrim(self.X, idx)
            elif self._correction == 'phc_median':
                PHC_coefficients, PHC_std = PHC_median(self.X, idx)
            elif self._correction == 'phc_robust':
                PHC_coefficients, PHC_std = PHC_robustTrim(self.X, idx)
            else:
                pass
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
                if self.correction.lower() == "mean":
                    correction_[:,ii] = np.mean(np.var((self.X_tilde - C_mat) @ modes[0], axis=0))
                    scores_factor = np.multiply(sq_rec_err, correction_)
                    self.__activateCorrection = True
                elif self._correction.lower() == "max":
                    correction_[:,ii] = np.max(np.var((self.X_tilde - C_mat) @ modes[0], axis=0))
                    scores_factor = np.multiply(sq_rec_err, correction_)
                    self.__activateCorrection = True
                elif self._correction.lower() == "min":
                    correction_[:,ii] = np.min(np.var((self.X_tilde - C_mat) @ modes[0], axis=0))
                    scores_factor = np.multiply(sq_rec_err, correction_)
                    self.__activateCorrection = True
                elif self._correction.lower() == "std":
                    correction_[:,ii] = np.std(np.var((self.X_tilde - C_mat) @ modes[0], axis=0))
                    scores_factor = np.multiply(sq_rec_err, correction_)
                    self.__activateCorrection = True
                elif self._correction.lower() == 'phc_standard' or self._correction.lower() == 'phc_median' or self._correction.lower() == 'phc_robust':
                    local_homogeneity = PHC_coefficients[ii]
                    scores_factor = np.add(sq_rec_err, local_homogeneity)
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
            self._k = max(idx)+1
        print("Convergence reached in {} iterations.".format(iteration))
        lpca.plot_residuals(iteration, residuals)
        return idx


class VQclassifier:
    '''
    For the classification task the following steps are accomplished:
    0. Preprocessing: The set of new observations Y is centered and scaled with the centering and scaling factors
    computed from the training dataset, X.
    1. For each cluster of the training matrix, computes the Principal Components.
    2. Assign each observation y \in Y to the cluster which minimizes the local reconstruction error.
    '''
    def __init__(self, X, idx, Y):
        self.X = X
        self._cent_crit = 'mean'
        self._scal_crit = 'auto'
        self.idx = idx
        self.k = max(self.idx) +1
        self.Y = Y
        self.nPCs = round(self.Y.shape[1] - (self.Y.shape[1]) /5) #Use a very high number of PCs to classify,removing only the last 20% which contains noise

    @property
    def centering(self):
        return self._cent_crit

    @centering.setter
    def centering(self, new_centering):
        self._cent_crit = new_centering

    @property
    def scaling(self):
        return self._scal_crit

    @scaling.setter
    def scaling(self, new_scaling):
        self._scal_crit = new_scaling



    def fit(self):
        '''
        Classify a new set of observations on the basis of a previous
        LPCA partitioning.
        '''
        print("Classifying the new observations...")
        # Compute the centering/scaling factors of the training matrix
        mu = center(self.X, self._cent_crit)
        sigma = scale(self.X, self._scal_crit)
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


class fpca:
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
    def __init__(self, X, condVec):
        self.X = X
        self.condVec = condVec

        self._nPCs = self.X.shape[1]-1

        #Decide if the input matrix must be centered:
        self._center = True
        #Set the centering method:
        self._centering = 'mean'                                                                    #'mean' or 'min' are available
        #Decide if the input matrix must be scaled:
        self._scale = True
        #Set the scaling method:
        self._scaling = 'auto'

    @property
    def clusters(self):
        return self._k

    @clusters.setter
    @accepts(object, int)
    def clusters(self, new_number):
        self._k = new_number

        if self._nPCs <= 0:
            raise Exception("The number of clusters must be a positive integer. Exiting..")
            exit()

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


class KMeans:
    '''
    X must be centered and scaled --- to change ---
    '''
    def __init__(self,X):
        #Initialize matrix and number of clusters.
        self.X = X
        self._k = 2
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
    def initMode(self):
        return self._initMode

    @initMode.setter
    @accepts(object, bool)
    def initMode(self, new_bool):
        self._initMode = new_bool

        if self._initMode:
            self.__convergeTol = 1E-08

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
    def merge_clusters(X, idx):
        '''
        Remove a cluster if it is empty, or not statistically meaningful.
        '''
        for jj in range(0, max(idx)+1):
            cluster_ = get_cluster(X, idx, jj)
            if cluster_.shape[0] < 2: #cluster_.shape[1]:
                pos = np.where(idx != 0)
                idx[pos] -= 1
                print("WARNING:")
                print("\tAn empty cluster was found:")
                print("\tThe number of cluster was lowered to ensure statistically meaningful results.")
                print("\tThe current number of clusters is equal to: {}".format(max(idx)))
                break
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
            dist = cdist(self.X, C_mat)
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

            #Consider only statistical meaningful groups of points: if there
            #are empty cluster, delete them. The algorithm's iterations will
            #be re-initialized each time a cluster is deleted.
            idx = KMeans.merge_clusters(self.X, idx)
            check_ = np.max(idx)
            if check_+1 != self._k:
                self._k = max(idx)
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
    def __init__(self,X, conditioning):
        self.X = X
        self.condVec = conditioning

        super().__init__(X)

    def partition(self):
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


def main():
    import clustering

    file_options = {
        "path_to_file"              : "/Users/giuseppedalessio/Dropbox/GitHub/data",
        "input_file_name"           : "dns_syngas_thermochemical.csv",
    }

    mesh_options = {
        "path_to_file"              : "/Users/giuseppedalessio/Dropbox/GitHub/data",
        "mesh_file_name"            : "dns_syngas_mesh.csv",
    }

    settings = {
        "centering_method"          : "MEAN",
        "scaling_method"            : "AUTO",
        "initialization_method"     : "observations",
        "number_of_clusters"        : 8,
        "number_of_eigenvectors"    : 4,
        "adaptive_PCs"              : False,
        "classify"                  : False,
        "write_on_txt"              : False,
        "plot_on_mesh"              : True,
    }


    X = readCSV(file_options["path_to_file"], file_options["input_file_name"])
    X_tilde = center_scale(X, center(X, method=settings["centering_method"]), scale(X, method=settings["scaling_method"]))


    model = clustering.lpca(X)
    model.centering = 'mean'
    model.scaling = 'auto'
    model.clusters = settings["number_of_clusters"]
    model.eigens = settings["number_of_eigenvectors"]
    model.initialization = settings["initialization_method"]
    model.correction = "mean" # 'phc'
    model.adaptivePCs = settings["adaptive_PCs"]

    index = model.fit()

    PHC_coeff, PHC_deviations = PHC_index(X, index)

    DB = evaluate_clustering_DB(X_tilde, index) #evaluate the clustering solution by means of the Davies-Bouldin index
    print(DB)

    text_file = open("stats_training_correction_.txt", "wt")
    DB_index = text_file.write("DB index equal to: {} \n".format(DB))
    PHC_coeff = text_file.write("Average PHC is: {} \n".format(np.mean(PHC_coeff)))
    text_file.close()

    if settings["write_on_txt"]:
        np.savetxt("idx_training.txt", index)


    if settings["plot_on_mesh"]:
        matplotlib.rcParams.update({'font.size' : 6, 'text.usetex' : True})
        mesh = np.genfromtxt(mesh_options["path_to_file"] + "/" + mesh_options["mesh_file_name"], delimiter= ',')

        fig = plt.figure()
        axes = fig.add_axes([0.2,0.15,0.7,0.7], frameon=True)
        axes.scatter(mesh[:,0], mesh[:,1], c=index,alpha=0.5)
        axes.set_xlabel('X [m]')
        axes.set_ylabel('Y [m]')
        plt.show()


    if settings["classify"]:

        file_options_classifier = {
            "path_to_file"              : "/home/peppe/Dropbox/GitHub/data",
            "test_file_name"            : "thermoC_timestep.csv",
        }

        try:
            print("Reading test matrix..")
            Y = np.genfromtxt(file_options_classifier["path_to_file"] + "/" + file_options_classifier["test_file_name"], delimiter= ',')
        except OSError:
            print("Could not open/read the selected file: " + "/" + file_options["test_file_name"])
            exit()

        # Input to the classifier: X = training matrix, Y = test matrix
        classifier = clustering.VQclassifier(X, index, Y)

        classifier.centering = settings["centering_method"]
        classifier.scaling = settings["scaling_method"]

        classification_vector = classifier.fit()

        if settings["write_on_txt"]:
            np.savetxt("idx_test.txt", classification_vector)


def mainK():
    import clustering

    file_options = {
        "path_to_file"              : "/Users/giuseppedalessio/Dropbox/GitHub/data",
        "input_file_name"           : "concentrations.csv",
    }


    settings = {
        "centering_method"          : "MEAN",
        "scaling_method"            : "AUTO",
        "number_of_clusters"        : 24,
    }


    mesh_options = {
        "path_to_file"              : "/Users/giuseppedalessio/Dropbox/GitHub/data",
        "mesh_file_name"           : "mesh.csv",
    }


    X = readCSV(file_options["path_to_file"], file_options["input_file_name"])
    X_tilde = center_scale(X, center(X, method=settings["centering_method"]), scale(X, method=settings["scaling_method"]))


    model = clustering.KMeans(X_tilde)
    model.clusters = settings["number_of_clusters"]
    index = model.fit()
    print(index)
    print("MIN IDX: {}".format(np.min(index)))
    print("MAX IDX: {}".format(np.max(index)))

    print("YO END")

    mesh = np.genfromtxt(mesh_options["path_to_file"] + "/" + mesh_options["mesh_file_name"], delimiter= ',')
    plt.scatter(mesh[:,0], mesh[:,1], c=index,alpha=0.5)
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.show()

def main_comb():
    import clustering

    file_options = {
        "path_to_file"              : "/Users/giuseppedalessio/Dropbox/GitHub/data",
        "input_file_name"           : "f10A25.csv",
    }


    settings = {
        "centering_method"          : "MEAN",
        "scaling_method"            : "AUTO",
        "number_of_clusters"        : 12,
    }


    mesh_options = {
        "path_to_file"              : "/Users/giuseppedalessio/Dropbox/GitHub/data",
        "mesh_file_name"            : "meshf10A25.csv",
    }


    X = readCSV(file_options["path_to_file"], file_options["input_file_name"])
    
    T = X[:,0]
    X = X[:,1:]

    X_tilde = center_scale(X, center(X, method=settings["centering_method"]), scale(X, method=settings["scaling_method"]))
    
    model = clustering.multistageLPCA(X, T)
    model.clusters = 4
    model.eigens = 11
    idx= model.partition()


    unique, counts = np.unique(idx, return_counts=True)
    print("UNIQUE: {}".format(unique))
    print("COUNTS: {}".format(counts))

    
    mesh = np.genfromtxt(mesh_options["path_to_file"] + "/" + mesh_options["mesh_file_name"], delimiter= ',')
    plt.scatter(mesh[:,0], mesh[:,1], c=idx,alpha=0.5)
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.show()

    
    
    DB = evaluate_clustering_DB(X_tilde, idx) #evaluate the clustering solutions by means of the Davies-Bouldin algorithm
    print("The DB index value for the multi is: {}".format(DB))

    PHC_coeff, PHC_deviations = PHC_robustTrim(X, idx)
    print("The PHC index value for the multi is: {}".format(np.mean(PHC_coeff)))
    


    model2= clustering.fpca(X,T)
    model2.clusters = 8

    idFPCA = model2.condition()

    unique2, counts2 = np.unique(idFPCA, return_counts=True)
    print("UNIQUE: {}".format(unique2))
    print("COUNTS: {}".format(counts2))

    mesh = np.genfromtxt(mesh_options["path_to_file"] + "/" + mesh_options["mesh_file_name"], delimiter= ',')
    plt.scatter(mesh[:,0], mesh[:,1], c=idFPCA,alpha=0.5)
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.show()

    DB2 = evaluate_clustering_DB(X_tilde, idFPCA) #evaluate the clustering solutions by means of the Davies-Bouldin algorithm
    print("The DB index value for the condT is: {}".format(DB2))

    PHC_coeff2, PHC_deviations = PHC_robustTrim(X, idFPCA)
    print("The PHC index value for the condT is: {}".format(np.mean(PHC_coeff2)))



if __name__ == '__main__':
    main_comb()
