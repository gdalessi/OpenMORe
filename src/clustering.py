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
        #Adaptive PCs per cluster:
        self._adaptive = False                                              #Available options: True or False (boolean)


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


    @staticmethod
    def initialize_clusters(X, k, method):
        '''
        The clustering solution must be initialized. Two methods are available,
        a random allocation (RANDOM) or a previous clustering solution (KMEANS).
        '''
        if method.lower() == 'random':
            idx = np.random.random_integers(0, k, size=(X.shape[0], 1))
        elif method.lower() == 'kmeans':
            init = KMeans(X)
            init.clusters = k
            init.initMode =True
            idx = init.fit()
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
        os.mkdir(newDirName)
        os.chdir(newDirName)


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


    def fit(self):
        '''
        Group the observations depending on the PCA reconstruction error.
        '''
        print("Fitting Local PCA model...")
        lpca.set_environment()
        lpca.write_recap_text(self._k, self._nPCs, self._correction, self._method)
        # Initialization
        iteration, eps_rec, residuals, iter_max, eps_tol = lpca.initialize_parameters()
        rows, cols = np.shape(self.X)
        # Initialize solution
        idx = lpca.initialize_clusters(self.X, self._k, self._method)
        residuals = np.array(0)
        if self._correction != "off":
            correction_ = np.zeros((rows, self._k), dtype=float)
            scores_factor = np.zeros((rows, self._k), dtype=float)
        # Iterate
        while(iteration < iter_max):
            sq_rec_oss = np.zeros((rows, cols), dtype=float)
            sq_rec_err = np.zeros((rows, self._k), dtype=float)
            for ii in range(0, self._k):
                cluster = get_cluster(self.X, idx, ii)
                centroids = get_centroids(cluster)
                local_model = model_order_reduction.PCA(cluster)
                local_model.to_center = False
                local_model.to_scale = False
                if not self._adaptive:
                    local_model.eigens = self._nPCs
                else:
                    local_model.set_PCs()
                modes = local_model.fit()
                C_mat = np.matlib.repmat(centroids, rows, 1)
                rec_err_os = (self.X - C_mat) - (self.X - C_mat) @ modes[0] @ modes[0].T
                sq_rec_oss = np.power(rec_err_os, 2)
                sq_rec_err[:,ii] = sq_rec_oss.sum(axis=1)
                if self._correction != "off":
                    if self.correction == "mean":
                        correction_[:,ii] = np.mean(np.var((self.X - C_mat) @ modes[0], axis=0))
                    elif self._correction == "max":
                        correction_[:,ii] = np.max(np.var((self.X - C_mat) @ modes[0], axis=0))
                    elif self._correction == "min":
                        correction_[:,ii] = np.min(np.var((self.X - C_mat) @ modes[0], axis=0))
                    elif self._correction == "std":
                        correction_[:,ii] = np.std(np.var((self.X - C_mat) @ modes[0], axis=0))
                    scores_factor = np.multiply(sq_rec_err, correction_)
            # Update idx
            if self._correction != "off":
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
            idx = lpca.merge_clusters(self.X, idx)
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


class spectral:
    '''
    The spectral clustering algorithm is based on the following steps:
    1) Construct a similarity graph, with A its weighted adjacency matrix.
    2) Compute the unnormalized laplacian matrix L from A.
    3) Decompose the L matrix, computing its eigenvalues and eigenvectors.
    4) Compute the matrix U (n x k), where the columns are the first 'k' eigenvectors from the decomposition.
    5) Apply k-Means on the U matrix.

    WARNING: It is extremely expensive from a CPU point of view. It cannot be applied to large matrices.sr
    '''
    def __init__(self, X, k, sigma=False):
        self.X = X
        self.k = k
        if not sigma:                       # sigma is the neighborhood radius. Generally its value must be optimized by means of a grid search.
            self.sigma = 0.5
        else:
            self.sigma = sigma


    def fit(self):
        from sklearn.neighbors import radius_neighbors_graph
        from scipy.sparse import csgraph

        # Compute the adjancency matrix from the training dataset (epsilon-neighborhood graph)
        print("Computing Adjacency matrix..")
        A = radius_neighbors_graph(self.X, self.sigma, mode='distance', metric='euclidean')
        A = A.toarray()

        # Compute the unnormalized Laplacian
        L = csgraph.laplacian(A, normed=False)

        # Laplacian decomposition - retain only the first 'k' eigenvectors
        eigval, eigvec = np.linalg.eig(L)
        eigvec = eigvec[:, :self.k]

        # Apply k-Means on the new points representation
        kmeans = KMeans(n_clusters=self.k, random_state=0).fit(eigvec)
        idx = kmeans.labels_
        centroids = kmeans.cluster_centers_

        return idx, centroids, eigvec


class KMeans:
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


    def fit(self):
        from scipy.spatial.distance import euclidean, cdist
        if not self._initMode:
            print("Fitting kmeans model..")
        else:
            print("Initializing clusters via KMeans algorithm..")
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



def main():
    import clustering

    file_options = {
        "path_to_file"              : "/Users/giuseppedalessio/Dropbox/GitHub/data",
        "input_file_name"           : "concentrations.csv",
    }

    mesh_options = {
        "path_to_file"              : "/Users/giuseppedalessio/Dropbox/GitHub/data",
        "mesh_file_name"           : "mesh.csv",
    }

    settings = {
        "centering_method"          : "MEAN",
        "scaling_method"            : "AUTO",
        "initialization_method"     : "KMEANS",
        "number_of_clusters"        : 16,
        "number_of_eigenvectors"    : 11,
        "adaptive_PCs"              : False,
        "classify"                  : False,
        "write_on_txt"              : True,
        "plot_on_mesh"              : True,
    }


    X = readCSV(file_options["path_to_file"], file_options["input_file_name"])
    X_tilde = center_scale(X, center(X, method=settings["centering_method"]), scale(X, method=settings["scaling_method"]))


    model = clustering.lpca(X_tilde)

    model.clusters = settings["number_of_clusters"]
    model.eigens = settings["number_of_eigenvectors"]
    model.initialization = settings["initialization_method"]
    model.correction = "off"
    model.adaptivePCs = settings["adaptive_PCs"]

    index = model.fit()

    DB = evaluate_clustering_DB(X_tilde, index) #evaluate the clustering solutions by means of the Davies-Bouldin algorithm
    print(DB)

    if settings["write_on_txt"]:
        np.savetxt("idx_training.txt", index)


    if settings["plot_on_mesh"]:
        mesh = np.genfromtxt(mesh_options["path_to_file"] + "/" + mesh_options["mesh_file_name"], delimiter= ',')
        plt.scatter(mesh[:,0], mesh[:,1], c=index,alpha=0.5)
        plt.xlabel("X [m]")
        plt.ylabel("Y [m]")
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




if __name__ == '__main__':
    main()
