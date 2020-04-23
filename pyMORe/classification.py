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
from . import model_order_reduction
from . import clustering

import numpy as np
import numpy.matlib
import matplotlib
import matplotlib.pyplot as plt


class VQclassifier(clustering.lpca):
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
        self.nPCs = round(self.Y.shape[1] - (self.Y.shape[1]) /5) #Use a very high number of PCs to classify,removing only the last 20% which contains noise

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
        sq_rec_oss = np.empty((rows, cols), dtype=float)
        sq_rec_err = np.empty((rows, self.k), dtype=float)
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



if __name__ == '__main__':

    file_options = {
        "path_to_file"              : "/Users/giuseppedalessio/Dropbox/GitHub/data",
        "input_file_name"           : "cfdf.csv",

        "idx_name"                  : "idx.txt",
    }

    X = readCSV(file_options["path_to_file"], file_options["input_file_name"])
    idx = np.genfromtxt(file_options["path_to_file"] + "/" + file_options["idx_name"], delimiter= '\n')


    file_options_classifier = {
        "path_to_file"              : "/Users/giuseppedalessio/Dropbox/GitHub/data",
        "test_file_name"            : "laminar2D.csv",
    }

    mesh_options = {
        "path_to_file"              : "/Users/giuseppedalessio/Dropbox/GitHub/data",
        "mesh_file_name"            : "mesh.csv",
    }

    try:
        print("Reading test matrix..")
        Y = np.genfromtxt(file_options_classifier["path_to_file"] + "/" + file_options_classifier["test_file_name"], delimiter= ',')
    except OSError:
        print("Could not open/read the selected file: " + "/" + file_options_classifier["test_file_name"])
        exit()


    classifier = VQclassifier(X, idx, Y)
    classification_vector = classifier.fit()

    matplotlib.rcParams.update({'font.size' : 6, 'text.usetex' : True})
    mesh = np.genfromtxt(mesh_options["path_to_file"] + "/" + mesh_options["mesh_file_name"], delimiter= ',')

    fig = plt.figure()
    axes = fig.add_axes([0.2,0.15,0.7,0.7], frameon=True)
    axes.scatter(mesh[:,0], mesh[:,1], c=classification_vector, alpha=0.5)
    axes.set_xlabel('X [m]')
    axes.set_ylabel('Y [m]')
    plt.show()
