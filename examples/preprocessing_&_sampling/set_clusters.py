import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import PyTROMode.clustering as clustering
from PyTROMode.utilities import *


file_options = {
    "path_to_file"              : "/Users/giuseppedalessio/Dropbox/GitHub/data",
    "input_file_name"           : "laminar2D.csv",
}

settings = {
    #centering and scaling options
    "center"                    : True,
    "centering_method"          : "mean",
    "scale"                     : True,
    "scaling_method"            : "auto",

    #set the initialization method (random, observations, kmeans, pkcia, uniform)
    "initialization_method"     : "uniform",

    #set the number of PCs in each cluster
    "number_of_eigenvectors"    : 12,

    #enable additional options:
    "adaptive_PCs"              : False,    # --> use a different number of PCs in each cluster (to test)
    "correction_factor"         : "off",    # --> enable eventual corrective coefficients for the LPCA algorithm:
                                            #     'off', 'mean', 'min', 'max', 'std', 'phc_standard', 'phc_median', 'phc_robust', 'medianoids', 'medoids' are available

    "classify"                  : False,    # --> classify a new matrix Y on the basis of the lpca clustering
    "write_on_txt"              : False,     # --> write the idx vector with the class for each observation
    "evaluate_clustering"       : False,     # --> enable the calculation of indeces to evaluate the goodness of the clustering


    "initial_k"                 : 5,
    "final_k"                   : 8,
}

X = readCSV(file_options["path_to_file"], file_options["input_file_name"])
X_tilde = center_scale(X, center(X, method=settings["centering_method"]), scale(X, method=settings["scaling_method"]))

num_of_k = np.linspace(settings["initial_k"], settings["final_k"], settings["final_k"]-settings["initial_k"])
DB_scores = [None]*len(num_of_k)
idxs = [None]*len(num_of_k)

for ii in range(0,len(num_of_k)):
    model = clustering.lpca(X, settings)
    model.clusters = int(num_of_k[ii])
    index = model.fit() 
    
    idxs[ii] = index
    DB_scores[ii] = evaluate_clustering_DB(X_tilde, index)

print(DB_scores)
print("According to the Davies-Bouldin index, the best number of clusters to use for the given dataset is: {}".format(num_of_k[np.argmin(DB_scores)]))
print("Corresponding idx: {}".format(idxs[np.argmin(DB_scores)]))


