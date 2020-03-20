'''
Variables selection via Principal Component Analysis and Procustes Analysis

@Authors: 
    G. D'Alessio [1,2], A. Parente[1]
    [1]: Université Libre de Bruxelles, Aero-Thermo-Mechanics Laboratory, Bruxelles, Belgium
    [2]: CRECK Modeling Lab, Department of Chemistry, Materials and Chemical Engineering, Politecnico di Milano

@Contacts:
    giuseppe.dalessio@ulb.ac.be

@Brief: 
    Variable selection via Principal Component Analysis and Procustes Analysis.

@Details: 
    In many applications, rather than reducing the dimensionality considering a
    new set of coordinates which are linear combination of the original ones, the main
    interest is to achieve a dimensionality reduction selecting a subset of m variables
    from the original set of p variables. One of the possible ways to accomplish this
    task is to couple the PCA dimensionality reduction with a Procustes Analysis.
    The iterative Local Principal Component Analysis clustering algorithm is based on the following steps:
    0. Preprocessing: The training matrix X is centered and scaled, after being loaded. Four scaling are available,
    AUTO, VAST, PARETO, RANGE - Two centering are available, MEAN and MIN;
    1. The dimensionality of m is initially set equal to p.
    2. Each variable is deleted from the matrix X, obtaining p ~X matrices. The
    corresponding scores matrices are computed by means of PCA. For each
    of them, a Procustes Analysis is performed with respect to the scores of the original matrix X, 
    and the corresponding M2 coeffcient is computed.
    3. The variable which, once excluded, leads to the smallest M2 coefficient is deleted from the matrix.
    4. Steps 2 and 3 are repeated until m variables are left.

@Cite:
    - Algorithm for variable selection via PCA and Procustes Analysis:
    [a] Wojtek J Krzanowski. Selection of variables to preserve multivariate data structure, using principal components. Journal of the Royal Statistical Society: Series C (Applied Statistics), 36(1):22{33, 1987.
    [b] Ian Jolliffe. Principal component analysis. Springer, 2011.

    - Preprocessing effects on PCA:
    [c] Parente, Alessandro, and James C. Sutherland. "Principal component analysis of turbulent combustion data: Data pre-processing and manifold sensitivity." Combustion and flame 160.2 (2013): 340-350.

@Additional notes:
    This cose is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; 
    Please report any bug to: giuseppe.dalessio@ulb.ac.be

'''



import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from utilities import *
import reduced_order_modelling 
from reduced_order_modelling import *
import clustering


file_options = {
    "path_to_file"              : "/Users/giuseppedalessio/Dropbox/GitHub/data",
    "file_name"                 : "concentrations.csv",
    "labels_name"               : "labels_species.csv"
}


settings = {
    "centering_method"          : "MEAN",
    "scaling_method"            : "AUTO",
    "n_variables_retained"      : 20,
    "number_of_eigenvectors"    : 15,
}

settings_local = {
    "Local_algorithm"           : False,
    "initialization_method"     : "KMEANS",
    "number_of_clusters"        : 16,
}


try:
    print("Reading the training matrix..")
    X = pd.read_csv(file_options["path_to_file"] + "/" + file_options["file_name"], sep = ',', header = None) 
except OSError:
    print("Could not open/read the selected file: " + "/" + file_options["file_name"])
    exit()

try:
    print("Reading the variables' labels..")
    labels = np.array(pd.read_csv(file_options["path_to_file"] + "/" + file_options["labels_name"], sep = ',', header = None))
except OSError:
    print("Could not open/read the selected file: " + "/" + file_options["labels_name"])
    exit()


X_tilde = np.array(center_scale(X, center(X, method=settings["centering_method"]), scale(X, method=settings["scaling_method"])))

if not settings_local["Local_algorithm"]:
    model_reduction = reduced_order_modelling.variables_selection(X_tilde, labels)

    model_reduction.eigens = settings["number_of_eigenvectors"]
    model_reduction.retained = settings["n_variables_retained"]

    retained_variables = model_reduction.fit()
    print(retained_variables)

elif settings_local["Local_algorithm"]:
    model_cluster = clustering.lpca(X_tilde)

    model_cluster.clusters = settings_local["number_of_clusters"]
    model_cluster.eigens = settings["number_of_eigenvectors"]
    model_cluster.initialization = settings_local["initialization_method"]

    index = model_cluster.fit()
    for ii in range (0, settings_local["number_of_clusters"]):
        cluster_X = get_cluster(X_tilde, index, ii)
        model_reduction = reduced_order_modelling.variables_selection(cluster_X, labels)
        
        model_reduction.eigens = settings["number_of_eigenvectors"]
        model_reduction.retained = settings["n_variables_retained"]
        
        retained_variables = model_reduction.fit()

        print("Cluster number {}".format(ii))
        print("\Principal variables: {}".format(retained_variables))
else:
    raise Exception("Please specify if the global or the local version of the algorithm must be adopted.")
    exit()