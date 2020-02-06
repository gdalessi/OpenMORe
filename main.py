'''
PROGRAM: lpca.py

@Authors: 
    G. D'Alessio [1,2], G. Aversano [1], A. Parente[1]
    [1]: Université Libre de Bruxelles, Aero-Thermo-Mechanics Laboratory, Bruxelles, Belgium
    [2]: CRECK Modeling Lab, Department of Chemistry, Materials and Chemical Engineering, Politecnico di Milano

@Contacts:
    giuseppe.dalessio@ulb.ac.be

@Brief: 
    Clustering via Local Principal Component Analysis.

@Details: 
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

@Cite:
    - Local algorithm for dimensionality reduction:
    [a] Kambhatla, Nandakishore, and Todd K. Leen. "Dimension reduction by local principal component analysis.", Neural computation 9.7 (1997): 1493-1516.

    - Clustering applications:
    [b] D’Alessio, Giuseppe, et al. "Adaptive chemistry via pre-partitioning of composition space and mechanism reduction.", Combustion and Flame 211 (2020): 68-82.

    - Data analysis applications:
    [c] Parente, Alessandro, et al. "Investigation of the MILD combustion regime via principal component analysis." Proceedings of the Combustion Institute 33.2 (2011): 3333-3341.
    [d] D'Alessio, Giuseppe, et al. "Analysis of turbulent reacting jets via Principal Component Analysis", Data Analysis in Direct Numerical Simulation of Turbulent Combustion, Springer (2020).
    [e] Bellemans, Aurélie, et al. "Feature extraction and reduced-order modelling of nitrogen plasma models using principal component analysis." Computers & chemical engineering 115 (2018): 504-514.

    - Preprocessing effects on PCA:
    [f] Parente, Alessandro, and James C. Sutherland. "Principal component analysis of turbulent combustion data: Data pre-processing and manifold sensitivity." Combustion and flame 160.2 (2013): 340-350.

    - Model order reduction:
    [g] Parente, Alessandro, et al. "Identification of low-dimensional manifolds in turbulent flames." Proceedings of the Combustion Institute. 2009 Jan 1;32(1):1579-86.
    [h] Aversano, Gianmarco, et al. "Application of reduced-order models based on PCA & Kriging for the development of digital twins of reacting flow applications." Computers & chemical engineering 121 (2019): 422-441.

@Additional notes:
    This cose is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; 
    Please report any bug to: giuseppe.dalessio@ulb.ac.be

'''


import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from preprocessing import *
from initialization import *

import clustering


file_options = {
    "path_to_file"              : "/Users/giuseppedalessio/Dropbox/python_course/LPCA/",
    "file_name"                 : "f10A25.csv",
}


settings = {
    "centering_method"          : "MEAN",
    "scaling_method"            : "AUTO",
    "initialization_method"     : "KMEANS",
    "number_of_clusters"        : 8,
    "number_of_eigenvectors"    : 15
}


try:
    print("Reading the training matrix..")
    X = pd.read_csv(file_options["path_to_file"] + file_options["file_name"], sep = ',', header = None) 
    print("The training matrix has been read successfully!")
except OSError:
    print("Could not open/read the selected file: " + file_options["file_name"])
    exit()



X_tilde = center_scale(X, center(X, method=settings["centering_method"]), scale(X, method=settings["scaling_method"]))

model = clustering.lpca(X_tilde, settings["number_of_clusters"], settings["number_of_eigenvectors"], settings["initialization_method"])
index = model.fit()

np.savetxt("idx_new_version_python.txt", index)



