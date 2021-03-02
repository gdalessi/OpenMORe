import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os

import OpenMORe.clustering as clustering
from OpenMORe.utilities import *
import OpenMORe.classification as classification


############################################################################
# In this example it's shown how to classify a new matrix Y (laminar2D.csv)
# via VQPCA,  on the basis of the clustering on a matrix X obtained
# via 1D CFDF flames (cfdf.csv)
############################################################################

# dictionary to load the .csv files:
file_options = {
    "path_to_file"              : os.path.abspath(os.path.join(__file__ ,"../../../data/reactive_flow/")),
    "input_file_name"           : "cfdf.csv",
    "test_file_name"            : "laminar2D.csv",
}

# dictionary to load the mesh, found in .csv format:
mesh_options = {
    "path_to_file"              : os.path.abspath(os.path.join(__file__ ,"../../../data/reactive_flow/")),
    "mesh_file_name"            : "mesh_laminar.csv",
}

# dictionary for the clustering step:
settings_clustering = {
    #centering and scaling options
    "center"                    : True,
    "centering_method"          : "mean",
    "scale"                     : True,
    "scaling_method"            : "auto",

    #set the initialization method (random, observations, kmeans, pkcia, uniform)
    "initialization_method"     : "uniform",

    #set the number of clusters and PCs in each cluster
    "number_of_clusters"        : 8,
    "number_of_eigenvectors"    : 5,

    #enable additional options:
    "correction_factor"         : "off",    # --> enable eventual corrective coefficients for the LPCA algorithm:
                                            #     'off', 'c_range', 'uncorrelation', 'local_variance', 'phc_multi', 'local_skewness' are available

    "classify"                  : False,    # --> call the method to classify a new matrix Y on the basis of the lpca clustering
    "write_on_txt"              : True,     # --> write the idx vector containing the label for each observation
    "evaluate_clustering"       : True,     # --> enable the calculation of indeces to evaluate the goodness of the clustering
}

# Load the matrices and the mesh, to plot the classification results
X = readCSV(file_options["path_to_file"], file_options["input_file_name"])
Y = readCSV(file_options["path_to_file"], file_options["test_file_name"])
mesh = np.genfromtxt(mesh_options["path_to_file"] + "/" + mesh_options["mesh_file_name"], delimiter= ',')

# Start the clustering step
model = clustering.lpca(X, settings_clustering)
index = model.fit()

# Start the classification step
classifier = classification.VQPCA(X, index, Y)
classification_vector = classifier.fit()


# Plot the results
matplotlib.rcParams.update({'font.size' : 14, 'text.usetex' : True})
fig = plt.figure()
axes = fig.add_axes([0.15,0.15,0.7,0.7], frameon=True)
axes.scatter(mesh[:,0], mesh[:,1], c=classification_vector, alpha=1, cmap='gnuplot')
axes.set_xlabel('$x [m]$')
axes.set_ylabel('$y [m]$')
plt.show()