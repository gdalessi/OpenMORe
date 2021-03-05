import OpenMORe.clustering as clustering
from OpenMORe.utilities import *

import matplotlib 
import matplotlib.pyplot as plt
import os
import numpy as np

############################################################################
# In this example it's shown how to cluster a matrix X (moons.csv) via
# Spectral Clustering. 
############################################################################

# dictionary to load the .csv file:
file_options = {
    #set the training matrix file options
    "path_to_file"              : os.path.abspath(os.path.join(__file__ ,"../../../data/dummy_data/")),
    "input_file_name"           : "moons.csv",
}


# dictionary with the options of spectral clustering:
settings = {
    #centering and scaling options
    "center"                    : False,
    "centering_method"          : "mean",
    "scale"                     : False,
    "scaling_method"            : "auto",

    #clustering options: choose the number of clusters
    "number_of_clusters"        : 2,
    "sigma"                     : 0.1,

    #write clustering solution on txt
    "write_on_txt"              : False,
    "evaluate_clustering"       : False,
}

# Load the input matrix to be clustered and the corresponding mesh
X = readCSV(file_options["path_to_file"], file_options["input_file_name"])


# Start the clustering step: call the class and the Kfit() method
# It is returned the idx, a vector containing the clustering solution
model = clustering.spectralClustering(X, settings)
idx = model.fit()


# Plot the clustering results 
matplotlib.rcParams.update({'font.size' : 12, 'text.usetex' : True})
fig = plt.figure()
axes = fig.add_axes([0.2,0.15,0.7,0.7], frameon=True)
axes.scatter(X[:,0], X[:,1], c=idx,alpha=0.9, cmap='gnuplot')
axes.set_xlabel('$x\ [-]$')
axes.set_ylabel('$y\ [-]$')
plt.show()