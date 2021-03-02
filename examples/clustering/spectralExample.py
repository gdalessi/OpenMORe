import OpenMORe.clustering as clustering
from OpenMORe.utilities import *

import matplotlib 
import matplotlib.pyplot as plt
import os
import numpy as np

############################################################################
# In this example, it's shown how to cluster a matrix X (turbo2D.csv) via
# Spectral Clustering. 
############################################################################

# dictionary to load the .csv file:
file_options = {
    #set the training matrix file options
    "path_to_file"              : os.path.abspath(os.path.join(__file__ ,"../../../data/reactive_flow/")),
    "input_file_name"           : "turbo2D.csv",
}

# dictionary to load the mesh, found in .csv format:
mesh_options = {
    #set the mesh file options
    "path_to_file"              : os.path.abspath(os.path.join(__file__ ,"../../../data/reactive_flow/")),
    "mesh_file_name"            : "mesh.csv",
}

# dictionary with the options of spectral clustering:
settings = {
    #centering and scaling options
    "center"                    : True,
    "centering_method"          : "mean",
    "scale"                     : True,
    "scaling_method"            : "auto",

    #clustering options: choose the number of clusters
    "number_of_clusters"        : 16,
    "sigma"                     : 1,

    #write clustering solution on txt
    "write_on_txt"              : False,
    "evaluate_clustering"       : False,
}

# Load the input matrix to be clustered and the corresponding mesh
X = readCSV(file_options["path_to_file"], file_options["input_file_name"])
mesh = np.genfromtxt(mesh_options["path_to_file"] + "/" + mesh_options["mesh_file_name"], delimiter= ',')

# Start the clustering step: call the class and the Kfit() method
# It is returned the idx, a vector containing the clustering solution
model = clustering.spectralClustering(X, settings)
idx = model.Kfit()


# Plot the clustering results on the mesh
matplotlib.rcParams.update({'font.size' : 12, 'text.usetex' : True})
fig = plt.figure()
axes = fig.add_axes([0.2,0.15,0.7,0.7], frameon=True)
axes.scatter(mesh[:,0], mesh[:,1], c=idx,alpha=0.9, cmap='gnuplot')
axes.set_xlabel('$x\ [m]$')
axes.set_ylabel('$y\ [m]$')
plt.show()