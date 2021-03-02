import OpenMORe.model_order_reduction as model_order_reduction
from OpenMORe.utilities import *

import matplotlib.pyplot as plt
import numpy as np
import os

############################################################################
# In this example, it's shown how to perform dimensionality reduction and 
# feature extraction on a matrix X (turbo2D.csv) via Principal Component 
# Analysis (PCA).
############################################################################

# Dictionary to load the input matrix, found in .csv format
file_options = {
    "path_to_file"              : os.path.abspath(os.path.join(__file__ ,"../../../data/reactive_flow/")),
    "input_file_name"           : "turbo2D.csv",
}

# Dictionary to load the mesh, also found in .csv format
mesh_options = {
    #set the mesh file options (the path goes up twice - it's ok)
    "path_to_file"              : os.path.abspath(os.path.join(__file__ ,"../../../data/reactive_flow/")),
    "mesh_file_name"            : "mesh_turbo.csv",

    #eventually enable the clustering solution plot on the mesh
    "plot_on_mesh"              : True,
}

# Dictionary with the instruction for the PCA algorithm:
settings ={
    #centering and scaling options
    "center"                    : True,
    "centering_method"          : "mean",
    "scale"                     : True,
    "scaling_method"            : "auto",

    #set the final dimensionality
    "number_of_eigenvectors"    : 7,
    
    #enable to plot the cumulative explained variance
    "enable_plot_variance"      : True,
    
    #set the number of the variable whose reconstruction must be plotted
    "variable_to_plot"          : 0,

}

# Load the input matrix 
X = readCSV(file_options["path_to_file"], file_options["input_file_name"])

# Start the dimensionality reduction and the feature extraction step:
# call the PCA class and give in input X and the dictionary with the instructions
model = model_order_reduction.PCA(X, settings)


# Perform the dimensionality reduction via Principal Component Analysis,
# and return the eigenvectors of the reduced manifold 
PCs = model.fit()


# Compute the projection of the original points on the reduced
# PCA manifold, obtaining the scores matrix Z
Z = model.get_scores()


# Assess the percentage of explained variance if the number of PCs has not
# been set automatically, and plot the result
model.get_explained()


# Reconstruct the matrix from the reduced PCA manifold
X_recovered = model.recover()


# Compare the reconstructed chosen variable "set_num_to_plot" with the
# original one, by means of a parity plot
model.plot_parity()
model.plot_PCs()


# Plot the first 3 scores on the mesh, to see the structures in the lower
# dimensional space
plt.rcParams.update({'font.size' : 12, 'text.usetex' : True})
mesh = np.genfromtxt(mesh_options["path_to_file"] + "/" + mesh_options["mesh_file_name"], delimiter= ',')

fig = plt.figure()
axes = fig.add_axes([0.2,0.15,0.7,0.7], frameon=True)
axes.scatter(mesh[:,0], mesh[:,1], c=Z[:,0],alpha=0.9, cmap='gnuplot')
axes.set_xlabel('$x [m]$')
axes.set_ylabel('$y [m]$')
plt.show()

fig = plt.figure()
axes = fig.add_axes([0.2,0.15,0.7,0.7], frameon=True)
axes.scatter(mesh[:,0], mesh[:,1], c=Z[:,1],alpha=0.9, cmap='gnuplot')
axes.set_xlabel('$x [m]$')
axes.set_ylabel('$y [m]$')
plt.show()

fig = plt.figure()
axes = fig.add_axes([0.2,0.15,0.7,0.7], frameon=True)
axes.scatter(mesh[:,0], mesh[:,1], c=Z[:,2],alpha=0.9, cmap='gnuplot')
axes.set_xlabel('$x [m]$')
axes.set_ylabel('$y [m]$')
plt.show()