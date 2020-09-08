import OpenMORe.model_order_reduction as model_order_reduction
from OpenMORe.utilities import *

import matplotlib.pyplot as plt
import numpy as np
import os

file_options = {
    "path_to_file"              : os.path.abspath(os.path.join(__file__ ,"../../../data/reactive_flow/")),
    "input_file_name"           : "flameD.csv",
}

mesh_options = {
    #set the mesh file options (the path goes up twice - it's ok)
    "path_to_file"              : os.path.abspath(os.path.join(__file__ ,"../../../data/reactive_flow/")),
    "mesh_file_name"            : "mesh.csv",

    #eventually enable the clustering solution plot on the mesh
    "plot_on_mesh"              : True,
}

settings ={
    #centering and scaling options
    "center"                    : True,
    "centering_method"          : "mean",
    "scale"                     : True,
    "scaling_method"            : "auto",

    #set the final dimensionality
    "number_of_PCs"             : 'auto',
}


X = readCSV(file_options["path_to_file"], file_options["input_file_name"])


model = model_order_reduction.PCA(X)
model.to_center = settings["center"]
model.centering = settings["centering_method"]
model.to_scale = settings["scale"]
model.scaling = settings["scaling_method"]


if settings["number_of_PCs"] is "auto":

    #if the "auto" setting is chosen, the number of PCs
    #is set on the basis of the explained variance (95% min)
    #or on the basis of the NRMSE with the reconstructed
    #matrix (<10% error with the original matrix, on average)
    model.set_PCs_method = 'var'
    model.set_PCs()
else:
    #otherwise the user input is used
    model.eigens = settings["number_of_PCs"]


#perform the dimensionality reduction via Principal Component Analysis,
#and return the eigenvectors of the reduced manifold
PCs = model.fit()


#compute the projection of the original points on the reduced
#PCA manifold, obtaining the scores matrix Z
Z = model.get_scores()


#assess the percentage of explained variance if the number of PCs has not
#been set automatically, and plot the result
model.get_explained()


#reconstruct the matrix from the reduced PCA manifold
X_recovered = model.recover()


#compare the reconstructed chosen variable "set_num_to_plot" with the
#original one, by means of a parity plot
model.set_num_to_plot = 0
model.plot_parity()
model.plot_PCs()


plt.rcParams.update({'font.size' : 12, 'text.usetex' : True})
mesh = np.genfromtxt(mesh_options["path_to_file"] + "/" + mesh_options["mesh_file_name"], delimiter= ',')

fig = plt.figure()
axes = fig.add_axes([0.2,0.15,0.7,0.7], frameon=True)
axes.scatter(mesh[:,0], mesh[:,1], c=Z[:,0],alpha=0.9, cmap='gnuplot')
axes.set_xlabel('X [m]')
axes.set_ylabel('Y [m]')
plt.show()

fig = plt.figure()
axes = fig.add_axes([0.2,0.15,0.7,0.7], frameon=True)
axes.scatter(mesh[:,0], mesh[:,1], c=Z[:,1],alpha=0.9, cmap='gnuplot')
axes.set_xlabel('X [m]')
axes.set_ylabel('Y [m]')
plt.show()

fig = plt.figure()
axes = fig.add_axes([0.2,0.15,0.7,0.7], frameon=True)
axes.scatter(mesh[:,0], mesh[:,1], c=Z[:,2],alpha=0.9, cmap='gnuplot')
axes.set_xlabel('X [m]')
axes.set_ylabel('Y [m]')
plt.show()