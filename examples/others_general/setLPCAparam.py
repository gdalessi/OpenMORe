import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os

import OpenMORe.clustering as clustering
from OpenMORe.utilities import *

from mpl_toolkits.mplot3d import Axes3D

#Function to plot the surface
def surface_plot (matrix, numEig, numK, **kwargs):
    # acquire the cartesian coordinate matrices from the matrix
    # x is cols, y is rows
    #(x, y) = np.meshgrid(np.arange(matrix.shape[0]), np.arange(matrix.shape[1]))
    (x, y) = np.meshgrid(numEig, numK)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(x, y, matrix, **kwargs)
    return (fig, ax, surf)


# Dictionary to load the input matrix, found in .csv format
file_options = {
    "path_to_file"              : os.path.abspath(os.path.join(__file__ ,"../../../data/reactive_flow/")),
    "input_file_name"           : "turbo2D.csv",
}

# Dictionary to load the mesh
mesh_options = {
    #set the mesh file options
    "path_to_file"              : os.path.abspath(os.path.join(__file__ ,"../../../data/reactive_flow/")),
    "mesh_file_name"            : "mesh_turbo.csv",

    #eventually enable the clustering solution plot on the mesh
    "plot_on_mesh"              : True,
}

# Dictionary with the instructions for the clustering
settings = {
    #centering and scaling options
    "center"                    : True,
    "centering_method"          : "mean",
    "scale"                     : True,
    "scaling_method"            : "auto",

    #set the initialization method (random, observations, kmeans, pkcia, uniform)
    "initialization_method"     : "uniform",

    #Initial and final number of clusters for the parameterization
    "initial_k"                 : 4,
    "final_k"                   : 16,

    #Initial and final number of PCs for the parameterizatin
    "initial_eig"               : 2,
    "final_eig"                 : 10,

    #Additional options
    "write_stats"               : False,
}



# Load the input matrix:
X = readCSV(file_options["path_to_file"], file_options["input_file_name"])
# Preprocess the input matrix: center and scale:
X_tilde = center_scale(X, center(X, method=settings["centering_method"]), scale(X, method=settings["scaling_method"]))

# Initialize the parameterization vectors
num_of_k = np.linspace(settings["initial_k"], settings["final_k"], settings["final_k"]-settings["initial_k"]+1)
num_of_eig = np.linspace(settings["initial_eig"], settings["final_eig"], settings["final_eig"]-settings["initial_eig"]+1)
DB_scores = np.empty((len(num_of_k), len(num_of_eig)), dtype=float)

# Initialize best clustering coefficient
DB_low = 1000
# Initialize worst clustering coefficient
DB_high = 0.5
# Here we put the idx corresponding to the best clustering
idxs = [None]
# Here we put the idx corresponding to the worst clustering
idx_worse = [None]

# Perform the clustering algorithm k*PCs times to see what's the best clustering solution
for ii in range(0,len(num_of_k)):
    for jj in range(0, len(num_of_eig)):
        model = clustering.lpca(X, settings)
        model.clusters = int(num_of_k[ii])
        model.eigens = int(num_of_eig[jj])
        index = model.fit() 
        
        DB_scores[ii,jj] = evaluate_clustering_DB(X_tilde, index)
        # If the DB is lower than the previous one, store it as the new
        # best clustering solution
        if DB_scores[ii, jj] < DB_low:
            DB_low = DB_scores[ii, jj]
            print("Num of eig: {}".format(num_of_eig[jj]))
            print("Num of k: {}".format(num_of_eig[ii]))
            print("Storing idx..")
            idxs = index
        # If the DB is lower than the previous one, store it as the new
        # worst clustering solution
        if DB_scores[ii, jj] > DB_high:
            DB_high = DB_scores[ii, jj]
            idx_worse = index


# Plot the surface for DB
plt.rcParams.update({'font.size' : 12, 'text.usetex' : True})
(fig, ax, surf) = surface_plot(DB_scores, num_of_eig, num_of_k, cmap="gnuplot")
fig.colorbar(surf)
ax.set_xlabel('$Principal\ Components\ [-]$')
ax.set_ylabel('$Number\ of\ clusters\ [-]$')
ax.set_zlabel('$DB\ index\ [-]$')
kTicks = [4, 6, 8, 10, 12, 14, 16]
PCsTicks = [2, 4, 6, 8, 10]
ax.set_xticks(PCsTicks)
ax.set_yticks(kTicks)
plt.savefig("DBparameter.png", format='png', dpi=300)
plt.show()


# Eventually plot the clustering solution on the mesh
if mesh_options["plot_on_mesh"]:
    matplotlib.rcParams.update({'font.size' : 14, 'text.usetex' : True})
    mesh = np.genfromtxt(mesh_options["path_to_file"] + "/" + mesh_options["mesh_file_name"], delimiter= ',')

    fig = plt.figure()
    axes = fig.add_axes([0.2,0.15,0.7,0.7], frameon=True)
    axes.scatter(mesh[:,0], mesh[:,1], c=idxs,alpha=0.9, cmap='gnuplot')
    axes.set_xlabel('$X\ [m]$')
    axes.set_ylabel('$Y\ [m]$')
    plt.savefig("bestClust.png", format='png', dpi=300)
    plt.show()


    fig = plt.figure()
    axes = fig.add_axes([0.2,0.15,0.7,0.7], frameon=True)
    axes.scatter(mesh[:,0], mesh[:,1], c=idx_worse,alpha=0.9, cmap='gnuplot')
    axes.set_xlabel('$X\ [m]$')
    axes.set_ylabel('$Y\ [m]$')
    plt.savefig("worseClust.png", format='png', dpi=300)
    plt.show()
