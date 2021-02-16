import OpenMORe.clustering as clustering
from OpenMORe.utilities import *

import matplotlib 
import matplotlib.pyplot as plt
import os


'''
file_options = {
    "path_to_file"              : os.path.abspath(os.path.join(__file__ ,"../../../data/dummy_data/")),
    "input_file_name"           : "moons.csv",
}
'''

file_options = {
    #set the training matrix file options
    "path_to_file"              : os.path.abspath(os.path.join(__file__ ,"../../../data/reactive_flow/")),
    #"path_to_file"              : "/Users/giuseppedalessio/Dropbox/GitHub/data",
    "input_file_name"           : "flameD.csv",
}


mesh_options = {
    #set the mesh file options
    "path_to_file"              : os.path.abspath(os.path.join(__file__ ,"../../../data/reactive_flow/")),
    #"path_to_file"              : "/Users/giuseppedalessio/Dropbox/GitHub/data",
    "mesh_file_name"            : "mesh.csv",

    #eventually enable the clustering solution plot on the mesh
    "plot_on_mesh"              : True,
}

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

X = readCSV(file_options["path_to_file"], file_options["input_file_name"])

model = clustering.spectralClustering(X, settings)
#idx = model.fit()
idx = model.Kfit()


'''
matplotlib.rcParams.update({'font.size' : 12, 'text.usetex' : True})

fig = plt.figure()
axes = fig.add_axes([0.2,0.15,0.7,0.7], frameon=True)

cmap = matplotlib.colors.ListedColormap(['darkred', 'midnightblue'])
sc = axes.scatter(X[:,0], X[:,1], c=idx,alpha=0.9, cmap=cmap, edgecolor ='none')
bounds = [0, 1]
axes.set_title("Spectral clustering solution")
axes.set_xlabel('X [-]')
axes.set_ylabel('Y [-]')
#plt.colorbar(sc, extendfrac='auto',spacing='uniform')
cb = plt.colorbar(sc, spacing='uniform', ticks=bounds)
cb.set_ticks(ticks=range(2))
plt.show()
'''
import numpy as np

matplotlib.rcParams.update({'font.size' : 12, 'text.usetex' : True})
mesh = np.genfromtxt(mesh_options["path_to_file"] + "/" + mesh_options["mesh_file_name"], delimiter= ',')

fig = plt.figure()
axes = fig.add_axes([0.2,0.15,0.7,0.7], frameon=True)
axes.scatter(mesh[:,0], mesh[:,1], c=idx,alpha=0.9, cmap='gnuplot')
axes.set_xlabel('X [m]')
axes.set_ylabel('Y [m]')
plt.show()