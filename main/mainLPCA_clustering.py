import sys
sys.path.insert(1, '../src')

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import clustering
from utilities import *

file_options = {
    "path_to_file"              : "../data",
    "input_file_name"           : "flameD.csv",
}


mesh_options = {
    #set the mesh file options (the path goes up twice - it's ok)
    "path_to_file"              : "../../data",
    "mesh_file_name"            : "mesh.csv",

    #eventually enable the clustering solution plot on the mesh
    "plot_on_mesh"              : True,
}


settings = {
    #centering and scaling options
    "centering_method"          : "mean",
    "scaling_method"            : "auto",

    #set the initialization method (random, observations, kmeans)
    "initialization_method"     : "observations",

    #set the number of clusters and PCs in each cluster
    "number_of_clusters"        : 8,
    "number_of_eigenvectors"    : 5,

    #enable additional options:
    "adaptive_PCs"              : False,    #use a different number of PCs in each cluster (to test)
    "classify"                  : False,    #classify a new matrix Y on the basis of the lpca clustering
    "write_on_txt"              : True,     #write the idx vector with the class for each observation
    
    "evaluate_clustering"       : True,     #enable the calculation of indeces to evaluate the goodness of the clustering
}

algorithm = {
    #enable eventual corrective coefficients for the LPCA algorithm:
    #'off', 'mean', 'min', 'max', 'std', 'phc_standard', 'phc_median', 'phc_robust', 'medianoids', 'medoids' are available
    "correction_factor"         : "off"
}

X = readCSV(file_options["path_to_file"], file_options["input_file_name"])

model = clustering.lpca(X)

model.centering = settings["centering_method"]
model.scaling = settings["scaling_method"]
model.initialization = settings["initialization_method"]
model.clusters = settings["number_of_clusters"]
model.eigens = settings["number_of_eigenvectors"]
model.adaptivePCs = settings["adaptive_PCs"]
model.correction = algorithm["correction_factor"] 

index = model.fit()

if settings["write_on_txt"]:
    np.savetxt("idx.txt", index)


#eventually evaluate the goodness of the clustering solution
if settings["evaluate_clustering"]:

    #evaluate the clustering solution
    PHC_coeff, PHC_deviations = PHC_index(X, index)
    
    #evaluate the clustering solution by means of the Davies-Bouldin index
    X_tilde = center_scale(X, center(X, method=settings["centering_method"]), scale(X, method=settings["scaling_method"]))
    DB = evaluate_clustering_DB(X_tilde, index) 


    text_file = open("stats_clustering_solution.txt", "wt")
    DB_index = text_file.write("DB index equal to: {} \n".format(DB))
    PHC_coeff = text_file.write("Average PHC is: {} \n".format(np.mean(PHC_coeff)))
    text_file.close()


#eventually plot the clustering solution on the mesh
if mesh_options["plot_on_mesh"]:
    matplotlib.rcParams.update({'font.size' : 12, 'text.usetex' : True})
    mesh = np.genfromtxt(mesh_options["path_to_file"] + "/" + mesh_options["mesh_file_name"], delimiter= ',')

    fig = plt.figure()
    axes = fig.add_axes([0.2,0.15,0.7,0.7], frameon=True)
    axes.scatter(mesh[:,0], mesh[:,1], c=index,alpha=0.5, cmap='gnuplot')
    axes.set_xlabel('X [m]')
    axes.set_ylabel('Y [m]')
    plt.show()