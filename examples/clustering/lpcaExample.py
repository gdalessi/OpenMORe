import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os

import OpenMORe.clustering as clustering
from OpenMORe.utilities import *


file_options = {
    #set the training matrix file options
    "path_to_file"              : os.path.abspath(os.path.join(__file__ ,"../../../data/reactive_flow/")),
    "input_file_name"           : "flameD.csv",
}


mesh_options = {
    #set the mesh file options
    "path_to_file"              : os.path.abspath(os.path.join(__file__ ,"../../../data/reactive_flow/")),
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


X = readCSV(file_options["path_to_file"], file_options["input_file_name"])

model = clustering.lpca(X, settings)
index = model.fit()

if settings["write_on_txt"]:
    np.savetxt("idx.txt", index)


#eventually evaluate the goodness of the clustering solution
if settings["evaluate_clustering"]:

    #evaluate the clustering solution
    PHC_coeff, PHC_deviations = evaluate_clustering_PHC(X, index)
    print(PHC_coeff)

    #evaluate the clustering solution by means of the Davies-Bouldin index. 
    #the input matrix must be preprocessed first
    X_tilde = center_scale(X, center(X, method=settings["centering_method"]), scale(X, method=settings["scaling_method"]))
    DB = evaluate_clustering_DB(X_tilde, index)

    #save the stats in a txt file for a recap
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
