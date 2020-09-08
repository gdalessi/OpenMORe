import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os

import OpenMORe.model_order_reduction as model_order_reduction
import OpenMORe.clustering as clustering
from OpenMORe.utilities import *

file_options = {
    "path_to_file"              : os.path.abspath(os.path.join(__file__ ,"../../../data/reactive_flow/")), 
    "input_file_name"           : "flameD.csv",
}

settings = {
    #centering and scaling options
    
    "center"                    : True,
    "centering_method"          : "mean",
    "scale"                     : True,
    "scaling_method"            : "auto",

    #variables selection options
    "select_variables"          : True,
    "method"                    : "procustes", #b2, b4, procustes, procustes rotation
    "number_of_PCs"             : 12,
    "number_of_variables"       : 30,
    "path_to_labels"            : os.path.abspath(os.path.join(__file__ ,"../../../data/reactive_flow/")),
    "labels_name"               : "none.csv",

    #clustering options
    #set the number of clusters
    "number_of_clusters"        : 8,
    #set the initialization method
    "initialization_method"     : "uniform",
    #enable additional options:
    "write_stats"               : False,
    "evaluate_clustering"       : True,


}

X = readCSV(file_options["path_to_file"], file_options["input_file_name"])
print("matrix dimensions: {}".format(X.shape))

if settings["select_variables"]:
    PVs = model_order_reduction.variables_selection(X, settings)
    labels, numbers = PVs.fit()


    redX = X[:,numbers]

print("The new data dimensions are: {}".format(X.shape))

model = clustering.lpca(redX)
model.clusters = settings["number_of_clusters"]
model.eigens = settings["number_of_PCs"]
model.initialization = settings["initialization_method"]

index = model.fit()


if settings["evaluate_clustering"]:

    #evaluate the clustering solution
    PHC_coeff, PHC_deviations = evaluate_clustering_PHC(X, index, method='PHC_standard')
    print(PHC_coeff)

    #evaluate the clustering solution by means of the Davies-Bouldin index
    X_tilde = center_scale(X, center(X, method=settings["centering_method"]), scale(X, method=settings["scaling_method"]))
    DB = evaluate_clustering_DB(X_tilde, index)


    text_file = open("stats_clustering_solution.txt", "wt")
    DB_index = text_file.write("DB index equal to: {} \n".format(DB))
    PHC_coeff = text_file.write("Average PHC is: {} \n".format(np.mean(PHC_coeff)))
    text_file.close()


