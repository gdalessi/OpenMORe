import numpy as np

import PyTROModelling.model_order_reduction as model_order_reduction
from PyTROModelling.utilities import *


file_options = {
    "path_to_file"                  :   "../data",
    "input_file_name"               :   "flameD.csv",
}

settings ={
    #centering and scaling options
    "centering"                     :   "mean",
    "scaling"                       :   "auto",

    #set the number of PCs:
    "number_of_PCs"                 :   5,
}

#it is possible to plot one or more local PCs, given a number of cluster
#and the number of the PC to plot in the chosen cluster:
plotting_options = {
    #the number of cluster where you want to plot:
    "cluster_to_plot"               :   1,

    #the local principal component you want to plot:
    "PC_to_plot"                    :   1,
}

X = readCSV(file_options["path_to_file"], file_options["input_file_name"])

model = model_order_reduction.LPCA(X)
model.centering = settings["centering"]
model.scaling = settings["scaling"]
model.eigens = settings["number_of_PCs"]
model.path_to_idx = file_options["path_to_file"] #WARNING: the name 'idx.txt' is mandatory for the file

LPCs, u_scores, Leigen, centroids = model.fit()
X_rec_lpca = model.recover()

rec_err_local = NRMSE(X, X_rec_lpca)
print(np.mean(rec_err_local))

model.plot_parity()
model.clust_to_plot = plotting_options["cluster_to_plot"]
model.set_num_to_plot = plotting_options["PC_to_plot"]
model.plot_PCs()
