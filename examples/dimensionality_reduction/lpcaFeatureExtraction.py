import numpy as np
import os

import OpenMORe.model_order_reduction as model_order_reduction
from OpenMORe.utilities import *


file_options = {
    "path_to_file"                  :   os.path.abspath(os.path.join(__file__ ,"../../../data/reactive_flow/")),
    "input_file_name"               :   "flameD.csv",
}

settings ={
    #centering and scaling options
    "center"                        :   True,
    "centering_method"              :   "mean",
    "scale"                         :   True,
    "scaling_method"                :   "auto",

    #set the number of PCs:
    "number_of_eigenvectors"        :   5,

    #set the path to the partitioning file:
    #WARNING: the file name "idx.txt" is mandatory
    "path_to_idx"                   : file_options["path_to_file"],

    #the number of cluster where you want to plot:
    "cluster_to_plot"               :   1,

    #the local principal component you want to plot:
    "PC_to_plot"                    :   0,
}


X = readCSV(file_options["path_to_file"], file_options["input_file_name"])

model = model_order_reduction.LPCA(X, settings)


LPCs, u_scores, Leigen, centroids = model.fit()
X_rec_lpca = model.recover()

model.plot_parity()
model.plot_PCs()

rec_err_local = NRMSE(X, X_rec_lpca)
print(np.mean(rec_err_local))