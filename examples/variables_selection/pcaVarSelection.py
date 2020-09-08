import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os

import OpenMORe.model_order_reduction as model_order_reduction
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
    "method"                    : "procustes",
    "number_of_PCs"             : 8,
    "number_of_variables"       : 25,
    "path_to_labels"            : os.path.abspath(os.path.join(__file__ ,"../../../data/reactive_flow/")),
    "labels_name"               : "labels.csv",
    "include_temperature"       : False
}

X = readCSV(file_options["path_to_file"], file_options["input_file_name"])
if not settings["include_temperature"]: 
    X = X[:,1:]

PVs = model_order_reduction.variables_selection(X, settings)
labels = PVs.fit()

print(labels)