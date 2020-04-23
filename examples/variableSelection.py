import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import pyMORe.model_order_reduction as model_order_reduction
from pyMORe.utilities import *

file_options = {
    "path_to_file"              : "../data",
    "input_file_name"           : "flameD.csv",
}

settings = {
    #centering and scaling options
    "center"                    : True,
    "centering_method"          : "mean",
    "scale"                     : True,
    "scaling_method"            : "auto",

    #variables selection options
    "method"                    : "b4",
    "number_of_PCs"             : 8,
    "number_of_variables"       : 15,
    "path_to_labels"            : "../data",
    "labels_name"               : "labels.csv",
}

X = readCSV(file_options["path_to_file"], file_options["input_file_name"])


PVs = model_order_reduction.variables_selection(X, settings)
labels = PVs.fit()

print(labels)
