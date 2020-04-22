import sys
sys.path.insert(1, '../src')

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import model_order_reduction
from utilities import *

file_options = {
    "path_to_file"              : "../data",
    "input_file_name"           : "flameD.csv",

    "labels_name"               : "labels.csv",    
}

settings = {
    #centering and scaling options
    "center"                    : True,
    "centering_method"          : "mean",
    "scale"                     : True,        
    "scaling_method"            : "auto",

    #variables selection options
    "method"                    : "procustes_rotation",
    "number_of_PCs"             : 8,
    "number_of_variables"       : 15,       
}

X = readCSV(file_options["path_to_file"], file_options["input_file_name"])


PVs = model_order_reduction.variables_selection(X)

PVs.method = settings["method"]
PVs.path_to_labels = file_options["path_to_file"]
PVs.labels_file_name = file_options["labels_name"]
PVs.eigens = settings["number_of_PCs"]
PVs.retained = settings["number_of_variables"]

labels = PVs.fit()

print(labels)