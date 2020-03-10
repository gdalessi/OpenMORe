import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from utilities import *
from reduced_order_modelling import *

import ANN 

file_options = {
    "path_to_file"              : "/Users/giuseppedalessio/Dropbox/GitHub/data",
    "input_file_name"           : "CF_pasr_Z.csv",
    "output_file_name"          : "idx_CFP.csv"
}

try:
    print("Reading training matrix..")
    X = np.genfromtxt(file_options["path_to_file"] + "/" + file_options["input_file_name"], delimiter= ',')
except OSError:
    print("Could not open/read the selected file: " + "/" + file_options["input_file_name"])
    exit()

try:
    print("Reading training matrix..")
    Y = np.genfromtxt(file_options["path_to_file"] + "/" + file_options["output_file_name"], delimiter= ',')
except OSError:
    print("Could not open/read the selected file: " + "/" + file_options["output_file_name"])
    exit()

#model = ANN.MLP_classifier(X,Y, 10)
#index = model.fit_network()

model = ANN.Autoencoder(X, 5)
model.fit()


print("end OK!!")