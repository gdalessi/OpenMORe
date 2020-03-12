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
    print("Reading training input..")
    X = np.genfromtxt(file_options["path_to_file"] + "/" + file_options["input_file_name"], delimiter= ',')
except OSError:
    print("Could not open/read the selected file: " + "/" + file_options["input_file_name"])
    exit()

try:
    print("Reading training output..")
    Y = np.genfromtxt(file_options["path_to_file"] + "/" + file_options["output_file_name"], delimiter= ',')
except OSError:
    print("Could not open/read the selected file: " + "/" + file_options["output_file_name"])
    exit()

### CLASSIFICATION ###
model = ANN.MLP_classifier(X,Y)
model.neurons = 50
model.activation = 'relu'
print(model.activation)
model.n_epochs = 100
#model.batch_size = 1
index = model.fit_network()

### DIMENSIONALITY REDUCTION ###
'''model = ANN.Autoencoder(X)
model.neurons = 50
model.activation = 'relu'
print(model.activation)
model.n_epochs = 100
#model.batch_size = 1
model.fit()'''

### REGRESSION ###
'''model = ANN.MLP_regressor(X,Y)
model.neurons = 50
model.activation_function = 'relu'
model.n_epochs = 100
model.batch_size = 100
yo = model.fit_network()'''


print("end OK!!")