'''
PROGRAM: main_neural.py

@Authors:
    G. D'Alessio [1,2]
    [1]: Université Libre de Bruxelles, Aero-Thermo-Mechanics Laboratory, Bruxelles, Belgium
    [2]: CRECK Modeling Lab, Department of Chemistry, Materials and Chemical Engineering, Politecnico di Milano

@Contacts:
    giuseppe.dalessio@ulb.ac.be

@Brief:
    Clustering via Local Principal Component Analysis and classification of new observations by means of the same metrics.

@Additional notes:
    This cose is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
    Please report any bug to: giuseppe.dalessio@ulb.ac.be

'''

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

training_options = {
    "number_of_layers"          : 2,
    "number_of_neurons"         : 50,
    "batch_size"                : 32,
    "activation_function"       : "relu",
    "number_of_epochs"          : 200,
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
'''model = ANN.MLP_classifier(X,Y, True)

model.neurons = training_options["number_of_neurons"]
model.layers = training_options["number_of_layers"]
model.activation = training_options["activation_function"]
model.n_epochs = training_options["number_of_epochs"]
model.batch_size = training_options["batch_size"]
model.dropout = 0.2

index = model.fit_network()'''

### DIMENSIONALITY REDUCTION ###
'''
model = ANN.Autoencoder(X)

model.neurons = training_options["number_of_neurons"]
model.activation = training_options["activation_function"]
model.n_epochs = training_options["number_of_epochs"]
#model.batch_size = training_options["batch_size"]

model.fit()
'''

### REGRESSION ###

model = ANN.MLP_regressor(X,Y)

model.neurons = training_options["number_of_neurons"]
model.layers = training_options["number_of_layers"]
model.activation_function = training_options["activation_function"]
model.n_epochs = training_options["number_of_epochs"]
model.batch_size = training_options["batch_size"]
model.dropout = 0.2

yo = model.fit_network()
