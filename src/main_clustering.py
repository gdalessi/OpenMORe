'''
PROGRAM: main.py

@Authors:
    G. D'Alessio [1,2], G. Aversano [1], A. Parente[1]
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

import clustering


file_options = {
    "path_to_file"              : "/Users/giuseppedalessio/Dropbox/GitHub/data",
    "input_file_name"           : "laminar2D.csv",
}

mesh_options = {
    "path_to_file"              : "/Users/giuseppedalessio/Dropbox/GitHub/data",
    "mesh_file_name"           : "mesh.csv",
}


settings = {
    "centering_method"          : "MEAN",
    "scaling_method"            : "AUTO",
    "initialization_method"     : "KMEANS",
    "number_of_clusters"        : 16,
    "number_of_eigenvectors"    : 15,
    "classify"                  : False,
    "write_on_txt"              : True,
    "plot_on_mesh"              : True,
}


try:
    print("Reading training matrix..")
    X = np.genfromtxt(file_options["path_to_file"] + "/" + file_options["input_file_name"], delimiter= ',')
except OSError:
    print("Could not open/read the selected file: " + file_options["input_file_name"])
    exit()
check_dummy(X, settings["number_of_clusters"], settings["number_of_eigenvectors"])


X_tilde = center_scale(X, center(X, method=settings["centering_method"]), scale(X, method=settings["scaling_method"]))


model = clustering.lpca(X_tilde)

model.clusters = settings["number_of_clusters"]
model.eigens = settings["number_of_eigenvectors"]
model.initialization = settings["initialization_method"]

index = model.fit()

if settings["write_on_txt"]:
    np.savetxt("idx_training.txt", index)


if settings["plot_on_mesh"]:
    mesh = np.genfromtxt(mesh_options["path_to_file"] + "/" + mesh_options["mesh_file_name"], delimiter= ',')
    plt.scatter(mesh[:,0], mesh[:,1], c=index,alpha=0.5)
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.show()


if settings["classify"]:

    file_options_classifier = {
        "path_to_file"              : "/home/peppe/Dropbox/GitHub/data",
        "test_file_name"            : "thermoC_timestep.csv",
    }

    try:
        print("Reading test matrix..")
        Y = np.genfromtxt(file_options_classifier["path_to_file"] + "/" + file_options_classifier["test_file_name"], delimiter= ',')
    except OSError:
        print("Could not open/read the selected file: " + "/" + file_options["test_file_name"])
        exit()

    # Input to the classifier: X = training matrix, Y = test matrix
    classifier = clustering.VQclassifier(X, index, Y)

    classifier.centering = settings["centering_method"]
    classifier.scaling = settings["scaling_method"]
    
    classification_vector = classifier.fit()

    if settings["write_on_txt"]:
        np.savetxt("idx_test.txt", classification_vector)
