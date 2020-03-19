'''
PROGRAM: main_MOR.py

@Authors: 
    G. D'Alessio [1,2]
    [1]: Université Libre de Bruxelles, Aero-Thermo-Mechanics Laboratory, Bruxelles, Belgium
    [2]: CRECK Modeling Lab, Department of Chemistry, Materials and Chemical Engineering, Politecnico di Milano

@Contacts:
    giuseppe.dalessio@ulb.ac.be

@Brief: 
    Main function for the model_order_reduction module

@Additional notes:
    This code is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; 
    Please report any bug to: giuseppe.dalessio@ulb.ac.be

'''

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from utilities import *
import model_order_reduction


file_options = {
    "path_to_file"              : "/Users/giuseppedalessio/Dropbox/GitHub/data",
    "input_file_name"           : "cfdf.csv",
}



try:
    print("Reading training matrix..")
    X = np.genfromtxt(file_options["path_to_file"] + "/" + file_options["input_file_name"], delimiter= ',')
except OSError:
    print("Could not open/read the selected file: " + file_options["input_file_name"])
    exit()


model = model_order_reduction.PCA(X)
model.eigens = 15

PCs = model.fit()                                      # OK


X_recovered = model.recover()                          # OK


model.set_PCs_method = False
model.set_PCs()                                        # OK     
model.get_explained()                                  # OK
model.set_num_to_plot = 5
model.plot_PCs()                                       # OK
model.plot_parity()                                    # OK

local_model = model_order_reduction.LPCA(X)

local_model.eigens = 10
local_model.centering = 'mean'
local_model.scaling = 'auto'
local_model.path_to_idx = '/Users/giuseppedalessio/Dropbox/GitHub/Clustering_and_Red_Ord_Modelling/src'
local_model.set_num_to_plot = 7
LPCs, u_scores, Leigen, centroids = local_model.fit()
X_rec_lpca = local_model.recover()

recon_local = NRMSE(X, X_rec_lpca)

print(np.mean(recon_local))

local_model.plot_parity()
local_model.clust_to_plot = 3
local_model.plot_PCs()

print("done")




