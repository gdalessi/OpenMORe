import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os

import OpenMORe.model_order_reduction as model_order_reduction
from OpenMORe.utilities import *

############################################################################
# In this example it's shown how to sample an input matrix X via OpenMORe. 
############################################################################

# Dictionary to load the input matrix, found in .csv format
file_options = {
    "path_to_file"              : os.path.abspath(os.path.join(__file__ ,"../../../data/reactive_flow/")),
    "input_file_name"           : "turbo2D.csv",

    "labels_name"               : "labels.csv",
}

# Dictionary with the instructions for the sampling class:
settings = {
    #set the method which has to be used for the sampling.
    #available options: "random", "cluster", "conditioned", "multistage"
    "method"                    : "conditioned",

    #set the final size of the sampled dataset
    "final_size"                : 1500,

    #enable the option to plot the accessed space (mkdir and save the images in the folder)
    "plot_accessed"             : False,
}

# Load the input matrix 
X = readCSV(file_options["path_to_file"], file_options["input_file_name"])

# Feed the function with the conditioning variables (e.g., mixture fraction)
# you want to use.
condVar = X[:,0]

reduceSize = model_order_reduction.SamplePopulation(X, settings)
reduceSize.set_conditioning = condVar
miniX = reduceSize.fit()

print("Training matrix sampled. New size: {}x{}".format(miniX.shape[0],miniX.shape[1]))
print("Original matrix size: {}x{}".format(X.shape[0],X.shape[1]))


if settings["plot_accessed"]:
    #first of all, create a folder to store the sampled matrix and the images of the accessed
    #space to assess the sampling quality.

    import datetime
    import sys
    import os
    import pandas as pd

    try:
        names_var= np.array(pd.read_csv(file_options["path_to_file"] + '/' + file_options["labels_name"], sep = '\n', header = None))
    except OSError:
        print("Could not open/read the selected file: " + file_options["labels_name"])
        exit()


    print("YO: {}".format(names_var))
    now = datetime.datetime.now()
    try:
        newDirName = "Accessed space fig - " + now.strftime("%Y_%m_%d-%H%M")
        os.mkdir(newDirName)
        os.chdir(newDirName)
    except FileExistsError:
        newDirName = "Accessed space fig 2nd try- " + now.strftime("%Y_%m_%d-%H%M")
        os.mkdir(newDirName)
        os.chdir(newDirName)


    now = datetime.datetime.now()
    try:
        newDirName = "Accessed space fig - " + now.strftime("%Y_%m_%d-%H%M")
        os.mkdir(newDirName)
        os.chdir(newDirName)
    except FileExistsError:
        newDirName = "Accessed space fig 2nd try- " + now.strftime("%Y_%m_%d-%H%M")
        os.mkdir(newDirName)
        os.chdir(newDirName)

    #now plot the full accessed space (black squares) and the one accessed by the sampled
    #matrix (blue circles). All the images will be saved in the corresponding folder in .png format.
    for ii in range(1,len(names_var)):
        matplotlib.rcParams.update({'font.size' : 16, 'text.usetex' : True})
        fig = plt.figure()
        axes = fig.add_axes([0.2,0.15,0.7,0.7], frameon=True)
        axes.scatter(Temperature, X[:,ii], 2, color= 'k', marker='s')
        axes.scatter(miniX[:,0], miniX[:,ii], 2, color= 'b')

        axes.set_xlabel('$T$')
        axes.set_ylabel(str(names_var[ii]))
        axes.set_xlim(min(X[:,0]), max(X[:,0]))
        axes.set_ylim(min(X[:,ii]), max(X[:,ii]))
        matplotlib.rcParams.update({'font.size' : 16, 'text.usetex' : True})
        axes.legend(('Full training data', 'Sampled training data'), markerscale =7)
        plt.savefig('sampledVar' + str(names_var[ii]) + '.png')
        plt.show()
        plt.close(fig)