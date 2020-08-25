import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import PyTROModelling.model_order_reduction as model_order_reduction
from PyTROModelling.utilities import *
import PyTROModelling.model_order_reduction as model_order_reduction

import datetime
import sys
import os



now = datetime.datetime.now()
try:
    newDirName = "Training - " + now.strftime("%Y_%m_%d-%H%M")
    os.mkdir(newDirName)
    os.chdir(newDirName)
except FileExistsError:
    newDirName = "Training 2nd try- " + now.strftime("%Y_%m_%d-%H%M")
    os.mkdir(newDirName)
    os.chdir(newDirName)


file_options = {
    #"path_to_file"              : "/Users/giuseppedalessio/Dropbox/GitLab/PyTROModelling/data",
    "path_to_file"              : "/home/peppe/Dropbox/GitLab/PyTROModelling/data",
    "input_file_name"           : "pasr_full.csv",
}

settings = {
    #centering and scaling options
    "center"                    : True,
    "centering_method"          : "mean",
    "scale"                     : True,
    "scaling_method"            : "auto",

    #variables selection options
    "number_of_PCs"             : 7,
    "number_of_variables"       : 72,
    "path_to_labels"            : "/home/peppe/Dropbox/GitLab/PyTROModelling/data",
    "labels_name"               : "full_labels.csv",

    #set the method which has to be used for the sampling.
    #available options: "random", "cluster", "stratified", "multistage"
    "method"                    : "multistage",

    #set the final size of the sampled dataset
    "final_size"                : 5000,

    #enable the option to plot the accessed space (mkdir and save the images in the folder)
    "plot_accessed"             : True,

    "sample_training"           : False,
}

X = readCSV(file_options["path_to_file"], file_options["input_file_name"])
Temperature = X[:,0]




if settings["sample_training"]:
    reduceSize = model_order_reduction.SamplePopulation(X)

    reduceSize.sampling_strategy = settings["method"]
    reduceSize.set_size = settings["final_size"]
    reduceSize.set_conditioning = Temperature

    miniX = reduceSize.fit()

    if settings["plot_accessed"]:
        #first of all, create a folder to store the sampled matrix and the images of the accessed
        #space to assess the sampling quality.

        import datetime
        import sys
        import os
        import pandas as pd

        try:
            names_var= np.array(pd.read_csv(file_options["path_to_file"] + '/' + settings["labels_name"], sep = '\n', header = None))
        except OSError:
            print("Could not open/read the selected file: " + settings["labels_name"])
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
            matplotlib.rcParams.update({'font.size' : 14, 'text.usetex' : True})
            axes.legend(('Full training data', 'Sampled training data'), markerscale =7)
            plt.savefig('sampledVar' + str(names_var[ii]) + '.png')
            
    np.savetxt("sampled_matrix_thermochemical.txt", miniX)
    X = miniX

#ONLY ON MY DATASET: DISCARD TEMPERATURE --> VAR SELECTION ON CHEMICAL SPECIES  
X = X[:,1:]


KPVs = model_order_reduction.KPCA(X)
KPVs.eigens = settings["number_of_PCs"]
KPVs.retained = settings["number_of_variables"]
KPVs.path_to_labels = "/home/peppe/Dropbox/GitLab/PyTROModelling/data"
KPVs.labels_file_name = "full_labels.csv"
KPVs.kernel_type = 'polynomial'


labels = KPVs.select_variables()
labels.tofile('foo.csv',sep=',')


print(labels)
