'''
MODULE: preprocessing.py

@Authors: 
    G. D'Alessio [1,2], G. Aversano [1], A. Parente[1]
    [1]: Université Libre de Bruxelles, Aero-Thermo-Mechanics Laboratory, Bruxelles, Belgium
    [2]: CRECK Modeling Lab, Department of Chemistry, Materials and Chemical Engineering, Politecnico di Milano

@Contacts:
    giuseppe.dalessio@ulb.ac.be

@Brief: 
    Centering and scaling functions for multivariate data.

@Details: 
    In multivariate data-sets the variables are characterized by different units and ranges,
    thus preprocessing in the form of centering and scaling is a mandatory operation.
    Centering consists of subtracting the mean/min value of each variable to all data-set
    observations. Scaling is achieved by dividing each variable by a given scaling factor. Therefore, the
    i-th observation of the j-th variable, x_{i,j} can be
    centered and scaled by means of:

    \tilde{x_{i,j}} = (x_{i,j} - mu_{j}) / (sig_{j}),

    where mu_{j} and sig_{j} are the centering and scaling factor for the considered j-th variable, respectively.

    AUTO: the standard deviation of each variable is used as a scaling factor.
    PARETO: the squared root of the standard deviation is used as a scaling f.
    RANGE: the difference between the minimum and the maximum value is adopted as a scaling f.
    VAST: the ratio between the variance and the mean of each variable is used as a scaling f.

@Cite:
    [a] D'Alessio, Giuseppe, et al. "Analysis of turbulent reacting jets via Principal Component Analysis", Data Analysis in Direct Numerical Simulation of Turbulent Combustion, Springer (2020).
    [b] Parente, Alessandro, and James C. Sutherland. "Principal component analysis of turbulent combustion data: Data pre-processing and manifold sensitivity." Combustion and flame 160.2 (2013): 340-350.


@Additional notes:
    This cose is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; 
    Please report any bug to: giuseppe.dalessio@ulb.ac.be

'''

__all__ = ["center", "scale", "center_scale"]

import numpy as np
import pandas as pd



# -----------------
# Functions
# -----------------
def center(X, method):
    '''
    Compute the mean/min value (mu) of each variable of all data-set observations.
    '''
    # Main
    if method == 'MEAN' or method == 'mean' or method == 'Mean':
        mu = np.mean(X, axis = 0)
    elif method == 'MIN' or method == 'min' or method == 'Min':
        mu = np.min(X, axis = 0)
    else:
        raise Exception("Caught an exception: unsupported centering option. Supported centering: MEAN or MIN.")
    return mu


def scale(X, method):
    '''
    Compute the scaling factor (sig) for each variable of the data-set
    '''
    # Main
    if method == 'AUTO' or method == 'auto' or method == 'Auto':
        sig = np.std(X, axis = 0)
    elif method == 'PARETO' or method == 'pareto' or method == 'Pareto':
        sig = np.sqrt(np.std(X, axis = 0))
    elif method == 'VAST' or method == 'vast' or method == 'Vast':
        variances = np.var(X, axis = 0)
        means = np.mean(X, axis = 0)
        sig = variances / means
    elif method == 'RANGE' or method == 'range' or method == 'Range':
        maxima = np.max(X, axis = 0)
        minima = np.min(X, axis = 0)
        sig = maxima - minima
    else:
        raise Exception("Caught an exception: unsupported scaling option. Supported centering: AUTO, PARETO, VAST or RANGE.")
    return sig


def center_scale(X, mu, sig):
    '''
    Center and scale a given multivariate data-set X.
    Centering consists of subtracting the mean/min value of each variable to all data-set
    observations. Scaling is achieved by dividing each variable by a given scaling factor. Therefore, the
    i-th observation of the j-th variable, x_{i,j} can be
    centered and scaled by means of:

    \tilde{x_{i,j}} = (x_{i,j} - mu_{j}) / (sig_{j}),

    where mu_{j} and sig_{j} are the centering and scaling factor for the considered j-th variable, respectively.

    AUTO: the standard deviation of each variable is used as a scaling factor.
    PARETO: the squared root of the standard deviation is used as a scaling f.
    RANGE: the difference between the minimum and the maximum value is adopted as a scaling f.
    VAST: the ratio between the variance and the mean of each variable is used as a scaling f.
    '''
    TOL = 1E-10
    X0 = X - mu
    X0 = X0 / (sig + TOL)
    return X0


# -------------------
# Standalone option:
# -------------------


if __name__ == "__main__":

    file_options = {
    "path_to_file"              : "/Users/giuseppedalessio/Dropbox/python_course/LPCA/",
    "file_name"                 : "cfdf.csv",
    "output_name"               : "X_tilde.csv"
    }

    settings = {
    "centering_method"          : "MEAN",
    "scaling_method"            : "AUTO",
    "write_centscal"            : True,
}

    try:
        print("Reading the training matrix..")
        X = pd.read_csv(file_options["path_to_file"] + file_options["file_name"], sep = ',', header = None) 
        print("The training matrix has been read successfully!")
    except OSError:
        print("Could not open/read the selected file: " + file_options["file_name"])
        exit()

    X_tilde = center_scale(X, center(X, method=settings["centering_method"]), scale(X, method=settings["scaling_method"]))

    if settings["write_centscal"]:
        np.savetxt(file_options["output_name"], X_tilde)

