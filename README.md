# Code structure:

In the /src/ folder, the Python modules which are necessary for Reduced Order Modelling (ROM) and Clustering are contained.

In the /main_files/ folder, the main files to accomplish a specific task are contained. Just copy the main file for the task in the same folder where the other modules are contained, and run it from the terminal. 


# Required external libraries:
Numpy and Matplotlib are required for almost all the ROM functions. For some function Scipy or Sklearn are also necessary.


# Module: clustering.py
Three main classes are available for clustering/classification tasks:

- Local Principal Component Analysis (LPCA) clustering.
- Vector Quantization algorithm for the classification of new observations, by means of a previous LPCA solution
- Spectral clustering.


# Module: reduced_order_modelling.py
A wide range of PCA-based functions for reduced-order-modelling are available in this module:

- Principal Component Analysis
- Kernel Principal Component Analysis
- Local Principal Component Analysis

Moreover, a class for a PCA-based variable selection (coupled with Procustes Analysis) is also available.


# Cite:
If you use these scripts for research purposes, these are few reference you can cite, depending on your application.

- PCA
    [0] Ian Jolliffe. Principal component analysis. Springer, 2011.

- Local algorithm for dimensionality reduction:
    [a] Kambhatla, Nandakishore, and Todd K. Leen. "Dimension reduction by local principal component analysis.", Neural computation 9.7 (1997): 1493-1516.
    
- LPCA-based clustering applications:
    [b] D’Alessio, Giuseppe, et al. "Adaptive chemistry via pre-partitioning of composition space and mechanism reduction.", Combustion and Flame 211 (2020): 68-82.
    
- Data analysis applications:
    [c] Parente, Alessandro, et al. "Investigation of the MILD combustion regime via principal component analysis." Proceedings of the Combustion Institute 33.2 (2011): 3333-3341.
    [d] D'Alessio, Giuseppe, et al. "Analysis of turbulent reacting jets via Principal Component Analysis", Data Analysis in Direct Numerical Simulation of Turbulent Combustion, Springer (2020).
    [e] Bellemans, Aurélie, et al. "Feature extraction and reduced-order modelling of nitrogen plasma models using principal component analysis." Computers & chemical engineering 115 (2018): 504-514.
    
- Preprocessing effects on PCA:
    [f] Parente, Alessandro, and James C. Sutherland. "Principal component analysis of turbulent combustion data: Data pre-processing and manifold sensitivity." Combustion and flame 160.2 (2013): 340-350.
    
- PCA-based model order reduction:
    [g] Parente, Alessandro, et al. "Identification of low-dimensional manifolds in turbulent flames." Proceedings of the Combustion Institute. 2009 Jan 1;32(1):1579-86.
    [h] Aversano, Gianmarco, et al. "Application of reduced-order models based on PCA & Kriging for the development of digital twins of reacting flow applications." Computers & chemical engineering 121 (2019): 422-441.
    
- Variables selection via PCA and Procustes Analysis:
[i]  Wojtek J Krzanowski. Selection of variables to preserve multivariate data structure, using principal components. Journal of the Royal Statistical Society: Series C (Applied Statistics), 36(1):22{33, 1987.
