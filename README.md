# clustering
Local Principal Component Analysis clustering algorithm.

@Details:
The iterative Local Principal Component Analysis clustering algorithm is based on the following steps:

    0. Preprocessing: The training matrix X is centered and scaled, after being loaded. Four scaling are available, AUTO, VAST, PARETO, RANGE - Two centering are available, MEAN and MIN;
    
    1. Initialization: The cluster centroids are initializated: a random allocation (RANDOM)
    or a previous clustering solution (KMEANS) can be chosen to compute the centroids initial values; 
    
    2. Partition: Each observation is assigned to a cluster k such that the local reconstruction
    error is minimized;
    
    3. PCA: The Principal Component Analysis is performed in each of the clusters found
    in the previous step. A new set of centroids is computed after the new partitioning
    step, their coordinates are calculated as the mean of all the observations in each
    cluster;
    
    4. Iteration: All the previous steps are iterated until convergence is reached. The convergence
    criterion is that the variation of the global mean reconstruction error between two consecutive
    iterations must be below a fixed threshold.
    
@How to use the script:
1) Clone or download the repo in your working directory.
2) Open the file "main.py", and set the path and the name of your training matrix.
3) Set the pre-processing options (i.e., centering and scaling method), as well as the number of clusters and PCs to retain.
4) Run "python main.py" from the terminal.
5) Wait until the algorithm converges. A txt file with the clustering solution will be automatically saved in the working dir.

@Required libraries:
The following python standard libraries are required:
- Pandas
- Numpy
- Matplotlib

@Cite:
If you use these scripts for research purposes, these are few reference you can cite, depending on your application.

    - Local algorithm for dimensionality reduction:
    [a] Kambhatla, Nandakishore, and Todd K. Leen. "Dimension reduction by local principal component analysis.", Neural computation 9.7 (1997): 1493-1516.
    
    - Clustering applications:
    [b] D’Alessio, Giuseppe, et al. "Adaptive chemistry via pre-partitioning of composition space and mechanism reduction.", Combustion and Flame 211 (2020): 68-82.
    
    - Data analysis applications:
    [c] Parente, Alessandro, et al. "Investigation of the MILD combustion regime via principal component analysis." Proceedings of the Combustion Institute 33.2 (2011): 3333-3341.
    [d] D'Alessio, Giuseppe, et al. "Analysis of turbulent reacting jets via Principal Component Analysis", Data Analysis in Direct Numerical Simulation of Turbulent Combustion, Springer (2020).
    [e] Bellemans, Aurélie, et al. "Feature extraction and reduced-order modelling of nitrogen plasma models using principal component analysis." Computers & chemical engineering 115 (2018): 504-514.
    
    - Preprocessing effects on PCA:
    [f] Parente, Alessandro, and James C. Sutherland. "Principal component analysis of turbulent combustion data: Data pre-processing and manifold sensitivity." Combustion and flame 160.2 (2013): 340-350.
    
    - Model order reduction:
    [g] Parente, Alessandro, et al. "Identification of low-dimensional manifolds in turbulent flames." Proceedings of the Combustion Institute. 2009 Jan 1;32(1):1579-86.
    [h] Aversano, Gianmarco, et al. "Application of reduced-order models based on PCA & Kriging for the development of digital twins of reacting flow applications." Computers & chemical engineering 121 (2019): 422-441.
