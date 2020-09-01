'''
MODULE: test_clustering.py

@Authors:
    G. D'Alessio [1,2]
    [1]: Université Libre de Bruxelles, Aero-Thermo-Mechanics Laboratory, Bruxelles, Belgium
    [2]: CRECK Modeling Lab, Department of Chemistry, Materials and Chemical Engineering, Politecnico di Milano

@Contacts:
    giuseppe.dalessio@ulb.ac.be


@Additional notes:
    This code is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
    Please report any bug to: giuseppe.dalessio@ulb.ac.be
'''

import unittest

import numpy as np
from numpy import linalg as LA
import matplotlib
import matplotlib.pyplot as plt

import OpenMORe.clustering as clustering
from OpenMORe.utilities import *


class testClustering(unittest.TestCase):

    def setUp(self):
        self.X = np.random.rand(100,5)
        self.nPCtest = 2
        self.nKtest = 3

    def tearDown(self):
        pass
    
    def test_VQPCA(self):
        VQPCA = clustering.lpca(self.X)
        VQPCA.eigens = self.nPCtest
        VQPCA.clusters = self.nKtest
        VQPCA.initialization = 'uniform'
        VQPCA.writeFolder = False

        try:
            idx = VQPCA.fit()
            passed = True
        except:
            passed = False
        
        self.assertEqual(passed, True)

    def test_KMeans(self):
        model = clustering.KMeans(self.X)
        model.initMode = True
        model.clusters = self.nKtest

        try:
            idx = model.fit()
            passed = True
        except:
            passed = False 
        
        self.assertEqual(passed, True)
    
    def test_Spectral(self):
        model = clustering.spectralClustering(self.X)
        model.clusters = self.nKtest
        model.affinity = 'rbf'

        try:
            idx = model.fit()
            passed = True 
        except:
            passed = False 
        
        self.assertEqual(passed, True)
