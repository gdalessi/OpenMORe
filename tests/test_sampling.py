'''
MODULE: test_sampling.py

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

import OpenMORe.model_order_reduction as model_order_reduction
from OpenMORe.utilities import *


class testSampling(unittest.TestCase):

    def setUp(self):
        self.X = np.random.rand(50,3)
        self.dimensions = 10


    def tearDown(self):
        pass

    def test_sampleRandom(self):
        method = 'random'
        sample = model_order_reduction.SamplePopulation(self.X)
        sample.sampling_strategy = method
        sample.set_size = self.dimensions

        miniX = sample.fit()

        self.assertEqual(miniX.shape[0], self.dimensions)

    def test_sampleStratified(self):
        method = 'stratified'
        sample = model_order_reduction.SamplePopulation(self.X)
        sample.sampling_strategy = method
        sample.set_size = self.dimensions

        miniX = sample.fit()

        self.assertEqual(miniX.shape[0], self.dimensions)