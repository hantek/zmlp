"""
Here are some classifiers.
"""
import numpy
import theano
import theano.tensor as T
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model import Model

class Classifier(Model):
    def __init__(self, n_in, n_out, varin=None):
        super().__init__(n_in, n_out, varin=varin)

    # define a bunch of result analysis methods here.
    def error(self, ):


class LogisticRegression(Classifier):
    def __init__(self):
        #
        # TODO: Should it go into classifiers file?
        #
        pass
    def cost(self):
    def error():
class Perceptron(Classifier):
    def __init__(self):
        #
        # TODO: Should it go into classifiers file?
        #
        pass

class LinearRegression(Classifier):
    def __init__(self):
        #
        # TODO: Should it go into classifiers file?
        #
        pass

