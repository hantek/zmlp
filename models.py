"""
This file defines some basic models. No cost is defined here. It should go into
the training part.
"""
import theano
import theano.tensor as T
import numpy
import layer

class AutoEncoder(object):
    def __init__(self, n_in, n_out, varin=None, tie=True, 
                 init_w=None, init_wT=None, init_b=None, init_bT=None, 
                 npy_rng=None):
        self.n_in = n_in
        self.n_out = n_out

        if tie:
            assert init_wT == None, "Tied autoencoder do not accept init_wT."
            assert init_bT == None, "Tied autoencoder do not accept init_bT."
            init_wT = init_w
            init_bT = init_b
        
        self.encoder = layer.SigmoidLayer(n_in, n_out, 
            varin=varin, init_w=init_w, init_b=init_b, npy_rng=npy_rng)
        self.decoder = layer.SigmoidLayer(n_out, n_in, 
            varin=self.encoder.get_output(), init_w=init_wT, init_b=init_bT, 
            npy_rng=npy_rng)
    
    def get_reconstruction(self):
        return self.decoder.get_output()

class StackedAutoEncoder(object):
    def __init__():
        
