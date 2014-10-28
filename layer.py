"""
All parameters (excluding superparameters) in the model should be in theano var-
iables or theano shared values. In the training part, these variables should be
organized into "theano.function"s. So there should be no theano.function in the 
definition of models here.
"""
import numpy
import theano
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class layer(object):
    def __init__(self, n_in, n_out, init_w=None):
        self.n_in = n_in
        self.n_out = n_out
        self.w = init_w
        self.fig_weight = None

    def get_fanin(self):
        raise NotImplementedError("Must be implemented by subclass.")
    
    def get_output(self):
        raise NotImplementedError("Must be implemented by subclass.")
    
    def draw_weight(self, verbose=False, filename='default_weight_drawing.png'):
        assert self.w

        if not self.fig_weight:
            self.fig_weight = plt.gcf()
            plt1 = self.fig_weight.add_subplot(121)
            p1 = plt1.imshow(self.w)
            plt2 = self.fig_weight.add_subplot(122)
            n, bins, patches = plt2.hist(self.w, face_color='blue')
            plt.clim()
        else:
            p1.set_data(self.w)
            n, bins, patches = plt2.hist(self.w, face_color='blue')
        if verbose:
            plt.pause(0.5)
        else:
            plt.savefig(filename)


class sigmoid_layer(layer):
class linear_layer(layer):
class zerobias_layer(layer):

