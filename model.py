"""
This file defines some basic models. A default cost is defined here. By overidi-
ng it subclasses can have more complex forms of costs.

Nominators with "_get_" stands for they are dealing with numpy arrays.
"""
import numpy
import theano
import theano.tensor as T
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from layer import Layer, SigmoidLayer, LinearLayer,ZerobiasLayer


class Model(Layer):
    """
    A base class for models, which by inheritation can turn to be a concreate
    classifier or regressor.
    """
    def __init__(self, n_in, n_out, varin=None):
        super(Model, self).__init__(n_in, n_out, varin=varin)
    
    def fanin(self):
        print "Calling the fan-in of a base model might have no meaning."

    def output(self):
        print "Calling the output of a base model might have no meaning."

    def cost(self):
        raise NotImplementedError("Must be implemented by subclass.")


class AutoEncoder(Model):
    def __init__(self, n_in, n_out, vistype, varin=None, tie=True, 
                 init_w=None, init_wT=None, init_b=None, init_bT=None, 
                 npy_rng=None):
        """
        By default the encoder and decoder are both set to be sigmoid layers. 
        Users can overide it to be other kind of layers in ther subclass.
        """
        super(AutoEncoder, self).__init__(n_in, n_out, varin)

        assert ((init_w == None) or \
            isinstance(init_w, theano.compile.sharedvalue.SharedVariable))
        assert ((init_b == None) or \
            isinstance(init_b, theano.compile.sharedvalue.SharedVariable))
        assert ((init_bT == None) or \
            isinstance(init_bT, theano.compile.sharedvalue.SharedVariable))
        
        if tie:
            assert init_wT == None, "Tied autoencoder do not accept init_wT."
            init_wT = init_w.T
        else:
            assert isinstance(init_wT, 
                              theano.compile.sharedvalue.SharedVariable)
        
        self.vistype = vistype
        if self.vistype == 'binary':
            self.encoder = SigmoidLayer(n_in, n_out, varin=self.varin, 
                init_w=init_w, init_b=init_b, npy_rng=npy_rng)
            self.decoder = SigmoidLayer(
                n_out, n_in, varin=self.encoder.output(), 
                init_w=init_wT, init_b=init_bT, npy_rng=npy_rng)
        elif self.vistype == 'real':
            self.encoder = LinearLayer(n_in, n_out, varin=self.varin, 
                init_w=init_w, init_b=init_b, npy_rng=npy_rng)
            self.decoder = LinearLayer(
                n_out, n_in, varin=self.encoder.output(), 
                init_w=init_wT, init_b=init_bT, npy_rng=npy_rng)
        else:
            raise ValueError("vistype has to be either binary or real.")
        
        if tie:
            self.params = self.encoder.params + [self.decoder.b]
        else:
            self.params = self.encoder.params + self.decoder.params

        # for plotting
        self.rangex=[0., 1.]
        self.rangey=[0., 1.]

    def hidden(self):
        return self.encoder.output()

    def output(self):
        """
        Setting the output in this manner helps stacking the encoders in a 
        convenient way.
        """
        return self.hidden()

    def reconstruction(self):
        return self.decoder.output()

    def cost(self):
        """
        By default it presents a cross entropy cost for binary data and MSE for
        real valued data. We can still overide it if we were using a different 
        training criteria.
        """
        if self.vistype == 'binary':
            x_head = self.reconstruction()
            cost_per_case = - T.sum(self.varin * T.log(x_head) + \
                (1 - self.varin) * T.log(1 - x_head), axis=1)
            return T.mean(cost_per_case)
        elif self.vistype == 'real':
            cost_per_case = T.sum((self.reconstruction() - self.varin) ** 2, 
                                  axis=1)
            return T.mean(cost_per_case)
        else:
            raise ValueError("vistype has to be either binary or real.")


    # Following are for analysis -----------------------------------------------
    
    def score(self, fig_handle=None):
        assert self.n_in == 2, "Cannot draw score for higher dimentions."
        #
        # TODO: plot the score of autoencoder for a 2-D input, if fig_handle 
        # given, plot on the given figure.
        #
        pass

    def quiver(self, rangex=[0., 1.], rangey=[0., 1.], dpi=50): 
        assert self.n_in == 2, "Cannot draw quiver for higher dimentions."

        if (self.rangex != rangex or self.rangey != rangey):
            if not hasattr(self, '_fig_quiver'):  # first call
                self._fig_quiver = plt.gcf()
                self.ax = self._fig_quiver.add_subplot(111)
                self._get_reconstruction = theano.function(
                    [self.encoder.varin], self.reconstruction())
            
            self.rangex = rangex
            self.rangey = rangey
            np_x, np_y = numpy.meshgrid(
                numpy.linspace(rangex[0], rangex[1], num=dpi),
                numpy.linspace(rangey[0], rangey[1], num=dpi))
            self.mesh_points = numpy.concatenate(
                (np_x[:, :, numpy.newaxis], np_y[:, :, numpy.newaxis]), 
                axis=2).reshape(-1, 2)
            recons = self._get_reconstruction(self.mesh_points)
            diff = recons - self.mesh_points
            quiver_x = diff[:, 0].reshape(dpi, dpi)
            quiver_y = diff[:, 1].reshape(dpi, dpi)

            self.Q = self.ax.quiver(quiver_x, quiver_y)
            self._fig_quiver.canvas.draw()

        else:
            recons = self._get_reconstruction(self.mesh_points)
            diff = recons - self.mesh_points
            quiver_x = diff[:, 0].reshape(dpi, dpi)
            quiver_y = diff[:, 1].reshape(dpi, dpi)
            self.Q.set_UVC(quiver_x, quiver_y)
            self._fig_quiver.canvas.draw()
        

class ZerobiasAutoencoder(AutoEncoder):
    def __init__(self, n_in, n_out, vistype, threshold=1.0, varin=None, 
                 tie=True, init_w=None, init_wT=None, init_bT=None, 
                 npy_rng=None):
        super(ZerobiasAutoencoder, self).__init__(
            n_in, n_out, vistype, varin=varin, tie=tie, 
            init_w=init_w, init_wT=init_wT, init_b=None, init_bT=None,
            npy_rng=None
        )

        self.encoder = ZerobiasLayer(n_in, n_out, threshold=threshold, 
            varin=self.varin, init_w=self.init_w, npy_rng=npy_rng)
        
        if self.vistype == 'binary':
            self.decoder = SigmoidLayer(
                n_out, n_in, varin=self.encoder.output(), 
                init_w=self.init_wT, init_b=self.init_bT, npy_rng=npy_rng)
        elif self.vistype == 'real':
            self.decoder = LinearLayer(
                n_out, n_in, varin=self.encoder.output(),
                init_w=self.init_wT, init_b=self.init_bT, npy_rng=npy_rng)
        else:
            raise ValueError("vistype has to be either binary or real.")

        if tie:
            self.params = self.encoder.params + [self.decoder.b]
        else:
            self.params = self.encoder.params + self.decoder.params
