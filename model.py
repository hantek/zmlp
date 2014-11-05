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
    def __init__(self, n_in, n_hid, vistype, varin=None):
        """
        The autoencoder ADT defines a 2-layer structure (both layer might be
        a StackedLayer object, in that way it will have more than 2 actual
        layers) which allows for reconstructing the input from middle hidden
        layers.

        It's better to construct an autoencoder by constructing an instance of a
        certain autoencoder class instead of using "+" between two already
        instanciated encoder and decoder. Because that helps dealing with a lot
        of parameter settings and guarantees a bunch of autoencoder-specific
        methods, along with analyzing & visualizing methods.
        """

        super(AutoEncoder, self).__init__(n_in, n_hid, varin=varin)
        self.n_hid = self.n_out
        
        self.encoder = None
        self.decoder = None

        # Note that we define this self.params_private member data to stand for
        # the "private" parameters for the autoencoder itself. i.e., parameters
        # in BOTH the encoder and decoder. This is ONLY for layerwise training
        # the autoencoder itself  while it were stacked into a deep structure.
        self.params_private = self.params + []

        # While, for self.params, we only include parameters of the encoder.
        # Defining self.params and self.params_private in this way helps
        # collecting "params" member data (which is for training the whole
        # network in the final fine-tuning stage.) while stacking layers.

        # for plotting
        self.vistype = vistype
        if vistype == 'binary':
            self.rangex=[0., 1.]
            self.rangey=[0., 1.]
        elif vistype == 'real':
            self.rangex=[-1., 1.]
            self.rangey=[-1., 1.]
        else:
            raise ValueError("vistype has to be either 'binary' or 'real'.")

    def encoder(self):
        raise NotImplementedError("Must be implemented by subclass.")
    
    def decoder(self):
        raise NotImplementedError("Must be implemented by subclass.")
    
    def fanin(self):
        """
        The fanin of the first actual layer might still have sense. so keep it
        here.
        """
        return self.encoder.fanin()

    def hidden(self):
        return self.encoder.output()

    def output(self):
        """
        Setting the output in this manner helps stacking the encoders in a 
        convenient way. That allows you to get a StackedLayer object which only
        contains the encoder part in the returned deep structure while calling
            ... + Autoencoder(...) + Layer(...) + ...
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
    def __init__(self, n_in, n_hid, vistype, threshold=1.0, varin=None,
                 tie=True, init_w=None, init_wT=None, init_bT=None, 
                 npy_rng=None):
        super(ZerobiasAutoencoder, self).__init__(
            n_in, n_hid, vistype=vistype, varin=varin
        )

        # TODO. The following assetion is complaining about an assertion
        # error while passing w.T to init_w. Considering using a more
        # robust way of assertion in the future.
        # assert ((init_w == None) or \
        #     isinstance(init_w, theano.compile.sharedvalue.SharedVariable))
        self.encoder = ZerobiasLayer(self.n_in, self.n_hid, threshold=threshold,
            varin=self.varin, init_w=init_w, npy_rng=npy_rng)

        assert ((init_bT == None) or \
            isinstance(init_bT, theano.compile.sharedvalue.SharedVariable))
        if tie:
            assert init_wT == None, "Tied autoencoder do not accept init_wT."
            init_wT = self.encoder.w.T
        # else:
        # TODO. The following assetion is complaining about an assertion
        # error while passing w.T to init_w. Considering using a more
        # robust way of assertion in the future.
        #     assert isinstance(init_wT,
        #                       theano.compile.sharedvalue.SharedVariable)
        if self.vistype == 'binary':
            self.decoder = SigmoidLayer(
                self.n_hid, self.n_in, varin=self.encoder.output(),
                init_w=init_wT, init_b=init_bT, npy_rng=npy_rng
            )
        elif self.vistype == 'real':
            self.decoder = LinearLayer(
                self.n_hid, self.n_in, varin=self.encoder.output(),
                init_w=init_wT, init_b=init_bT, npy_rng=npy_rng
            )

        self.params = self.encoder.params
        if tie:
            self.params_private = self.encoder.params + [self.decoder.b]
        else:
            self.params_private = self.encoder.params + self.decoder.params
