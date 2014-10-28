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
        self._fig_weight = None

    def get_fanin(self):
        raise NotImplementedError("Must be implemented by subclass.")
    
    def get_output(self):
        raise NotImplementedError("Must be implemented by subclass.")
    
    def draw_weight(self, verbose=False, 
                    filename='default_draw_baselayer_w.png'):
        assert self.w

        if not self._fig_weight:
            self._fig_weight = plt.gcf()
            plt1 = self._fig_weight.add_subplot(121)
            p1 = plt1.imshow(self.w)
            plt2 = self._fig_weight.add_subplot(122)
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
    def __init__(self):
        raise NotImplementedError("Not implemented yet..")


class linear_layer(layer):
    def __init__(n_in, n_out, varin=None, init_w=None, init_b=None, 
                 npy_rng=None):
        super().__init__(n_in, n_out, init_w)
        if not npy_rng:
            npy_rng = numpy.random.RandomState(123)
        assert isinstance(npy_rng, numpy.random.RandomState)
        self._npy_rng = npy_rng

        if not varin:
            varin = T.matrix('varin')
        assert isinstance(varin, T.var.TensorVariable)
        self.varin = varin

        if not init_w:
            init_w = numpy.asarray(self._npy_rng.uniform(
                      low = -4 * numpy.sqrt(6. / (n_in + n_out)),
                      high = 4 * numpy.sqrt(6. / (n_in + n_out)),
                      size=(n_in, n_out)), dtype=theano.config.floatX)
        assert init_w.shape == (n_in, n_out)
        self.w = theano.shared(value=init_w, name='w', borrow=True)

        if not init_b:
            init_b = numpy.zeros(n_out)
        assert init_b.shape == (n_out,)
        self.b = theano.shared(value=init_b, name='b', borrow=True)

    def get_fanin(self):
        return T.dot(self.varin, self.w) + self.b
    
    def get_output(self):
        return get_fanin()


class zerobias_layer(layer):

