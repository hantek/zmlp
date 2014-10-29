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

class Layer(object):
    def __init__(self, n_in, n_out, varin=None):
        """
        Parameters
        -----------
        n_in : int
        n_out : int
        varin : theano.tensor.TensorVariable, optional
        init_w : theano.tensor.TensorType, optional
            We initialise the weights to be zero here, but it can be initialized
            into a proper random distribution by set_value() in the subclass.
        """
        self.n_in = n_in
        self.n_out = n_out
        
        if not varin:
            varin = T.matrix('varin')
        assert isinstance(varin, T.TensorVariable)
        self.varin = varin
        
        self.w = None  # to be implemented by subclass
        self._fig_weight = None

    def get_fanin(self):
        raise NotImplementedError("Must be implemented by subclass.")
    
    def get_output(self):
        raise NotImplementedError("Must be implemented by subclass.")
   
    def __add__(self, other):
        """
        It is used for conveniently construct stacked layers.
        """
        assert isinstance(other, layer)
        assert other.n_in == self.n_out
        other.varin = self.get_output()
        return other.get_output()

    def draw_weight(self, verbose=False, 
                    filename='default_draw_baselayer_w.png'):
        """
        Parameters
        -----------
        verbose : bool
        filename : string
    
        Returns
        -----------
        Notes
        -----------
        """
        assert self.w

        if not self._fig_weight:
            self._fig_weight = plt.gcf()
            plt1 = self._fig_weight.add_subplot(311)
            p1 = plt1.imshow(self.w.get_value())
            plt2 = self._fig_weight.add_subplot(312)
            n, bins, patches = plt2.hist(self.w.get_value(), face_color='blue')
            plt3 = self._fig_weight.add_subplot(313)
            p3 = plt3.imshow(T.dot(self.w.T, self.w).eval())
            plt.clim()
        else:
            p1.set_data(self.w.get_value())
            n, bins, patches = plt2.hist(self.w.get_value(), face_color='blue')
            p3.set_data(T.dot(self.w.T, self.w).eval())
        if verbose:
            plt.pause(0.5)
        else:
            plt.savefig(filename)


class SigmoidLayer(Layer):
    def __init__(self, n_in, n_out, varin=None, init_w=None, init_b=None, 
                 npy_rng=None):
        super().__init__(n_in, n_out)
        if not npy_rng:
            npy_rng = numpy.random.RandomState(123)
        assert isinstance(npy_rng, numpy.random.RandomState)
        self._npy_rng = npy_rng

        if not init_w:
            w = numpy.asarray(self._npy_rng.uniform(
                low = -4 * numpy.sqrt(6. / (n_in + n_out)),
                high = 4 * numpy.sqrt(6. / (n_in + n_out)),
                size=(n_in, n_out)), dtype=theano.config.floatX)
            self.w = theano.shared(value=w, name='w_sigmoid', borrow=True)
        else:
            assert init_w.get_value().shape == (n_in, n_out)
            self.w = init_w

        if not init_b:
            init_b = numpy.zeros(n_out)
        assert init_b.shape == (n_out,)
        self.b = theano.shared(value=init_b, name='b_sigmoid', borrow=True)

    def get_fanin(self):
        return T.dot(self.varin, self.w) + self.b

    def get_output(self):
        return T.nnet.sigmoid(get_fanin())


class LinearLayer(Layer):
    def __init__(self, n_in, n_out, varin=None, init_w=None, init_b=None, 
                 npy_rng=None):
        super().__init__(n_in, n_out)
        if not npy_rng:
            npy_rng = numpy.random.RandomState(123)
        assert isinstance(npy_rng, numpy.random.RandomState)
        self._npy_rng = npy_rng

        if not init_w:
            w = numpy.asarray(self._npy_rng.uniform(
                low = -4 * numpy.sqrt(6. / (n_in + n_out)),
                high = 4 * numpy.sqrt(6. / (n_in + n_out)),
                size=(n_in, n_out)), dtype=theano.config.floatX)
            self.w = theano.shared(value=w, name='w_linear', borrow=True)
        else:
            assert init_w.get_value().shape == (n_in, n_out)
            self.w = init_w

        if not init_b:
            init_b = numpy.zeros(n_out)
        assert init_b.shape == (n_out,)
        self.b = theano.shared(value=init_b, name='b_linear', borrow=True)

    def get_fanin(self):
        return T.dot(self.varin, self.w) + self.b

    def get_output(self):
        return get_fanin()


class ZerobiasLayer(Layer):
    def __init__(n_in, n_out, threshold=1.0, varin=None, init_w=None, 
                 npy_rng=None):
        super().__init__(n_in, n_out)
        if not npy_rng:
            npy_rng = numpy.random.RandomState(123)
        assert isinstance(npy_rng, numpy.random.RandomState)
        self._npy_rng = npy_rng

        if not init_w:
            w = numpy.asarray(self._npy_rng.uniform(
                low=-4 * numpy.sqrt(6. / (n_in + n_out)),
                high=4 * numpy.sqrt(6. / (n_in + n_out)),
                size=(n_in, n_out)), dtype=theano.config.floatX)
            self.w = theano.shared(value=w, name='w_zerobias', borrow=True)
        else:
            assert init_w.get_value().shape == (n_in, n_out)
            self.w = init_w
        self.threshold = threshold

    def get_fanin(self):
        return T.dot(self.varin, self.w)

    def get_output(self):
        return (get_fanin() > self.threshold) * get_fanin()


class GatedLinearLayer(Layer):
    def __init__(self):
        raise NotImplementedError("Not implemented yet...")


if __name__ == "__main__":
    a = Layer(2, 3)

