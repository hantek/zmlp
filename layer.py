"""
All parameters (excluding superparameters) in the model should be in theano var-
iables or theano shared values. In the training part, these variables should be
organized into "theano.function"s. So there should be no theano.function in the 
definition of models here.
"""
import numpy
import theano
import theano.tensor as T
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

        self.params = []  # to be implemented by subclass

    def fanin(self):
        raise NotImplementedError("Must be implemented by subclass.")
    
    def output(self):
        raise NotImplementedError("Must be implemented by subclass.")
   
    def __add__(self, other):
        """
        It is used for conveniently construct stacked layers.
        """
        assert isinstance(other, layer)
        assert other.n_in == self.n_out
        other.varin = self.output()
        return other.output()


    # Following are for analysis ----------------------------------------------

    def draw_weight(self, verbose=True, 
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
        assert hasattr(self, 'w'), "The layer need to have weight defined."

        if not hasattr(self, '_fig_weight'):
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
        super().__init__(n_in, n_out, varin=varin)
        if not npy_rng:
            npy_rng = numpy.random.RandomState(123)
        assert isinstance(npy_rng, numpy.random.RandomState)

        if not init_w:
            w = numpy.asarray(npy_rng.uniform(
                low = -4 * numpy.sqrt(6. / (n_in + n_out)),
                high = 4 * numpy.sqrt(6. / (n_in + n_out)),
                size=(n_in, n_out)), dtype=theano.config.floatX)
            self.w = theano.shared(value=w, name='w_sigmoid', borrow=True)
        else:
            assert init_w.get_value().shape == (n_in, n_out)
            self.w = init_w

        if not init_b:
            self.b = theano.shared(value=numpy.zeros(n_out),
                                   name='b_sigmoid', borrow=True)
        else:
            assert init_b.get_value().shape == (n_out,)
            self.b = init_b

        self.params = [self.w, self.b]

    def fanin(self):
        return T.dot(self.varin, self.w) + self.b

    def output(self):
        return T.nnet.sigmoid(self.fanin())


class LinearLayer(Layer):
    def __init__(self, n_in, n_out, varin=None, init_w=None, init_b=None, 
                 npy_rng=None):
        super().__init__(n_in, n_out, varin=varin)
        if not npy_rng:
            npy_rng = numpy.random.RandomState(123)
        assert isinstance(npy_rng, numpy.random.RandomState)

        if not init_w:
            w = numpy.asarray(npy_rng.uniform(
                low = -4 * numpy.sqrt(6. / (n_in + n_out)),
                high = 4 * numpy.sqrt(6. / (n_in + n_out)),
                size=(n_in, n_out)), dtype=theano.config.floatX)
            self.w = theano.shared(value=w, name='w_linear', borrow=True)
        else:
            assert init_w.get_value().shape == (n_in, n_out)
            self.w = init_w

        if not init_b:
            self.b = theano.shared(value=numpy.zeros(n_out),
                                   name='b_linear', borrow=True)
        else:
            assert init_b.get_value().shape == (n_out,)
            self.b = init_b

        self.params = [self.w, self.b]

    def fanin(self):
        return T.dot(self.varin, self.w) + self.b

    def output(self):
        return self.fanin()


class ZerobiasLayer(Layer):
    def __init__(n_in, n_out, threshold=1.0, varin=None, init_w=None, 
                 npy_rng=None):
        super().__init__(n_in, n_out, varin=varin)
        if not npy_rng:
            npy_rng = numpy.random.RandomState(123)
        assert isinstance(npy_rng, numpy.random.RandomState)

        if not init_w:
            w = numpy.asarray(npy_rng.uniform(
                low=-4 * numpy.sqrt(6. / (n_in + n_out)),
                high=4 * numpy.sqrt(6. / (n_in + n_out)),
                size=(n_in, n_out)), dtype=theano.config.floatX)
            self.w = theano.shared(value=w, name='w_zerobias', borrow=True)
        else:
            assert init_w.get_value().shape == (n_in, n_out)
            self.w = init_w
        self.params = [self.w]

        self.threshold = threshold

    def fanin(self):
        return T.dot(self.varin, self.w)

    def output(self):
        return (self.fanin() > self.threshold) * self.fanin()

class TanhLayer(Layer):
    def __init__(self):
        raise NotImplementedError("Not implemented yet...")


class GatedLinearLayer(Layer):
    def __init__(self):
        raise NotImplementedError("Not implemented yet...")


if __name__ == "__main__":
    a = Layer(2, 3)

