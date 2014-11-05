import theano
import numpy
import theano.tensor as T


class Layer(object):
    def __init__(self, n_in, n_out, varin=None):
        self.n_in = n_in
        self.n_out = n_out

        if not varin:
            varin = T.matrix('varin')
        self.varin = varin
        self.w = theano.shared(numpy.zeros((n_in, n_out)).astype(theano.config.floatX))
        self.params = [self.w]

    def fanin(self):
        raise NotImplementedError("Must be implemented by subclass.")

    def output(self):
        return T.dot(self.varin, self.w)

    def __add__(self, other):
        return StackedLayer(models_stack=[self, other], varin=self.varin)

class StackedLayer(Layer):
     def __init__(self, models_stack=[], varin=None):
        super(StackedLayer, self).__init__(
            n_in=models_stack[0].n_in, n_out=models_stack[-1].n_out,
            varin=varin
        )

        previous_layer = None
        self.params = []
        for layer_model in models_stack:
            if not previous_layer:  # First layer
                layer_model.varin = self.varin
            else:
                layer_model.varin = previous_layer.output()
            previous_layer = layer_model
            self.params += layer_model.params
        self.models_stack = models_stack

snn = Layer(2, 3) + Layer(3, 2)

consta = Layer(2, 3)
consta.__init__(5, 7)