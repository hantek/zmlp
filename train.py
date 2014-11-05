import pylab
import numpy
import numpy.random
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams


class GraddescentMinibatch(object):
    def __init__(self, varin, data, cost, params, 
                 truth=None, truth_data=None, supervised=False,
                 batchsize=100, learningrate=0.1, momentum=0.9, 
                 rng=None, verbose=True):
        """
        Using stochastic gradient descent on data in a minibatch update manner.
        """
        # TODO: check datatype and dependencies between varin, cost, and param.
        self.varin         = varin
        self.data          = data
        self.cost          = cost  # Be careful with it. It might be the return
                                   # value of a method which have "truth" 
                                   # in its parameter.
        self.params        = params
        
        self.verbose       = verbose
        self.batchsize     = batchsize
        self.numbatches    = self.data.get_value().shape[0] / batchsize
        self.momentum      = momentum 
        self.supervised    = supervised
        if rng is None:
            rng = numpy.random.RandomState(1)
        assert isinstance(rng, numpy.random.RandomState), \
            "rng has to be a random number generater."
        self.rng = rng

        self.epochcount = 0
        self.index = T.lscalar('batch_index_in_trainer') 
        self.incs = dict([(
            p, 
            theano.shared(value=numpy.zeros(p.get_value().shape, 
                                            dtype=theano.config.floatX),
                          name='inc_' + p.name)
        ) for p in self.params])

        if supervised:
            assert isinstance(truth_data, T.sharedvar.TensorSharedVariable)
            assert isinstance(truth, T.TensorVariable)
            self.truth_data = truth_data
            self.truth = truth
        self.grad = T.grad(self.cost, self.params)

        self.set_learningrate(learningrate)


    def set_learningrate(self, learningrate):
        """
        TODO: set_learningrate() is not known to be working after 
        initialization. Not checked. A unit test should be written on it.
        """
        self.learningrate  = learningrate
        self.inc_updates = []  # inc_updates stands for how much we should 
                               # update our parameters during each epoch.
                               # Due to momentum, the increasement itself is
                               # changing between epochs. Its increasing by:
                               # from (key) inc_params 
                               # to (value) momentum * inc_params - lr * grad
                               
        self.updates = []  # updates the parameters of model during each epoch.
                           # from (key) params
                           # to (value) params + inc_params
                           
        for _param, _grad in zip(self.params, self.grad):
            self.inc_updates.append(
                (self.incs[_param],
                 self.momentum * self.incs[_param] - self.learningrate * _grad
                )
            )
            self.updates.append((_param, _param + self.incs[_param]))

        if not self.supervised:
            self._updateincs = theano.function(
                inputs = [self.index], 
                outputs = self.cost, 
                updates = self.inc_updates,
                givens = {
                    self.varin : self.data[self.index * self.batchsize: \
                                           (self.index+1)*self.batchsize]
                }
            )
        else:
            self._updateincs = theano.function(
                inputs = [self.index],
                outputs = self.cost,
                updates = self.inc_updates,
                givens = {
                    self.varin : self.data[self.index * self.batchsize: \
                                           (self.index+1)*self.batchsize],
                    self.truth : self.truth_data[self.index * self.batchsize: \
                                                 (self.index+1)*self.batchsize]
                }
            )

        self.n = T.scalar('n')
        self.noop = 0.0 * self.n
        self._trainmodel = theano.function([self.n], self.noop, 
                                           updates = self.updates)

    def step(self):
        # def inplaceclip(x):
        #     x[:,:] *= x>0.0
        #     return x

        # def inplacemask(x, mask):
        #     x[:,:] *= mask
        #     return x

        cost = 0.0
        stepcount = 0.0
        for batch_index in self.rng.permutation(self.numbatches - 1):
            stepcount += 1.0
            cost = (1.0 - 1.0/stepcount) * cost + \
                   (1.0/stepcount) * self._updateincs(batch_index)
            self._trainmodel(0)

        self.epochcount += 1
        if self.verbose:
            print 'epoch: %d, cost: %f' % (self.epochcount, cost)
