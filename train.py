import numpy
import numpy.random
import theano
import theano.tensor as T

SharedCPU = theano.tensor.sharedvar.TensorSharedVariable
try:
    SharedGPU = theano.sandbox.cuda.var.CudaNdarraySharedVariable
except:
    SharedGPU=SharedCPU


class GraddescentMinibatch(object):
    def __init__(self, varin, data, cost, params, 
                 truth=None, truth_data=None, supervised=False,
                 batchsize=100, learningrate=0.1, momentum=0.9, 
                 rng=None, verbose=True):
        """
        Using stochastic gradient descent with momentum on data in a minibatch
        update manner.
        """
        
        # TODO: check dependencies between varin, cost, and param.
        
        assert isinstance(varin, T.TensorVariable)
        if (not isinstance(data, SharedCPU)) and \
           (not isinstance(data, SharedGPU)):
            raise TypeError("\'data\' needs to be a theano shared variable.")
        assert isinstance(cost, T.TensorVariable)
        assert isinstance(params, list)
        self.varin         = varin
        self.data          = data
        self.cost          = cost
        self.params        = params
        
        if supervised:
            if (not isinstance(truth_data, SharedCPU)) and \
               (not isinstance(truth_data, SharedGPU)):
                raise TypeError("\'truth_data\' needs to be a theano " + \
                                "shared variable.")
            assert isinstance(truth, T.TensorVariable)
            self.truth_data = truth_data
            self.truth = truth
        
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
        self.index = T.lscalar('batch_index_in_sgd') 
        self.incs = dict([(
            p, 
            theano.shared(value=numpy.zeros(p.get_value().shape, 
                                            dtype=theano.config.floatX),
                          name='inc_' + p.name,
                          broadcastable=p.broadcastable)
        ) for p in self.params])

        self.grad = T.grad(self.cost, self.params)

        self.set_learningrate(learningrate)


    def set_learningrate(self, learningrate):
        """
        TODO: set_learningrate() is not known to be working after 
        initialization. Not checked. A unit test should be written on it.
        """
        self.learningrate  = learningrate
        self.inc_updates = []  # updates the parameter increasements (i.e. 
                               # value in the self.incs dictionary.). Due to 
                               # momentum, the increasement itself is
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

        stepcount = 0.0
        cost = 0.
        for batch_index in self.rng.permutation(self.numbatches - 1):
            stepcount += 1.0
            # This is Roland's way of computing cost, still mean over all
            # batches. It saves space and don't harm computing time... 
            # But a little bit unfamilliar to understand at first glance.
            cost = (1.0 - 1.0/stepcount) * cost + \
                   (1.0/stepcount) * self._updateincs(batch_index)
            self._trainmodel(0)

        self.epochcount += 1
        if self.verbose:
            print 'epoch: %d, lr: %f, cost: %f' % (
                self.epochcount, self.learningrate, cost
            )
        return cost


class FeedbackAlignment(object):
    def __init__(self, model, data, truth_data, 
                 batchsize=100, learningrate=0.1, rng=None, verbose=True):
        """
        It works for both linear and nonlinear layers.

        Cost is defined intrinsicaly as the MSE between target y vector and 
        real y vector at the top layer.

        Parameters:
        ------------
        model : StackedLayer

        data : theano.compile.SharedVariable

        truth_data : theano.compile.SharedVariable
        """
        if (not isinstance(data, SharedCPU)) and \
           (not isinstance(data, SharedGPU)):
            raise TypeError("\'data\' needs to be a theano shared variable.")
        if (not isinstance(truth_data, SharedCPU)) and \
           (not isinstance(truth_data, SharedGPU)):
            raise TypeError("\'truth_data\' needs to be a theano shared variable.")
        self.varin         = model.models_stack[0].varin
        self.truth         = T.lmatrix('trurh_fba')
        self.data          = data
        self.truth_data    = truth_data

        self.model         = model
        self.output        = model.models_stack[-1].output()
        self.verbose       = verbose
        self.batchsize     = batchsize
        self.numbatches    = self.data.get_value().shape[0] / batchsize
 
        if rng is None:
            rng = numpy.random.RandomState(1)
        assert isinstance(rng, numpy.random.RandomState), \
            "rng has to be a random number generater."
        self.rng = rng

        self.error = (self.truth - self.output) * \
                     self.model.models_stack[-1].activ_prime()

        # set fixed random matrix
        self.fixed_B = [None, ]
        for imod in self.model.models_stack[1:]:
            i_layer_B = []
            for ipar in imod.params:
                rnd = numpy.asarray(
                    self.rng.uniform(
                        low = -4 * numpy.sqrt(6. / (imod.n_in + imod.n_out)),
                        high = 4 * numpy.sqrt(6. / (imod.n_in + imod.n_out)),
                        size = ipar.get_value().shape
                    ), 
                    dtype=ipar.dtype
                )
 
                i_layer_B.append(
                    theano.shared(value = rnd, name=ipar.name + '_fixed',
                                  borrow=True)
                )
            self.fixed_B.append(i_layer_B)

        self.epochcount = 0
        self.index = T.lscalar('batch_index_in_fba') 
        self._get_cost = theano.function(
            inputs = [self.index],
            outputs = T.sum(self.error ** 2),
            givens = {
                 self.varin : self.data[self.index * self.batchsize: \
                                        (self.index+1)*self.batchsize],
                 self.truth : self.truth_data[self.index * self.batchsize: \
                                              (self.index+1)*self.batchsize]
            }
        )

        self.set_learningrate(learningrate)


    def set_learningrate(self, learningrate):
        self.learningrate = learningrate

        layer_error = self.error
        self.layer_learning_funcs = []
        for i in range(len(self.model.models_stack) - 1, -1, -1):
            iupdates = []
            iupdates.append((
                 self.model.models_stack[i].w,
                 self.model.models_stack[i].w + self.learningrate * \
                     T.dot(self.model.models_stack[i].varin.T, layer_error)
            ))  # w
            iupdates.append((
                 self.model.models_stack[i].b,
                 self.model.models_stack[i].b + self.learningrate * \
                     T.mean(layer_error, axis=0)
            ))  # b
            if i > 0:  # exclude the first layer.
                layer_error = T.dot(layer_error, self.fixed_B[i][0].T) * \
                    self.model.models_stack[i-1].activ_prime()
            
            self.layer_learning_funcs.append(
                theano.function(
                    inputs = [self.index],
                    outputs = self.model.models_stack[i].output(),
                    updates = iupdates,
                    givens = {
                        self.varin : self.data[
                            self.index * self.batchsize: \
                            (self.index+1)*self.batchsize
                        ],
                        self.truth : self.truth_data[
                            self.index * self.batchsize: \
                            (self.index+1)*self.batchsize
                        ]
                    }
                )
            )  


    def step(self):
        stepcount = 0.
        cost = 0.
        for batch_index in self.rng.permutation(self.numbatches - 1):
            stepcount += 1.
            cost = (1.0 - 1.0/stepcount) * cost + \
                   (1.0/stepcount) * self._get_cost(batch_index)
            for layer_learn in self.layer_learning_funcs:
                layer_learn(batch_index)

        self.epochcount += 1
        if self.verbose:
            print 'epoch: %d, lr: %f, cost: %f' % (
                self.epochcount, self.learningrate, cost
            )
        return cost
