import cPickle
import gzip
import os
import sys
import time
import cPickle
import numpy
import matplotlib.pyplot as plt

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

if not "../DeepLearningTutorials/code/" in sys.path:
    sys.path.append("../DeepLearningTutorials/code/")

from logistic_sgd import LogisticRegression, load_data
from mlp import HiddenLayer
from dA import dA

class HiddenZaeLayer():
    def __init__(self, rng=None, input=None, W=None,
                 n_in=None, n_out=None, selectionthreshold=1.):
        self.input = input

        if W is None:
            W_values = numpy.asarray(rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)), dtype=theano.config.floatX)
            W = theano.shared(value=W_values, name='W', borrow=True)
        self.W = W

        lin_output = T.dot(input, self.W)
        self.output = (lin_output > selectionthreshold) * lin_output

        self.layer_cost = T.sum((self.input - T.dot(self.output, self.W.T))**2)
        
        # parameters of the model
        self.params = [self.W,]



class SZAE(object):
    def __init__(self, numpy_rng, theano_rng=None, n_ins=784,
                 hidden_layers_sizes=[500, 500], n_outs=10, lambda_zae=0.0001):
        self.zae_layers = []
        self.zae_cost = []
        self.params = []
        self.n_layers = len(hidden_layers_sizes)

        assert self.n_layers > 0

        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        self.x = T.matrix('x')  # the data is presented as rasterized images
        self.y = T.ivector('y')  # the labels are presented as 1D vector of
                                 # [int] labels

        for i in xrange(self.n_layers):

            if i == 0:
                input_size = n_ins
            else:
                input_size = hidden_layers_sizes[i - 1]

            if i == 0:
                layer_input = self.x
            else:
                layer_input = self.zae_layers[-1].output

            zae_layer = HiddenZaeLayer(rng=numpy_rng,
                                       input=layer_input,
                                       n_in=input_size,
                                       n_out=hidden_layers_sizes[i])
            # add the layer to our list of layers
            self.zae_layers.append(zae_layer)
            self.zae_cost.append(zae_layer.layer_cost)
            self.params.extend(zae_layer.params)

        # We now need to add a logistic layer on top of the MLP
        self.logLayer = LogisticRegression(
            input=self.zae_layers[-1].output,
            n_in=hidden_layers_sizes[-1], n_out=n_outs)

        self.params.extend(self.logLayer.params)
        # construct a function that implements one step of finetunining

        # compute the cost for second phase of training,
        # defined as the negative log likelihood
        self.finetune_cost = self.logLayer.negative_log_likelihood(self.y)

        for i in self.zae_cost:
            self.finetune_cost = self.finetune_cost + i * lambda_zae
        
        # symbolic variable that points to the number of errors made on the
        # minibatch given by self.x and self.y
        self.errors = self.logLayer.errors(self.y)

    def build_finetune_functions(self, datasets, batch_size, learning_rate):

        (train_set_x, train_set_y) = datasets[0]
        (valid_set_x, valid_set_y) = datasets[1]
        (test_set_x, test_set_y) = datasets[2]

        # compute number of minibatches for training, validation and testing
        n_train_batches = train_set_x.get_value(borrow=True).shape[0]
        n_train_batches /= batch_size
        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
        n_valid_batches /= batch_size
        n_test_batches = test_set_x.get_value(borrow=True).shape[0]
        n_test_batches /= batch_size

        index = T.lscalar('index')  # index to a [mini]batch

        # compute the gradients with respect to the model parameters
        gparams = T.grad(self.finetune_cost, self.params)

        # compute list of fine-tuning updates
        updates = []
        for param, gparam in zip(self.params, gparams):
            updates.append((param, param - gparam * learning_rate))

        train_fn = theano.function(inputs=[index],
              outputs=self.finetune_cost,
              updates=updates,
              givens={
                self.x: train_set_x[index * batch_size:
                                    (index + 1) * batch_size],
                self.y: train_set_y[index * batch_size:
                                    (index + 1) * batch_size]},
              name='train')

        train_score_i = theano.function([index], self.errors,
                 givens={
                   self.x: train_set_x[index * batch_size:
                                       (index + 1) * batch_size],
                   self.y: train_set_y[index * batch_size:
                                       (index + 1) * batch_size]},
                      name='train')
        
        test_score_i = theano.function([index], self.errors,
                 givens={
                   self.x: test_set_x[index * batch_size:
                                      (index + 1) * batch_size],
                   self.y: test_set_y[index * batch_size:
                                      (index + 1) * batch_size]},
                      name='test')

        valid_score_i = theano.function([index], self.errors,
              givens={
                 self.x: valid_set_x[index * batch_size:
                                     (index + 1) * batch_size],
                 self.y: valid_set_y[index * batch_size:
                                     (index + 1) * batch_size]},
                      name='valid')

        # Create a function that scans the entire train set
        def train_score():
            return [train_score_i(i) for i in xrange(n_train_batches)]
        
        # Create a function that scans the entire validation set
        def valid_score():
            return [valid_score_i(i) for i in xrange(n_valid_batches)]

        # Create a function that scans the entire test set
        def test_score():
            return [test_score_i(i) for i in xrange(n_test_batches)]

        return train_fn, train_score, valid_score, test_score
    
    
    def save_params(self, filename):
        output = open(filename, 'wb')
        for p in range(len(self.params)):
            cPickle.dump(self.params[p].get_value(), output)
        output.close()

    def load_params(self, filename):
        pkl_file = open(filename, 'rb')
        param_num = len(self.params)
        for p in range(param_num):
            self.params[p].set_value(cPickle.load(pkl_file))
        pkl_file.close()















class MLP(object):
    def __init__(self, numpy_rng, theano_rng=None, n_ins=784,
                 hidden_layers_sizes=[500, 500], n_outs=10):
        self.sigmoid_layers = []
        self.params = []
        self.n_layers = len(hidden_layers_sizes)

        assert self.n_layers > 0

        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
        # allocate symbolic variables for the data
        self.x = T.matrix('x')  # the data is presented as rasterized images
        self.y = T.ivector('y')  # the labels are presented as 1D vector of
                                 # [int] labels

        for i in xrange(self.n_layers):
            # construct the sigmoidal layer

            # the size of the input is either the number of hidden units of
            # the layer below or the input size if we are on the first layer
            if i == 0:
                input_size = n_ins
            else:
                input_size = hidden_layers_sizes[i - 1]

            # the input to this layer is either the activation of the hidden
            # layer below or the input of the SdA if you are on the first
            # layer
            if i == 0:
                layer_input = self.x
            else:
                layer_input = self.sigmoid_layers[-1].output

            sigmoid_layer = HiddenLayer(rng=numpy_rng,
                                        input=layer_input,
                                        n_in=input_size,
                                        n_out=hidden_layers_sizes[i],
                                        activation=T.nnet.sigmoid)
            # add the layer to our list of layers
            self.sigmoid_layers.append(sigmoid_layer)
            # its arguably a philosophical question...
            # but we are going to only declare that the parameters of the
            # sigmoid_layers are parameters of the StackedDAA
            # the visible biases in the dA are parameters of those
            # dA, but not the SdA
            self.params.extend(sigmoid_layer.params)

        # We now need to add a logistic layer on top of the MLP
        self.logLayer = LogisticRegression(
                         input=self.sigmoid_layers[-1].output,
                         n_in=hidden_layers_sizes[-1], n_out=n_outs)

        self.params.extend(self.logLayer.params)
        # construct a function that implements one step of finetunining

        # compute the cost for second phase of training,
        # defined as the negative log likelihood
        self.finetune_cost = self.logLayer.negative_log_likelihood(self.y)
        
        # compute the gradients with respect to the model parameters
        # symbolic variable that points to the number of errors made on the
        # minibatch given by self.x and self.y
        self.errors = self.logLayer.errors(self.y)

    def build_finetune_functions(self, datasets, batch_size, learning_rate):
        '''Generates a function `train` that implements one step of
        finetuning, a function `validate` that computes the error on
        a batch from the validation set, and a function `test` that
        computes the error on a batch from the testing set

        :type datasets: list of pairs of theano.tensor.TensorType
        :param datasets: It is a list that contain all the datasets;
                         the has to contain three pairs, `train`,
                         `valid`, `test` in this order, where each pair
                         is formed of two Theano variables, one for the
                         datapoints, the other for the labels

        :type batch_size: int
        :param batch_size: size of a minibatch

        :type learning_rate: float
        :param learning_rate: learning rate used during finetune stage
        '''

        (train_set_x, train_set_y) = datasets[0]
        (valid_set_x, valid_set_y) = datasets[1]
        (test_set_x, test_set_y) = datasets[2]

        # compute number of minibatches for training, validation and testing
        n_train_batches = train_set_x.get_value(borrow=True).shape[0]
        n_train_batches /= batch_size
        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
        n_valid_batches /= batch_size
        n_test_batches = test_set_x.get_value(borrow=True).shape[0]
        n_test_batches /= batch_size

        index = T.lscalar('index')  # index to a [mini]batch

        # compute the gradients with respect to the model parameters
        gparams = T.grad(self.finetune_cost, self.params)

        # compute list of fine-tuning updates
        updates = []
        for param, gparam in zip(self.params, gparams):
            updates.append((param, param - gparam * learning_rate))

        train_fn = theano.function(inputs=[index],
              outputs=self.finetune_cost,
              updates=updates,
              givens={
                self.x: train_set_x[index * batch_size:
                                    (index + 1) * batch_size],
                self.y: train_set_y[index * batch_size:
                                    (index + 1) * batch_size]},
              name='train')

        train_score_i = theano.function([index], self.errors,
                 givens={
                   self.x: train_set_x[index * batch_size:
                                       (index + 1) * batch_size],
                   self.y: train_set_y[index * batch_size:
                                       (index + 1) * batch_size]},
                      name='train')
        
        test_score_i = theano.function([index], self.errors,
                 givens={
                   self.x: test_set_x[index * batch_size:
                                      (index + 1) * batch_size],
                   self.y: test_set_y[index * batch_size:
                                      (index + 1) * batch_size]},
                      name='test')

        valid_score_i = theano.function([index], self.errors,
              givens={
                 self.x: valid_set_x[index * batch_size:
                                     (index + 1) * batch_size],
                 self.y: valid_set_y[index * batch_size:
                                     (index + 1) * batch_size]},
                      name='valid')

        # Create a function that scans the entire train set
        def train_score():
            return [train_score_i(i) for i in xrange(n_train_batches)]
        
        # Create a function that scans the entire validation set
        def valid_score():
            return [valid_score_i(i) for i in xrange(n_valid_batches)]

        # Create a function that scans the entire test set
        def test_score():
            return [test_score_i(i) for i in xrange(n_test_batches)]

        return train_fn, train_score, valid_score, test_score
    
    
    def save_params(self, filename):
        output = open(filename, 'wb')
        for p in range(len(self.params)):
            cPickle.dump(self.params[p].get_value(), output)
        output.close()

    def load_params(self, filename):
        pkl_file = open(filename, 'rb')
        param_num = len(self.params)
        for p in range(param_num):
            self.params[p].set_value(cPickle.load(pkl_file))
        pkl_file.close()

def test(finetune_lr=0.1, training_epochs=1000, 
             hidden_layers_sizes = [500, 500, 500],
             dataset='../DeepLearningTutorials/data/mnist.pkl.gz', batch_size=100):

    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_train_batches /= batch_size

    # numpy random generator
    numpy_rng = numpy.random.RandomState(89677)
    print '... building the model'
    # construct the stacked denoising autoencoder class
    mlp = MLP(numpy_rng=numpy_rng, n_ins=28 * 28,
              hidden_layers_sizes=hidden_layers_sizes,
              n_outs=10)
    zae = SZAE(numpy_rng=numpy_rng, n_ins=28 * 28,
              hidden_layers_sizes=hidden_layers_sizes,
              n_outs=10)

    # get the training, validation and testing function for the model
    print '... getting the finetuning functions'
    train_fn, train_score, validate_model, test_model = mlp.build_finetune_functions(
                datasets=datasets, batch_size=batch_size,
                learning_rate=finetune_lr)
    train_fn_zae, train_score_zae, validate_model_zae, test_model_zae = \
        zae.build_finetune_functions(datasets=datasets, batch_size=batch_size,
                                     learning_rate=finetune_lr)

    print '... finetunning the model'
    validation_frequency = n_train_batches * 5 
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    start_time = time.clock()
    
    train_err = []
    valid_err = []
    train_err_zae = []
    valid_err_zae = []
    epoch = 0
    while (epoch < training_epochs):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):
            minibatch_avg_cost = train_fn(minibatch_index)
            minibatch_avg_cost_zae = train_fn_zae(minibatch_index)
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                train_loss = numpy.mean(train_score())
                this_validation_loss = numpy.mean(validate_model())
                train_err.append(train_loss)
                valid_err.append(this_validation_loss)
                
                train_loss_zae = numpy.mean(train_score_zae())
                this_validation_loss_zae = numpy.mean(validate_model_zae())
                train_err_zae.append(train_loss_zae)
                valid_err_zae.append(this_validation_loss_zae)
                
                print('epoch %i, training error %f %%, validation error %f %%' %
                      (epoch, train_loss * 100., this_validation_loss * 100.))

                print('     zae, training error %f %%, validation error %f %%' %
                      (train_loss_zae * 100., this_validation_loss_zae * 100.))

    end_time = time.clock()
    plt.figure()
    x = numpy.arange(len(train_err))
    plt.plot(x, numpy.asarray(train_err),"cx--",label="train_MLP",linewidth=2)
    plt.plot(x, numpy.asarray(valid_err),"co--",label="valid_MLP",linewidth=2)
    plt.plot(x, numpy.asarray(train_err_zae),"mx--",label="train_ZAE",linewidth=2)
    plt.plot(x, numpy.asarray(valid_err_zae),"mo--",label="valid_ZAE",linewidth=2)
    plt.legend()
    filename = 'fig_hid_' + '_'.join([str(i) for i in hidden_layers_sizes]) + \
               '.png'
    plt.savefig(filename)

    print >> sys.stderr, ('The training code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))














if __name__ == '__main__':
    hidden_layers = [500, ] * int(sys.argv[1])
    train_epc = int(sys.argv[2])
    test(training_epochs=train_epc, hidden_layers_sizes=hidden_layers)


