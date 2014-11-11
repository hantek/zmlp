import os
import gzip
import numpy
import theano
import theano.tensor as T
import cPickle

from layer import SigmoidLayer
from classifier import LogisticRegression
from train import GraddescentMinibatch, FeedbackAlignment
from datasets import convert_to_onehot

import pdb

def load_data(dataset):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''

    #############
    # LOAD DATA #
    #############

    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        import urllib
        origin = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        print 'Downloading data from %s' % origin
        urllib.urlretrieve(origin, dataset)

    print '... loading data'

    # Load the dataset
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    # train_set, valid_set, test_set format: tuple(input, target)
    # input is an numpy.ndarray of 2 dimensions (a matrix)
    # witch row's correspond to an example. target is a
    # numpy.ndarray of 1 dimensions (vector)) that have the same length as
    # the number of rows in the input. It should give the target
    # target to the example with the same index in the input.

    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(
            numpy.asarray(data_x, dtype=theano.config.floatX),
            borrow=borrow
        )
        shared_y = theano.shared(
            numpy.asarray(data_y, dtype='int64'),
            borrow=borrow
        )
        return shared_x, shared_y

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval









#############
# LOAD DATA #
#############

# datasets = load_data('/home/hantek/data/mnist.pkl.gz')
datasets = load_data('/data/lisa/data/mnist.pkl.gz')

train_set_x, train_set_y = datasets[0]
valid_set_x, valid_set_y = datasets[1]
test_set_x, test_set_y = datasets[2]

train_set_y_onehot, _ = convert_to_onehot(train_set_y.get_value())
train_set_y_onehot = theano.shared(train_set_y_onehot.astype('int64'),
                                   name='y_onehot')
npy_rng = numpy.random.RandomState(123)

######################
# FEEDBACK ALIGNMENT #
######################

model_fba = SigmoidLayer(
    784, 500, npy_rng = npy_rng
) + SigmoidLayer(
    500, 10, npy_rng=npy_rng
)

index = T.lscalar()
error_rate_fba = theano.function(
    [index], 
    T.mean(T.neq(T.argmax(model_fba.models_stack[-1].output(), axis=1), test_set_y)),
    givens = {model_fba.models_stack[0].varin : test_set_x[index:]},
)
"""
Build the model in this way doesn't work quite well. Cost jumps to zero befor
the error rate become smaller than 0.5. But error rate keeps decreasing 
anyway.

model_fba = SigmoidLayer(
    784, 500, npy_rng = npy_rng
) + LogisticRegression(
    500, 10, npy_rng=npy_rng
)

index = T.lscalar()
error_rate_fba = theano.function(
    [index], 
    T.mean(T.neq(model_fba.models_stack[-1].predict(), test_set_y)),
    givens = {model_fba.models_stack[0].varin : test_set_x[index:]},
)
"""

print "Begin Feedback Alignment"

fb_trainer = FeedbackAlignment(
    model=model_fba, 
    data=train_set_x, truth_data=train_set_y_onehot,
    batchsize=1, learningrate=0.02, 
    rng=npy_rng, verbose=True
)
for epoch in xrange(5):
    fb_trainer.step()
    print "    error rate: %f" % (error_rate_fba(0))


#############
# BACK-PROP #
#############

model = SigmoidLayer(
    784, 500, npy_rng = npy_rng
) + LogisticRegression(
    500, 10, npy_rng=npy_rng
)

error_rate = theano.function(
    [index], 
    T.mean(T.neq(model.models_stack[-1].predict(), test_set_y)),
    givens = {model.models_stack[0].varin : test_set_x[index:]},
)

print "\n\nBegin normal backprop"
bp_trainer = GraddescentMinibatch(
    varin=model.varin, data=train_set_x, 
    truth=model.models_stack[-1].vartruth, truth_data=train_set_y,
    supervised=True, cost=model.models_stack[-1].cost(),
    params=model.params,
    batchsize=1, learningrate=0.02, momentum=0., 
    rng=npy_rng
)
for epoch in xrange(5):
    bp_trainer.step()
    print "    error rate: %f" % (error_rate(0))
