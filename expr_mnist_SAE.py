import os
import numpy
import theano
import theano.tensor as T

from model import ClassicalAutoencoder
from classifier import LogisticRegression
from train import GraddescentMinibatch
from dataset import load_mnist


#############
# LOAD DATA #
#############

train_set, valid_set, test_set = load_mnist()
train_x, train_y = train_set
test_x, test_y = test_set
train_x = theano.shared(
    value = train_x.astype(theano.config.floatX),
    name = 'train_x',
    borrow = True
)
train_y = theano.shared(
    value = train_y.astype('int64'),
    name = 'train_y',
    borrow = True
)
test_x = theano.shared(
    value = test_x.astype(theano.config.floatX),
    name = 'test_x',
    borrow = True
)
test_y = theano.shared(
    value = test_y.astype('int64'),
    name = 'test_y',
    borrow = True
)
npy_rng = numpy.random.RandomState(123)

###############
# BUILD MODEL #
###############

model = ClassicalAutoencoder(
    784, 1000, vistype = 'binary', npy_rng = npy_rng
) + ClassicalAutoencoder(
    1000, 1000, vistype = 'binary', npy_rng = npy_rng
) + ClassicalAutoencoder(
    1000, 1000, vistype = 'binary', npy_rng = npy_rng
) + LogisticRegression(
    1000, 10, npy_rng = npy_rng
)
model.print_layer()

error_rate = theano.function(
    [], 
    T.mean(T.neq(model.models_stack[-1].predict(), test_set_y)),
    givens = {model.models_stack[0].varin : test_set_x},
)

#############
# PRE-TRAIN #
#############

for i in range(len(model.models_stack)-1):
    print "\n\nPre-training layer %d:" % i
    trainer = GraddescentMinibatch(
        varin=model.varin, data=train_set_x,   #theano.shared(train_set_x.get_value()[:1000, :].astype(theano.config.floatX)),  #
        cost=model.models_stack[i].cost(),
        params=model.models_stack[i].params_private,
        supervised=False,
        batchsize=1, learningrate=0.001, momentum=0., rng=npy_rng
    )

    for epoch in xrange(15):
        trainer.step()
        # model.models_stack[i].encoder().draw_weight()

#############
# FINE-TUNE #
#############

print "\n\nBegin fine-tune: normal backprop"
bp_trainer = GraddescentMinibatch(
    varin=model.varin, data=train_set_x, 
    truth=model.models_stack[-1].vartruth, truth_data=train_set_y,
    supervised=True, cost=model.models_stack[-1].cost(),
    params=model.params,
    batchsize=1, learningrate=0.1, momentum=0., 
    rng=npy_rng
)
for epoch in xrange(1000):
    bp_trainer.step()
    print "    error rate: %f" % (error_rate())
