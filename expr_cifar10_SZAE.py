import os
import numpy
import theano
import theano.tensor as T
import cPickle

from layer import ZerobiasLayer, ReluLayer
from classifier import LogisticRegression
from model import ZerobiasAutoencoder
from dispims_color import dispims_color
from datasets import convert_to_onehot
import train
SMALL = 0.001

import pdb

#############
# LOAD DATA #
#############

def unpickle(file):
    fo = open(file, 'rb')
    dictionary = cPickle.load(fo)
    fo.close()
    return dictionary

crnt_dir = os.getcwd()
# os.chdir('/home/hantek/data/cifar-10-batches-py')
os.chdir('/data/lisa/data/cifar10/cifar-10-batches-py')
npy_rng = numpy.random.RandomState(123)
train_x_list = []
train_y_list = []
for i in ["1"]:  #["1", "2", "3", "4", "5"]:
    dicti = unpickle('data_batch_' + i)
    train_x_list.append(dicti['data'])
    train_y_list.append(dicti['labels'])
dicti = unpickle('test_batch')
test_x = dicti['data']
test_y = dicti['labels']
os.chdir(crnt_dir)

train_x = theano.shared(
    value = (
        numpy.concatenate(train_x_list) / 255.
    ).astype(theano.config.floatX),
    name = 'train_x',
    borrow = True
)
train_y = theano.shared(
    value = numpy.concatenate(train_y_list),
    name = 'train_y',
    borrow = True
)
train_y_onehot, _ = convert_to_onehot(train_y.get_value())
train_y_onehot = theano.shared(
    value = train_y_onehot.astype('int64'),
    name = 'y_onehot',
    borrow = True
)

test_x = theano.shared(
    value = (test_x / 255.).astype(theano.config.floatX),
    name = 'test_x',
    borrow = True
)

pdb.set_trace()
test_y = theano.shared(
    value = numpy.asarray(test_y).astype('int64'),
    name = 'test_y',
    borrow = True
)

###############
# BUILD MODEL #
###############

hid_layer_sizes = [1000, 1000]
model = ZerobiasAutoencoder(
    3072, hid_layer_sizes[0], 
    threshold=1., vistype='real', tie=True, npy_rng=npy_rng
)

for i in range(len(hid_layer_sizes)-1):
    model = model + ZerobiasAutoencoder(
        hid_layer_sizes[i], hid_layer_sizes[i+1],
        threshold=1., vistype='real', tie=True, npy_rng=npy_rng
    )
model = model + ReluLayer(hid_layer_sizes[-1], 10, npy_rng=npy_rng)


#############
# PRE-TRAIN #
#############
for i in range(len(model.models_stack)-1):
    trainer = train.GraddescentMinibatch(
        varin=model.varin, data=train_x, 
        cost=model.models_stack[i].cost(),
        params=model.models_stack[i].params_private,
        supervised=False,
        batchsize=100, learningrate=0.01, momentum=0.9, rng=npy_rng
    )

    for epoch in xrange(2):
        trainer.step()
        if epoch % 10 == 0 and epoch > 0:
            trainer.set_learningrate(trainer.learningrate*0.8)
            dispims_color(
                numpy.dot(
                    model.w.get_value().T, pca_forward.T
                ).reshape(-1, patchsize, patchsize, 3), 
                1
            )


#############
# FINE-TUNE #
#############
for imodel in model.models_stack[:-1]:
    imodel.threshold = 0. 

index = T.lscalar()
error_rate = theano.function(
    [index], 
    T.mean(T.neq(T.argmax(model.models_stack[-1].output(), axis=1), 
                 test_y)),
    givens = {model.models_stack[0].varin : test_x[index:]},
)
truth = T.lmatrix('truth')
trainer = train.GraddescentMinibatch(
    varin=model.varin, data=train_x, 
    truth=truth, truth_data=train_y_onehot,
    supervised=True,
    cost=T.mean(
        T.sum(model.models_stack[-1].output - truth, axis=1),
        axis=0
    ), 
    params=model.params,
    batchsize=100, learningrate=0.01, momentum=0.9, rng=npy_rng
)

for epoch in xrange(1000):
    trainer.step()
    print "    error rate: %f" % (error_rate(0))
    if epoch % 10 == 0 and epoch > 0:
        trainer.set_learningrate(trainer.learningrate*0.8)
        """
        dispims_color(
            numpy.dot(
                model.w.get_value().T, pca_forward.T
            ).reshape(-1, patchsize, patchsize, 3), 
            1
        )
        """
