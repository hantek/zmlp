import os
import numpy
import theano
import cPickle

from layer import ZerobiasLayer
from classifier import LogisticRegression
from model import ZerobiasAutoencoder
from dispims_color import dispims_color
import train
SMALL = 0.001


def unpickle(file):
    fo = open(file, 'rb')
    dictionary = cPickle.load(fo)
    fo.close()
    return dictionary

def crop_patches_color(image, keypoints, patchsize):
    patches = numpy.zeros((len(keypoints), 3*patchsize**2))
    for i, k in enumerate(keypoints):
        patches[i, :] = image[
            k[0] - patchsize/2 : k[0] + patchsize/2, 
            k[1] - patchsize/2 : k[1] + patchsize/2,
            :
        ].flatten()
    return patches


def pca(data, var_fraction, whiten=True):
    """ principal components analysis of data (columnwise in array data), retai-
    ning as many components as required to retain var_fraction of the variance 
    """
    u, v = numpy.linalg.eigh(numpy.cov(data, rowvar=0, bias=1))
    v = v[:, numpy.argsort(u)[::-1]]
    u.sort()
    u = u[::-1]
    u = u[u.cumsum()<=(u.sum()*var_fraction)]
    numprincomps = u.shape[0]
    u[u<SMALL] = SMALL
    if whiten: 
        backward_mapping = (
            (u**(-0.5))[:numprincomps][numpy.newaxis, :] * v[:, :numprincomps]
        ).T
        forward_mapping = (u**0.5)[:numprincomps][numpy.newaxis, :] \
                        * v[:, :numprincomps]
    else: 
        backward_mapping = v[:,:numprincomps].T
        forward_mapping = v[:,:numprincomps]
    return backward_mapping, forward_mapping, \
           numpy.dot(v[:,:numprincomps], backward_mapping), \
           numpy.dot(forward_mapping, v[:,:numprincomps].T)


#############
# LOAD DATA #
#############

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
os.chdir(crnt_dir)

train_x = (
    numpy.concatenate(train_x_list).reshape(-1, 3, 32, 32) / 255.
).astype(theano.config.floatX)[:1000]

train_y = (numpy.concatenate(train_y_list))[:1000]

#CROP PATCHES
print "cropping patches"
patchsize = 12
trainpatches = numpy.concatenate([
    crop_patches_color(
        im.reshape(3, 32, 32).transpose(1,2,0), 
        numpy.array([numpy.random.randint(patchsize/2, 32-patchsize/2, 40), 
                     numpy.random.randint(patchsize/2, 32-patchsize/2, 40)]).T, 
        patchsize) for im in train_x
])
R = npy_rng.permutation(trainpatches.shape[0])
trainpatches = trainpatches[R, :]
train_y = train_y.repeat(40)
print "numpatches: ", trainpatches.shape[0]
print "done"

#LEARN WHITENING MATRICES 
print "whitening"
meanstd = trainpatches.std()
trainpatches -= trainpatches.mean(1)[:,None]  # subtract mean of each patch
# The latter term is just for avoiding zero devidor error 
trainpatches /= trainpatches.std(1)[:,None] + 0.1 * meanstd  
trainpatches_mean = trainpatches.mean(0)[None,:]
trainpatches_std = trainpatches.std(0)[None,:] 
trainpatches -= trainpatches_mean  # subtract mean of each pixel 
trainpatches /= trainpatches_std + 0.1 * meanstd
pca_backward, _, _, _ = pca(trainpatches, 0.9, whiten=True)
trainpatches_whitened = numpy.dot(
    trainpatches, pca_backward.T
).astype(theano.config.floatX)

###############
# BUILD MODEL #
###############

trainpatches_theano = theano.shared(value=trainpatches_whitened,
                                    name='input_x_data',
                                    borrow=True)
train_y_theano = theano.shared(value=train_y, 
                               name='target_y_data',
                               borrow=True)

model = ZerobiasAutoencoder(
    43, 100, threshold=1., vistype='real', tie=True, npy_rng=npy_rng
) + LogisticRegression(
    100, 10, npy_rng=npy_rng
)

#############
# PRE-TRAIN #
#############

# DO SOME STEPS WITH SMALL LEARNING RATE TO MAKE SURE THE INITIALIZATION IS IN 
# A REASONABLE RANGE
trainer = train.GraddescentMinibatch(
    varin=model.varin, data=trainpatches_theano, 
    cost=model.models_stack[0].cost(),
    params=model.models_stack[0].params_private,
    supervised=False,
    batchsize=100, learningrate=0.0001, momentum=0.9, rng=npy_rng
)
trainer.step(); trainer.step(); trainer.step()

# TRAIN THE MODEL FOR REAL, AND SHOW FILTERS
trainer = train.GraddescentMinibatch(
    varin=model.varin, data=trainpatches_theano, 
    cost=model.models_stack[0].cost(),
    params=model.models_stack[0].params_private,
    supervised=False,
    batchsize=100, learningrate=0.01, momentum=0.9, rng=npy_rng
)

for epoch in xrange(10):
    trainer.step()
    if epoch % 10 == 0 and epoch > 0:
        trainer.set_learningrate(trainer.learningrate*0.8)
        dispims_color(
            numpy.dot(
                model.W.get_value().T, pca_forward.T
            ).reshape(-1, patchsize, patchsize, 3), 
            1
        )
        pylab.draw(); pylab.show()

#############
# FINE-TUNE #
#############
trainer = train.GraddescentMinibatch(
    varin=model.varin, data=trainpatches_theano, 
    cost=model.models_stack[-1].cost(),
    params=model.params,
    truth=model.models_stack[-1].vartruth,
    truth_data=train_y_theano,
    supervised=True,
    batchsize=100, learningrate=0.01, momentum=0.9, rng=npy_rng
)

for epoch in xrange(10):
    trainer.step()
    if epoch % 10 == 0 and epoch > 0:
        trainer.set_learningrate(trainer.learningrate*0.8)
        dispims_color(
            numpy.dot(
                model.W.get_value().T, pca_forward.T
            ).reshape(-1, patchsize, patchsize, 3), 1)
        pylab.draw(); pylab.show()
