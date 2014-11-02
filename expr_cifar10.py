from layer import ZerobiasLayer
from classifier import LogisticRegression
import numpy
import cPickle
#############
# LOAD DATA #
#############
def unpickle(file):
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

crnt_dir = os.getcwd()
os.chdir('/home/hantek/data/cifar-10-batches-py')
for i in range(6):
    

os.chdir(crnt_dir)
###############
# BULID MODEL #
###############


model = Zero

#########
# TRAIN #
#########

