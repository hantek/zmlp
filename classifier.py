"""
Here are some classifiers.
"""
import numpy
import theano
import theano.tensor as T
from sklearn.metrics import confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model import Model
from layer import SigmoidLayer


class Classifier(Model):
    def __init__(self, n_in, n_out, varin=None, vartruth=None):
        super(Classifier, self).__init__(n_in, n_out, varin=varin)
        if not vartruth:
            vartruth = T.ivector('truth')
        assert isinstance(vartruth, T.TensorVariable)
        self.vartruth = vartruth

    def output(self):
        """The input and output should always be theano variables."""
        raise NotImplementedError("Must be implemented by subclass.")

    
    # Following are for analysis ----------------------------------------------
    
    def analyze_performance(self, data, truth, verbose=True):
        """
        TODO: WRITEME
        
        data : numpy.ndarray
        truth : numpy.ndarray
        verbose : bool
        """
        assert data.shape[0] == truth.shape[0], "Data and truth shape mismatch."
        if not hasattr(self, '_get_output'):
            self._get_output = theano.function([self.varin], self.output())
        
        cm = confusion_matrix(truth, self.get_output(data))
        pr_a = cm.trace()*1.0 / test_truth.size
        pr_e = ((cm.sum(axis=0)*1.0 / test_truth.size) * \
            (cm.sum(axis=1)*1.0 / test_truth.size)).sum()
        k = (pr_a - pr_e) / (1 - pr_e)
        print "OA: %f, kappa index of agreement: %f" % (pr_a, k)
        if verbose: # Show confusion matrix
            if not hasattr(self, '_fig_confusion'):
                self._fig_confusion = plt.gcf()
                self.ax = self._fig_confusion.add_subplot(111)
                self.confmtx = self.ax.matshow(cm)
                plt.title("Confusion Matrix")
                self._fig_confusion.canvas.show()
            else:
                self.confmtx.set_data(cm)
                self._fig_confusion.canvas.draw()
            print "confusion matrix:"
            print cm


class LogisticRegression(Classifier):
    def __init__(self, n_in, n_out, varin=None, vartruth=None, 
                 init_w=None, init_b=None, npy_rng=None):
        super(LogisticRegression, self).__init__(n_in, n_out, 
                                                 varin=varin,
                                                 vartruth=vartruth)
        self.layer = SigmoidLayer(
            n_in, n_out, varin=varin, 
            init_w=init_w, init_b=init_b, npy_rng=npy_rng
        )
        self.params = self.layer.params
    
    def p_y_given_x(self):
        return T.nnet.softmax(self.layer.fanin())

    def cost(self):
        """
        y_truth : theano.tensor.TensorType
        The truth value of data. Usually y_truth = ivector('y_truth')
        """
        return -T.mean(
            T.log(self.p_y_given_x())[
                T.arange(self.vartruth.shape[0]), y]
        )

    def output(self):
        return T.argmax(self.p_y_given_x(), axis=1)


class Perceptron(Classifier):
    def __init__(self):
        #
        # TODO:
        #
        pass


class LinearRegression(Classifier):
    def __init__(self):
        #
        # TODO:
        #
        pass
