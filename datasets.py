import numpy

def convert_to_onehot(truth_data):
    """
    truth_data is a numpy array.
    """
    labels = numpy.unique(truth_data)
    data = numpy.zeros((truth_data.shape[0], len(labels)))
    data[numpy.arange(truth_data.shape[0]), 
         truth_data.reshape(truth_data.shape[0])] = 1
    return data, labels
