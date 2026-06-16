import numpy as np


def load_and_prepare_data(filename:str):
    """
    Load the galaxy dataset and prepare the feature matrix and binary class labels

    Returns
    -------
    features : ndarray, shape (m, 4)
        Matrix containing the four rescaled input features
        Each feature should have mean 0 and standard deviation 1

    labels : ndarray, shape (m,)
        A value of 1 corresponds to spiral galaxies
        A value of 0 corresponds to elliptical galaxies
    """
    data = np.loadtxt(filename)
    
    # extract the features and labels from the data
    features = data[:, :4]  
    labels = data[:, 4].astype(np.int8)  
    
    # rescale the features
    features -= np.mean(features, axis=1) # subtract the mean -> mean becomes 0
    features /= np.std(features, axis=1) # divide the std -> std becomes 1
    
    return features, labels



