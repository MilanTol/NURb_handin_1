import numpy as np


def load_and_prepare_data(filename: str):
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

    # note that the features in the 3rd and 4th column live in logspace:
    features[:, 2] = np.log(features[:, 2] - np.min(features[:, 2]) + 1)
    features[:, 3] = np.log(features[:, 3] - np.min(features[:, 3]) + 1)

    # artificially remove the outlier of the line emission by clipping the data.
    # in this galaxy there is some absorption going on that is ruining the rest of the data.
    # we will clip it by demanding that it is within some typical distance to the median
    # compute the median OFFSET to the median:
    median = np.median(features[:, 3])
    median_offset = np.median(np.abs(features[:, 3] - median))
    tolerance = 100
    lower_bound = median - tolerance * median_offset
    upper_bound = median + tolerance * median_offset
    features[:, 3] = np.clip(features[:, 3], lower_bound, upper_bound)

    # rescale the features
    features -= np.mean(features, axis=0)  # subtract the mean -> mean becomes 0
    features /= np.std(features, axis=0)  # divide the std -> std becomes 1

    return features, labels


def load_and_prepare_data_old(filename: str):
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
    features -= np.mean(features, axis=0)  # subtract the mean -> mean becomes 0
    features /= np.std(features, axis=0)  # divide the std -> std becomes 1

    return features, labels
