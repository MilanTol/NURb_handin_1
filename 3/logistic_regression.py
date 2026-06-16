import numpy as np
import os


def sigmoid(z):
    return 1/(1+np.exp(-z))


def cost(theta:np.ndarray, features:np.ndarray, labels:np.ndarray)->float:
    """
    Returns cost function of a parametrization theta.

    Args:
        theta (np.ndarray): 
            parametrization
        features (np.ndarray): 
            features of the data
        labels (np.ndarray):
            true outcomes of the data

    Returns:
        (float):
            cost function value
    """
    components = (
        labels*np.log(sigmoid(theta.T@features))
        - (1-labels)*np.log( (1 - sigmoid(theta.T@features)) )
    )
    return -1/len(labels) * np.sum(components)


def logistic_regression(
    features, labels, feature_combinations, learning_rate=0.1, n_iterations=30
):
    """
    This function should select a chosen set
    of input feature columns, then fit a logistic regression model to classify
    galaxies as spirals or ellipticals.

    Parameters
    ----------
    features : ndarray, shape (m, 4)
        Rescaled feature matrix

    labels : ndarray, shape (m,)
        1 corresponds to spiral galaxies
        0 corresponds to elliptical galaxies

    feature_combinations : list of tuple
        example: [(0, 1), (0, 2)]

    learning_rate : float, optional
        Step size used in gradient descent

    n_iterations : int, optional
        Number of minimisation iterations

    Returns
    -------
    cost_function : ndarray, shape (n_iterations, n_combinations)
        Cost function values for every iteration and every feature combination

    theta_values : list of ndarray
        Best-fit parameters for each feature combination"""
    return np.random.rand(n_iterations, len(feature_combinations)), [
        np.random.rand(len(columns) + 1) for columns in feature_combinations
    ]  


def test_logistic_regression(features, labels, theta, feature_columns, output_dir):
    """
    Compute the number of true/false positives/negatives, as well as the F1 score, and save them for your report

    Parameters
    ----------
    features : ndarray, shape (m, 4)
        Rescaled feature matrix

    labels : ndarray, shape (m,)
        True binary class labels

    theta : ndarray
        Logistic regression parameters returned by logistic_regression()

    feature_columns : list or tuple
        Feature columns corresponding to parameters used by the trained model
    output_dir : str
        Directory where to save the results

    Returns
    -------
    predictions : ndarray, shape (m,)
        Predicted class labels

    true_positive : int

    false_positive : int

    true_negative : int

    false_negative : int

    f1_score : float
    """
    predictions = np.random.randint(0, 2, size=labels.shape)  # REPLACE
    true_positive = np.random.randint(0, 50)  # REPLACE
    false_positive = np.random.randint(0, 50)  # REPLACE
    true_negative = np.random.randint(0, 50)  # REPLACE
    false_negative = np.random.randint(0, 50)  # REPLACE

    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)

    f1_score = 2 * precision * recall / (precision + recall)

    # save txt
    with open(os.path.join(output_dir, "logistic_regression_metrics.txt"), "w") as f:
        f.write(f"True Positives: {true_positive}\n")
        f.write(f"False Positives: {false_positive}\n")
        f.write(f"True Negatives: {true_negative}\n")
        f.write(f"False Negatives: {false_negative}\n")
        f.write(f"F1 Score: {f1_score:.4f}\n")

    return (
        predictions,
        true_positive,
        false_positive,
        true_negative,
        false_negative,
        f1_score,
    )
