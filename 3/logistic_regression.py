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
        labels*np.log(sigmoid(theta@features.T))
        + (1-labels)*np.log( (1 - sigmoid(theta@features.T)) )
    )
    return -1/len(labels) * np.sum(components)


def cost_gradient(theta:np.ndarray, features:np.ndarray, labels:np.ndarray)->np.ndarray:
    """
    Returns the gradient of the cost function

    Args:
        theta (np.ndarray): 
            parametrization
        features (np.ndarray): 
            features of the data
        labels (np.ndarray):
            true outcomes of the data

    Returns:
        (np.ndarray):
            gradient of cost function
    """
    return 1/len(labels) * features.T @ ( sigmoid(theta@features.T) - labels )


def logistic_regression_single_feature_combination(
    features:np.ndarray, labels:np.ndarray, learning_rate=0.1, n_iterations=30
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

    learning_rate : float, optional
        Step size used in gradient descent

    n_iterations : int, optional
        Number of minimisation iterations

    Returns
    -------
    cost_function : ndarray, shape (n_iterations,)
        Cost function values for every iteration 

    theta : list of ndarray, shape(n_features+1)
        best fit_values for theta.
        This also includes the bias, hence the shape is n_features+1.
        The bias is given in the last entry: bias = theta[-1]
    """
        
    # add bias to the features  
    cost_vals = [] # initialize a list in which to store cost values
    theta = np.ones_like(features[0]) # initialize all theta vals to 1
    
    for i in range(n_iterations): 
        # update theta_value using gradient descent
        theta -= learning_rate*cost_gradient(theta, features, labels)
        # store cost value
        cost_vals.append(cost(theta, features, labels))
        
    return np.array(cost_vals), theta



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

    theta_values : list of ndarray, shape(n_combinations, 1+n_features)
        Best-fit parameters for each feature combination
    """
    
    # add bias to features, set them inside the first column
    features = np.hstack([np.ones((features.shape[0],1)), features])

    cost_vals = []
    thetas = []
    for feature_combination in feature_combinations:
        # add bias to feature_combinations:
        feature_combination = (0,) + feature_combination
        
        features_temp = features[:,feature_combination]
        cost_vals_temp, theta = logistic_regression_single_feature_combination(features_temp, labels, learning_rate, n_iterations)
        cost_vals.append(cost_vals_temp)
        thetas.append(theta)
    
    # transpose cost_vals to get shape (n_iterations, n_combinations)
    return np.array(cost_vals).T, thetas


def test_logistic_regression(features:np.ndarray, labels:np.ndarray, theta:np.ndarray, feature_columns, output_dir):
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
    
    # add bias to the features
    features = np.hstack([np.ones((features.shape[0],1)), features])
    # add inclusion of bias in feature_columns:
    # add 0 index to feature_columns and shift all indices by 1.
    feature_columns = (0,) + feature_columns
        
    sigmoid_vals = sigmoid(theta@features[:,feature_columns].T)
    predictions = np.array([sigmoid_vals > 0.5])
    
    true_positive = np.sum((predictions==1) & (labels==1))
    false_positive = np.sum((predictions==1) & (labels==0))
    false_negative = np.sum((predictions==0) & (labels==1))
    true_negative = np.sum((predictions==0) & (labels==0))

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
