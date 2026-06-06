import numpy as np
import matplotlib.pyplot as plt
import itertools
import os

# Question 3: Spiral and elliptical galaxies


def load_and_prepare_data(filename):
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
    features = np.random.rand(100, 4)  # REPLACE, base on "data"
    labels = np.random.randint(0, high=2, size=100)  # REPLACE
    return features, labels


# Make your own implementation of logistic regression


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
    ]  # REPLACE


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


def main() -> None:
    output_dir = "Plots"
    os.makedirs(output_dir, exist_ok=True)

    # Problem 3.a
    features, labels = load_and_prepare_data("Data/galaxy_data.txt")
    np.savetxt(
        os.path.join(output_dir, "galaxy_data_scaled.txt"),
        features,
        header="kappa_CO Color Extended Emission_line_flux",
    )
    fig, ax = plt.subplots(2, 2, figsize=(10, 8))
    ax[0, 0].hist(features[:, 0], bins=20)
    ax[0, 0].set(ylabel="N", xlabel=r"$\kappa_{CO}$")
    ax[0, 1].hist(features[:, 1], bins=20)
    ax[0, 1].set(xlabel="Color")
    ax[1, 0].hist(features[:, 2], bins=20)
    ax[1, 0].set(ylabel="N", xlabel="Extended")
    ax[1, 1].hist(features[:, 3], bins=20)
    ax[1, 1].set(xlabel="Emission line flux")
    plt.savefig(os.path.join(output_dir, "fig3a.png"), dpi=300)
    plt.close()

    # Problem 3.b
    cost_function = np.random.rand(30, 2)  # REPLACE using the function above
    fig, ax = plt.subplots(1, 1, figsize=(10, 5), constrained_layout=True)
    ax.plot(np.arange(0, len(cost_function)), cost_function[:, 0], label="Features 1+2")
    ax.plot(
        np.arange(0, len(cost_function)), cost_function[:, 1], label="Features 1+3"
    )  # SAME IDEA FOR THE OTHER FEATURE COMBINATIONS
    # ...........
    ax.set(xlabel="Number of iterations", ylabel="Cost function")
    plt.legend(loc=(1.05, 0))
    plt.savefig(os.path.join(output_dir, "fig3b.png"), dpi=300)
    plt.close()

    # Problem 3.c

    # For every pair of features, plot the two features against each other and indicate the decision boundary
    (
        predictions,
        true_positive,
        false_positive,
        true_negative,
        false_negative,
        f1_score,
    ) = test_logistic_regression(
        features,
        labels,
        theta=np.random.rand(3),
        feature_columns=(0, 1),
        output_dir=output_dir,
    )  # REPLACE with the parameters corresponding to the trained model using features 1 and 2; then repeat for the other feature combinations
    fig, ax = plt.subplots(3, 2, figsize=(10, 15))
    names = [r"$\kappa_{CO}$", "Color", "Extended", "Emission line flux"]
    plot_idx = [[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [2, 1]]
    for i, comb in enumerate(itertools.combinations(np.arange(0, 4), 2)):
        ax[plot_idx[i][0], plot_idx[i][1]].scatter(
            features[:, comb[0]], features[:, comb[1]], c=labels
        )
        ax[plot_idx[i][0], plot_idx[i][1]].set(
            xlabel=names[comb[0]], ylabel=names[comb[1]]
        )
        ax[plot_idx[i][0], plot_idx[i][1]].plot(
            [0.5, 0.5], [0, 1], "k--"
        )  # REPLACE with the decision boundary from the trained logistic regression model
    plt.savefig(os.path.join(output_dir, "fig3c.png"), dpi=300)
    plt.close()


if __name__ == "__main__":
    main()
