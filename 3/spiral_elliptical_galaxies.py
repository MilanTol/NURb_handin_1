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
