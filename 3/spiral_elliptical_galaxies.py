import numpy as np
import matplotlib.pyplot as plt
import itertools
import os

from logistic_regression import cost, logistic_regression, test_logistic_regression
from data_processing import load_and_prepare_data

# Question 3: Spiral and elliptical galaxies


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
    ax[1, 0].set_yscale('log')
    ax[1, 1].hist(features[:, 3], bins=20)
    ax[1, 1].set(xlabel="Emission line flux")
    ax[1, 1].set_yscale('log')
    plt.savefig(os.path.join(output_dir, "fig3a.png"), dpi=300)
    plt.close()

    # Problem 3.b
    feature_combinations = [
        (0,1), 
        (0,2),
        (0,3),
        (1,2),
        (1,3),
        (2,3),
    ]
    cost_vals, thetas = logistic_regression(
        features, 
        labels,
        feature_combinations,
        learning_rate=0.1,
        n_iterations=5000,
    )  
    fig, ax = plt.subplots(1, 1, figsize=(10, 5), constrained_layout=True)
    for i, comb in enumerate(feature_combinations):
        ax.plot(
            np.arange(0, len(cost_vals)), 
            cost_vals[:, i], 
            label=f"Features {comb}"
        )

    ax.set(xlabel="Number of iterations", ylabel="Cost function")
    plt.legend(loc=(1.05, 0))
    plt.savefig(os.path.join(output_dir, "fig3b.png"), dpi=300)
    plt.close()

    # Problem 3.c

    # for every feature_combination, compute the test_statistics
    for i, comb in enumerate(feature_combinations):
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
            theta=thetas[i],
            feature_columns=comb,
            output_dir=output_dir,
        ) 
    
    # For every pair of features, plot the two features against each other and indicate the decision boundary
    fig, ax = plt.subplots(3, 2, figsize=(10, 15))
    names = [r"$\kappa_{CO}$", "Color", "Extended", "Emission line flux"]
    plot_idx = [[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [2, 1]]
    
    # loop over feature combinations
    for i, comb in enumerate(feature_combinations):
        
        # scatter plot the galaxies according to feature values,
        # color them according to true labels
        ax[plot_idx[i][0], plot_idx[i][1]].scatter(
            features[:, comb[0]], features[:, comb[1]], c=labels
        )
        # set xlabel and ylabel
        ax[plot_idx[i][0], plot_idx[i][1]].set(
            xlabel=names[comb[0]], ylabel=names[comb[1]]
        )
        
        # plot decision boundary
        x0_vals = np.array([features[:, comb[0]].min(), features[:, comb[0]].max()])
        theta = thetas[i]  # the theta corresponding to this feature combination
        # note that if theta@features = 0 then sigmoid(theta@features) returns 0.5,
        # this corresponds to the dicision boundary
        # so if feat2 = -1/theta2 * (bias + theta1*feat1)
        x1_vals = -(theta[-1] + theta[0] * x0_vals) / theta[1]
        ax[plot_idx[i][0], plot_idx[i][1]].plot(
            x0_vals, x1_vals, "k--"
        )
    plt.savefig(os.path.join(output_dir, "fig3c.png"), dpi=300)
    plt.close()

if __name__ == "__main__":
    main()
