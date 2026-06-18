import os
import numpy as np
import matplotlib.pyplot as plt


def plot_initial_positions(
    positions: np.ndarray,
    body_names: list[str],
    output_dir: str,
    filename: str,
) -> None:
    """
    Plot initial positions in the (x,y) and (x,z) planes.

    Parameters
    ----------
    positions : ndarray
        Initial positions, shape (N_bodies, 3)
    body_names : list of str
        Names of the bodies
    output_dir : str
        Directory to save the plots
    """
    # convert positions shape (Nbodies, 3) -> (3, Nbodies)
    # then extract positions
    x, y, z = positions.T
    fig, ax = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for i, obj in enumerate(body_names):
        ax[0].scatter(x[i], y[i], label=obj)
        ax[1].scatter(x[i], z[i], label=obj)
    ax[0].set_aspect("equal", "box")
    ax[1].set_aspect("equal", "box")
    ax[0].set(xlabel="X [AU]", ylabel="Y [AU]")
    ax[1].set(xlabel="X [AU]", ylabel="Z [AU]")
    plt.legend(loc=(1.05, 0))
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close(fig)


def plot_orbits_xy(
    positions: np.ndarray,
    body_names: list[str],
    output_dir: str,
    filename: str,
) -> None:
    """
    Plot the orbits of all bodies in the x-y plane.

    Parameters
    ----------
    positions : ndarray
        Positions at all time steps, shape (N_steps, N_bodies, 3)
    body_names : list of str
        Names of the bodies
    output_dir : str
        Directory where plot is saved
    """

    # For visibility, you may want to do two versions of this plot:
    # one with all planets, and another zoomed in on the four inner planets
    x, y, z = positions.T
    fig, ax = plt.subplots(1, 1, figsize=(6, 5), constrained_layout=True)
    for i, obj in enumerate(body_names):
        ax.plot(x[i, :], y[i, :], label=obj)
    ax.set_aspect("equal", "box")
    ax.set(xlabel="X [AU]", ylabel="Y [AU]")
    plt.legend(loc=(1.05, 0))
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close(fig)


def z_vs_time(
    times: np.ndarray,
    positions: np.ndarray,
    body_names: list[str],
    output_dir: str,
    filename: str,
) -> None:
    """
    Plot z position as a function of time.

    Parameters
    ----------
    times : ndarray
        Times, shape (N_steps,)
    positions : ndarray
        Positions, shape (N_steps, N_bodies, 3)
    body_names : list of str
        Names of the bodies
    output_dir : str
        Directory where to save the plot
    filename : str
        Output filename
    """

    x, y, z = positions.T
    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    for i, obj in enumerate(body_names):
        ax.plot(times, z[i, :], label=obj)
    ax.set(
        xlabel="Time [yr]",
        ylabel="z [AU]",
        title="z position as a function of time",
    )
    ax.legend(fontsize=8)
    plt.legend(loc=(1.05, 0))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close(fig)


def plot_x_difference_vs_time(
    times: np.ndarray,
    positions_a: np.ndarray,
    positions_b: np.ndarray,
    body_names: list[str],
    output_dir: str,
    filename: str,
) -> None:
    """
    Plot difference in x positions between two integration methods.

    Parameters
    ----------
    times : ndarray
        Times, shape (N_steps,)
    positions_a : ndarray
        Positions from first method, shape (N_steps, N_bodies, 3)
    positions_b : ndarray
        Positions from second method, shape (N_steps, N_bodies, 3)
    body_names : list of str
        Names of the bodies
    output_dir : str
        Directory where plot is saved
    filename : str
        Output filename
    """

    x1 = positions_a.T[0]
    delta_x1 = (x1 - x1[0]) 
    x2 = positions_b.T[0]
    delta_x2 = (x1 - x2)/(x1+x2) #(x2 - x2[0]) 
    fig, ax = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for i, obj in enumerate(np.flip(body_names)):
        ax[0].plot(times, delta_x1[i, :], label=obj)
        ax[1].plot(times, delta_x2[i, :], label=obj)
    ax[0].set(xlabel="Time [yr]", ylabel="Z [AU]", title="Leapfrog")
    ax[1].set(xlabel="Time [yr]", ylabel="Z [AU]", title="Other method")
    plt.legend(loc=(1.05, 0))
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close(fig)
