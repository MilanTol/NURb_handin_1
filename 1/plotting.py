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
        ax.plot(x[i, :], y[i, :], label=obj, linewidth=0.5)
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
    ax.set(xlim=(0, 300))
    ax.legend(fontsize=8)
    plt.legend(loc=(1.05, 0))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close(fig)


def plot_r_difference_vs_time(
    times: np.ndarray,
    positions_a: np.ndarray,
    positions_b: np.ndarray,
    body_names: list[str],
    output_dir: str,
    filename: str,
) -> None:
    """
    Plot difference in r positions between two integration methods.

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

    x1 = positions_a[:, :, 0]
    x2 = positions_b[:, :, 0]
    y1 = positions_a[:, :, 1]
    y2 = positions_b[:, :, 1]
    r1 = np.sqrt(x1 * x1 + y1 * y1)
    r2 = np.sqrt(x2 * x2 + y2 * y2)

    delta_x = (r2 - r1) / np.max(r1)  # compute relative difference in radius
    # note that for this we use the maximum of the radii the object attains with leapfrog.
    # since leapfrog is the more accurate integrator, this sets the typical scale for the object.
    fig, ax = plt.subplots(1, 1, figsize=(12, 5), constrained_layout=True)
    for i, obj in enumerate(body_names):
        ax.plot(times, delta_x[:, i], label=obj)
    ax.set(xlabel="Time [yr]", ylabel=r"$\delta r$", title="RK4 vs Leapfrog")
    ax.set(xlim=(0, 300))
    plt.legend(loc=(1.05, 0))
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close(fig)
