# imports
import os
import numpy as np
import matplotlib.pyplot as plt

from astropy.time import Time


from a import get_initial_conditions, plot_initial_positions
from b import compute_accelerations, leapfrog_integrator
from c import another_integrator

bodies_with_masses = {
    "sun": 1.0,
    "mercury": 1.660e-7,
    "venus": 2.447e-6,
    "earth": 3.003e-6,
    "mars": 3.227e-7,
    "jupiter": 9.545e-4,
    "saturn": 2.858e-4,
    "uranus": 4.366e-5,
    "neptune": 5.151e-5,
}
# in M_sun

# Question 1: Simulating the solar system

##### Plots #####

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
    x, y, z = np.random.rand(3, 9, 10) * 10 - 5  # REPLACE
    time = x.copy() * 0 + np.linspace(0, 200, 10)  # REPLACE
    fig, ax = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for i, obj in enumerate(body_names):
        ax[0].plot(x[i, :], y[i, :], label=obj)
        ax[1].plot(time[i, :], z[i, :], label=obj)
    ax[0].set_aspect("equal", "box")
    ax[0].set(xlabel="X [AU]", ylabel="Y [AU]")
    ax[1].set(xlabel="Time [yr]", ylabel="Z [AU]")
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

    x, y, z = np.random.rand(3, 9, 10) * 10 - 5  # REPLACE
    time = x.copy() * 0 + np.linspace(0, 200, 10)  # REPLACE
    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    for i, obj in enumerate(body_names):
        ax.plot(time[i, :], z[i, :], label=obj)
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

    x, y, z = np.random.rand(3, 9, 10) * 10 - 5  # REPLACE
    time = x.copy() * 0 + np.linspace(0, 200, 10)  # REPLACE
    fig, ax = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for i, obj in enumerate(np.flip(body_names)):
        ax[0].plot(time[i, :], z[i, :], label=obj)
        ax[1].plot(time[i, :], z[i, :], label=obj)
    ax[0].set(xlabel="Time [yr]", ylabel="Z [AU]", title="Leapfrog")
    ax[1].set(xlabel="Time [yr]", ylabel="Z [AU]", title="Other method")
    plt.legend(loc=(1.05, 0))
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close(fig)



def main() -> None:
    output_dir = "Plots"
    os.makedirs(output_dir, exist_ok=True)

    body_names = list(bodies_with_masses.keys())
    masses = np.array(list(bodies_with_masses.values()))

    # current time
    time = Time.now()

    positions_init, velocities_init = get_initial_conditions(time=time, bodies=body_names)

    # (a)
    plot_initial_positions(
        positions=positions_init,
        body_names=body_names,
        output_dir=output_dir,
        filename="initial_positions.png",
    )

    # Parameters for the exercise
    total_time = 300.0  # years
    dt = 0.8 / 365.25  # 0.8 days in years
    N_steps = int(total_time / dt)

    times = np.arange(N_steps + 1) * dt

    # (b)
    positions_lf, velocities_lf = leapfrog_integrator(
        positions_init=positions_init,
        velocities_init=velocities_init,
        masses=masses,
        dt=dt,
        N_steps=N_steps,
    )

    plot_orbits_xy(
        positions=positions_lf,
        body_names=body_names,
        output_dir=output_dir,
        filename="orbits_xy_leapfrog.png",
    )

    z_vs_time(
        times=times,
        positions=positions_lf,
        body_names=body_names,
        output_dir=output_dir,
        filename="z_vs_time_leapfrog.png",
    )
    # (c)
    positions_another, velocities_another = another_integrator(
        positions_init=positions_init,
        velocities_init=velocities_init,
        masses=masses,
        dt=dt,
        N_steps=N_steps,
    )

    plot_orbits_xy(
        positions=positions_another,
        body_names=body_names,
        output_dir=output_dir,
        filename="orbits_xy_another_method.png",
    )

    z_vs_time(
        times=times,
        positions=positions_another,
        body_names=body_names,
        output_dir=output_dir,
        filename="z_vs_time_another_method.png",
    )

    plot_x_difference_vs_time(
        times=times,
        positions_a=positions_lf,
        positions_b=positions_another,
        body_names=body_names,
        output_dir=output_dir,
        filename="x_difference_another_method_minus_leapfrog.png",
    )

    # (d): optional

    # movie_path = make_movie_with_matplotlib(
    #     positions=positions_lf,
    #     body_names=body_names,
    #     output_dir="Plots",
    #     frame_interval=10,
    #     movie_name="solar_system_movie.mp4",
    #     fps=30,
    # )


if __name__ == "__main__":
    main()
