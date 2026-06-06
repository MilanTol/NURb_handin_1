# imports
import os
import numpy as np
import matplotlib.pyplot as plt

from astropy.time import Time

from matplotlib.animation import FuncAnimation
from matplotlib import animation

from a import get_initial_conditions, plot_initial_positions

G = 4.0 * np.pi**2  # in AU^3 / (M_sun yr^2)

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


def compute_accelerations(
    positions: np.ndarray,
    masses: np.ndarray,
) -> np.ndarray:
    """
    Compute the accelerations using equation 1 in the hand in.

    Parameters
    ----------
    positions : ndarray
        Positions of all bodies, shape (N_bodies, 3)
    masses : ndarray
        Masses of all bodies, shape (N_bodies,)

    Returns
    -------
    accelerations : ndarray
        Accelerations of all bodies, shape (N_bodies, 3)
    """

    # for i in range(N_bodies):
    #     for j in range(i + 1, N_bodies):
    # TODO: implement this function using the description in the hand in
    return np.zeros_like(positions)


def leapfrog_integrator(
    positions_init: np.ndarray,
    velocities_init: np.ndarray,
    masses: np.ndarray,
    dt: float,  # use 0.8 days
    N_steps: int,  # 300 years / 0.8 days
) -> tuple[np.ndarray, np.ndarray]:
    """
    The leapfrog integrator.

    Parameters
    ----------
    positions_init : ndarray
        Initial positions, shape (N_bodies, 3).
    velocities_init : ndarray
        Initial velocities, shape (N_bodies, 3).
    masses : ndarray
        Masses of the bodies, shape (N_bodies,).
    dt : float
        Time step in years.
    N_steps : int
        Number of integration steps.

    Returns
    -------
    positions : ndarray
        Positions at all time steps, shape (N_steps + 1, N_bodies, 3).
    velocities : ndarray
        Velocities at all time steps, shape (N_steps + 1, N_bodies, 3).
    """
    # use the equations for xi+1 and vi+1/2 provided in class for the leapfrog algorithm,
    # do not forget to kick (i.e. apply the acceleration) your initial conditions for the velocity

    # for step in range(N_steps):
    # drift
    # acceleration
    # kick
    # store approximate velocity for the full time-step
    return np.zeros((N_steps + 1, len(masses), 3)), np.zeros(
        (N_steps + 1, len(masses), 3)
    )


def another_integrator(
    positions_init: np.ndarray,
    velocities_init: np.ndarray,
    masses: np.ndarray,
    dt: float,
    N_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the trajectories using another integration method.

    Parameters
    ----------
    positions_init : ndarray
        Initial positions, shape (N_bodies, 3)
    velocities_init : ndarray
        Initial velocities, shape (N_bodies, 3)
    masses : ndarray
        Masses of the bodies, shape (N_bodies,)
    dt : float
        Time step in years
    N_steps : int
        Number of integration steps

    Returns
    -------
    positions : ndarray
        Positions at all time steps, shape (N_steps + 1, N_bodies, 3)
    velocities : ndarray
        Velocities at all time steps, shape (N_steps + 1, N_bodies, 3)
    """
    # TODO: implement this function using a different integration method than leapfrog, e.g. RK4
    return np.zeros((N_steps + 1, len(masses), 3)), np.zeros(
        (N_steps + 1, len(masses), 3)
    )


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


def make_movie_with_matplotlib(
    positions: np.ndarray,
    body_names: list[str],
    output_dir: str,
    frame_interval: int,
    movie_name: str = "solar_system_movie.mp4",
    fps: int = 30,
) -> str:
    """
    Create a Solar System animation directly with Matplotlib.

    Parameters
    ----------
    positions : ndarray
        Positions, shape (N_steps, N_bodies, 3)
    body_names : list of str
        Names of the bodies
    output_dir : str
        Directory where the movie is saved
    frame_interval : int
        Number of simulation steps between animation frames
    movie_name : str, optional
        Name of the output movie file
    fps : int, optional
        Frames per second of the output movie

    Returns
    -------
    movie_path : str
        Path to the created movie
    """
    os.makedirs(output_dir, exist_ok=True)
    movie_path = os.path.join(output_dir, movie_name)

    positions_plot = positions.copy()  # shape (N_steps, N_bodies, 3)
    positions_plot = (
        positions_plot - positions[:, 0, :][:, None, :]
    )  # shift relative to the Sun

    frame_steps = list(range(0, positions.shape[0], frame_interval))

    fig, ax = plt.subplots()

    scatters = []
    for name in body_names:
        scatter = ax.scatter([], [], label=name)
        scatters.append(scatter)

    ax.set(
        xlabel="x [AU]",
        ylabel="y [AU]",
        title="Solar System movie",
        aspect="equal",
        xlim=(-35, 35),
        ylim=(-35, 35),
    )
    ax.legend(fontsize=8)
    plt.tight_layout()

    def init():
        for scatter in scatters:
            scatter.set_offsets(np.empty((0, 2)))
        return scatters

    def update(frame_idx):
        step = frame_steps[frame_idx]

        for i, scatter in enumerate(scatters):
            x = positions_plot[step, i, 0]
            y = positions_plot[step, i, 1]
            scatter.set_offsets(np.array([[x, y]]))

        ax.set_title(f"Solar System, step = {step}")
        return scatters

    ani = FuncAnimation(
        fig,
        update,
        frames=len(frame_steps),
        init_func=init,
        blit=False,
    )

    writer = animation.FFMpegWriter(fps=fps)
    ani.save(movie_path, writer=writer, dpi=150)

    plt.close(fig)

    return movie_path


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
