import os
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation
from matplotlib import animation

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