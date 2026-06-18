# imports
import os
import numpy as np
import matplotlib.pyplot as plt

from astropy.time import Time


from a import get_initial_conditions
from b import leapfrog_integrator
from c import RK4
from d import make_movie_with_matplotlib
from plotting import (
    plot_initial_positions,
    plot_orbits_xy,
    z_vs_time,
    plot_x_difference_vs_time,
)

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


def main() -> None:
    output_dir = "Plots"
    os.makedirs(output_dir, exist_ok=True)

    body_names = list(bodies_with_masses.keys())
    masses = np.array(list(bodies_with_masses.values()))

    # current time
    time = Time.now()

    positions_init, velocities_init = get_initial_conditions(
        time=time, bodies=body_names
    )

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
    dt = 8 / 365.25  # 8 days in years
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
    positions_another, velocities_another = RK4(
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

    # (d): 
        
    # we must choose our frame_interval such that we do not exceed 30s
    # the total frames is given by the length of positions_lf
    # the total duration of our move is then given by:
    # T = frames / (frame_interval * fps)
    # so frame_interval =  frames / (T*fps)
    fps = 24 # frames per second
    T = 20 # seconds
    frames = len(positions_lf)
    
    frame_interval = frames // (fps*T)
    
    movie_path = make_movie_with_matplotlib(
        positions=positions_lf,
        body_names=body_names,
        output_dir="Plots",
        frame_interval=frame_interval,
        movie_name="solar_system_movie.mp4",
        fps=fps,
    )


if __name__ == "__main__":
    main()
