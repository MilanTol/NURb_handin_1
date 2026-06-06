# imports
import os
import numpy as np
import matplotlib.pyplot as plt

from astropy.time import Time
from astropy.coordinates import solar_system_ephemeris
from astropy.coordinates import get_body_barycentric_posvel
import astropy.units as u


def get_initial_conditions(time:Time, bodies:list[str]) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate initial conditions for the Solar System using get_body_barycentric_posvel.

    Positions are in AU, while velocities are in AU/yr.

    Parameters
    ----------
    time : astropy.time.Time
        Time at which the initial conditions are evaluated.
    bodies : dictionary
        Dictionary containing names of bodies as keys

    Returns
    -------
    positions : ndarray
        Initial positions, shape (N_bodies, 3).
    velocities : ndarray
        Initial velocities, shape (N_bodies, 3).
    """
    # instantiate lists for positions and velocities
    positions = []
    velocities = []
    
    # loop over all bodies
    for body in bodies:
        # load in position and velocity of body
        with solar_system_ephemeris.set("jpl"):
            body_posvel = get_body_barycentric_posvel(body, time)
        body_pos = body_posvel[0]
        body_vel = body_posvel[1]
        # convert position units to AU
        x = body_pos.x.to_value(u.AU)
        y = body_pos.y.to_value(u.AU)
        z = body_pos.z.to_value(u.AU)
        body_pos = np.array([x, y, z])
        # convert velocity units to AU/yr
        v_x = body_vel.x.to_value(u.AU / u.yr)
        v_y = body_vel.y.to_value(u.AU / u.yr)
        v_z = body_vel.z.to_value(u.AU / u.yr)
        body_vel = np.array([v_x, v_y, v_z])
        # store position and velocity in respective lists
        positions.append(body_pos)
        velocities.append(body_vel)
        
    # convert position and velocitiy lists to arrays
    positions = np.array(positions)
    velocities = np.array(velocities)
    return positions, velocities



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
