import numpy as np
from b import compute_accelerations

def RK4(
    positions_init: np.ndarray,
    velocities_init: np.ndarray,
    masses: np.ndarray,
    dt: float,
    N_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the trajectories using 4th order Runge-Kutta.

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

    # instantiate lists in which positions and velocities are stored
    positions_list = []
    velocities_list = []
    positions_list.append(positions_init.copy())
    velocities_list.append(velocities_init.copy())

    velocities = velocities_init.copy()
    positions = positions_init.copy()

    # first we compute all the k coefficients 
    for step in range(N_steps):
        k1_v = dt * compute_accelerations(positions, masses)
        k1_x = dt * velocities

        k2_v = 0.5*dt * compute_accelerations(positions + 0.5*k1_x, masses)
        k2_x = 0.5*dt * (velocities + 0.5*k1_v)

        k3_v = 0.5*dt * compute_accelerations(positions + 0.5*k2_x, masses)
        k3_x = 0.5*dt * (velocities + 0.5*k2_v)

        k4_v = dt * compute_accelerations(positions + k3_x, masses)
        k4_x = dt * (velocities + k3_v)

        # combine coefficients to compute velocities and positions at t + dt
        velocities += (1 / 6) * (k1_v + 2*k2_v + 2*k3_v + k4_v)
        positions += (1 / 6) * (k1_x + 2*k2_x + 2*k3_x + k4_x)
        
        velocities_list.append(velocities.copy())
        positions_list.append(positions.copy())

    positions_arr = np.array(positions_list)
    velocities_arr = np.array(velocities_list)

    return positions_arr, velocities_arr
