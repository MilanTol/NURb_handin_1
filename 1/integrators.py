import numpy as np
from astropy.constants import G, M_sun
import astropy.units as u

# convert G to number in AU^3 / (M_sun yr^2)
G = G.to_value(u.AU**3 / (M_sun * u.yr**2))


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
    N_bodies = len(positions)

    # instantiate force matrix:
    f = np.zeros((N_bodies, N_bodies, 3))

    for i in range(N_bodies):  # loop over rows
        for j in range(i + 1, N_bodies):  # loop over columns
            # compute difference in position
            r_ij = positions[j] - positions[i]
            # compute absolute distance
            r2 = r_ij[0] * r_ij[0] + r_ij[1] * r_ij[1] + r_ij[2] * r_ij[2]
            r = np.sqrt(r2)
            f_ij = G * r_ij / (r * r * r)
            # fill f matrix using antisymmetric property
            f[i][j] = f_ij
            f[j][i] = -f_ij

    # create array which contains mass i in each row of the ith column
    m = masses[None, :, None]

    # create acceleration matrix
    A = m * f

    # The total acceleration the ith particle feels is then just the sum
    # over the ith column of A:
    accelerations = np.sum(A, axis=1)

    return accelerations


def leapfrog_integrator(
    positions_init: np.ndarray,
    velocities_init: np.ndarray,
    masses: np.ndarray,
    dt: float,
    N_steps: int,
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
    positions_arr : ndarray
        Positions at all time steps, shape (N_steps + 1, N_bodies, 3).
    velocities_arr : ndarray
        Velocities at all time steps, shape (N_steps + 1, N_bodies, 3).
    """
    # use the equations for xi+1 and vi+1/2 provided in class for the leapfrog algorithm,
    # do not forget to kick (i.e. apply the acceleration) your initial conditions for the velocity

    # instantiate lists in which positions and velocities are stored
    positions_list = []
    velocities_list = []
    positions_list.append(positions_init.copy())
    velocities_list.append(velocities_init.copy())

    velocities = velocities_init.copy()
    positions = positions_init.copy()
    
    # we first have to kick our velocities to v_i+1/2 from v_i
    # for this we will use RK4 on a step size of 0.5*dt
    dt = 0.5*dt # set dt to 0.5*dt
    
    # first we compute all the k coefficients 
    k1_v = dt * compute_accelerations(positions, masses)
    k1_x = dt * velocities

    k2_v = 0.5*dt * compute_accelerations(positions + 0.5*k1_x, masses)
    k2_x = 0.5*dt * (velocities + 0.5*k1_v)

    k3_v = 0.5*dt * compute_accelerations(positions + 0.5*k2_x, masses)
    k3_x = 0.5*dt * (velocities + 0.5*k2_v)

    k4_v = dt * compute_accelerations(positions + k3_x, masses)
    k4_x = dt * (velocities + k3_v)

    # combine coefficients to compute velocities at t + 0.5*dt
    velocities += (1 / 6) * (k1_v + 2*k2_v + 2*k3_v + k4_v)

    # set dt = 2*dt (so that it is back to the original dt)
    dt = 2*dt
    
    for step in range(N_steps):
        positions += dt * velocities  # drift
        accelerations = compute_accelerations(positions, masses)  # acceleration
        # compute approximate velocity for the full time-step
        velocities_full_time_step = velocities + 0.5 * dt * accelerations
        velocities += dt * accelerations  # kick

        # store positions
        positions_list.append(positions.copy())
        velocities_list.append(velocities_full_time_step.copy())

    positions_arr = np.array(positions_list)
    velocities_arr = np.array(velocities_list)

    return positions_arr, velocities_arr


def RK4_integrator(
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

    for step in range(N_steps):
        # first we compute all the k coefficients 
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
