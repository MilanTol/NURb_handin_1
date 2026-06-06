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
    
    for i in range(N_bodies): # loop over rows
        for j in range(i + 1, N_bodies): # loop over columns
            # compute difference in position
            r_ij = positions[j] - positions[i]
            # compute absolute distance
            r2 = r_ij[0]*r_ij[0] + r_ij[1]*r_ij[1] + r_ij[2]*r_ij[2]
            r = np.sqrt(r2)
            f_ij = G * r_ij / (r*r*r)
            # fill f matrix using antisymmetric property
            f[i][j] = f_ij
            f[j][i] = -f_ij
            
    # create array which contains mass i in each row of the ith column
    m = masses[None, :, None]
    
    # create acceleration matrix
    A = m*f
    
    # The total acceleration the ith particle feels is then just the sum 
    # over the ith column of A:
    accelerations = np.sum(A, axis=1)
            
    return accelerations


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
