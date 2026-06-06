import numpy as np

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
