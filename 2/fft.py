import numpy as np
import copy

# non-recursively


def fft(x: np.ndarray, inverse: bool = False) -> np.ndarray:
    """
    computes the fourier transform of x using a bottom-up approach.

    Args:
        x (np.ndarray):
            array of samples.
            They must be evenly spaced with a length that is a power of 2.
        inverse (bool):
            whether to use inverse fourier transform.

    Returns:
        np.ndarray: fourier-transform of samples.
    """
    N = len(x)
    # swap the indices using bit reversal
    # bit_reversal_swap also checks whether N is a power of 2
    x = bit_reversal_swap(x, N)
    x = x.astype(np.complex64)

    N_j = 2

    sign = 1
    if inverse:
        sign = -1

    while N_j < N + 1:

        for n in range(0, N, N_j):
            # compute the theta, alpha, beta arguments
            theta = sign * 2 * np.pi / N_j  # use negative sign if inverse
            alpha = 2 * np.sin(0.5 * theta) * np.sin(0.5 * theta)
            beta = np.sin(theta)
            # instantiate cos and sin values
            cos = 1
            sin = 0
            # store half N_j value as integer
            half_N_j = int(0.5 * N_j)

            for k in range(0, half_N_j):
                m = n + k
                t = x[m]

                # compute the complex exponent
                exp = np.complex64(cos + 1j * sin)

                # update the array values
                x[m] = t + exp * x[m + half_N_j]
                x[m + half_N_j] = t - exp * x[m + half_N_j]

                # update the cos values
                new_cos = cos - alpha * cos - beta * sin
                new_sin = sin - alpha * sin + beta * cos
                cos, sin = new_cos, new_sin

        N_j *= 2

    return x


def fft_frequencies(Nsamples:int, L:float)->np.ndarray:
    """
    Computes the frequencies which a fft would return

    Args:
        Nsamples (int): number of samples must be a power of 2.
        L (float): Size of box

    Returns:
        k (np.ndarray): fourier_space frequencies.
    """

    # compute the frequencies at which the fourier_transform computes.
    Delta_k = 2*np.pi / L
    
    k = np.arange(Nsamples)
    # frequencies at indices N/2 +1 ,..., N - 1
    # are actually -N+1 ,..., -1
    k[int(0.5*Nsamples + 1):] -= Nsamples
    
    k = Delta_k * k.astype("float32")
    
    return k

def bit_reversal_swap(x: np.ndarray, N: int) -> np.ndarray:
    """
    swaps elements of x on index i, with elements of x on
    index j, where j is the binary representation of i in reverse.
    (eg: i = 11001, j= 10011)

    Args:
        x (np.ndarray): array to be swapped, must have length of power of 2.
        N (int): length of array x

    Returns:
        np.ndarray: the permuted version of x
    """
    x = copy.deepcopy(x)
    bits = int(np.log2(N))  # number of bits for each index

    # check whether x contains 2^a elements.
    if bits - np.log2(N) != 0:
        raise Exception("x array does not have length of power of 2")

    for i in range(N):
        # represent the index in binary as a string
        # ignore the first 2 elements as this is "sign b": "0b00000"
        # zfill pads 0's until length of bits is reached
        i_bin_rep = bin(i)[2:].zfill(bits)
        # reverse the binary representation
        i_rev = i_bin_rep[::-1]
        # convert back to integer in base 10
        i_rev = int(i_rev, 2)
        # ensure we only swap once per pair
        if i < i_rev:
            x[i], x[i_rev] = x[i_rev], x[i]

    return x


def fft3d(x: np.ndarray, inverse: bool = False) -> np.ndarray:
    """
    computes the 3d fourier transform of x using a bottom-up approach.
    Done by applying the 1d fft along each axis separately.

    Args:
        x (np.ndarray):
            array of samples.
            3D array of shape (N, N, N) where N is a power of 2.
        inverse (bool):
            whether to use inverse fourier transform.

    Returns:
        np.ndarray: fourier-transform of samples.
    """
    result = x.astype(np.complex64)
    # store number of samples in each dimension
    N0, N1, N2 = result.shape

    # transform rows along x
    for j in range(N1):
        for k in range(N2):
            result[:, j, k] = fft(result[:, j, k], inverse=inverse)

    # transform into screen rows along y
    for i in range(N0):
        for k in range(N2):
            result[i, :, k] = fft(result[i, :, k], inverse=inverse)

    # transform columns along z
    for i in range(N0):
        for j in range(N1):
            result[i, j, :] = fft(result[i, j, :], inverse=inverse)

    return result
