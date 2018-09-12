import numpy as np
from scipy import linalg, signal


def moving_average(x, N, axis=-1):
    # adapted from https://stackoverflow.com/questions/13728392/moving-average-or-running-mean/27681394#27681394
    cumsum = np.cumsum(np.insert(x, 0, 0, axis), axis)
    s1 = [np.s_[:]] * x.ndim
    s2 = s1.copy()
    s1[axis] = np.s_[N:]
    s2[axis] = np.s_[:-N]
    return (cumsum[tuple(s1)] - cumsum[tuple(s2)]) / float(N)


def ssa(x: np.ndarray, L: int):
    """
    Performs Singular Spectrum Analysis as found in Hossein Hassani (2007),
    Bonizzi et al (2012) (the "basic version they mention"), and
    https://www.mathworks.com/matlabcentral/fileexchange/58967-singular-spectrum-analysis-beginners-guide
    among other places.

    Args:
        x (numpy.ndarray): the array to decompose (it will be transformed to a 1-dimensional array)
        L (int): the window length, an analogue to spectral resolution

    Returns:
        Q (numpy.ndarray): a n_components x n_time matrix of SSA components
        s (numpy.ndarray): a 1-dimensional array with the spectral contributions of each component
    """
    x = np.array(x).reshape(-1)
    x = (x - x.mean()) / x.var()
    N = len(x)
    trajectory_matrix = linalg.hankel(c=x[:L], r=x[L-1:]) # L x N-L+1
    U, s, Vh = linalg.svd(trajectory_matrix, full_matrices=False)
    Q = np.zeros((len(s), N))
    for i in range(len(s)):
        # compute rank 1 matrix
        A = s[i] * np.dot(U[:,i].reshape(-1,1), Vh[i,:].reshape(1,-1))
        # compute anti-diagonal average to get N-dimensional component
        Q[i,:] = [np.diag(A[:, ::-1], N-L-j).mean() for j in range(N)]
    return Q, s


def hilbert_spectrum(x: np.ndarray, L: int, Fs: float):
    x = np.array(x).reshape(-1)
    Q, s = ssa(x, L)  # n_components x time
    s = s / s.sum()
    z = signal.hilbert(np.dot(np.diag(s), Q))  # n_components x time

    freq = np.diff(np.unwrap(np.angle(z))) * Fs / (2*np.pi)
    freq_bins = np.linspace(-Fs/2+Fs/len(x), Fs/2, len(x)) # include negative frequencies
    freq_labs = np.digitize(freq, freq_bins)-1  # n_components x time

    mags = moving_average(np.abs(z), 2)  # n_components x time-1
    S = np.zeros((len(freq_bins)-1, mags.shape[-1]))  # n_freqs-1 x time-1
    for i in range(Q.shape[0]):
        idxs = np.logical_and(freq_labs[i,:] >= 0, freq_labs[i,:] < S.shape[0])
        m = mags[i, idxs]
        f = freq_labs[i,idxs]
        S[f,idxs] += m
    t = moving_average(np.arange(len(x)) / Fs, 2)
    f = moving_average(freq_bins, 2)
    return S, t, f
