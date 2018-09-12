import numpy as np
import matplotlib.pyplot as plt

from ssa import ssa, hilbert_spectrum

if __name__ == '__main__':
    N = 1000
    Fs = 100
    amp = np.array([2,4,10]).reshape(-1,1)
    L = Fs

    t = np.linspace(0, N/Fs, N).reshape(1,-1)
    y = np.sin(2*np.pi*np.dot(amp, t)).sum(0) + \
            np.random.randn(*t.shape)

    Q, s = ssa(y, L)
    fig = plt.figure(1)
    plt.plot(t.T, Q[:6,:].T)
    plt.title("First 6 SSA Components")
    plt.xlabel("Time (s)")
    fig.show()

    S, t, f = hilbert_spectrum(y, L, Fs)
    fig = plt.figure(2)
    plt.imshow(S, origin='lower')
    t_idxs = np.arange(0, len(t), 300)
    plt.xticks(t_idxs, ["{:.2f}".format(x) for x in t[t_idxs]])
    f_idxs = np.arange(0, len(f), 100)
    plt.yticks(f_idxs, ["{:.2f}".format(x) for x in f[f_idxs]])
    plt.ylim(len(f)/2, np.argmax(f > 31))
    plt.title("Hilbert Spectrum")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    fig.show()

    fig = plt.figure(3)
    plt.plot(f, 10*np.log10(np.sum(S, axis=-1)))
    plt.title("Marginal Hilbert Spectrum")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power (AU)")
    fig.show()

    input()
