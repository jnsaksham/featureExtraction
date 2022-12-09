import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.fftpack import fft
import math
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from scipy.signal import hamming, blackmanharris
tol = 1e-14


def isPower2(num):
    """
    Check if num is power of two
    """
    return ((num & (num - 1)) == 0) and num > 0


def dftAnal(x, w, N):
    """
    Analysis of a signal using the discrete Fourier transform
    x: input signal, w: analysis window, N: FFT size
    returns mX, pX: magnitude and phase spectrum
    """

    if not(isPower2(N)):                                 # raise error if N not a power of two
        raise ValueError("FFT size (N) is not a power of 2")

    if (w.size > N):                                        # raise error if window size bigger than fft size
        raise ValueError("Window size (M) is bigger than FFT size")

    hN = (N // 2) + 1                                           # size of positive spectrum, it includes sample 0
    hM1 = (w.size + 1) // 2                                     # half analysis window size by rounding
    hM2 = w.size // 2                                         # half analysis window size by floor
    fftbuffer = np.zeros(N)                                 # initialize buffer for FFT
    w = w / sum(w)                                          # normalize analysis window
    xw = x * w                                                # window the input sound
    fftbuffer[:hM1] = xw[hM2:]                              # zero-phase window in fftbuffer
    fftbuffer[-hM2:] = xw[:hM2]
    X = fft(fftbuffer)                                      # compute FFT
    absX = abs(X[:hN])                                      # compute ansolute value of positive side
    absX[absX < np.finfo(float).eps] = np.finfo(float).eps    # if zeros add epsilon to handle log
    mX = (absX)                                # magnitude spectrum of positive frequencies in dB
    X[:hN].real[np.abs(X[:hN].real) < tol] = 0.0            # for phase calculation set to 0 the small values
    X[:hN].imag[np.abs(X[:hN].imag) < tol] = 0.0            # for phase calculation set to 0 the small values
    pX = np.unwrap(np.angle(X[:hN]))                        # unwrapped phase spectrum of positive frequencies
    return mX, pX


def stftAnal(x, w, N, H):
    """
    Analysis of a sound using the short-time Fourier transform
    x: input array sound, w: analysis window, N: FFT size, H: hop size
    returns xmX, xpX: magnitude and phase spectra
    """
    if (H <= 0):                                   # raise error if hop size 0 or negative
        raise ValueError("Hop size (H) smaller or equal to 0")

    M = w.size                                      # size of analysis window
    hM1 = (M + 1) // 2                                  # half analysis window size by rounding
    hM2 = M // 2                                      # half analysis window size by floor
    x = np.append(np.zeros(hM2), x)                  # add zeros at beginning to center first window at sample 0
    x = np.append(x, np.zeros(hM2))                  # add zeros at the end to analyze last sample
    pin = hM1                                       # initialize sound pointer in middle of analysis window
    pend = x.size - hM1                               # last sample to start a frame
    w = w / sum(w)                                  # normalize analysis window
    xmX = []                                       # Initialise empty list for mX
    xpX = []                                       # Initialise empty list for pX
    while pin <= pend:                                # while sound pointer is smaller than last sample
        x1 = x[pin - hM1:pin + hM2]                     # select one frame of input sound
        mX, pX = dftAnal(x1, w, N)              # compute dft
        xmX.append(np.array(mX))                    # Append output to list
        xpX.append(np.array(pX))
        pin += H                                    # advance sound pointer
    xmX = np.array(xmX)                             # Convert to numpy array
    xpX = np.array(xpX)
    return xmX, xpX


def HFC(x, w, N, H):
    """
    Onset detection using High Frequency Content
    x: input array sound, w: analysis window, N: FFT size, H: hop size
    returns Energy: Matrix containing sum(K * mX**2)/N for all frames
    """
    if (H <= 0):                                   # raise error if hop size 0 or negative
        raise ValueError("Hop size (H) smaller or equal to 0")

    M = w.size                                      # size of analysis window
    hM1 = (M + 1) // 2                                  # half analysis window size by rounding
    hM2 = M // 2                                      # half analysis window size by floor
    x = np.append(np.zeros(hM2), x)                  # add zeros at beginning to center first window at sample 0
    x = np.append(x, np.zeros(hM2))                  # add zeros at the end to analyze last sample
    pin = hM1                                       # initialize sound pointer in middle of analysis window
    pend = x.size - hM1                               # last sample to start a frame
    w = w / sum(w)                                  # normalize analysis window
    Energy = []                                       # Initialise empty list for mX
    frame_frequencies = np.arange(N / 2 + 1)
    while pin <= pend:                                # while sound pointer is smaller than last sample
        x1 = x[pin - hM1:pin + hM2]                     # select one frame of input sound
        mX, pX = dftAnal(x1, w, N)
        energy_frame = mX * mX * frame_frequencies
        hfc_frame = sum(energy_frame) / N
        Energy.append(hfc_frame)                    # Append output to list
        pin += H                                    # advance sound pointer
    Energy = np.array(Energy)                             # Convert to numpy array
    return Energy


def SpectralFlux(x, w, N, H):
    """
    Onset detection using Spectral Difference by L1-Norm
    x: input signal, w: analysis window, N: FFT size
    returns flux: array spectral flux between subsequent frames by L1-Norm
    """
    xmX, xpX = stftAnal(x, w, N, H)
    log_xmX = np.log1p(100 * xmX)

    # cmap = plt.get_cmap('prism')
    """
    # levels = MaxNLocator(nbins=15).tick_values(xmX.min(), xmX.max())
    # norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
    """

    # plt.figure(1, figsize=(9.5, 6))

    # plt.subplot(311)
    # plt.plot(x)

    # plt.subplot(312)
    # numFrames = int(xmX[:, 0].size)
    # frmTime = H * np.arange(numFrames) / float(sr)
    # binFreq = np.arange(N / 2 + 1) * float(sr) / N
    # plt.pcolormesh(frmTime, binFreq, np.transpose(xmX), cmap=cmap)
    # plt.title('mX (piano.wav), M=1001, N=1024, H=256')
    # plt.autoscale(tight=True)

    # plt.subplot(313)
    # numFrames = int(log_xmX[:, 0].size)
    # frmTime = H * np.arange(numFrames) / float(sr)
    # binFreq = np.arange(N / 2 + 1) * float(sr) / N
    # plt.pcolormesh(frmTime, binFreq, np.transpose(log_xmX), cmap=cmap)
    # plt.title('mX (piano.wav), M=1001, N=1024, H=256')
    # plt.autoscale(tight=True)
    # plt.show()

    num_frames = int(xmX.size * 2 // N)
    frame = num_frames // 2
    k = N // 4
    print(xmX[frame + 1, k])
    print(xmX[frame, k])
    print(xmX[frame + 1, k] - xmX[frame, k])
    flux = np.array([])
    for frame in np.arange(num_frames - 1):                # Each xmX row corresponds to a frame and has N/2 frequency indices
        frame_flux = 0
        for k in np.arange(N // 2):
            k_flux = abs(xmX[frame + 1, k] - xmX[frame, k])
            frame_flux = frame_flux + k_flux
        flux = np.append(flux, frame_flux)

    return flux


y, sr = librosa.load('Beat.wav')
M = 2047
N = 2048
H = M // 4
w = hamming(M)

# HFC

Energy = HFC(y, w, N, H)

fig = plt.figure()
duration = y.size / float(sr)
time = np.arange(0, duration, 1 / float(sr))
yplot = fig.add_subplot(411)
yplot.plot(time, y)
frames = np.arange(len(Energy))
t = librosa.frames_to_time(frames, sr=sr)
onsetplot = fig.add_subplot(412)
onsetplot.plot(t, Energy)

flux = SpectralFlux(y, w, N, H)
avg_window_size = 7
avg_window = np.ones(avg_window_size)
flux_centered = np.append(np.ones(avg_window_size // 2), flux)
flux_centered = np.append(flux_centered, np.ones(avg_window_size // 2))
local_avg = np.array([])
for i in np.arange(avg_window_size // 2, flux_centered.size - (avg_window_size // 2), 1):
    avg = sum(flux[i:i + avg_window_size]) / avg_window_size
    local_avg = np.append(local_avg, avg)

peaks = flux - local_avg
duration = y.size / float(sr)
time = np.arange(0, duration, 1 / float(sr))
sdonsetplot = fig.add_subplot(413)
sdonsetplot.plot(flux)
peaksplot = fig.add_subplot(414)
peaksplot.plot(peaks)
plt.show()
