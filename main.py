import scipy.io.wavfile as wavfile
from scipy import signal
from mutagen.wave import WAVE
import numpy as np
import matplotlib.pyplot as plt


def DFT(x):
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n / N)
    return np.dot(M, x)

def frames(data: np.ndarray, data_size):
    shift = 512
    frame_size = 1024
    number_of_frames = int(data_size // shift)
    frm = [data[i * shift: frame_size + i * shift] for i in range(number_of_frames)]
    frm[-1] = np.pad(frm[-1], (0, frame_size - frm[-1].shape[0]), "constant")
    return np.array(frm)

def graph1(frame, length):
    frame_duration = length / (len(frame) / 2)
    new_t = np.arange(0, frame_duration, frame_duration / 1024)
    plt.figure(figsize=(16, 9))
    plt.title("Znely ramec (index 7)")
    plt.xlabel("f[HZ]")
    plt.ylabel("Amplituda[-]")
    plt.plot(new_t, frame[7])
    plt.show()

def graph2(frame, Fs):
    t = np.arange(1024)                           # build in DFT
    sp = np.fft.fft(frame[7])
    freq = np.fft.fftfreq(t.shape[-1])
    plt.figure(figsize=(18,8))
    plt.title("Numpy DFT")
    plt.xlabel("f[HZ]")
    plt.ylabel("Amplituda[-]")
    sp = np.split(abs(sp), 2)[0]
    freq = np.split(abs(freq), 2)[0]
    plt.plot(freq * Fs, abs(sp))
    plt.show()

    t = np.arange(1024)                             # custom DFT
    freq = np.fft.fftfreq(t.shape[-1])              # freq by mala byt rovnaka ako pri build in verzii,
    sp = DFT(frame[7])                              # snad je to legalne robit takto
    plt.figure(figsize=(18,8))
    plt.title("Custom DFT")
    plt.xlabel("f[HZ]")
    plt.ylabel("Amplituda[-]")
    sp = np.split(abs(sp), 2)[0]
    freq = np.split(abs(freq), 2)[0]
    plt.plot(freq * Fs, abs(sp))
    plt.tight_layout()
    plt.show()

def spectrogram(Fs, y):
    f, t, Sxx = signal.spectrogram(y, Fs, nperseg=1024, noverlap=512)
    plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading="gouraud", cmap="jet")
    plt.colorbar()
    plt.xlabel("Čas[t]")
    plt.ylabel("Frekvencia[Hz]")
    plt.title("Spektogram")
    plt.show()

def load(name):
    length = WAVE(name).info.length                 # dlzka signalu
    print(f'Total Duration: {format(length)}s')

    Fs, y = wavfile.read('xvalen27.wav')            # nacitanie
    y = y - np.mean(y)                              # ustrednenie
    y /= max(abs(y))                                # normalizacia
    print(y.min(), y.max())

    frame = frames(y, Fs * length)                  # rozdelenie na frames
    return frame, Fs, y, length


if __name__ == '__main__':
    frame, Fs, y, length = load('xvalen27.wav')
    #graph1(frame, length)                           # graf zneleho frame
    #graph2(frame, Fs)                               # graf build in a custom DFT
    spectrogram(Fs, y)
