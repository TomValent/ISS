import scipy.io.wavfile as wavfile
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

def graph(result, frame, Fs, length):
    for i in range (60, 68):
        t = np.arange(1024)                             # build in DFT
        sp = frame[i]
        freq = length / len(frame[i])/2
        plt.figure(figsize=(18,8))
        plt.title("Numpy DFT")
        plt.xlabel = ("f[HZ]")
        plt.ylabel = ("Amplituda[-]")
        plt.plot(freq * Fs, abs(sp))
        plt.tight_layout()
        plt.show()

    t = np.arange(1024)                             # custom DFT
    sp = result[50]
    freq = np.fft.fftfreq(t.shape[-1])
    plt.figure(figsize=(18,8))
    plt.title("Custom DFT")
    plt.xlabel = ("f[HZ]")
    plt.ylabel = ("Amplituda[-]")
    plt.plot(freq * Fs, abs(sp))
    plt.tight_layout()

def graph2(frame, Fs, length):
    for i in range(len(frames)):
        frame_duration = length / (len(frames) / 2)
        new_t = np.arange(0, frame_duration, frame_duration / 1024)

        print("frame number: ", i)

def load(name):
    length = WAVE(name).info.length                 # dlzka signalu
    print(f'Total Duration: {format(length)}s')

    Fs, y = wavfile.read('xvalen27.wav')            # nacitanie
    y = y - np.mean(y)                              # ustrednenie
    y /= max(abs(y))                                # normalizacia
    print(y.min(), y.max())

    frame = frames(y, Fs * length)                  # rozdelenie na frames
    graph2(frame, Fs, length)

    return DFT(frame), frame, Fs, y, length


if __name__ == '__main__':
    result, frame, Fs, y, length = load('xvalen27.wav')

