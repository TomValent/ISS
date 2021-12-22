import scipy.io.wavfile as wavfile
from scipy import signal
from mutagen.wave import WAVE
import numpy as np
import matplotlib.pyplot as plt
import wavio


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

def bad_freq():
    f1 = 750
    f2 = 1500
    f3 = 2250
    f4 = 3000
    if ((f1 * 2 == f2) and (f1 * 3 == f3) and (f1 * 4 == f4)):
        print("Bad frequences are harmoniously related")
    return f1, f2, f3, f4

def new_sound(length, Fs, f1, f2, f3, f4):

    time = np.linspace(0, int(length), int(length * Fs), endpoint=False)

    out_cos1 = np.cos(2 * np.pi * f1 * time)
    out_cos2 = np.cos(2 * np.pi * f2 * time)
    out_cos3 = np.cos(2 * np.pi * f3 * time)
    out_cos4 = np.cos(2 * np.pi * f4 * time)

    output_total = out_cos1 + out_cos2 + out_cos3 + out_cos4
    wavio.write("../audio/4cos.wav", output_total, Fs, sampwidth=3)

#---------------------------------------------------------------------------
#                       Spektrogram zleho zvuku


    f, t, Sxx = signal.spectrogram(output_total, Fs, nperseg=1024, noverlap=512)
    plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading="gouraud", cmap="jet")
    plt.colorbar()
    plt.xlabel("Čas[t]")
    plt.ylabel("Frekvencia[Hz]")
    plt.title("Spektogram ruchu")
    plt.show()


def load(name):
    length = WAVE(name).info.length                 # dlzka signalu
    print(f'Total Duration: {format(length)}s')

    Fs, y = wavfile.read(name)                      # nacitanie
    y = y - np.mean(y)                              # ustrednenie
    y /= max(abs(y))                                # normalizacia
    print(y.min(), y.max())

    frame = frames(y, Fs * length)                  # rozdelenie na frames
    return frame, Fs, y, length


if __name__ == '__main__':
    frame, Fs, y, length = load("../audio/xvalen27.wav")
    graph1(frame, length)                           # graf zneleho frame
    graph2(frame, Fs)                               # graf build in a custom DFT
    spectrogram(Fs, y)                              # f1 - f4 = 750*n
    f1, f2, f3, f4 = bad_freq()
    new_sound(length, Fs, f1, f2, f3, f4)