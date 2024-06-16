import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import random

def generateSignal(E, sko, freq):
    clearSignal = norm.rvs(E,sko,freq)
    return clearSignal

def generateNoise(freq):
    whiteNoise = []
    for i in range(0,freq):
        whiteNoise.append(random.randint(-5,5))
    return whiteNoise


if __name__ == "__main__":
    signal = generateSignal(0,5,10000)
    fig = plt.figure(1, dpi=80)

    hist, bins = np.histogram(signal, bins='auto')

    #plt.plot(signal[0:100])
    #plt.title('Шумовой сигнал')
    #plt.yticks(np.arange(-14,14,1))
    #plt.grid(True)
    #plt.plot(bins[:-1], hist)
    plt.hist(signal, bins='auto', edgecolor = 'black')
    plt.title('Гистограмма шумового сигнала')
    fig.savefig('полезный.png')
    plt.show()