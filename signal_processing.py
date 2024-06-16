from wave_generator import generateNoise
from wave_generator import generateSignal
from wave_shift_model import calculateDelta
from directivity_characteristic import calculateDirectivityCharacteristic
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn import preprocessing


def comp(a1,a2):
    return 100*np.sum(np.abs(a1-a2))/np.sum((np.abs(a1+a2)/2))
    return np.sum(np.abs(a1-a2)/(np.abs(a1+a2)/2))*100

def getMax(deltaList):
    buffer = -100
    for delta in deltaList:
        if delta is not np.inf and delta > buffer:
            buffer = delta
    return buffer

def addSignalShift(alpha):
    clearSignal = generateSignal(0,5,10000)
    deltaList = calculateDelta(alpha)
    dirCharacteristic = []
    for phi in np.arange(-177.5, 180.0, 5.0):
        dirCharacteristic.append(calculateDirectivityCharacteristic(alpha,phi))
    maxDelta = getMax(deltaList)
    print(maxDelta)
    newTime = np.arange(maxDelta*(10000), len(clearSignal)-maxDelta*(10000), 1.0)
    X = [[]] * 72
    for i in range(len(X)):
        X[i] = np.interp(newTime - deltaList[i]*(10000), np.arange(0,10000,1), clearSignal)*dirCharacteristic[i]
        
    print(X[40], X[41])
    return X




def getFftLists(X):
    F = [[]] * 72
    for i in range(len(X)):
        F[i] = np.fft.rfft(X[i])
    return F

def lastTransform(Y):
    W = np.empty(len(Y))
    summ = 0
    for i in range(len(Y)):
        for j in range(len(Y[0])):
            summ = summ + np.abs(Y[i][j]) * np.abs(Y[i][j])
        W[i] = summ
        summ = 0
    print(len(W))
    return W

def findMainWindow(deltaList):
    maxD = 100.0
    countMax = 0
    index = 0
    window = []
    for i in range(len(deltaList)):
        if deltaList[i] == maxD:
            countMax = countMax + 1
        if deltaList[i] < maxD:
            countMax = 1
            maxD = deltaList[i]
            index = i
    if countMax == 2:
        for i in range(24):
            window.append(deltaList[index-11+i])
    return window


def convolution(F, deltaList):
    koef = 0
    Y = np.empty((49, len(F[0])), dtype=np.dtype('c16'))
    summ = 0
    mainW = findMainWindow(deltaList)
    for window in range(0,49):
        for f in range(len(F[0])):
            for m in range(0,24):
                koef = np.exp(2j*np.pi*f*mainW[m])
                summ = summ + F[m+window][f]*koef                
            Y[window][f] = summ
            summ = 0
    return Y


def plotShifts(X):
    fig, ax = plt.subplots()
    z1 = []
    z = []
    z1.append([])
    for i in range(len(X)):
        for j in X[i]:
            z1[i].append(j)
        if i!=71:
            z1.append([])
    for i in range(len(z1)):
        z.append(tuple(z1[i][:200]))
    Z = tuple(z)
    axisX = np.arange(0,len(Z),1)
    axisY = np.arange(0,200,1)
    plt.pcolormesh(axisY, axisX, Z, shading = 'auto')
    ax.set_ylabel('Номер ПЭ')
    ax.set_xlabel('Время')
    fig.savefig('последний.png')
    plt.show()
    return


def calcY(F, deltaList):
    ds = 0
    f = 4500
    Y = np.empty(49, dtype=np.dtype('c16'))
    mainW = findMainWindow(deltaList)
    for window in range(0,49):
        summ = 0
        for m in range(0,24):
            koef = np.exp(2j*np.pi*f*mainW[m])
            summ = summ + F[m+window][f]*koef                
        Y[window] = summ
    koef = [0]*49
    freq = np.empty(len(F), dtype=np.dtype('c16'))
    for i in range(len(F)):
        freq[i] = F[i][f]
    fftFreq = np.fft.fft(freq)
    for m in range(0,24):
        koef[m] = np.exp(2j*np.pi*f*mainW[m])
    fftKoef = np.fft.fft(koef, len(fftFreq))
    bigY = np.fft.ifft(fftFreq*fftKoef)[23:]

    fig, ax = plt.subplots()
    plt.plot(np.arange(0,len(bigY),1), np.abs(bigY), 'b', markersize=6,  label = 'Быстрая свёртка')
    plt.plot(np.arange(0,len(Y),1), np.abs(Y), 'r+', markersize=6, label = 'Обычная свёртка')
    plt.grid(True)
    ax.set_ylabel('Амплитуда')
    ax.set_xlabel('Номер ПК')
    plt.legend(loc='upper right')
    fig.savefig('primer.png')
    plt.show()
    return

def fastConvolution(F, deltaList):
    freq = np.empty(len(F), dtype=np.dtype('c16'))
    koef = [0]*49
    Y = np.empty((49, len(F[0])), dtype=np.dtype('c16'))
    for f in range(len(F[0])):
        for i in range(len(F)):
            freq[i] = F[i][f]
        for m in range(0,24):
            koef[m] = np.exp(2j*np.pi*f*deltaList[m])
        fftFreq = np.fft.fft(freq)
        koef[m] = np.exp(2j*np.pi*f*deltaList[m])
        fftKoef = np.fft.fft(koef, len(fftFreq))
        for i in range(0,49):
            Y[i][f] = np.fft.ifft(fftFreq*fftKoef)[23+i]
    return Y

if __name__ == "__main__":
    deltaList = calculateDelta(0)
    X = addSignalShift(0)
    #plotShifts(X)
    F = getFftLists(X)
    freq = np.fft.rfftfreq(10000, 1/10000)
    #print(len(F[0]))
    calcY(F, deltaList)

    start_time = time.time()

    Y = convolution(F,deltaList)

    end_time = time.time()
    execution_time = end_time - start_time
    print("Время выполнения обычной свёртки:", execution_time, "сек.")
    
    start_time = time.time()
    
    YFast = fastConvolution(F,deltaList)
    
    end_time = time.time()
    execution_time = end_time - start_time
    print("Время выполнения быстрой свёртки:", execution_time, " сек.")
    W = lastTransform(Y)
    WFast = lastTransform(YFast)
    time = np.arange(0,10000,1)
    #plt.subplot(1, 2, 1)
    fig, ax = plt.subplots()
    ax.set_ylabel('Амплитуда')
    ax.set_xlabel('Частота')
    ax.set_title('Выходной эффект системы обработки')
    freq = np.fft.rfftfreq(10000, 1/10000)
    #hist, bins = np.histogram(WFast, bins='auto')
    plt.plot(np.arange(0,len(W),1), np.abs(W), 'b', markersize=6,  label = 'Быстрая свёртка')
    plt.plot(np.arange(0,len(W),1), np.abs(W), 'r+', markersize=6, label = 'Обычная свёртка')
    ax.set_ylabel('Амплитуда')
    ax.set_xlabel('Номер ПК')
    plt.legend(loc='upper right')
    plt.grid(True)
    fig.savefig('W.png')
    plt.show()
    '''
    plt.subplot(1, 2, 2)
    hist, bins = np.histogram(WFast, np.linspace(-40, 40, 80))
    plt.plot(bins[:-1], hist)
    plt.plot(freq, np.abs(WFast))
    plt.grid(True)
    '''
    fig, ax = plt.subplots()
    ax.set_ylabel('Амплитуда')
    ax.set_xlabel('Частота')
    ax.set_title('Выходной эффект системы обработки быстрой свёрткой')
    hist, bins = np.histogram(WFast, bins='auto')
    plt.plot(np.arange(0,len(np.abs(WFast)), 1), np.abs(WFast))
    plt.grid(True)
    fig.savefig('WFast.png')
    plt.show()