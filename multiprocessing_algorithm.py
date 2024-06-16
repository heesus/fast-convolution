import multiprocessing
import numpy as np

def fastConvolution(F, deltaList, freqForProc, proc):
    freq = np.empty(len(F), dtype=np.dtype('c16'))
    koef = [0]*49
    for f in range(proc*freqForProc, (proc+1)*freqForProc):
        if f < len(F[0]):
            for i in range(len(F)):
                freq[i] = F[i][f]
            for m in range(0,24):
                koef[m] = np.exp(2j*np.pi*f*deltaList[m])
            fftFreq = np.fft.fft(freq)
            koef[m] = np.exp(2j*np.pi*f*deltaList[m])
            fftKoef = np.fft.fft(koef, len(fftFreq))
            for i in range(0,49):
                #Y[i][f] = 
                np.fft.ifft(fftFreq*fftKoef)[23+i]
    print("Процесс № ", proc, " закончился")

def processesed(procs, calc, F, deltaList):
    processes = []
    for proc in range(procs):
        p = multiprocessing.Process(target=fastConvolution, args=(F, deltaList, calc, proc))
        processes.append(p)
        p.start()
    for p in processes:
        p.join()
    
    

def startMultiProc(F,deltaList):
    n_proc = multiprocessing.cpu_count()
    calc = len(F[0]) // n_proc + 1
    Y = np.empty((49, len(F[0])), dtype=np.dtype('c16'))
    processesed(n_proc, calc, F, deltaList)
    return Y

if __name__ == "__main__":
    print(1)