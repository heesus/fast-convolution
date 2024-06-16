import numpy as np
import matplotlib.pyplot as plt
from directivity_characteristic import calculateDirectivityCharacteristic

def generateSinusoid(delta):
    time = np.linspace(delta,0.001,10000)
    sinusoid = np.sin((time-delta)*10000)
    return time, sinusoid

def calculateDelta(alpha): #Радиус антенны = 1, угол расположения датчиков
    radius = 1
    firstSensor = -180
    distance = 360
    sensors = np.arange(-177.5, 180.0, 5.0)
    deltaList = [0] * len(sensors)
    characteristic = []
    for s in sensors:
        if distance > np.abs(firstSensor - alpha):
            distance = np.abs(firstSensor - alpha)
            firstSensor = s
    for i in range(len(sensors)):
        deltaList[i] = radius * (np.cos(np.deg2rad(firstSensor-alpha)) - np.cos(np.deg2rad(sensors[i]-alpha))) / 1500
    return deltaList

if __name__ == "__main__":
    fig = plt.figure(dpi=80)
    deltaList = calculateDelta(0)
    #print(deltaList)
    for i in range(30,40):
        time, sinusoid = generateSinusoid(deltaList[i])
        plt.plot(time, sinusoid, color='C'+str(i))
    plt.grid(True)
    fig.savefig('shift.png')
    plt.show()
    