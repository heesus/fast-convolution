import numpy as np
import matplotlib.pyplot as plt

def calculateDirectivityCharacteristic(alpha, phi, mu = 0.4):
    character = (mu + np.cos(np.deg2rad(alpha) - np.deg2rad(phi))) / (1 + mu)
    if character > 0:
        return character
    else:
        return 0


if __name__ == "__main__":
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    dirCharacter = []
    i = 0 
    theta = np.deg2rad(np.arange(-180, 180, 1))
    #for phi in np.arange(-177.5, 180.0, 5.0):
    #    dirCharacter.append(calculateDirectivityCharacteristic(0, phi))
    #print(dirCharacter)
    for alpha in range(-180, 180, 1):
        dirCharacter.append(calculateDirectivityCharacteristic(alpha, 0))
    ax.plot(theta, dirCharacter, color='C'+str(i))
    ax.set_thetamin(-180)
    ax.set_thetamax(180)
    i = i + 1
    dirCharacter = []
    fig.savefig("ХН.png")
    plt.show()