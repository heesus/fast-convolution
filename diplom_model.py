import vpython
from vpython import *
import numpy as np
from math import cos,sin,radians

def generateSensors():
    pointsPos = []
    y = -1.0
    for i in range(5):
        for alpha in np.arange(-177.5, 180.0, 5.0):
            pointsPos.append(sphere(pos = vector(cos(radians(alpha)), y, -sin(radians(alpha))), radius = 0.02, color = color.red))
        y = y + 0.5
    return pointsPos

signal = box(pos = vector(0,0,-1.3),length=4, height=4, width=0.01, color = color.cyan, opacity = 0.5)
antenna = cylinder(pos = vector(0,-1,0), axis = vector(0,2,0), color = color.yellow)
#center = sphere(pos = vector(0,0,0), radius = 0.01, color = color.red)
animationVector = vector(0,0,-1.3)

sensors = generateSensors()

while True:
    pass
    while animationVector.z < 0.2:
        rate(20)
        for s in sensors:
            if abs(signal.pos.z - s.pos.z) < s.radius:
                s.color = color.green
        signal.pos = animationVector
        #center.pos = animationVector
        animationVector.z = animationVector.z + 0.01
    animationVector = vector(0,0,-1.3)
    for s in sensors:
        s.color = color.red