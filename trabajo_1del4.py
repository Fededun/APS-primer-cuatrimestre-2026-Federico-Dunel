# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 20:14:12 2026

@author: feder
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import fft

N = 8
nn = np.arange(0, N, 1)
fs = 100
tt = nn/fs

x = 4 + 3*np.sin((np.pi/2)*nn)

X = fft.fft(x)

plt.figure(1)
plt.plot(nn,np.abs(X),'x')
plt.show()

fase = np.angle(X)

plt.figure(2)
plt.plot(nn,fase, 'x')
plt.show()

plt.figure(3)
plt.plot(nn,((1/N)**2)*np.abs(X)**2,'x')
plt.show()


g = 4 + 3*np.sin((3*np.pi/5)*nn)
G = fft.fft(g)

plt.figure(4)
plt.plot(nn,np.abs(G),'x')
plt.show()

faseg = np.angle(G)

plt.figure(5)
plt.plot(nn,faseg, 'x')
plt.show()

plt.figure(6)
plt.plot(nn,((1/N)**2)*np.abs(G)**2,'x')
plt.show()
