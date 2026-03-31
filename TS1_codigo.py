# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 20:52:56 2026

@author: feder
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

pi = np.pi
N = 1000 #cantidad de muestras, podria ser la cantidad de memoria disponible
fs = 50000 #frecuencia de muestreo
nn = np.arange(0,N,1)
tt = nn/fs
amp = np.sqrt(2)

ff1 = 2000

def sen(nn, fs, amp = np.sqrt(2), ff = ff1, ph = 0, dc = 0):
    tt = nn/fs
    xx = amp*np.sin((2*np.pi)*ff*tt + ph) + dc
    return tt, xx




tt1, xx1 = sen(nn = nn, fs = fs, ff = 2000)
Pxx1 = np.mean(np.abs(xx1)**2)
Per1 = 1/ff1

plt.figure(1)
plt.plot(tt1, xx1, color = 'red')
plt.title("señal de 2000Hz")
plt.xlim([0,1/ff1])
plt.xlabel("Tiempo [s]")
plt.ylabel("Voltaje [V]")
plt.show()

print("la potencia de la señal es", Pxx1, "con un periodo de", Per1)

tt2, xx2 = sen(nn = nn, fs = fs, amp = amp*np.sqrt(2), ff = 2000, ph = (np.pi)/2)
Pxx2 = np.mean(np.abs(xx2)**2)
Per2 = 1/ff1

plt.figure(2)
plt.plot(tt1, xx2, color = 'green')
plt.title("señal de 2000Hz amplificada 3db")
plt.xlabel("Tiempo [s]")
plt.ylabel("Voltaje [V]")
plt.xlim([0,1/ff1])
plt.show()

print("la potencia de la señal es", Pxx2, "con un periodo de", Per2)

tt3, xx3 = sen(nn = nn, fs = fs, ff = 1000)
XX3 = xx1 * xx3
PXX3 = np.mean(np.abs(XX3)**2)
Per3 = 1/1000

plt.figure(3)
plt.plot(tt3, XX3, color = 'blue')
plt.title("señal modulada con sinusoidal de 1000Hz")
plt.xlabel("Tiempo [s]")
plt.ylabel("Voltaje [V]")
plt.xlim([0, (1/1000)*2])
plt.show()

print("la potencia de la señal es", PXX3, "con un periodo de", Per3)

xx4 = np.clip(xx1, a_min = -amp*0.75, a_max= amp*0.75)
Pxx4 = np.mean(np.abs(xx4)**2)
Per4 = 1/ff1

plt.figure(4)
plt.plot(tt1, xx4, color = 'red')
plt.title("señal original de 2000Hz amplitud limitada a un 75%")
plt.xlabel("Tiempo [s]")
plt.ylabel("Voltaje [V]")
plt.xlim([0,1/ff1])
plt.show()

print("la potencia de la señal es", Pxx4, "con un periodo de", Per4)

def funpulso (ff, nn, fs, amp = 1, dc = 0, ph = 0, duty = 0.5):
    tt = nn/fs
    xx = dc + amp * signal.square(2 * np.pi * ff * tt + ph, duty=duty)
    
    return tt, xx

tt5 , xx5 =funpulso(ff = 4000, nn = nn, fs = 100000, duty = 0.5)
Pxx5 = np.mean(np.abs(xx5)**2)
Per5 = 1/4000

plt.figure(5)
plt.plot(tt5, xx5, color = 'yellow')
plt.title("señal cuadrada de 4KHz")
plt.xlim([0,3/4000])
plt.xlabel("Tiempo [s]")
plt.ylabel("Voltaje [V]")
plt.show()

print("la potencia de la señal es", Pxx5, "con un periodo de", Per5)

tt6, xx6 =funpulso(ff = 50, nn = nn, fs = 50000, amp = 1.0, duty = 0.5)
XX6 = np.clip(xx6, a_min = 0, a_max = None)
Pxx6 = np.mean(np.abs(XX6)**2)
Per6 = 0.01

plt.figure(6)
plt.plot(tt6, XX6, color = 'green')
plt.title("Pulso rectangular de 0.01s")
plt.xlabel("Tiempo [s]")
plt.ylabel("Voltaje [V]")
plt.show()

print("la potencia de la señal es", Pxx6, "con un periodo de", Per6)







