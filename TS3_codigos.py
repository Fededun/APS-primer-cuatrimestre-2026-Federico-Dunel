# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 18:46:25 2026

@author: feder
"""


import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft

# Función senoidal
def sen(ff, nn, vmax=1, dc=0, ph=0, fs=2):
    n = np.arange(0, nn)
    tt = n/fs
    w0 = 2 * np.pi * ff
    xx = dc + vmax * np.sin(w0 * tt + ph)
    return tt, xx

# Parámetros
N = 1000
Fs = N
df = Fs/N
vmx= np.sqrt(2)

# Definimos las tres frecuencias pedidas
f1 = (N/4) * df
f2 = (N/4 + 0.25) * df
f3 = (N/4 + 0.5) * df

#eje frecuencia

frec = np.arange(N) * df
frec_positiva = frec[:N//2 + 1]

tt, x1 = sen(ff=f1, nn=N, vmax= vmx,fs=Fs)
tt, x2 = sen(ff=f2, nn=N, vmax= vmx, fs=Fs)
tt, x3 = sen(ff=f3, nn=N, vmax= vmx, fs=Fs)

#PSD

X1 = fft(x1)/N
X2 = fft(x2)/N
X3 = fft(x3)/N

PSD1 = (np.abs(X1))**2
PSD2 = (np.abs(X2))**2
PSD3 = (np.abs(X3))**2

#graficos
plt.figure()
plt.plot(frec, 10*np.log(PSD1*2 + 1e-15), marker="o")
plt.xlim([0,Fs/2])
plt.title("PSD en db frec N/4")
plt.show()
print("potencia de la PSD1: ", np.sum(PSD1))

plt.figure()
plt.plot(frec, 10*np.log(PSD2*2 + 1e-15), marker="o")
plt.xlim([0,Fs/2])
plt.title("PSD en db frec N/4 + 0.25")
plt.show()
print("potencia de la PSD2: ", np.sum(PSD2))

plt.figure()
plt.plot(frec, 10*np.log(PSD3*2 + 1e-15), marker="o")
plt.xlim([0,Fs/2])
plt.title("PSD en db frec N/4 + 0.5")
plt.show()
print("potencia de la PSD3: ", np.sum(PSD3))

#Aplico zero-padding

cant_pad = 9
Npad = np.zeros(cant_pad*N)

x1_pad = np.concatenate((x1,Npad))
x2_pad = np.concatenate((x2,Npad))
x3_pad = np.concatenate((x3,Npad))

muestras_tot = cant_pad*N + N
factor_escala = muestras_tot/N

X1_pad = fft(x1_pad)/N
X2_pad = fft(x2_pad)/N
X3_pad = fft(x3_pad)/N

PSD1_pad = (np.abs(X1_pad))**2
PSD2_pad = (np.abs(X2_pad))**2
PSD3_pad = (np.abs(X3_pad))**2

frec_pad = np.arange(muestras_tot) * df/factor_escala

plt.figure()
plt.plot(frec_pad, 10*np.log(PSD1_pad*2 + 1e-15), marker='o')
plt.plot(frec, 10*np.log(PSD1*2), 'x')
plt.xlim([0,Fs/2])
plt.title("Zero padding en N/4")
plt.show()
print("potencia de la PSD1 con padding: ", np.sum(PSD1_pad)/factor_escala)

plt.figure()
plt.plot(frec_pad, 10*np.log(PSD2_pad*2 + 1e-15), marker='o')
plt.plot(frec, 10*np.log(PSD2*2), 'x')
plt.xlim([0,Fs/2])
plt.title("Zero padding en N/4 + 0.25")
plt.show()
print("potencia de la PSD2 con padding: ", np.sum(PSD2_pad)/factor_escala)

plt.figure()
plt.plot(frec_pad, 10*np.log(PSD3_pad*2 + 1e-15), marker='o')
plt.plot(frec, 10*np.log(PSD3*2), 'x')
plt.xlim([0,Fs/2])
plt.title("Zero padding en N/4 + 0.5")
plt.show()
print("potencia de la PSD3 con padding: ", np.sum(PSD3_pad)/factor_escala)









