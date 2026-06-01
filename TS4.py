# -*- coding: utf-8 -*-
"""
Created on Wed May  6 19:44:42 2026

@author: feder
"""

from numpy.fft import fft
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal.windows as win

N = 1000
fs = N
df = fs/N
R = 200 #cantidad señales de muestra

SNR = 10

amp = 2
Px = (amp**2)/2
Pn = Px/(10**(SNR/10))


#ruidos
n = np.random.normal(0,Pn,N)
Fr = np.random.uniform(-2,2,R)

# ejes para hacer matrices para llenar de senoidales

tt = np.arange(N) * 1/fs
tt = tt.reshape(-1,1)


F1 = N/4 + Fr
ff1 = F1.reshape(1,-1)


matriz_ruido = np.tile(n.reshape(-1,1), (1,R))



matriz_argumento = 2*np.pi*(fs/N)*ff1*tt

xx = amp * np.sin(matriz_argumento) + matriz_ruido

#hago las PSD

X= (1/N)*fft(xx, axis = 0)
# frec = np.arange(10*N) * df/10

Xabs = np.abs(X)

PSD = Xabs**2

PSDdb = 10*np.log(Xabs**2 + 1e-15)

plt.figure(1)
plt.plot(PSDdb)
plt.xlim(0,N/2)
plt.show()

#ventaneo

# armo ventanas

flattop = win.flattop(N).reshape(-1,1)
blackmanh = win.blackmanharris(N).reshape(-1,1)
hamming = win.hamming(N).reshape(-1,1)

#PSD ventaneada sin padding

XXflattop = fft(xx*flattop, axis=0, n=N)/N
XXblackmanh = fft(xx*blackmanh, axis=0, n=N)/N
XXhamming = fft(xx*hamming, axis=0, n=N)/N

PSDflattop = np.abs(XXflattop)**2
PSDblackmanh = np.abs(XXblackmanh)**2
PSDhamming = np.abs(XXhamming)**2


PSDdbflattop = 10*np.log(PSDflattop)
PSDdbblackmanh = 10*np.log(PSDblackmanh)
PSDdbhamming = 10*np.log(PSDhamming)


frec = np.arange(0, N)

# plt.figure(2)
# plt.plot(frec,PSDdbflattop)
# plt.xlim([(N/4)-15,(N/4)+15])
# plt.show()

# plt.figure(3)
# plt.plot(frec,PSDdbblackmanh)
# plt.xlim([(N/4)-15,(N/4)+15])
# plt.show()

# plt.figure(4)
# plt.plot(frec,PSDdbhamming)
# plt.xlim([(N/4)-15,(N/4)+15])
# plt.show()

#Estimadores de amplitud
amplitud_rectangular = Xabs[N//4,:]**2
amplitud_flattop = PSDflattop[(N//4),:]
amplitud_blackmanh = PSDblackmanh[(N//4),:]
amplitud_hamming = PSDhamming[(N//4),:]

# Estimadores de frecuencia
frec_rect = np.argmax(Xabs[:N//2,:]**2, axis=0)*df
frec_flattop = np.argmax(PSDflattop[:N//2,:], axis=0)*df
frec_blackmanh = np.argmax(PSDblackmanh[:N//2,:], axis=0)*df
frec_hamming = np.argmax(PSDhamming[:N//2,:], axis=0)*df





#PSD ventaneada con padding

XXflattop = fft(xx*flattop, axis=0, n=10*N)/N
XXblackmanh = fft(xx*blackmanh, axis=0, n=10*N)/N
XXhamming = fft(xx*hamming, axis=0, n=10*N)/N

PSDflattop = np.abs(XXflattop)**2
PSDblackmanh = np.abs(XXblackmanh)**2
PSDhamming = np.abs(XXhamming)**2


PSDdbflattop = 10*np.log(PSDflattop)
PSDdbblackmanh = 10*np.log(PSDblackmanh)
PSDdbhamming = 10*np.log(PSDhamming)

factor_escala = 10
frec_padding = np.arange(0, factor_escala*N)/factor_escala

plt.figure(2)
plt.plot(frec_padding,PSDdbflattop)
plt.xlim([(N/4)-15,(N/4)+15])
plt.show()

plt.figure(3)
plt.plot(frec_padding,PSDdbblackmanh)
plt.xlim([(N/4)-15,(N/4)+15])
plt.show()

plt.figure(4)
plt.plot(frec_padding,PSDdbhamming)
plt.xlim([(N/4)-15,(N/4)+15])
plt.show()

# armo estimadores de amplitud y frecuencia

def esperanza (muestras):
    mu = (1/len(muestras))*np.sum(muestras)
    return mu



# Sesgo y varianza de amplitud

amp_teorica = 1
frec_teorica = 250

sesgo_amp_flattop = np.mean(amplitud_flattop) - amp_teorica
var_amp_flattop = np.var(amplitud_flattop)

sesgo_amp_blackmanh = np.mean(amplitud_blackmanh) - amp_teorica
var_amp_blackmanh = np.var(amplitud_blackmanh)

sesgo_amp_hamming = np.mean(amplitud_hamming) - amp_teorica
var_amp_hamming = np.var(amplitud_hamming)


# Sesgo y varianza de frecuencia

sesgo_frec_flattop = np.mean(frec_flattop) - frec_teorica
var_frec_flattop = np.var(frec_flattop)

sesgo_frec_blackmanh = np.mean(frec_blackmanh) - frec_teorica
var_frec_blackmanh = np.var(frec_blackmanh)

sesgo_frec_hamming = np.mean(frec_hamming) - frec_teorica
var_frec_hamming = np.var(frec_hamming)





plt.figure(figsize=(8,5))

plt.hist(amplitud_rectangular, bins=15, alpha=0.5, label='Rectangular')

plt.hist(amplitud_flattop, bins=15, alpha=0.5, label='Flattop')

plt.hist(amplitud_blackmanh, bins=15, alpha=0.5, label='Blackman-Harris')

plt.hist(amplitud_hamming, bins=15, alpha=0.5, label='Hamming')

plt.title('Histogramas de amplitud')
plt.xlabel('Amplitud')
plt.ylabel('Cantidad')
plt.legend()
plt.grid(True)

plt.show()

plt.figure(figsize=(8,5))

plt.hist(frec_rect, bins=15, alpha=0.5, label='Rectangular')

plt.hist(frec_flattop, bins=15, alpha=0.5, label='Flattop')

plt.hist(frec_blackmanh, bins=15, alpha=0.5, label='Blackman-Harris')

plt.hist(frec_hamming, bins=15, alpha=0.5, label='Hamming')

plt.title('Histogramas de frecuencia')
plt.xlabel('Frecuencia')
plt.ylabel('Cantidad')
plt.legend()
plt.grid(True)

plt.show()










