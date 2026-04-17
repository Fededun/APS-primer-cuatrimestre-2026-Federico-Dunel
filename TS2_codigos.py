# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 14:18:36 2026

@author: feder
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import fft

pi = np.pi
N = 1000
fs = 1000
nn = np.arange(0, N, 1)
tt = nn / fs
amp = np.sqrt(2)

# Parámetros de cuantización
Vf = 2
B = 8
q =  2 * Vf / (2**B)

# Ruido
Pq = (q**2) / 12
k = 10
Pr = k * Pq
n = np.random.normal(0, np.sqrt(Pr), N)

def sen(nn, fs, amp = 1, ff=1, ph=0):
    tt = nn / fs
    return amp * np.sin(2 * np.pi * ff * tt + ph)

# Señales
s = sen(nn, fs=fs, amp=2, ff=fs/N*600, ph=0)
Sr = s + n
Sq = np.round(Sr / q) * q

# Error de cuantización
e_q = Sr - Sq
e_q = e_q - np.mean(e_q)

# Densidades epectrales de potencia

SQ = (1/N) * fft.fft(Sq)
PSD_Sq = np.abs(SQ)**2
SQdb = 10 * np.log10(PSD_Sq + 1e-12)

NR = (1/N) * fft.fft(n)
PSD_n = np.abs(NR)**2
NRdb = 10 * np.log10(PSD_n + 1e-12)

EQ = (1/N) * fft.fft(e_q)
PSD_eq = np.abs(EQ)**2
EQdb = 10 * np.log10(PSD_eq + 1e-12)

#Eje de frecuencias

f = np.arange(0, N) * fs / N


# 1. Señal con ruido
plt.figure()
plt.plot(tt, Sr, color='blue')
plt.title("Señal con ruido (Sr)")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud [V]")
plt.grid()
plt.show()

# 2. Señal cuantizada
plt.figure()
plt.plot(tt, Sq, color='red')
plt.title("Señal cuantizada (Sq)")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud [V]")
plt.grid()
plt.show()

# 3. FFT de la señal cuantizada

plt.figure()
plt.plot(f[:N//2], SQdb[:N//2], label='Señal cuantizada', color='green')
plt.plot(f[:N//2], NRdb[:N//2], label='Ruido n', color='orange')
plt.plot(f[:N//2], EQdb[:N//2], label='Ruido cuantizacion (q)', color='purple')

plt.title("FFT - Señal y ruidos")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Magnitud [dB]")
plt.legend()
plt.grid()

plt.show()

# 4. Histograma del error de cuantización
plt.figure()
plt.hist(e_q, bins=10, color='purple')
plt.title("Histograma del error de cuantización")
plt.xlabel("Error")
plt.ylabel("Frecuencia")
plt.grid()

plt.show()

print(np.mean(EQdb))

S = (1/N) * fft.fft(Sr)
PSD_S = np.abs(S)**2
Sdb = 10 * np.log10(PSD_S + 1e-12)

