import numpy as np
import matplotlib.pyplot as plt

# Parámetros generales
fs = 1000          # Frecuencia de muestreo (Hz)
t = np.linspace(-1, 1, fs)

# =========================
# 1. Definición de señales
# =========================

# Pulso rectangular
pulso = np.where(np.abs(t) <= 0.2, 1, 0)

# Función escalón
escalon = np.heaviside(t, 1)

# Señal senoidal
f = 5  # Frecuencia de la señal (Hz)
seno = np.sin(2 * np.pi * f * t)

# =========================
# 2. Transformada de Fourier
# =========================

def calcular_fft(signal):
    fft_signal = np.fft.fft(signal)
    freqs = np.fft.fftfreq(len(signal), 1/fs)
    return freqs, fft_signal

freq_pulso, fft_pulso = calcular_fft(pulso)
freq_escalon, fft_escalon = calcular_fft(escalon)
freq_seno, fft_seno = calcular_fft(seno)

# =========================
# 3. Visualización
# =========================

def graficar(signal, fft_signal, freqs, titulo):
    plt.figure(figsize=(12,5))

    # Dominio del tiempo
    plt.subplot(1,2,1)
    plt.plot(t, signal)
    plt.title(f"{titulo} - Dominio del tiempo")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Amplitud")

    # Dominio de la frecuencia
    plt.subplot(1,2,2)
    plt.plot(freqs, np.abs(fft_signal))
    plt.title(f"{titulo} - Dominio de la frecuencia")
    plt.xlabel("Frecuencia (Hz)")
    plt.ylabel("Magnitud")

    plt.tight_layout()
    plt.show()

graficar(pulso, fft_pulso, freq_pulso, "Pulso rectangular")
graficar(escalon, fft_escalon, freq_escalon, "Función escalón")
graficar(seno, fft_seno, freq_seno, "Señal senoidal")