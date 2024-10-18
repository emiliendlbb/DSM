import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.integrate import cumulative_trapezoid

current_dir = os.path.dirname(__file__)

freq_path = os.path.join(current_dir, 'files', 'P2024_frf_acc.txt')
acc_path = os.path.join(current_dir, 'files', 'P2024_irf_acc.txt')

data_freq = np.loadtxt(freq_path)
freq = data_freq[:, 0]
Re_FRF = data_freq[:, 1]
Im_FRF = data_freq[:, 2]
data_acc = np.loadtxt(acc_path)
time = data_acc[:, 0]
acc = data_acc[:, 1]

plt.plot(time, data_acc)
plt.show()


def damped_natural_frequency(data_acc):
    time = data_acc[:, 0]
    acc = data_acc[:, 1]

    peaks, _ = find_peaks(acc)

    time_peaks = time[peaks]
    # print(time_peaks)

    time_peaks_difference = np.diff(time_peaks)
    # print(time_peaks_difference)

    time_peaks_difference = time_peaks_difference[7:]
    # print(time_peaks_difference)

    avg_time_peaks_difference = np.mean(time_peaks_difference)

    damped_natural_frequency = 1/avg_time_peaks_difference

    return damped_natural_frequency


def integrate_acc(data_acc):
    time = data_acc[:, 0]
    acc = data_acc[:, 1]

    velocity = cumulative_trapezoid(acc, time, initial=0)

    displacement = cumulative_trapezoid(velocity, time, initial=0)

    return displacement, velocity

def log_method(displacement):
    peaks, _ = find_peaks(displacement)

    displacement_peaks = displacement[peaks]
    # print(displacement_peaks)

    displacement_peaks = displacement_peaks[1:6]
    # print(displacement_peaks)

    log_quotient = np.log(np.abs(displacement_peaks[:-1] / displacement_peaks[1:]))
    # print(log_quotient)

    mean_log = np.mean(log_quotient)

    damping_ratio = mean_log / (2 * np.pi)

    return damping_ratio

    



natural_freq = damped_natural_frequency(data_acc)
print(natural_freq)

natural_omega = 2*np.pi*natural_freq
print(natural_omega)

displacement, velocity = integrate_acc(data_acc)
# print(displacement)

estimated_damping_ratio = log_method(displacement)
print(estimated_damping_ratio)

plt.plot(time, displacement)
plt.show()


# Construire la fonction de transfert complexe
fonction_transfert = Re_FRF + 1j * Im_FRF

# Calculer l'amplitude (module) et la phase (argument)
amplitude = np.abs(fonction_transfert)
phase = np.angle(fonction_transfert)

# Tracer l'amplitude et la phase
plt.figure()

# Amplitude
plt.subplot(2, 1, 1)
plt.plot(freq, amplitude)
plt.title("Amplitude de la fonction de transfert")
plt.xlabel("Fréquence (Hz)")
plt.ylabel("Amplitude")

# Phase
plt.subplot(2, 1, 2)
plt.plot(freq, phase)
plt.title("Phase de la fonction de transfert")
plt.xlabel("Fréquence (Hz)")
plt.ylabel("Phase (radians)")

plt.tight_layout()
plt.show()


#nyquist

# Tracer le diagramme de Nyquist
plt.figure()

plt.plot(Re_FRF/(natural_omega)**2, Im_FRF/(natural_omega)**2, label='Nyquist Plot')
# plt.plot(Re_FRF, -Im_FRF, linestyle='--', label='Conjugate symmetry')

# Ajout des axes pour plus de clarté
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)

# Titres et légendes
plt.title("Diagramme de Nyquist")
plt.xlabel("Partie réelle")
plt.ylabel("Partie imaginaire")
plt.legend()

plt.grid(True)
plt.axis('equal')  # Pour avoir des échelles identiques sur les deux axes
plt.show()
