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
omega_squared = (2 * np.pi * freq) ** 2
Re_FRF = data_freq[:, 1] / omega_squared
Im_FRF = data_freq[:, 2] / omega_squared
#Vu qu'on divise par 0 pour le premier élément, on a un nan à l'indice 0,
#Mise à 0 pour les graphiques
Re_FRF[0] = Re_FRF[1]
Im_FRF[0] = Re_FRF[1]
data_acc = np.loadtxt(acc_path)
time = data_acc[:, 0]
acc = data_acc[:, 1]

def damped_natural_frequency(data_acc):
    time = data_acc[:, 0]
    acc = data_acc[:, 1]

    peaks, _ = find_peaks(acc)

    time_peaks = time[peaks]

    time_peaks_difference = np.diff(time_peaks)

    time_peaks_difference = time_peaks_difference[7:]

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

    


#methode avec l'accélération

natural_freq_peaks = damped_natural_frequency(data_acc)
print(f"Estimated natural frequency from peaks: {natural_freq_peaks} Hz")

natural_omega_peaks = 2*np.pi*natural_freq_peaks
print(f"Estimated natural pulsation from peaks: {natural_omega_peaks} rad/s")

# displacement, velocity = integrate_acc(data_acc)
# # print(displacement)

damping_ratio_log_method = log_method(acc)
print(f"Estimated damping ratio from log method: {damping_ratio_log_method}")

plt.plot(time, acc)
plt.show()





#méthode avec diagramme de Bode

fonction_transfert = (Re_FRF + 1j * Im_FRF)

# Calculer l'amplitude (module) et la phase (argument)
amplitude = np.abs(fonction_transfert)
phase = np.angle(fonction_transfert)

pi_over_2_phase_index = np.argmin(np.abs(np.abs(phase)-np.pi/2))
natural_frequency_bode = freq[pi_over_2_phase_index]

peak_bode = np.argmax(amplitude)
max_amplitude_frequency_bode = freq[peak_bode]

print(f"Estimated natural frequency from Bode plot: {natural_frequency_bode} Hz")
print(f"Maximum amplitude frequency from Bode plot: {max_amplitude_frequency_bode} Hz")

plt.figure()

# Amplitude
plt.subplot(2, 1, 1)
plt.plot(freq, amplitude, label="Amplitude")
plt.axvline(x=max_amplitude_frequency_bode, color='r', linestyle='--', label=f"Max Amplitude @ {max_amplitude_frequency_bode:.2f} Hz")
plt.axvline(x=natural_frequency_bode, color='g', linestyle='--', label=f"Phase = pi/2 @ {natural_frequency_bode:.2f} Hz")
plt.title("Amplitude de la fonction de transfert")
plt.xlabel("Fréquence (Hz)")
plt.ylabel("Amplitude")

# Phase
plt.subplot(2, 1, 2)
plt.plot(freq, phase, label="Phase")
plt.axvline(x=natural_frequency_bode, color='g', linestyle='--', label=f"Phase = pi/2 @ {natural_frequency_bode:.2f} Hz")
plt.xlim(left=freq[0] - 4)
xmin, xmax = plt.gca().get_xlim()
plt.plot([xmin, natural_frequency_bode], [np.pi/2, np.pi/2], 'g--', label=f"Phase = pi/2")
plt.yticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi], [r'$0$', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$', r'$\frac{3\pi}{4}$', r'$\pi$'])
plt.title("Phase de la fonction de transfert")
plt.xlabel("Fréquence (Hz)")
plt.ylabel("Phase (radians)")

plt.tight_layout()
plt.show()


#half-power method

half_power_amplitude = amplitude[peak_bode] / np.sqrt(2)
half_power_indices = np.where(amplitude >= half_power_amplitude)[0]
bandwidth_freqs = freq[half_power_indices]

delta_f = bandwidth_freqs[-1] - bandwidth_freqs[0]
print(f"Bandwidth Δf: {delta_f} Hz")

Q_factor = natural_frequency_bode / delta_f
print(f"Estimated Quality Factor (Q): {Q_factor}")

damping_ratio_half_power = 1 / (2 * Q_factor)
print(f"Estimated damping ratio using half-power method: {damping_ratio_half_power}")



natural_frequency_nyquist = freq[102]
print(f"Estimated natural frequency from Nyquist plot: {natural_frequency_nyquist} Hz")

nyquist_amplitude_natural_frequency = np.sqrt(Im_FRF[102]**2 + Re_FRF[102]**2)
equivalent_mass = 87.5
damping_ratio_nyquist = 1/(2*equivalent_mass*nyquist_amplitude_natural_frequency)
print(f"Estimated damping ratio using Nyquist plot: {damping_ratio_nyquist}")

#nyquist
# Tracer le diagramme de Nyquist
plt.figure()

plt.plot(Re_FRF, Im_FRF, label='Nyquist Plot')
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
