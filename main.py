import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.integrate import cumulative_trapezoid
import sympy as sp


def partie2():
    current_dir = os.path.dirname(__file__)

    freq_path = os.path.join(current_dir, 'files', 'P2024_frf_acc.txt')
    acc_path = os.path.join(current_dir, 'files', 'P2024_irf_acc.txt')

    data_freq = np.loadtxt(freq_path)
    freq = data_freq[:, 0]
    omega_squared = (2 * np.pi * freq) ** 2
    Re_FRF = - data_freq[:, 1] / omega_squared
    Im_FRF = - data_freq[:, 2] / omega_squared
    #Vu qu'on divise par 0 pour le premier élément, on a un nan à l'indice 0,
    #Mise à aux éléments 1 pour les graphiques
    Re_FRF[0] = Re_FRF[1]
    Im_FRF[0] = Im_FRF[1]
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

        damping_ratio_approx = mean_log / (2 * np.pi)

        x = sp.symbols('x')

        # Définir l'équation
        equation = sp.Eq(mean_log, (2 * sp.pi * x) / sp.sqrt(1 - x**2))

        # Résoudre l'équation pour x
        damping_ratio_exact = sp.solve(equation, x)

        return damping_ratio_approx, damping_ratio_exact

        


    #methode avec l'accélération

    natural_freq_peaks = damped_natural_frequency(data_acc)
    print(f"Estimated natural frequency from peaks: {natural_freq_peaks} Hz")

    natural_omega_peaks = 2*np.pi*natural_freq_peaks
    print(f"Estimated natural pulsation from peaks: {natural_omega_peaks} rad/s")

    displacement, velocity = integrate_acc(data_acc)
    # print(displacement)

    damping_ratio_log_method_approx, damping_ratio_log_method_exact = log_method(displacement)
    print(f"Estimated damping ratio from log method: {damping_ratio_log_method_approx}")
    print(f"Exact damping ratio from log method: {damping_ratio_log_method_exact}")
    print(f"Relative error from log method: {(damping_ratio_log_method_approx - damping_ratio_log_method_exact)/damping_ratio_log_method_exact}")


    plt.plot(time, acc)
    plt.grid()
    plt.show()
    plt.plot(time, velocity)
    plt.grid()
    plt.show()
    plt.plot(time, displacement)
    plt.grid()
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

    plt.figure(figsize=(10, 8))


    amplitude_dB = 20*np.log10(amplitude)
    max_amplitude_dB = np.max(amplitude_dB)
    half_power_amplitude_dB = max_amplitude_dB - 3  # Demi-puissance en dB

    half_power_indices = np.where(amplitude_dB >= half_power_amplitude_dB)[0]
    bandwidth_freqs = freq[half_power_indices]
    half_power_frequencies = (bandwidth_freqs[0], bandwidth_freqs[-1])
    min_amplitude_dB = np.min(amplitude_dB)
    natural_frequency_dB = amplitude_dB[np.argmin(np.abs(freq - natural_frequency_bode))]

    # Amplitude
    plt.subplot(2, 1, 1)
    plt.grid()

    # Configurer les graduations de l'axe y avec un intervalle de 10 dB
    y_min = np.floor(np.min(amplitude_dB) / 5) * 5
    y_max = np.ceil(np.max(amplitude_dB) / 5) * 5
    plt.yticks(np.arange(y_min, y_max + 1, 5))  # Intervalle de 10 dB

    plt.ylim([np.min(amplitude_dB) - 2, max_amplitude_dB + 2])
    x_min, x_max = plt.gca().get_xlim()
    y_min, y_max = plt.gca().get_ylim()

    plt.plot(freq, amplitude_dB, label="Amplitude (dB)")
    plt.plot([max_amplitude_frequency_bode, max_amplitude_frequency_bode], [y_min, max_amplitude_dB], 'r--')
    plt.plot([x_min, max_amplitude_frequency_bode], [max_amplitude_dB, max_amplitude_dB], 'r--')
    plt.plot(max_amplitude_frequency_bode, max_amplitude_dB, 'ro')
    plt.plot([x_min, half_power_frequencies[1]], [half_power_amplitude_dB, half_power_amplitude_dB], 'b--')
    plt.plot([half_power_frequencies[0], half_power_frequencies[0]], [y_min, half_power_amplitude_dB], 'b--')
    plt.plot([half_power_frequencies[1], half_power_frequencies[1]], [y_min, half_power_amplitude_dB], 'b--')
    plt.plot(half_power_frequencies[0], half_power_amplitude_dB, 'bo')
    plt.plot(half_power_frequencies[1], half_power_amplitude_dB, 'bo')
    plt.plot([natural_frequency_bode, natural_frequency_bode], [y_min, max_amplitude_dB - 0.4], 'g--')
    plt.plot(natural_frequency_bode, max_amplitude_dB - 0.4, 'go')
    plt.annotate(r'$\frac{{X}_{max}}{\sqrt{2}}$', xy=(max_amplitude_frequency_bode-90, half_power_amplitude_dB), color='blue')
    plt.annotate(r'${X}_{max}$', xy=(max_amplitude_frequency_bode-90, max_amplitude_dB), color='red')
    plt.annotate(r'$f_a$', xy=(max_amplitude_frequency_bode + 1.5, max_amplitude_dB-18), color='red')
    mid_freq = (half_power_frequencies[0] + half_power_frequencies[1]) / 2
    plt.annotate(r'$\Delta f$', xy=(mid_freq, half_power_amplitude_dB - 11), color='blue', ha='center')

    plt.annotate('', 
                xy=(half_power_frequencies[0], half_power_amplitude_dB - 12), 
                xytext=(half_power_frequencies[1], half_power_amplitude_dB - 12),
                arrowprops=dict(arrowstyle='<->', color='blue'))

    plt.title("Amplitude de la fonction de transfert")
    plt.xlabel("Fréquence (Hz)")
    plt.ylabel("Amplitude (dB)")

    # Phase
    plt.subplot(2, 1, 2)
    plt.grid()
    x_min, x_max = plt.gca().get_xlim()
    y_min, y_max = plt.gca().get_ylim()
    plt.plot(freq, phase + np.pi, label="Phase")
    plt.xlim(left=freq[0] - 4)
    plt.plot([natural_frequency_bode, natural_frequency_bode], [y_min, np.pi/2], 'g--')
    plt.plot([x_min - 4, natural_frequency_bode], [np.pi/2, np.pi/2], 'g--')
    plt.plot(natural_frequency_bode, np.pi/2, 'go')
    plt.annotate(r'$f_0$', xy=(natural_frequency_bode + 1.5, y_min), color='green')
    plt.yticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi], [r'$-\pi$', r'$-\frac{3\pi}{4}$', r'$-\frac{\pi}{2}$', r'$-\frac{\pi}{4}$', r'$0$'])
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


    # Nyquist en accélération
    # Re_FRF = data_freq[:, 1]
    # Im_FRF = data_freq[:, 2]

    # nyquist_amplitude_natural_frequency = np.sqrt(Im_FRF[102]**2 + Re_FRF[102]**2)
    # equivalent_mass = 87.5
    # damping_ratio_nyquist = 1/(2*equivalent_mass*nyquist_amplitude_natural_frequency)
    # print(f"Estimated damping ratio using Nyquist plot: {damping_ratio_nyquist}")

    # Nyquist en déplacement
    nyquist_amplitude_natural_frequency = np.sqrt(Im_FRF[102]**2 + Re_FRF[102]**2)
    equivalent_mass = 87.5
    damping_ratio_nyquist = 1/(2*equivalent_mass*nyquist_amplitude_natural_frequency*(2*np.pi*natural_frequency_nyquist)**2)
    print(f"Estimated damping ratio using Nyquist plot: {damping_ratio_nyquist}")

    #nyquist
    # Tracer le diagramme de Nyquist
    plt.figure()

    plt.plot(Re_FRF, Im_FRF, label='Nyquist Plot')
    plt.plot(Re_FRF[102], Im_FRF[102], 'ro')
    # plt.plot(Re_FRF, -Im_FRF, linestyle='--', label='Conjugate symmetry')
    plt.annotate("A", (Re_FRF[102], Im_FRF[102]), textcoords="offset points", xytext=(5,5), ha='center', color='red')

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



if __name__ == '__main__':
    partie2()