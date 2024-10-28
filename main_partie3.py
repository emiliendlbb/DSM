import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.integrate import cumulative_trapezoid
import sympy as sp

def load_data():
    current_dir = os.path.dirname(__file__)

    frf_path = os.path.join(current_dir, 'files', 'P2024_frf_Part3_f_ds.txt')
    modes_path = os.path.join(current_dir, 'files', 'P2024_modes_Part3.txt')
    damping_path = os.path.join(current_dir, 'files', 'P2024_f_eps_Part3.txt')

    frf_data = np.loadtxt(frf_path)
    modes_data = np.loadtxt(modes_path)
    damping_data = np.loadtxt(damping_path)

    natural_freq = damping_data[:, 0]
    damping_ratios = damping_data[:, 1]

    freq_frf = frf_data[:, 0]
    Re_frf = frf_data[:, 1]
    Im_frf = frf_data[:, 2]

    mode1 = modes_data[:, 0]
    mode2 = modes_data[:, 1]
    mode3 = modes_data[:, 2]
    mode4 = modes_data[:, 3]

    return natural_freq, damping_ratios, freq_frf, Re_frf, Im_frf, modes_data, mode1, mode2, mode3, mode4

def plot_excitation_force(F_0, wavelenght, speed, time_interval, sampling_rate=10000):
    Omega = speed / wavelenght * 2* np.pi

    plt.figure()

    time = np.linspace(0, time_interval, (int)(sampling_rate*time_interval))
    F_z = F_0*np.sin(Omega*time)
    plt.plot(time, F_z)

    plt.title("Force excitatrice")
    plt.xlabel("Temps [s]")
    plt.ylabel("Force [N]")

    plt.grid(True)
    plt.show()

    plt.show()   

def compute_FRF_matrix(natural_frequencies, damping_ratios, modes, frequencies_range):
    nb_modes = modes.shape[1]
    nb_points = modes.shape[0]
    nb_freq = len(frequencies_range)

    FRF_matrix = np.zeros((nb_points, nb_points, nb_freq), dtype=complex)

    masses = np.ones(nb_modes)  # les modes sont normalisés du point de vue de la masse
    raideurs = (2*np.pi*natural_frequencies)**2 * masses

    for i, freq in enumerate(frequencies_range):
        frf = np.zeros((nb_points, nb_points), dtype=complex)
        for mode in range(nb_modes):
            omega = 2*np.pi*freq
            omega_r = 2*np.pi*natural_frequencies[mode]
            mode_vector = modes[:, mode]
            numerator = np.outer(mode_vector, mode_vector)
            denominator = raideurs[mode] - masses[mode]*omega**2 + 2 * 1j * masses[mode] * damping_ratios[mode] * omega_r * omega
            frf += -omega**2 * numerator/denominator
        FRF_matrix[:, :, i] = frf

    return FRF_matrix

def plot_Bode_Nyquist(FRF_matrix, frequencies_range, first_point_index, second_point_index, freq_frf, Re_frf, Im_frf):
    plt.figure()

    amplitude_FRF = np.abs(FRF_matrix[first_point_index, second_point_index, :])
    fonction_de_transfert = Re_frf + 1j * Im_frf
    amplitude_data = np.abs(fonction_de_transfert)

    # Bode
    plt.subplot(2, 1, 1)
    plt.semilogx(frequencies_range, 20 * np.log10(amplitude_FRF), label='FRF_matrix')
    plt.semilogx(freq_frf, 20 * np.log10(amplitude_data), label='Data', linestyle='--')
    plt.title('Bode Amplitude')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude (dB)')
    plt.legend()
    plt.grid()

    # Nyquist
    plt.subplot(2, 1, 2)
    nyquist_real_FRF = np.real(FRF_matrix[first_point_index, second_point_index, :])
    nyquist_imag_FRF = np.imag(FRF_matrix[first_point_index, second_point_index, :])
    plt.plot(nyquist_real_FRF, nyquist_imag_FRF, label='FRF_matrix')
    plt.plot(Re_frf, Im_frf, label='Data', linestyle='--')

    plt.title('Nyquist')
    plt.xlabel('Real Part')
    plt.ylabel('Imaginary Part')
    plt.axhline(0, color='grey', lw=0.5, ls='--')
    plt.axvline(0, color='grey', lw=0.5, ls='--')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()

def max_amplitude_different_point(FRF_matrix, frequencies_range, F_0, wavelenght, speed, time_interval, sampling_rate=10000):
    Omega = speed / wavelenght * 2* np.pi

    time = np.linspace(0, time_interval, (int)(sampling_rate*time_interval))
    F_z = F_0*np.sin(Omega*time)

    # Fourier
    F_w = np.array([np.sum(F_z * np.exp(-1j * 2 * np.pi * f * time)) for f in frequencies_range])

    max_amplitudes = np.zeros(FRF_matrix.shape[0])

    for i in range(FRF_matrix.shape[0]):
        X_w = FRF_matrix[i, 0, :] * F_w
        max_amplitudes[i] = np.max(np.abs(X_w))

    return max_amplitudes

def max_amplitude_specific_point(FRF_matrix, frequencies_range, F_0, wavelenght, speed, time_interval, point_index, sampling_rate=10000):
    Omega = speed / wavelenght * 2* np.pi

    time = np.linspace(0, time_interval, (int)(sampling_rate*time_interval))
    F_z = F_0*np.sin(Omega*time)

    # Fourier
    F_w = np.array([np.sum(F_z * np.exp(-1j * 2 * np.pi * f * time)) for f in frequencies_range])

    
    X_w = FRF_matrix[point_index, 0, :] * F_w

    max_amplitude = np.max(np.abs(X_w))

    return max_amplitude

def plot_max_amplitude_different_point(max_amplitudes):
    plt.figure()
    plt.plot(range(len(max_amplitudes)), max_amplitudes, 'o', color='red')
    plt.xlabel("Accéléromètre indice")
    plt.ylabel(r"$\max |X_{i,0}(\omega)|$")
    plt.title("Réponse maximum en fréquentiel")
    plt.grid(True)
    plt.show()

def max_amplitude_different_speed(natural_freq, damping_ratios, modes, freq_frf, 
                                  F_0, wavelenght, time_interval,
                                  point_index=11, 
                                  min_speed=(50/3.6), max_speed=(70/3.6),
                                  sampling_rate=100):
    
    new_damping_ratios = np.full_like(damping_ratios, 0.02)
    new_FRF_matrix = compute_FRF_matrix(natural_freq, new_damping_ratios, modes, freq_frf)


    max_amplitudes = np.zeros(sampling_rate)
    for i in range(sampling_rate):
        speed = min_speed + i*(max_speed - min_speed)/sampling_rate
        max_amplitudes[i] = max_amplitude_specific_point(new_FRF_matrix, freq_frf, F_0, wavelenght, speed, time_interval, point_index)
    return max_amplitudes

def plot_max_amplitude_different_speed(max_amplitudes, sampling_rate=100):
    plt.figure()
    speeds = np.linspace(50/3.6, 70/3.6, sampling_rate)
    plt.plot(speeds, max_amplitudes, 'o', color='red')
    plt.xlabel("Vitesse [m/s]")
    plt.ylabel("Amplitude")
    plt.title("Réponse maximum en fréquentiel")
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    natural_freq, damping_ratios, freq_frf, Re_frf, Im_frf, modes, mode1, mode2, mode3, mode4 = load_data()
    modes = modes/1000      # m

    speed = 50/3.6          # m/s
    wavelenght = 0.2        # m
    F_0 = 450               # N

    point_distance = np.linspace(0, 1200, 13)
    point_distance = np.insert(point_distance, 5, 400)

    time_interval = 0.15    # s
    # plot_excitation_force(F_0, wavelenght, speed, time_interval)

    # crée une matrice 14*14*nb_freq
    FRF_matrix = compute_FRF_matrix(natural_freq, damping_ratios, modes, freq_frf)

    # plot_Bode_Nyquist(FRF_matrix, freq_frf, 0, 11, freq_frf, Re_frf, Im_frf)

    max_amplitudes_points = max_amplitude_different_point(FRF_matrix, freq_frf, F_0, wavelenght, speed, time_interval)

    plot_max_amplitude_different_point(max_amplitudes_points)

    max_amplitudes_speed = max_amplitude_different_speed(natural_freq, damping_ratios, modes, freq_frf,
                                                         F_0, wavelenght, time_interval)
    
    plot_max_amplitude_different_speed(max_amplitudes_speed)

        