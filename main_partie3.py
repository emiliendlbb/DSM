import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.integrate import cumulative_trapezoid
import sympy as sp
from scipy.interpolate import interp1d

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

def plot_excitation_force(F_0, wavelength, speed, time_interval, sampling_rate=10000):
    Omega = speed / wavelength * 2* np.pi

    plt.figure()

    time = np.linspace(0, time_interval, (int)(sampling_rate*time_interval))
    F_z = F_0*np.sin(Omega*time)
    plt.plot(time, F_z)

    plt.xlabel("Temps [s]")
    plt.ylabel("Force [N]")

    plt.grid(True)
    plt.show()

    plt.show()   

def compute_FRF_matrix(natural_omegas, damping_ratios, modes, omegas_range):
    nb_modes = modes.shape[1]
    nb_points = modes.shape[0]
    nb_omega = len(omegas_range)

    FRF_matrix = np.zeros((nb_points, nb_points, nb_omega), dtype=complex)

    masses = np.ones(nb_modes)  # les modes sont normalisés du point de vue de la masse
    raideurs = (natural_omegas)**2 * masses

    for i, omega in enumerate(omegas_range):
        frf = np.zeros((nb_points, nb_points), dtype=complex)
        for mode in range(nb_modes):
            omega_r = natural_omegas[mode]
            mode_vector = modes[:, mode]
            numerator = np.outer(mode_vector, mode_vector)
            denominator = raideurs[mode] - omega**2 + 2 * 1j * masses[mode] * damping_ratios[mode] * omega_r * omega
            frf += -omega**2 * numerator/denominator
        FRF_matrix[:, :, i] = frf

    return FRF_matrix

def plot_Bode_Nyquist(FRF_matrix, omega_range, first_point_index, second_point_index, omega_frf, Re_frf, Im_frf):
    amplitude_FRF = np.abs(FRF_matrix[first_point_index, second_point_index, :])
    fonction_de_transfert = Re_frf + 1j * Im_frf
    amplitude_data = np.abs(fonction_de_transfert)

    freq_range = omega_range/(2*np.pi)
    freq_range = freq_range[150:]
    amplitude_FRF = amplitude_FRF[150:]
    freq_frf = omega_frf/(2*np.pi)
    freq_frf = freq_frf[150:]
    amplitude_data = amplitude_data[150:]

    maxima_indices, _ = find_peaks(20 * np.log10(amplitude_FRF))
    minima_indices, _ = find_peaks(-20 * np.log10(amplitude_FRF)) 

    borne_min = np.argmin(np.abs(freq_range - 3.0))
    borne_max = np.argmax(np.abs(freq_range - 4.0))

    recherche = amplitude_FRF[borne_min:borne_max]
    print("Recherche : ", recherche)
    # Bode
    plt.figure()
    plt.plot(freq_range, 20 * np.log10(amplitude_FRF), label='Amplitude FRF via matrice')
    plt.plot(freq_frf, 20 * np.log10(amplitude_data), label='Amplitude FRF via frf_f_ds', linestyle='--')
    plt.plot(freq_range[maxima_indices], 20 * np.log10(amplitude_FRF[maxima_indices]), 'ro', label='maximas')
    plt.plot(freq_range[minima_indices], 20 * np.log10(amplitude_FRF[minima_indices]), 'go', label='minimas')
    plt.xlabel('Fréquence [Hz]')
    plt.ylabel('Amplitude [dB]')
    plt.legend()
    plt.grid()
    plt.show
    # Nyquist
    plt.figure()
    nyquist_real_FRF = np.real(FRF_matrix[first_point_index, second_point_index, :])
    nyquist_imag_FRF = np.imag(FRF_matrix[first_point_index, second_point_index, :])
    plt.plot(nyquist_real_FRF, nyquist_imag_FRF, label='Nyquist via matrice')
    plt.plot(Re_frf, Im_frf, label='Nyquist via frf_f_ds', linestyle='--')

    plt.xlabel('Partie réelle [m/(s^2*N)]')
    plt.ylabel('Partie imaginaire [m/(s^2*N)]')
    plt.axhline(0, color='grey', lw=0.5, ls='--')
    plt.axvline(0, color='grey', lw=0.5, ls='--')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()

def max_amplitude_different_point(FRF_matrix, omega_range, F_0, wavelength, speed, time_interval, sampling_rate=10000):
    Omega = speed / wavelength * 2 * np.pi

    time = np.linspace(0, time_interval, (int)(sampling_rate*time_interval))
    F_z = F_0*np.sin(Omega*time)

    # # Fourier
    # F_w = np.array([np.sum(F_z * np.exp(-1j * 2 * np.pi * f * time)) for f in frequencies_range])

    max_amplitudes = np.zeros(FRF_matrix.shape[0], dtype=complex)

    # for i in range(FRF_matrix.shape[0]):
    #     X_w = FRF_matrix[i, 0, :] # * F_w
    #     max_amplitudes[i] = np.max(np.abs(X_w))
    
    approx_omega = omega_range[np.argmin(np.abs(omega_range - Omega))]
    print(approx_omega)
    print(Omega)
    for i in range(FRF_matrix.shape[0]):
        X_w = FRF_matrix[i, 0, np.argmin(np.abs(omega_range - Omega))] * F_0
        max_amplitudes[i] = X_w
        max_amplitudes[i] = np.abs(X_w)

    return np.abs(max_amplitudes)

def max_amplitude_specific_point(FRF_matrix, omega_range, F_0, wavelength, speed, time_interval, point_index, sampling_rate=10000):
    Omega = speed / wavelength * 2* np.pi

    time = np.linspace(0, time_interval, (int)(sampling_rate*time_interval))
    F_z = F_0*np.sin(Omega*time)

    # Fourier
    # F_w = np.array([np.sum(F_z * np.exp(-1j * 2 * np.pi * f * time)) for f in frequencies_range])
    
    # X_w = FRF_matrix[point_index, 0, :] * F_w

    # max_amplitude = np.max(np.abs(X_w))
    FRF_interp = interp1d(omega_range, FRF_matrix[point_index, 0, :], kind='cubic', fill_value="extrapolate")

    X_w = FRF_interp(Omega) * F_0
    max_amplitude = X_w
    max_amplitude = np.abs(X_w)

    return max_amplitude

def plot_max_amplitude_different_point(max_amplitudes):
    plt.figure()
    plt.plot(range(len(max_amplitudes)), max_amplitudes, 'o', color='red')
    plt.xlabel("Accéléromètre indice")
    plt.ylabel(r"$\max |X_{i,0}(\omega)|$")
    plt.title("Réponse maximum en fréquentiel")
    plt.grid(True)
    plt.show()

def plot_time_response_specific_point(FRF_matrix, omega_range, F_0, wavelength, speed, time_interval, point_index, sampling_rate=10000):
    Omega = speed / wavelength * 2* np.pi
    time = np.linspace(0, 0.15, (int)(sampling_rate*time_interval))
    
    FRF_driver_seat = FRF_matrix[point_index, 0, np.argmin(np.abs(omega_range - Omega))]
    F_z = F_0*np.sin(Omega*time)

    "acceleration_time_response = np.real(FRF_driver_seat * F_z)"
    acceleration_time_response = np.imag(FRF_driver_seat*F_0*np.exp(1j*Omega*time))

    max_acceleration = np.max(acceleration_time_response)

    plt.figure(figsize=(10, 5))
    plt.plot(time, acceleration_time_response, color='teal', label='Réponse en accélération')
    plt.axhline(y=max_acceleration, color='orange', linestyle='--', label='Valeur maximale = {:.2f} m/s²'.format(max_acceleration))
    plt.xlabel("Temps (s)")
    plt.ylabel("Accélération [m/$s^2$]")
    plt.title("Réponse en accélération au siège du conducteur au cours du temps")
    plt.grid(True)



    plt.legend(loc='lower right')
    plt.show()

def max_amplitude_different_speed(natural_omega, damping_ratios, modes, omega_frf, 
                                  F_0, wavelenght, time_interval,
                                  point_index=11, 
                                  min_speed=(50/3.6), max_speed=(70/3.6),
                                  sampling_rate=1000):
    
    new_damping_ratios = np.full_like(damping_ratios, 0.02)
    new_FRF_matrix = compute_FRF_matrix(natural_omega, new_damping_ratios, modes, omega_frf)

    max_amplitudes = np.zeros(sampling_rate, dtype=complex)
    for i in range(sampling_rate):
        speed = min_speed + i*(max_speed - min_speed)/sampling_rate
        max_amplitudes[i] = max_amplitude_specific_point(new_FRF_matrix, omega_frf, F_0, wavelenght, speed, time_interval, point_index)
    return np.abs(max_amplitudes)

def plot_max_amplitude_different_speed(max_amplitudes, sampling_rate=1000):
    plt.figure()
    speeds = np.linspace(50, 70, sampling_rate)

    plt.plot(speeds, max_amplitudes, '-', color='red', label='Interpolated curve')  # Ligne continue
    plt.xlabel("Vitesse [m/s]")
    plt.ylabel("Amplitude de l'accélération [m/s²]")
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    natural_freq, damping_ratios, freq_frf, Re_frf, Im_frf, modes, mode1, mode2, mode3, mode4 = load_data()
    natural_omega = 2*np.pi*natural_freq
    omega_frf = 2*np.pi*freq_frf
    modes = modes/1000      # m

    speed = 50/3.6          # m/s
    wavelength = 0.2        # m
    F_0 = 450               # N

    point_distance = np.array([0, 100, 200, 300, 400, 400, 500, 600, 700, 800, 901, 1000, 1101, 1200])

    time_interval = 0.15    # s
    plot_excitation_force(F_0, wavelength, speed, time_interval)

    # crée une matrice 14*14*nb_freq
    FRF_matrix = compute_FRF_matrix(natural_omega, damping_ratios, modes, omega_frf)
    
    """
    FRF_matrix2 = np.zeros((14, 14, len(omega_frf)), dtype=complex)
    for j in range(14):
        for k, omega_exc in enumerate(omega_frf):
            FRF_matrix[0, j, k] = sum(
                -(modes[0, n] * modes[j, n] * omega_exc**2) /
                (natural_omega[n]**2 - omega_exc**2 + 1j * 2 * damping_ratios[n] * natural_omega[n] * omega_exc)
                for n in range(len(natural_omega))
            )

    if np.allclose(FRF_matrix, FRF_matrix2, atol=1e-8):
        print("Les matrices sont identiques (avec une tolérance).")
    else:
        print("Les matrices ne sont pas identiques (même avec une tolérance).")
        diff = np.abs(FRF_matrix - FRF_matrix2)
        
        # Trouver la plus grande différence
        max_diff = np.max(diff)
        print("La plus grande différence est :", max_diff)
        """

    plot_Bode_Nyquist(FRF_matrix, omega_frf, 11, 0, omega_frf, Re_frf, Im_frf)

    max_amplitudes_points = max_amplitude_different_point(FRF_matrix, omega_frf, F_0, wavelength, speed, time_interval)

    plot_max_amplitude_different_point(max_amplitudes_points)

    plot_time_response_specific_point(FRF_matrix, omega_frf, F_0, wavelength, speed, time_interval, 11)

    max_amplitudes_speed = max_amplitude_different_speed(natural_omega, damping_ratios, modes, omega_frf,
                                                         F_0, wavelength, time_interval)
    
    plot_max_amplitude_different_speed(max_amplitudes_speed)
