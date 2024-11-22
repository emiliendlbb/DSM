import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
import os

# Charger les fréquences naturelles et les ratios d'amortissement
current_dir = os.path.dirname(__file__)

file_path3 = os.path.join(current_dir, 'files', 'P2024_frf_Part3_f_ds.txt')
file_path_2 = os.path.join(current_dir, 'files', 'P2024_modes_Part3.txt')
file_path = os.path.join(current_dir, 'files', 'P2024_f_eps_Part3.txt')
data = np.loadtxt(file_path, delimiter='\t')
frequencies = data[:, 0]  # Fréquences naturelles en Hz
damping_ratios = data[:, 1]
natural_omegas = 2 * np.pi * frequencies  # Conversion en rad/s

# Charger les formes modales
mode_shapes = np.loadtxt(file_path_2, delimiter='\t') / 1000

# Définir les fréquences d'excitation
excitation_freqs = np.linspace(0, 1500, 10000)  # Fréquences d'excitation en Hz
excitation_omegas = 2 * np.pi * excitation_freqs  # Conversion en rad/s

# Initialisation de la matrice de FRF en termes d'accélération
num_points = mode_shapes.shape[0]
FRF_matrix = np.zeros((num_points, num_points, len(excitation_omegas)), dtype=complex)

# Calcul de la FRF pour chaque fréquence d'excitation
for j in range(num_points):
    for k, omega_exc in enumerate(excitation_omegas):
        FRF_matrix[0, j, k] = sum(
            -(mode_shapes[0, n] * mode_shapes[j, n] * omega_exc**2) /
            (natural_omegas[n]**2 - omega_exc**2 + 1j * 2 * damping_ratios[n] * natural_omegas[n] * omega_exc)
            for n in range(len(frequencies))
        )

# Extraire la FRF entre le point d'excitation (fourche) et le siège du conducteur
FRF_driver_seat = FRF_matrix[0, 11, :]

# Appliquer un masque pour retirer le point où la fréquence est nulle
non_zero_mask = excitation_freqs >= 0
excitation_freqs_non_zero = excitation_freqs[non_zero_mask]
FRF_driver_seat_non_zero = FRF_driver_seat[non_zero_mask]

# Calculer l'amplitude sans la convertir en dB
FRF_amplitude = np.abs(FRF_driver_seat_non_zero)

# Charger les données de data3 pour la vérification
data3 = np.loadtxt(file_path3, delimiter='\t')
frequencies_data3 = data3[:, 0]  # Fréquences
FRF_real_data3 = data3[:, 1]     # Partie réelle de la FRF
FRF_imag_data3 = data3[:, 2]     # Partie imaginaire de la FRF

# Appliquer un masque pour data3 pour retirer la fréquence nulle
non_zero_mask_data3 = frequencies_data3 >= 0
frequencies_data3_non_zero = frequencies_data3[non_zero_mask_data3]
FRF_amp_data3_non_zero = np.sqrt(FRF_real_data3[non_zero_mask_data3]**2 + FRF_imag_data3[non_zero_mask_data3]**2)

# --- Tracé du diagramme de Bode ---
plt.figure(figsize=(10, 5))
plt.semilogy(excitation_freqs_non_zero, FRF_amplitude, label='FRF Calculé')
plt.semilogy(frequencies_data3_non_zero, FRF_amp_data3_non_zero, '--', label='FRF data3 (données expérimentales)')

# Trouver les maxima et minima
max_indices = argrelextrema(FRF_amplitude, np.greater)[0]
min_indices = argrelextrema(FRF_amplitude, np.less)[0]

# Ajouter des points aux maxima et minima sur le diagramme
plt.scatter(excitation_freqs_non_zero[max_indices], FRF_amplitude[max_indices], color='red', label='Maxima', zorder=5)
plt.scatter(excitation_freqs_non_zero[min_indices], FRF_amplitude[min_indices], color='blue', label='Minima', zorder=5)

# Annotation des points
for i in max_indices:
    plt.annotate(f'{excitation_freqs_non_zero[i]:.1f} Hz', (excitation_freqs_non_zero[i], FRF_amplitude[i]), 
                 textcoords="offset points", xytext=(0,10), ha='center', color='red')

for i in min_indices:
    plt.annotate(f'{excitation_freqs_non_zero[i]:.1f} Hz', (excitation_freqs_non_zero[i], FRF_amplitude[i]), 
                 textcoords="offset points", xytext=(0,-15), ha='center', color='blue')

plt.xlabel('Fréquence d\'excitation (Hz)')
plt.ylabel('Amplitude de la FRF [m/$s^2$/N]')
plt.title('Diagramme de Bode de la FRF entre la fourche et le siège conducteur')
plt.grid(True)
plt.legend()
plt.show()

# --- Tracé du diagramme de Nyquist ---
plt.figure(figsize=(6, 6))
plt.plot(np.real(FRF_driver_seat), np.imag(FRF_driver_seat), label='FRF Calculé')
plt.plot(FRF_real_data3, FRF_imag_data3, '--', label='FRF données expérimentales')

plt.xlabel('Partie Réelle [m/$s^2$/N]')
plt.ylabel('Partie Imaginaire [m/$s^2$/N]')
plt.title('Diagramme de Nyquist de la FRF entre la fourche et le siège conducteur')
plt.grid(True)
plt.axis('equal')
plt.legend()

plt.show()

# Affichage des points où la partie réelle est proche de zéro
real_zero_indices = np.where(np.isclose(np.real(FRF_driver_seat), 0, atol=1e-5))[0]
for index in real_zero_indices:
    print(f"À la fréquence d'excitation {excitation_freqs_non_zero[index]:.1f} Hz, "
          f"Partie réelle ≈ 0, Partie imaginaire = {np.imag(FRF_driver_seat_non_zero[index]):.5f}")
    
    
    
FRF_amp_interp = np.interp(frequencies_data3_non_zero, excitation_freqs_non_zero, FRF_amplitude)

# Calculer la différence entre les FRF pour les fréquences > 1 Hz
FRF_difference = (FRF_amp_interp - FRF_amp_data3_non_zero)*100/FRF_amp_data3_non_zero

# Tracer la différence de la FRF en fonction de la fréquence (à partir de 1 Hz)
plt.figure(figsize=(10, 5))
plt.plot(frequencies_data3_non_zero[300:], FRF_difference[300:], label='Différence de FRF (Calculé - Expérimental)', color='purple')
plt.xlabel('Fréquence (Hz)')
plt.ylabel('Différence d\'amplitude de la FRF [%]')
plt.title('Différence entre les FRF calculée et expérimentale')
plt.grid(True)
plt.legend()
plt.show()



# Distance des 14 points de référence en mm
positions = np.array([0, 100, 200, 300, 400, 400, 500, 600, 700, 800, 901, 1000, 1101, 1200])
# Fréquence d'excitation fixe en rad/s
omega_exc_fixed = 436  # en rad/s
freq_exc_fixed = omega_exc_fixed / (2 * np.pi)  # Conversion en Hz

# Trouver l'indice correspondant à la fréquence d'excitation fixe
index_fixed_freq = np.argmin(np.abs(excitation_freqs - freq_exc_fixed))

# Initialisation des listes pour stocker les amplitudes maximales et les fréquences correspondantes
max_amplitudes_per_point_fixed_freq = []
fixed_freqs_per_point = []

# Calcul de l'amplitude maximale de l'accélération pour chaque point à la fréquence fixe
for j in range(len(positions)):
    # Extraction de la FRF pour chaque point j par rapport au point d'excitation (fourche)
    FRF_point = FRF_matrix[0, j, :]  # FRF entre la fourche et le point j
    
    # Calculer l'amplitude de la FRF à la fréquence d'excitation fixe
    FRF_amplitude_fixed = np.abs(FRF_point[index_fixed_freq])
    
    # Multiplier par l'amplitude de la force d'excitation (450 N)
    amplitude_fixed = FRF_amplitude_fixed * 450
    fixed_frequency = excitation_freqs[index_fixed_freq]  # Fréquence correspondante (devrait être proche de 69.41 Hz)
    # Stocker les valeurs

    max_amplitudes_per_point_fixed_freq.append(amplitude_fixed)
    fixed_freqs_per_point.append(fixed_frequency)
    
    # Afficher le résultat pour chaque point
    print(f"Point P{j+1}: Amplitude = {amplitude_fixed:.2f} m/s² à {fixed_frequency:.2f} Hz")

# Tracé du graphique pour les 14 points à la fréquence fixe
plt.figure(figsize=(10, 5))
plt.plot(range(1, 15), max_amplitudes_per_point_fixed_freq, marker='o', linestyle='-', color='g', label=f'Amplitude à {omega_exc_fixed:.2f} rad/s')
plt.xticks(ticks=range(1, 15), labels=[f"P{i+1}" for i in range(14)], rotation=45)
plt.xlabel("Points de mesure le long du cadre")
plt.ylabel("Amplitude de l'accélération (m/$s^2$)")
plt.title(f"Amplitude de l'accélération aux points de référence du cadre à {omega_exc_fixed:.2f} rad/s")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()



# Définir les paramètres
F0 = 450  # Amplitude de la force en Newtons
omega_exc = 436  # Fréquence d'excitation en rad/s

# Temps de simulation
time = np.linspace(0, 0.15, 1000)  # Intervalle de 0 à 0.15 seconde, échantillonné en 1000 points
force_excitation = F0 * np.sin(omega_exc * time)  # Force d'excitation dans le temps

# Index correspondant à la fréquence d'excitation dans le vecteur des fréquences
excitation_freq_hz = omega_exc / (2 * np.pi)
excitation_index = np.argmin(np.abs(excitation_freqs - excitation_freq_hz))

# Récupérer la FRF au siège du conducteur (point D, ici supposé être le point P12)
FRF_driver_seat = FRF_matrix[0, 11, excitation_index]  # Point P12 correspond à l'index 11 si on commence à 0

# Calcul de la réponse en accélération en multipliant la FRF par la force d'excitation
acceleration_time_response = np.real(FRF_driver_seat * force_excitation)

# Calcul de la valeur maximale
max_acceleration = np.max(acceleration_time_response)

# Tracé de la réponse en accélération au cours du temps
plt.figure(figsize=(10, 5))
plt.plot(time, acceleration_time_response, color='teal', label='Réponse en accélération')
plt.axhline(y=max_acceleration, color='orange', linestyle='--', label='Valeur maximale = {:.2f} m/s²'.format(max_acceleration))
plt.xlabel("Temps (s)")
plt.ylabel("Accélération [m/$s^2$]")
plt.title("Réponse en accélération au siège du conducteur au cours du temps")
plt.grid(True)



plt.legend(loc='lower right')
plt.show()





# Constants
F0 = 450  # Amplitude of the force in Newtons
wavelength = 0.2  # Wavelength of the paved road in meters

# Define speed range in km/h and convert to m/s
speeds_kmh = np.linspace(50, 70, 1000)  # Speeds from 50 to 70 km/h
speeds_ms = speeds_kmh * 1000 / 3600   # Convert to m/s

# Calculate excitation frequencies for each speed
excitation_frequencies = (2 * np.pi * speeds_ms) / wavelength  # Excitation frequency in rad/s
# print(excitation_frequencies/(2*np.pi))
# Initialize array to store max acceleration response for each speed
max_accelerations = np.zeros(len(speeds_ms))

# Calculate the maximum acceleration response at the driver seat for each speed
for i, omega_exc in enumerate(excitation_frequencies):
    # Find the closest excitation frequency in the FRF
    excitation_freq_hz = omega_exc / (2 * np.pi)
    excitation_index = np.argmin(np.abs(excitation_freqs - excitation_freq_hz))
    
    # Get the FRF at the driver seat (point D, assumed to be point P12 or index 11)
    FRF_driver_seat = FRF_matrix[0, 11, excitation_index]  # Point P12 at index 11

    # Calculate the maximum acceleration for this frequency
    max_accelerations[i] = np.abs(FRF_driver_seat) * F0

# print(max(max_accelerations))
# print(excitation_frequencies[np.argmax(max_accelerations)]/(2*np.pi))
# print(speeds_kmh[np.argmax(max_accelerations)])
# Plot the maximum acceleration response as a function of speed
plt.figure(figsize=(10, 5))
plt.plot(speeds_kmh, max_accelerations, '-o', color='teal')
plt.xlabel("Vitesse de la moto (km/h)")
plt.ylabel("Amplitude maximale de l'accélération [m/$s^2$]")
plt.title("Amplitude de l'accélération maximale du siège du conducteur en fonction de la vitesse de la moto")
plt.grid(True)
plt.show()

