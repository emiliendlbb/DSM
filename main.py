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
    print(displacement_peaks)

    displacement_peaks = displacement_peaks[1:6]
    print(displacement_peaks)

    log_quotient = np.log(np.abs(displacement_peaks[:-1] / displacement_peaks[1:]))
    print(log_quotient)

    mean_log = np.mean(log_quotient)

    damping_ratio = mean_log / (2 * np.pi)

    return damping_ratio

    



freq = damped_natural_frequency(data_acc)
print(freq)

displacement, velocity = integrate_acc(data_acc)
# print(displacement)

estimated_damping_ratio = log_method(displacement)
print(estimated_damping_ratio)

plt.plot(time, displacement)
plt.show()