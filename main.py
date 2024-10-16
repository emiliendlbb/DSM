import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

current_dir = os.path.dirname(__file__)

freq_path = os.path.join(current_dir, 'files', 'P2024_frf_acc.txt')
acc_path = os.path.join(current_dir, 'files', 'P2024_irf_acc.txt')

data_freq = np.loadtxt(freq_path)
data_acc = np.loadtxt(acc_path)
time = data_acc[:, 0]
acc = data_acc[:, 1]


def damped_natural_frequency(data_acc):
    time = data_acc[:, 0]
    acc = data_acc[:, 1]

    peaks, _ = find_peaks(acc)

    time_peaks = time[peaks]
    print(time_peaks)

    time_peaks_difference = np.diff(time_peaks)
    print(time_peaks_difference)

    time_peaks_difference = time_peaks_difference[7:]
    print(time_peaks_difference)

    avg_time_peaks_difference = np.mean(time_peaks_difference)

    damped_natural_frequency = 1/avg_time_peaks_difference

    return damped_natural_frequency


def log_method(data_acc):
    time = data_acc[:, 0]
    acc = data_acc[:, 1]

    



freq = damped_natural_frequency(data_acc)
print(freq)


plt.plot(time, acc)
plt.show()