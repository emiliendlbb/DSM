import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


def load_data():
    current_dir = os.path.dirname(__file__)

    freq_path = os.path.join(current_dir, 'files', 'P2024_f_Part4.txt')
    modes_path = os.path.join(current_dir, 'files', 'P2024_Modes_Part4.txt')

    freq_data = np.loadtxt(freq_path)
    modes_data = np.loadtxt(modes_path)

    mode1 = modes_data[:, 0]
    mode2 = modes_data[:, 1]
    mode3 = modes_data[:, 2]
    mode4 = modes_data[:, 3]

    return freq_data, modes_data, mode1, mode2, mode3, mode4

def plot_mode(mode):
    plt.plot(range(len(mode)), mode)
    plt.show()

if __name__ == "__main__":
    freqs_pitch_modes, modes, mode1, mode2, mode3, mode4 = load_data()
    # plot_mode(mode1)
    # plot_mode(mode2)
    # plot_mode(mode3)
    # plot_mode(mode4)