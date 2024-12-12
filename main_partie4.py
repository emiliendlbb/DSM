import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy import integrate
from scipy.linalg import eig
import sympy 
import matplotlib as mpl

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


x = sympy.symbols("x")

#Paramètres
E= 210e9
rho= 7850
l = 1.2 
e = 0.15
t = 0.01
A = e**2-(e-2*t)**2
I = ((e**4)/12)-((e-2*t)**4)/12
M_mot = 0.4*188
M_dv = 75
J_B = 1
J_D = 10
k1l = 10**4 
k1r = 10**9 
k2l = 10**5
k2r = 10**4
d_B = 0.25 #distance entre le chassis et la masse B
L = 0.8
a = 0.2

#Fonctions de Ritz
def phi(x,l,n):
    f = (x/l)**(n)
    return f

# N = nombre de fonctions d'approximation de Rayleigh
#Matrice des raideurs
def Stiff(N):
    K = np.zeros((N,N))
    for i in range(0,N):
        for j in range(0,N):
            phi_i = phi(x,l,i)
            phi_j = phi(x,l,j)
            phi_i_0 = phi_i.evalf(subs={x: 0})
            phi_j_0 = phi_j.evalf(subs={x: 0})
            phi_i_L = phi_i.evalf(subs={x: L})
            phi_j_L = phi_j.evalf(subs={x: L})
            
            dphi_i = sympy.diff(phi_i,x)
            dphi_j = sympy.diff(phi_j,x)
            dphi_i_0 = dphi_i.evalf(subs={x: 0})
            dphi_j_0 = dphi_j.evalf(subs={x: 0})
            dphi_i_L = dphi_i.evalf(subs={x: L})
            dphi_j_L = dphi_j.evalf(subs={x: L})
            
            d2phi_i = sympy.diff(phi_i,x,x)
            d2phi_j = sympy.diff(phi_j,x,x)
    
            kij = sympy.integrate(E*I*d2phi_i*d2phi_j, (x, 0, l)) + k1l * phi_i_0*phi_j_0 + k1r * dphi_i_0 * dphi_j_0 + k2l * phi_i_L * phi_j_L + k2r * dphi_i_L * dphi_j_L
        
            K[i,j] = kij
    return K

#Matrice des masses
def Mass(N):
    M = np.zeros((N,N))
    for i in range(0,N):
        for j in range(0,N):
            phi_i = phi(x,l,i)
            phi_j = phi(x,l,j)
            phi_i_B = phi_i.evalf(subs={x: L/2})
            phi_j_B = phi_j.evalf(subs={x: L/2})
            phi_i_D = phi_i.evalf(subs={x: L+a})
            phi_j_D = phi_j.evalf(subs={x: L+a})
            
            dphi_i = sympy.diff(phi_i,x)
            dphi_j = sympy.diff(phi_j,x)
            dphi_i_B = dphi_i.evalf(subs={x: L/2})
            dphi_j_B = dphi_j.evalf(subs={x: L/2})
            dphi_i_D = dphi_i.evalf(subs={x: L+a})
            dphi_j_D = dphi_j.evalf(subs={x: L+a})
            
            mij = sympy.integrate(rho*A*phi_i*phi_j, (x, 0, l)) + M_mot * phi_i_B * phi_j_B + M_dv * phi_i_D * phi_j_D + (J_B+M_mot*(d_B)**2)*dphi_i_B*dphi_j_B+J_D*dphi_i_D*dphi_j_D
            M[i,j] = mij
    return M

def frequency_convergence(nth_frequency):
    Nritz = 13 #  = Nombre de fonctions jusqu'auquel on teste la convergence
    OM = np.zeros(Nritz-nth_frequency+1)
    N_range = np.zeros(Nritz-nth_frequency+1)
    
    for N in range(nth_frequency,Nritz+1): # Au moins nth_frequency pour avoir la freq correspondante
        K = Stiff(N)
        M = Mass(N)
        omegas_squared,eigen_vects = eig(K,M)
        omegas = np.sort(np.abs(np.sqrt(omegas_squared)))/2/np.pi
        OM[N-nth_frequency] = omegas[nth_frequency-1]
        print(N,'omega :', OM[N-nth_frequency])
        N_range[N-nth_frequency]=N
        
    plt.plot(N_range,OM, 'b')
    plt.plot(N_range, OM, 'bo')
    plt.title(f"Convergence de la fréquence propre {nth_frequency}")
    plt.xlabel("Nombre de fonctions d'approximation")
    plt.ylabel("Fréquence naturelle approximée [Hz]")
    plt.grid('True')
    plt.show()
    
    
    relative_errors=np.zeros(len)
    for i in range(0,len(OM)):
        

def relative_errors(freq_approx, freq_num):
    relative_error = (freq_approx-freq_num)/freq_num
    
    return relative_error

def plot_mode(mode):
    plt.plot(range(len(mode)), mode)
    plt.show()




if __name__ == "__main__":
    freqs_pitch_modes, modes, mode1, mode2, mode3, mode4 = load_data()
    
    frequency_convergence(1)
    frequency_convergence(2)
    frequency_convergence(3)
    frequency_convergence(4)
    
    # plot_mode(mode1)
    # plot_mode(mode2)
    # plot_mode(mode3)
    # plot_mode(mode4)