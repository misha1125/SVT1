import numpy as np
from scipy.integrate import quad
from scipy.special import erf

dx = 1e-1
dy = 0.1
a = 20
c_0 = 1
N = 100


def under_int(x, y, tao):
    return tao ** (-1.5) * (erf((a + y) / np.sqrt(4 * dy * tao)) + erf((a - y) / np.sqrt(4 * dy * tao))) \
           * np.exp(-((x - tao) ** 2 / (4 * dx * tao)))

def under_integral_fun(x, y, tau):
    divider = np.sqrt(4*dy*tau)
    return np.power(tau, -1.5) * (erf((a+y)/divider) + erf((a-y)/divider))\
           * np.exp(-np.power(((x-tau)/np.sqrt(4*dx*tau)),2))

def integrated_function(x, y, T,  M = 200):
    n_j_array = [np.cos((2*j - 1)*np.pi/2/M) for j in range(1,M+1)]
    return x*c_0/(np.sqrt(16*np.pi*dx)) * np.pi * T / 2 / M * np.sum(np.array([np.sqrt(1-np.power(n_j,2))*\
                                       under_integral_fun(x, y, T*(n_j + 1)/2) for n_j in n_j_array]))

def to_coord(i, j):
    return i * 200 / N, j * 200 / N - 100

matrix = np.zeros((N, N))
for i in range(1, N):
    for j in range(1, N):
        x,y = to_coord(i, j)
        matrix[i-1,j-1] = integrated_function(x, y, 50,  M = 1000)

import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(matrix.T, cmap="viridis")
plt.show()
