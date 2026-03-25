# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 08:57:29 2026

@author: David
"""

import numpy as np
import matplotlib.pyplot as plt

D_min = 70.0       # minimum diameter nm
D_max = 300.0       # maximum diameter nm
N_bins = 500        # resolution

D = np.linspace(D_min, D_max, N_bins)

dD = D[1] - D[0]
D_F = 100.0        # filter pore diameter
w = 0.6 * D_F      # Gaussian width
D0 = 10.0          # minimum vesicle diameter
    
n_steps = 61       # number of extrusion passes

def initial_distribution(D):
    mu = np.log(200)
    sigma = 0.4
    P = (1/(D * sigma * np.sqrt(2*np.pi))) * np.exp(-(np.log(D) - mu)**2 / (2*sigma**2))
    return P / np.trapz(P, D)


def f(D):
    return (1 / (w * np.sqrt(np.pi))) * np.exp(-((D - D_F)**2) / w**2)


def extrusion_step(P):

    T = np.zeros_like(P)
    S = np.zeros_like(P)

    mass = D**2 * P

    for i, D_i in enumerate(D):

        lower_bound = np.sqrt(D_i**2 + D0**2)

        mask = D >= lower_bound
        mass_sum = np.sum(mass[mask]) * dD

        T[i] = f(D_i) * mass_sum / (D_i**2)

        if D_i > D0:
            arg = np.sqrt(D_i**2 - D0**2)
            S[i] = P[i] * f(arg)
        else:
            S[i] = 0.0

    P_new = P + T - S

    P_new = np.clip(P_new, 0, None)
    P_new /= np.trapz(P_new, D)

    return P_new


def mean_diameter(P):
    numerator = np.trapz(D**3 * P, D)
    denominator = np.trapz(D**2 * P, D)
    return numerator / denominator


def extrusion_mean_diameter():

    means = []
    
    P = initial_distribution(D)
    for n in range(n_steps):
        means.append(mean_diameter(P))
        P = extrusion_step(P)
    # plt.figure()
    # plt.plot(D, P, label=f'After {n_steps} extrusions')
    # plt.xlabel('Diameter (nm)')
    # plt.ylabel('P(D)')
    # plt.legend()
    # plt.title('Vesicle Size Distribution After Extrusion')
    
    # plt.figure()
    # plt.plot(range(n_steps), means, marker='o')
    # plt.xlabel('Extrusion step n')
    # plt.ylabel('Mean diameter (mass-weighted)')
    # plt.title('Mean Diameter vs Extrusion Number')
    
    # plt.show()
    return np.array(means)
    

means_diams = extrusion_mean_diameter()