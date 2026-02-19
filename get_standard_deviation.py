# -*- coding: utf-8 -*-
"""
Created on Tue Feb 17 09:39:11 2026

@author: David mawson
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd

#==============================================================================
def gaussian(x, A, mu, sigma):
    # gaussian equation 
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2))

def red_chi(y_data, y_fit, errs):
    chi = np.sum(((y_fit-y_data)/errs)**2)
    return chi / len(y_data)

def fit_gaussian(data, PLOT = False):

    sizes = data[:, 0]
    intensities = data[:, 1]

    # Remove zeros
    mask = intensities > 0
    sizes = sizes[mask]
    intensities = intensities[mask]

    if len(sizes) < 3:
        return np.nan

    x_data = np.log(sizes)
    y_data = intensities

    # Constant errors
    errors = np.full(len(y_data), 1.0)

    # Stronger initial guesses
    A0 = np.max(y_data)
    mu0 = np.average(x_data, weights=y_data)
    sigma0 = np.sqrt(np.average((x_data - mu0)**2, weights=y_data))

    p0 = [A0, mu0, sigma0]

    popt, pcov = curve_fit(
        gaussian,
        x_data,
        y_data,
        p0=p0,
        sigma=errors,
        absolute_sigma=True,
        maxfev=10000
    )

    A, mu, sigma = popt

    if PLOT:
        plot_gaussian(x_data, y_data,errors, popt)

    return np.exp(mu) * sigma

def plot_gaussian(x_data, y_data, errors, params):

    
    #x = np.linspace(np.min(data[:, 0]), np.max(data[:, 0]), 10000)
    x = np.linspace(0,6, 10000)
    
    y_gauss = gaussian(x, params[0], params[1], params[2])
    y_fit = gaussian(x_data, params[0], params[1], params[2])

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.axvspan(np.exp(params[1]) - (np.exp(params[1])*params[2]), np.exp(params[1]) + (np.exp(params[1])*params[2]), alpha=0.25, 
               label=r'$\mu \pm \sigma$')

    ax.axvline(np.exp(params[1]), color="blue", linestyle = "--", 
               label=r"$\mu$")
    
    ax.errorbar(np.exp(x_data), y_data, errors, fmt="rx")
    

    ax.plot(np.exp(x), y_gauss, color="black", linestyle = "--",
            label =rf"mean = e^{params[1]:.2f} = {np.exp(params[1]):.2f}, $\sigma = \mu$ x {params[2]:.2f} = {np.exp(params[1])* params[2]:.2f}")
    
    chi_2R = red_chi(y_data, y_fit, errors)
    ax.scatter(0,0, color="white", label=rf"$\chi^2_R$ = {chi_2R:.2f}")
    ax.set_title("")
    ax.set_xlabel("")
    ax.set_ylabel("")
    #ax.set_xscale("log")
    #ax.set_xlim(0,1000)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.legend()
    plt.tight_layout()
    plt.show()
    


#==============================================================================


'''
df = pd.read_csv("data_test.csv", sep="\t", encoding="latin1")


intens = df.iloc[36, 74:143]
intens2 = df.iloc[62, 74:143]
size = df.iloc[1, 3:72]

intens = intens.astype(float)
intens2 = intens2.astype(float)
size = size.astype(float)


data = np.column_stack((size.values, intens.values))
data2 = np.column_stack((size.values, intens2.values))
stand_dev = fit_gaussian(data)
stand_dev1 = fit_gaussian(data2)
'''



