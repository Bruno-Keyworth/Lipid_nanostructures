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

def fit_gaussian(data, PLOT = True):
    
    # data folows gaussian better when plotted on a log scale. 
    # fit is better when fitting to the log of x values
    x_data = np.log(data[:, 0])
    y_data = data[:, 1]
    errors = np.linspace(10,10,69)
    
    # avoid zeros
    mask = y_data > 0
    y_data = y_data[mask]
    x_data = x_data[mask]
    errors = errors[mask]
    
    # guesses 
    A0 = np.max(np.log(data[:, 1]))
    mu0 = np.mean(data)
    sigma0 = np.std(data, ddof=1)
    p0 = [A0, mu0, sigma0]
    
    #fitting procedure for a gaussian
    popt, pcov = curve_fit(
    gaussian,
    x_data,
    y_data,
    p0=p0,
    sigma=errors,
    absolute_sigma=True)
    
    A, mu, sigma = popt
    A_err, mu_err, sigma_err = np.sqrt(np.diag(pcov))
    params = [A, mu, sigma, A_err, mu_err, sigma_err]
    
    if PLOT:
        plot_gaussian(x_data, y_data, params)
    
    stan_dev = np.exp(params[1])* params[2]
    mean = np.exp(params[1])
    
    return stan_dev

def plot_gaussian(x_data, y_data, params):
    
    errors = np.linspace(0.5,0.5,69)
    
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



