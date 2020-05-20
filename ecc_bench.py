#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 07:26:24 2018

@author: cjburke
Utilties for handling of eccentricity for transit model
fitting.  Kepler equation solver plus 
  find the true anomaly that minimizes the separation between planets/stars
  This is needed to get true anomaly equivalent to the mid transit time
"""
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
import timeit
import scipy.interpolate as intp
from ke import E
import theano
import theano.tensor as tt
import exoplanet as xo


#@profile
def ekepl2(emin, ein, iterations=2):
    """
    Returns E that solves Kepler's equation em=E-esinE
    Based upon odell and gooding 1986 celestial mechanics
    Two or three iterations needed also vectorized with numpy
    """
    if np.isscalar(emin):
        em = np.atleast_1d(emin)
    else:
#        em = emin
        em = np.copy(emin)

    if np.isscalar(ein):
        e = np.atleast_1d(ein)
    else:
#        e = ein
        e = np.copy(ein)
    if len(em) > len(e):
        e = np.full_like(em, e[0])
    if len(e) > len(em):
        em = np.full_like(e, em[0])
    twopi = np.ones_like(em) * 2.0 * np.pi
    ahalf = 0.5
    asixth = ahalf / 3.0
    athird = asixth * 2.0
    a = (np.pi - 1.0)*(np.pi - 1.0)/ (np.pi + 2.0/3.0)
    b = 2.0*(np.pi - asixth)*(np.pi - asixth) / (np.pi + 2.0/3.0)
    
    # Reduce range in input em to between -pi to pi
    emr = np.mod(em, twopi)
    emr = np.where(emr < -np.pi, emr + twopi, emr)
    emr = np.where(emr > np.pi, emr - twopi, emr)
    ee = np.abs(emr)
    
    # emr is range reduced em and ee is its absolute value
    # Starter value of E
    ee = np.where(ee < asixth, np.power(6.0*ee, athird), ee)
    w = np.pi - ee
    ee = np.where(ee >= asixth, np.pi - a*w/(b-w), ee)
    ee = np.where(emr < 0.0, -ee, ee)
    ee = emr + (ee-emr)*e
    e1 = 1.0 - e
    L = (e1 + ee*ee / 6.0) >= 0.1
    anyL = len(L) - np.sum(L)
    f = np.zeros_like(em)
    fd = np.zeros_like(em)
    for k in range(iterations):
        fdd = e*np.sin(ee)
        fddd = e*np.cos(ee)
        if (anyL == 0):
            f = (ee-fdd)-emr
            fd = 1.0-fddd
        else:
            f[L] = ee[L] - fdd[L] - emr[L]
            fd[L] = 1.0 - fddd[L]
            notL = np.logical_not(L)
            f[notL] = emkepl(e[notL], ee[notL]) - emr[notL]
            fd[notL] = e1[notL] + 2.0*e[notL]*np.power(np.sin(ahalf*ee[notL]),2)
        dee = f*fd / (ahalf*f*fdd - fd*fd)
        w = fd + ahalf*dee*(fdd + athird*dee*fddd)
        fd = fd+dee*(fdd+ahalf*dee*fddd)
        ee = ee - (f-dee*(fd-w)) / fd
    return ee + (em-emr)

#@profile
def emkepl(e, ee):
    # Supporting function for ekepl2
    x = (1.0-e)*np.sin(ee)
    ee2 = -ee*ee
    term = ee
    d = np.zeros_like(ee)
    x0 = np.zeros_like(ee)
    while not (np.sum(x - x0) == 0.0):
        d = d+2.0
        term = term*ee2/ (e*(d+1.0))
        x0 = x
        x = x - term
    return x


if __name__ == '__main__':
    # Test native python only ekepl2 and show precision of results
    # Note that using single fixed value of ecc use_e rather than 
    #  a random or linspaced array because cordic uses scalar ecc.
    use_e = 0.95
    meanAnom = np.linspace(0.0, 2.0*np.pi, 5000) # with 5000 effective 5000^2 tests
    ecc = np.full_like(meanAnom, use_e)
    # ecc = np.linspace(0.0, 0.98, 5000)
    meanAnom_2d, ecc_2d = np.meshgrid(meanAnom, ecc)
    tmp1 = meanAnom_2d.flatten()
    tmp2 = ecc_2d.flatten()
    start_time = timeit.default_timer()
    tmp3 = ekepl2(tmp1, tmp2, iterations=3)
    print(timeit.default_timer() - start_time)
    tmp3 = np.reshape(tmp3, (len(ecc), len(meanAnom)))
    error_2d = np.abs(tmp3 - ecc_2d*np.sin(tmp3) - meanAnom_2d)
    log_error_2d = np.log10(error_2d + 1.0e-20)
    print('Log of Max Error: {:f}'.format(np.max(log_error_2d.flatten())))
    

    # Test Zechmeister's cordic-like solution to KepEq. https://github.com/mzechmeister/ke    
    # Note the function accepcts a scalar in eccentricity
    start_time = timeit.default_timer()
    tmp3 = E(tmp1, use_e, typ='E1N', n=22)
    print(timeit.default_timer() - start_time)
    tmp3 = np.reshape(tmp3, (len(ecc), len(meanAnom)))
    error_2d = np.abs(tmp3 - ecc_2d*np.sin(tmp3) - meanAnom_2d)
    log_error_2d = np.log10(error_2d + 1.0e-20)
    print('Log of Max Error: {:f}'.format(np.max(log_error_2d.flatten())))

    start_time = timeit.default_timer()
    op = xo.theano_ops.kepler.KeplerOp()
    M_t = tt.dvector()
    e_t = tt.dvector()
    func = theano.function([M_t, e_t], op(M_t, e_t))
    sinf0, cosf0 = func(tmp1, tmp2)
    print(timeit.default_timer() - start_time)
    # Givs results in terms of true anomaly ovr range -pi<th<pi convert back to ecc anom
    #  0<E<2pi
    #cosE = (tmp2 + cosf0)/(1.0 + tmp2*cosf0)
    tmp3 = np.arctan2(np.sqrt(1.0-tmp2*tmp2)*sinf0, tmp2+cosf0)
    idx = np.where(tmp3 < 0.0)[0]
    tmp3[idx] = tmp3[idx] + np.pi*2.0
    tmp3 = np.mod(tmp3, 2.0*np.pi)
    meanAnom_2d = np.mod(meanAnom_2d, 2.0*np.pi)
    tmp3 = np.reshape(tmp3, (len(ecc), len(meanAnom)))
    error_2d = np.abs(tmp3 - ecc_2d*np.sin(tmp3) - meanAnom_2d)
    log_error_2d = np.log10(error_2d + 1.0e-20)
    print('Log of Max Error: {:f}'.format(np.max(log_error_2d.flatten())))

    
