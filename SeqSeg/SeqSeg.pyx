#!python
#cython: boundscheck=False, wraparound=False
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 17:52:52 2017

@author: paulo.hubert@gmail.com

Copyright Paulo Hubert 2018

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/

Automatic signal segmentation
"""

import os
import time

import numpy as np
cimport numpy as np
import operator
import re

import scipy.stats as stats

cimport cython
from cython_gsl cimport *
from cython_gsl import *
from cython.parallel cimport prange, parallel
from cython.parallel import prange, parallel

# Type for Cython NumPy acceleration
DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

# External functions from math.h
cdef extern from "math.h" nogil:
    double lgamma(double)

cdef double gammaln(double x) nogil:
    return lgamma(x)

cdef extern from "math.h" nogil:
    double log(double)

cdef double Ln(double x) nogil:
    return log(x)

cdef double Abs(double x) nogil:
    if x < 0:
        return -x
    else:
        return x

cdef extern from "math.h" nogil:
    double sqrt(double)

cdef double Sqrt(double x) nogil:
    return sqrt(x)

cdef extern from "math.h" nogil:
    double exp(double)

cdef double Exp(double x) nogil:
    return exp(x)

# Random number generator - Mersenne Twister
cdef gsl_rng *r = gsl_rng_alloc(gsl_rng_mt19937)


# Cython pure C functions
cdef double cprior_t(long t, long tstart, long tend, long minlen = 0) nogil:
    ''' Prior distribution for the change point using SNR parameterization.
        Uniform prior over [tstart+minlen, tend-minlen].
    
        Arguments:
        
        t -- point to calculate posterior
        tstart -- start of the current segment
        tend -- end of the current segment
        minlen -- minimum allowed length to a segment
            
        Returns:
        d -- log-prior at t
    '''
    
    cdef double d
    
    # Uniform prior over [tstart + minlen, tend - minlen]
    if t >= tstart + minlen and t <= tend - minlen:
        d = 0.
    else:
        d = -1e+300
        
    return d


cdef double cposterior_t(long t, long tstart, long tend, double prior_v, double send, double sstart, double st, double st1) nogil:
    ''' Calculates the log-posterior distribution for t using variance based parameterization

        Arguments:

        t -- segmentation point
        tstart -- start index of current signal window
        tend -- end indef of current signal window
        prior_v -- prior value associated to t
        send -- sum of amplitude squared from 0 to tend
        sstart -- sum of amplitude squared from 0 t0 tstart
        st -- sum of amplitude squared from 0 to t
        st1 -- sum of amplitude squared from t+1 to tend
        
        Returns:
        
        post -- log unnormalized posterior at t
    '''

    cdef long adjt = t - tstart + 1
    cdef long Nw = tend - tstart + 1
    cdef double dif1 = st-sstart
    cdef double dif2 = send - st1
    cdef double arg1 = 0.5*adjt
    cdef double arg2 = 0.5*(Nw - adjt)
    cdef double post = prior_v - arg1*Ln(dif1) - arg2*Ln(dif2) + gammaln(arg1) + gammaln(arg2)

    return post

cdef double cposterior_full_laplace(double d, double s, long Nw, long N2, double beta, double sum1, double sum2, long iprior) nogil:
    ''' Full log-posterior kernel for MCMC sampling for quotient of variances parameterization

        Arguments:

        d -- current value for delta
        s -- current value for sigma
        Nw -- total signal size
        N2 -- size of second segment
        beta -- parameter for laplace prior
        sum1 -- sum of amplitudes squared for first segment
        sum2 -- sum of amplitudes squared for second segment
        iprior -- prior to use; 0 = laplace, 1 = gaussian, 2 = uniform
        
        Returns:
        
        post -- log-posterior at given point
    '''

    if d <= 0 or s <= 0:
        return -1e+300
    
    # Jeffreys' prior for sigma
    cdef double dpriors = -Ln(s)

    cdef double dpriord
    
    if iprior == 0:
        # Laplace prior for delta        
        dpriord = -Ln(beta) - Abs(d-1)/beta
    elif iprior == 1:
        # Gaussian prior for delta
        dpriord = -0.5*Ln(beta)-(d-1.)*(d-1.)/(2.*beta)
    else:
        # Uniform prior
        dpriord = 0.

    cdef double post = dpriord +  dpriors - Nw*Ln(s)-0.5*N2*Ln(d)
    post = post - sum1/(2*(s**2)) - sum2/(2*d*(s**2))

    return post

cdef double cposterior_full_laplace_log(double d, double s, long Nw, long N2, double beta, double sum1, double sum2) nogil:
    ''' Full log-posterior kernel for MCMC sampling for log of quotient of variances parameterization

        Arguments:

        d -- current value for delta
        s -- current value for sigma
        Nw -- total signal size
        N2 -- size of second segment
        beta -- parameter for laplace prior
        sum1 -- sum of amplitudes squared for first segment
        sum2 -- sum of amplitudes squared for second segment
        
        Returns:
        
        post -- log-posterior at given point
    '''

    if s <= 0:
        return -1e+300
    
    # Jeffreys' prior for sigma
    cdef double dpriors = -Ln(s)

    cdef double dpriord
    
    # Laplace prior for delta        
    dpriord = -Ln(beta) - Abs(d)/beta

    cdef double post = dpriord +  dpriors - Nw*Ln(s)-0.5*N2*d
    post = post - sum1/(2*(s**2)) - sum2/(2*Exp(d)*(s**2))

    return post

cdef double cposterior_full(double s1, double s2, long Nw, long N2, double sum1, double sum2) nogil:
    ''' Full log-posterior kernel for MCMC sampling using two variances parameterization

        Arguments:

        s1 -- current value for s1
        s2 -- current value for s2
        Nw -- total signal size
        N2 -- size of second segment
        sum1 -- sum of amplitudes squared for first segment
        sum2 -- sum of amplitudes squared for second segment
        
        Returns:
        
        post -- log-posterior at given point
    '''

    if s1 <= 0 or s2 <= 0:
        return -1e+300
    
    # Jeffreys' prior for sigma
    cdef double dpriors1 = -Ln(s1)

    cdef double dpriors2 = -Ln(s2)

    cdef double post = dpriors1 +  dpriors2 - (Nw-N2)*Ln(s1)-N2*Ln(s2)
    post = post - sum1/(2*(s1**2)) - sum2/(2*(s2**2))

    return post


cdef double cmcmc_laplace(int mcburn, int mciter, double p0, double beta, long N, long N2, double sum1, double sum2, long iprior) nogil:
    ''' Run MCMC for quotient of variances parameterization.

        Arguments:

        mcburn -- burn-in period for chain
        mciter -- number of points to sample
        p0 -- posterior under H0
        beta -- parameter for Laplace prior
        N -- total signal size
        N2 -- size of second segment
        sum1 -- sum of amplitude squared for the first segment
        sum2 -- sum of amplitude squared for the second segment
        
        Returns:
        
        ev -- proportion of points s for which P(s|y) > max_{H0}P(s|y) 

    '''
    cdef double pcur, pcand, scur, scand, dcur, dcand, a, u, ev
    cdef int i, t0, t
    cdef double dvar, svar, cov, sd, eps, u1, u2, dmean, dmeanant, smean, smeanant, cov0, sumdsq, sumssq
    cdef double accept, dvarmin, svarmin
    
    dcur = (sum2 / (N2-1))/(sum1 / (N-N2-1))
    scur = Sqrt(sum1 / (N-N2-1))

    # Standard deviations and covariance for random-walk candidates distributions
    dvar = (dcur / 3) ** 2
    svar = (scur / 3) ** 2
    cov = 0.0

    # To safeguard variances
    dvarmin = dvar
    svarmin = svar


    # Generating starting values for the chain
    dcur = Abs(dcur + gsl_ran_gaussian(r, Sqrt(dvar)))
    scur = Abs(scur + gsl_ran_gaussian(r, Sqrt(svar)))
    pcur = cposterior_full_laplace(dcur, scur, N, N2, beta, sum1, sum2, iprior)

    # Parameters for adaptive MH
    sd = (2.4*2.4)/2.0
    eps = 1e-30

    # Starting point for adaptive MH
    t0 = 1000

    dmean = 0.0
    smean = 0.0
    sumdsq = 0.0
    sumssq = 0.0
    cov0 = 0.0
    accept = 0
    for i in range(t0):

        # Generate candidates
        u1 = gsl_ran_ugaussian(r)
        dcand = dcur + u1*Sqrt(dvar)

        # Calculates full posterior
        pcand = cposterior_full_laplace(dcand, scur, N, N2, beta, sum1, sum2, iprior)

        # Acceptance ratio
        a = pcand - pcur

        if Ln(gsl_rng_uniform(r)) < a:
            dcur = dcand
            pcur = pcand
            accept = accept + 1
        #endif

        u2 = gsl_ran_ugaussian(r)
        scand = scur + Sqrt(svar)*u2

        # Calculates full posterior
        pcand = cposterior_full_laplace(dcur, scand, N, N2, beta, sum1, sum2, iprior)

        # Acceptance ratio
        a = pcand - pcur

        if Ln(gsl_rng_uniform(r)) < a:
            scur = scand
            pcur = pcand
            accept = accept + 1
        #endif

        dmean = dmean + dcur
        smean = smean + scur
        cov0 = cov0 + dcur*scur
        sumdsq = sumdsq + dcur*dcur
        sumssq = sumssq + scur*scur

    #endfor

    dvar = (sumdsq - (dmean*dmean)/t0)/(t0-1)
    svar = (sumssq - (smean*smean)/t0)/(t0-1)


    if svar < 0:
        with gil:
            # This shouldn't happen, but if it does we reset the variance to a valid value
            print("Posterior variance of signal power with negative value!")
        svar = svarmin

    if dvar < 0:
        with gil:
            # This shouldn't happen, but if it does we reset the variance to a valid value
            print("Posterior variance of delta with negative value!")
        dvar = dvarmin

    cov = (1/(t0-1))*(cov0 - dmean*smean/t0)
    rho = cov/Sqrt(dvar*svar)
    dmean = dmean / t0
    smean = smean / t0
    t = t0

    accept = 0
    for i in range(mcburn):

        # Generate candidates
        u1 = gsl_ran_ugaussian(r)
        u2 = gsl_ran_ugaussian(r)
        if Abs(rho) > 1:
            with gil:
                # This also shouldn't happen. If it does, we set the correlation to 0
                print("Adaptive covariance defective!")
            rho = 0
        u2 = rho*u1 + (1-rho)*u2


        dcand = dcur + u1*Sqrt(dvar)
        scand = scur + u2*Sqrt(svar)

        if dcand > 0 and scand > 0:
            # Calculates full posterior
            pcand = cposterior_full_laplace(dcand, scand, N, N2, beta, sum1, sum2, iprior)

            # Acceptance ratio
            a = pcand - pcur

            if Ln(gsl_rng_uniform(r)) < a:
                scur = scand
                dcur = dcand
                pcur = pcand
                accept = accept + 1
            #endif
        #endif

        # Updating covariance matrix
        dmeanant = dmean
        smeanant = smean
        dmean = (t*dmeanant + dcur) / (t + 1)
        smean = (t*smeanant + scur) / (t + 1)

        dvar =  (((t-1)*dvar)/t) + (sd/t)*(t*dmeanant*dmeanant - (t+1)*dmean*dmean + dcur*dcur + eps)
        svar =  (((t-1)*svar)/t) + (sd/t)*(t*smeanant*smeanant - (t+1)*smean*smean + scur*scur + eps)
        cov = (((t-1)*cov)/t) + (sd/t)*(t*dmeanant*smeanant - (t+1)*dmean*smean + dcur*scur)
        rho = cov/Sqrt(dvar*svar)
        t = t + 1
    #endfor

    ev = 0.0
    dtmp = 0.0
    stmp = 0.0
    accept = 0
    for i in range(mciter):
        # Generate candidates
        u1 = gsl_ran_ugaussian(r)
        u2 = gsl_ran_ugaussian(r)
        u2 = rho*u1 + (1-rho)*u2

        dcand = dcur + u1*Sqrt(dvar)
        scand = scur + u2*Sqrt(svar)

        if dcand > 0 and scand > 0:
            # Calculates full posterior
            pcand = cposterior_full_laplace(dcand, scand, N, N2, beta, sum1, sum2, iprior)

            # Acceptance ratio
            a = pcand - pcur

            if Ln(gsl_rng_uniform(r)) < a:
                dcur = dcand
                scur = scand
                pcur = pcand
                accept = accept + 1
            #endif
        #endif

        if pcur > p0:
            ev = ev + 1.0
        #endif
    #endfor


    ev = ev / mciter

    return ev

cdef double cmcmc_laplace_log(int mcburn, int mciter, double p0, double beta, long N, long N2, double sum1, double sum2) nogil:
    ''' Run MCMC for log of quotient of variances parameterization.

        Arguments:

        mcburn -- burn-in period for chain
        mciter -- number of points to sample
        p0 -- posterior under H0
        beta -- parameter for Laplace prior
        N -- total signal size
        N2 -- size of second segment
        sum1 -- sum of amplitude squared for the first segment
        sum2 -- sum of amplitude squared for the second segment
        
        Returns:
        
        ev -- proportion of points s for which P(s|y) > max_{H0}P(s|y) 

    '''
    cdef double pcur, pcand, scur, scand, dcur, dcand, a, u, ev
    cdef int i, t0, t
    cdef double dvar, svar, cov, sd, eps, u1, u2, dmean, dmeanant, smean, smeanant, cov0, sumdsq, sumssq
    cdef double accept, dvarmin, svarmin
    
    dcur = Ln((sum2 / (N2-1))/(sum1 / (N-N2-1)))
    scur = Sqrt(sum1 / (N-N2-1))

    # Standard deviations and covariance for random-walk candidates distributions
    dvar = (dcur / 3) ** 2
    svar = (scur / 3) ** 2
    cov = 0.0

    # To safeguard variances
    dvarmin = dvar
    svarmin = svar


    # Generating starting values for the chain
    dcur = Abs(dcur + gsl_ran_gaussian(r, Sqrt(dvar)))
    scur = Abs(scur + gsl_ran_gaussian(r, Sqrt(svar)))
    pcur = cposterior_full_laplace_log(dcur, scur, N, N2, beta, sum1, sum2)

    # Parameters for adaptive MH
    sd = (2.4*2.4)/2.0
    eps = 1e-30

    # Starting point for adaptive MH
    t0 = 1000

    dmean = 0.0
    smean = 0.0
    sumdsq = 0.0
    sumssq = 0.0
    cov0 = 0.0
    accept = 0
    for i in range(t0):

        # Generate candidates
        u1 = gsl_ran_ugaussian(r)
        dcand = dcur + u1*Sqrt(dvar)

        # Calculates full posterior
        pcand = cposterior_full_laplace_log(dcand, scur, N, N2, beta, sum1, sum2)

        # Acceptance ratio
        a = pcand - pcur

        if Ln(gsl_rng_uniform(r)) < a:
            dcur = dcand
            pcur = pcand
            accept = accept + 1
        #endif

        u2 = gsl_ran_ugaussian(r)
        scand = scur + Sqrt(svar)*u2

        # Calculates full posterior
        pcand = cposterior_full_laplace_log(dcur, scand, N, N2, beta, sum1, sum2)

        # Acceptance ratio
        a = pcand - pcur

        if Ln(gsl_rng_uniform(r)) < a:
            scur = scand
            pcur = pcand
            accept = accept + 1
        #endif

        dmean = dmean + dcur
        smean = smean + scur
        cov0 = cov0 + dcur*scur
        sumdsq = sumdsq + dcur*dcur
        sumssq = sumssq + scur*scur

    #endfor

    dvar = (sumdsq - (dmean*dmean)/t0)/(t0-1)
    svar = (sumssq - (smean*smean)/t0)/(t0-1)


    if svar < 0:
        with gil:
            # This shouldn't happen, but if it does we reset the variance to a valid value
            print("Posterior variance of signal power with negative value!")
        svar = svarmin

    if dvar < 0:
        with gil:
            # This shouldn't happen, but if it does we reset the variance to a valid value
            print("Posterior variance of delta with negative value!")
        dvar = dvarmin

    cov = (1/(t0-1))*(cov0 - dmean*smean/t0)
    rho = cov/Sqrt(dvar*svar)
    dmean = dmean / t0
    smean = smean / t0
    t = t0

    accept = 0
    for i in range(mcburn):

        # Generate candidates
        u1 = gsl_ran_ugaussian(r)
        u2 = gsl_ran_ugaussian(r)
        if Abs(rho) > 1:
            with gil:
                # This also shouldn't happen. If it does, we set the correlation to 0
                print("Adaptive covariance defective!")
            rho = 0
        u2 = rho*u1 + (1-rho)*u2


        dcand = dcur + u1*Sqrt(dvar)
        scand = scur + u2*Sqrt(svar)

        if dcand > 0 and scand > 0:
            # Calculates full posterior
            pcand = cposterior_full_laplace_log(dcand, scand, N, N2, beta, sum1, sum2)

            # Acceptance ratio
            a = pcand - pcur

            if Ln(gsl_rng_uniform(r)) < a:
                scur = scand
                dcur = dcand
                pcur = pcand
                accept = accept + 1
            #endif
        #endif

        # Updating covariance matrix
        dmeanant = dmean
        smeanant = smean
        dmean = (t*dmeanant + dcur) / (t + 1)
        smean = (t*smeanant + scur) / (t + 1)

        dvar =  (((t-1)*dvar)/t) + (sd/t)*(t*dmeanant*dmeanant - (t+1)*dmean*dmean + dcur*dcur + eps)
        svar =  (((t-1)*svar)/t) + (sd/t)*(t*smeanant*smeanant - (t+1)*smean*smean + scur*scur + eps)
        cov = (((t-1)*cov)/t) + (sd/t)*(t*dmeanant*smeanant - (t+1)*dmean*smean + dcur*scur)
        rho = cov/Sqrt(dvar*svar)
        t = t + 1
    #endfor

    ev = 0.0
    dtmp = 0.0
    stmp = 0.0
    accept = 0
    for i in range(mciter):
        # Generate candidates
        u1 = gsl_ran_ugaussian(r)
        u2 = gsl_ran_ugaussian(r)
        u2 = rho*u1 + (1-rho)*u2

        dcand = dcur + u1*Sqrt(dvar)
        scand = scur + u2*Sqrt(svar)

        if dcand > 0 and scand > 0:
            # Calculates full posterior
            pcand = cposterior_full_laplace_log(dcand, scand, N, N2, beta, sum1, sum2)

            # Acceptance ratio
            a = pcand - pcur

            if Ln(gsl_rng_uniform(r)) < a:
                dcur = dcand
                scur = scand
                pcur = pcand
                accept = accept + 1
            #endif
        #endif

        if pcur > p0:
            ev = ev + 1.0
        #endif
    #endfor


    ev = ev / mciter

    return ev

cdef double cmcmc(int mcburn, int mciter, double p0, long N, long N2, double sum1, double sum2) nogil:
    ''' Run MCMC for two variances parameterization

        Arguments:

        mcburn -- burn-in period for chain
        mciter -- number of points to sample
        p0 -- posterior under H0
        N -- total signal size
        N2 -- size of second segment
        sum1 -- sum of amplitude squared for the first segment
        sum2 -- sum of amplitude squared for the second segment
        
        Returns:
        
        ev -- proportion of points s for which P(s|y) > max_{H0}P(s|y) 

    '''
    cdef double pcur, pcand, s1cur, s1cand, s2cur, s2cand, a, u, ev
    cdef int i, t0, t
    cdef double s2var, s1var, cov, sd, eps, u1, u2, s2mean, s2meanant, s1mean, s1meanant, cov0, sums2sq, sums1sq
    cdef double accept, s2varmin, s1varmin
    
    s2cur = (sum2 / (N2-1))/(sum1 / (N-N2-1))
    s1cur = Sqrt(sum1 / (N-N2-1))

    # Standard deviations and covariance for random-walk candidates distributions
    s2var = (s2cur / 3) ** 2
    s1var = (s1cur / 3) ** 2
    cov = 0.0

    # To safeguard variances
    s2varmin = s2var
    s1varmin = s1var


    # Generating starting values for the chain
    s2cur = Abs(s2cur + gsl_ran_gaussian(r, Sqrt(s2var)))
    s1cur = Abs(s1cur + gsl_ran_gaussian(r, Sqrt(s1var)))
    pcur = cposterior_full(s2cur, s1cur, N, N2, sum1, sum2)

    # Parameters for adaptive MH
    sd = (2.4*2.4)/2.0
    eps = 1e-30

    # Starting point for adaptive MH
    t0 = 1000

    s2mean = 0.0
    s1mean = 0.0
    sums2sq = 0.0
    sums1sq = 0.0
    cov0 = 0.0
    accept = 0
    for i in range(t0):

        # Generate candidates
        u1 = gsl_ran_ugaussian(r)
        s2cand = s2cur + u1*Sqrt(s2var)

        # Calculates full posterior
        pcand = cposterior_full(s2cand, s1cur, N, N2, sum1, sum2)

        # Acceptance ratio
        a = pcand - pcur

        if Ln(gsl_rng_uniform(r)) < a:
            s2cur = s2cand
            pcur = pcand
            accept = accept + 1
        #endif

        u2 = gsl_ran_ugaussian(r)
        s1cand = s1cur + Sqrt(s1var)*u2

        # Calculates full posterior
        pcand = cposterior_full(s2cur, s1cand, N, N2, sum1, sum2)

        # Acceptance ratio
        a = pcand - pcur

        if Ln(gsl_rng_uniform(r)) < a:
            s1cur = s1cand
            pcur = pcand
            accept = accept + 1
        #endif

        s2mean = s2mean + s2cur
        s1mean = s1mean + s1cur
        cov0 = cov0 + s2cur*s1cur
        sums2sq = sums2sq + s2cur*s2cur
        sums1sq = sums1sq + s1cur*s1cur

    #endfor

    s2var = (sums2sq - (s2mean*s2mean)/t0)/(t0-1)
    s1var = (sums1sq - (s1mean*s1mean)/t0)/(t0-1)


    if s1var < 0:
        with gil:
            # This shouldn't happen, but if it does we reset the variance to a valid value
            print("Posterior variance of signal power with negative value!")
        s1var = s1varmin

    if s2var < 0:
        with gil:
            # This shouldn't happen, but if it does we reset the variance to a valid value
            print("Posterior variance of delta with negative value!")
        s2var = s2varmin

    cov = (1/(t0-1))*(cov0 - s2mean*s1mean/t0)
    rho = cov/Sqrt(s2var*s1var)
    s2mean = s2mean / t0
    s1mean = s1mean / t0
    t = t0

    accept = 0
    for i in range(mcburn):

        # Generate candidates
        u1 = gsl_ran_ugaussian(r)
        u2 = gsl_ran_ugaussian(r)
        if Abs(rho) > 1:
            with gil:
                # This also shouldn't happen. If it does, we set the correlation to 0
                print("Adaptive covariance defective!")
            rho = 0
        u2 = rho*u1 + (1-rho)*u2


        s2cand = s2cur + u1*Sqrt(s2var)
        s1cand = s1cur + u2*Sqrt(s1var)

        if s2cand > 0 and s1cand > 0:
            # Calculates full posterior
            pcand = cposterior_full(s2cand, s1cand, N, N2, sum1, sum2)

            # Acceptance ratio
            a = pcand - pcur

            if Ln(gsl_rng_uniform(r)) < a:
                s1cur = s1cand
                s2cur = s2cand
                pcur = pcand
                accept = accept + 1
            #endif
        #endif

        # Updating covariance matrix
        s2meanant = s2mean
        s1meanant = s1mean
        s2mean = (t*s2meanant + s2cur) / (t + 1)
        s1mean = (t*s1meanant + s1cur) / (t + 1)

        s2var =  (((t-1)*s2var)/t) + (sd/t)*(t*s2meanant*s2meanant - (t+1)*s2mean*s2mean + s2cur*s2cur + eps)
        s1var =  (((t-1)*s1var)/t) + (sd/t)*(t*s1meanant*s1meanant - (t+1)*s1mean*s1mean + s1cur*s1cur + eps)
        cov = (((t-1)*cov)/t) + (sd/t)*(t*s2meanant*s1meanant - (t+1)*s2mean*s1mean + s2cur*s1cur)
        rho = cov/Sqrt(s2var*s1var)
        t = t + 1
    #endfor

    ev = 0.0
    dtmp = 0.0
    stmp = 0.0
    accept = 0
    for i in range(mciter):
        # Generate candidates
        u1 = gsl_ran_ugaussian(r)
        u2 = gsl_ran_ugaussian(r)
        u2 = rho*u1 + (1-rho)*u2

        s2cand = s2cur + u1*Sqrt(s2var)
        s1cand = s1cur + u2*Sqrt(s1var)

        if s2cand > 0 and s1cand > 0:
            # Calculates full posterior
            pcand = cposterior_full(s2cand, s1cand, N, N2, sum1, sum2)

            # Acceptance ratio
            a = pcand - pcur

            if Ln(gsl_rng_uniform(r)) < a:
                s2cur = s2cand
                s1cur = s1cand
                pcur = pcand
                accept = accept + 1
            #endif
        #endif

        if pcur > p0:
            ev = ev + 1.0
        #endif
    #endfor


    ev = ev / mciter

    return ev



# Interface
cdef class SeqSeg:
    ''' class SeqSeg: implements the python interface for the sequential segmentation algorithm

        Hubert, P., Padovese, L., Stern, J. A sequential algorithm for signal segmentation, Entropy 20 (1) 55 (2018)

        Hubert, P., Padovese, L., Stern, J. Fast implementation of a Bayesian unsupervised segmentation algorithm, arXiv:1803.01801

        Please cite these papers if you use the algorithm.

	If you have any trouble or doubts using this code, or just want to discuss, feel free to send an e-mail to
	paulo.hubert@gmail.com
    '''

    cdef long N, tstart, tend, seed
    cdef int mciter, mcburn, nchains, minlen, tstep
    cdef double beta, alpha
    cdef np.ndarray wave, sumw2
    cdef bint data_fed, initialized


    def __init__(self, np.ndarray wave = None, replicate = False):

        self.wave = wave
        if wave is None:
            self.data_fed = False
        else:
            self.data_fed = True

        self.initialized = False

        self.initialize()

        np.seterr(over = 'ignore', under = 'ignore')

        # To replication purposes
        if replicate == True:
            self.seed = 1529365132
            gsl_rng_set(r, self.seed)
        else:
            self.seed = time.time()*1000
            gsl_rng_set(r, self.seed)



    def initialize(self, double beta = 2.9e-5, double alpha = 0.1, int mciter = 4000, int mcburn = 1000, int nchains = 1):
        ''' Initializes the segmenter
        
            Args:
            
            beta - prior hyperparamenter for Laplace prior for the variance quotient
            alpha - the threshold value for evidence againts H0; it evidence < alpha, the segments are accepted
            mciter - number of iterations for the MCMC chain to calculate the evidence value
            mcburn - number of iterations to burn 
            nchains - number of parallel chains to run

            Can be called explicitly to set parameters for the MCMC
        '''

        if self.wave is not None:

            # Stores the cumulative sum to speed up calculations
            self.sumw2 = np.cumsum(self.wave**2)
            self.sumw2 = np.insert(self.sumw2, 0, 0)
            self.N = len(self.wave)

            # Current segment start and end
            self.tstart = 0
            self.tend = self.N-1

        else:

            self.tstart = 0
            self.tend = 0
            self.sumw2 = None
            self.N = -1


        self.mciter = mciter
        self.mcburn = mcburn
        self.nchains = nchains

        self.beta = beta
        self.alpha = alpha

        self.initialized = True


    def feed_data(self, wave):
        ''' Stores the signal and updates internal variables.
        
        Arguments:
        
        wave -- the signal, as a numpy array
        
        Returns:
        
        None
        '''

        # Store the wave and precalculates the cumulative sums
        self.wave = wave
        self.N = len(wave)
        self.sumw2 = np.cumsum(self.wave**2)
        self.sumw2 = np.insert(self.sumw2, 0, 0)

        # Current segment start and end
        self.tstart = 0
        self.tend = self.N-1

        self.data_fed = True
        

    cpdef double tester_laplace(self, long tcut, int iprior = 0, bint normalize = False, bint regularize = False):
        ''' Tests if tcut is a significant cutpoint using the Laplace prior and quotient of variances parameterization.
            Can be called separately to test the current segment.
            
            Arguments:
            
            tcut -- the candidate changepoint
            iprior -- which prior to use; 0 = laplace, 1 = gaussian, 2 = uniform
            normalize -- whether to normalize the first segment to have variance = 1
            regularize -- whether to used the regularized e-value (significance value)
            
            Returns:
            
            ev -- evidence value in favor of H0 (equality of variances)
        '''
        cdef double s0, p0, ev, sum1, sum2, beta
        cdef long n, N, N2
        cdef int i, nburn, npoints
        cdef np.ndarray[DTYPE_t, ndim = 1] vev = np.repeat(0.0, self.nchains)

        # Calculating sum of squares of amplitudes for both segments
        sum1 = self.sumw2[tcut] - self.sumw2[self.tstart]
        sum2 = self.sumw2[self.tend] - self.sumw2[tcut]

        if normalize:
            sum2 = sum2 / sum1
            sum1 = 1.0

        N = self.N
        N2 = self.tend - tcut
        nburn = self.mcburn
        npoints = self.mciter
        beta = self.beta

        # Calculates maximum posterior under H0
        s0 = Sqrt((sum1 + sum2)/(N + 2.))
        p0 = cposterior_full_laplace(1., s0, N, N2, beta, sum1, sum2, iprior)

        # Run chains
        with nogil, parallel():
            for i in prange(self.nchains, schedule = 'static'):
                vev[i] = cmcmc_laplace(nburn, npoints, p0, beta, N, N2, sum1, sum2, iprior)

        # Evidence IN FAVOR OF null hypothesis (delta = 1)
        ev = 1 - sum(vev) / self.nchains
        
        if regularize == True:
            # One parameter more, since now we have the prior
            ev = 1-stats.chi2.cdf(stats.chi2.ppf(1 - ev, df = 2), 1)

        return ev
    
    
    cpdef double tester_laplace_log(self, long tcut, bint normalize = False, bint regularize = False):
        ''' Tests if tcut is a significant cutpoint using the Laplace prior and log of quotient of variances parameterization.
            Can be called separately to test the current segment.
            
            Arguments:
            
            tcut -- the candidate changepoint
            normalize -- whether to normalize the first segment to have variance = 1
            regularize -- whether to used the regularized e-value (significance value)
            
            Returns:
            
            ev -- evidence value in favor of H0 (equality of variances)
        '''
        cdef double s0, p0, ev, sum1, sum2, beta
        cdef long n, N, N2
        cdef int i, nburn, npoints
        cdef np.ndarray[DTYPE_t, ndim = 1] vev = np.repeat(0.0, self.nchains)

        # Calculating sum of squares of amplitudes for both segments
        sum1 = self.sumw2[tcut] - self.sumw2[self.tstart]
        sum2 = self.sumw2[self.tend] - self.sumw2[tcut]

        if normalize:
            sum2 = sum2 / sum1
            sum1 = 1.0

        N = self.N
        N2 = self.tend - tcut
        nburn = self.mcburn
        npoints = self.mciter
        beta = self.beta

        # Calculates maximum posterior under H0
        s0 = Sqrt((sum1 + sum2)/(N + 2.))
        p0 = cposterior_full_laplace_log(0., s0, N, N2, beta, sum1, sum2)

        # Run chains
        with nogil, parallel():
            for i in prange(self.nchains, schedule = 'static'):
                    vev[i] = cmcmc_laplace_log(nburn, npoints, p0, beta, N, N2, sum1, sum2)

        # Evidence IN FAVOR OF null hypothesis (delta = 1)
        ev = 1 - sum(vev) / self.nchains
        
        if regularize == True:
            ev = 1-stats.chi2.cdf(stats.chi2.ppf(1 - ev, df = 2), 1)

        return ev    

    cpdef double tester(self, long tcut, bint normalize = False, bint regularize = False):
        ''' Tests if tcut is a significant cutpoint using Jeffreys prior.
            Can be called separately to test the current segment.
            
            
            Arguments:
            
            tcut -- the candidate changepoint
            normalize -- whether to normalize the first segment to have variance = 1
            regularize -- whether to used the regularized e-value (significance value)
            
            Returns:
            
            ev -- evidence value for H0 (equality of variances)
                        
        '''
        cdef double s0, p0, ev, sum1, sum2, beta
        cdef long n, N, N2
        cdef int i, nburn, npoints
        cdef np.ndarray[DTYPE_t, ndim = 1] vev = np.repeat(0.0, self.nchains)

        # Calculating sum of squares of amplitudes for both segments
        sum1 = self.sumw2[tcut] - self.sumw2[self.tstart]
        sum2 = self.sumw2[self.tend] - self.sumw2[tcut]

        if normalize:
            sum2 = sum2 / sum1
            sum1 = 1.0

        N = self.N
        N2 = self.tend - tcut
        nburn = self.mcburn
        npoints = self.mciter
        beta = self.beta

        # Calculates maximum posterior under H0
        s0 = Sqrt((sum1 + sum2)/(N + 2.))
        p0 = cposterior_full(s0, s0, N, N2, sum1, sum2)

        # Run chains
        with nogil, parallel():
            for i in prange(self.nchains, schedule = 'static'):
                vev[i] = cmcmc(nburn, npoints, p0, N, N2, sum1, sum2)

        # Evidence IN FAVOR OF null hypothesis (delta = 1)
        ev = 1 - sum(vev) / self.nchains
        
        if regularize == True:
            ev = 1-stats.chi2.cdf(stats.chi2.ppf(1 - ev, df = 2), 1)        

        return ev

    def get_posterior(self, start, end, minlen = 0, res = 1):
        ''' Returns the posterior values for the changepoint using SNR parameterization.

            Arguments:

            start -- first point to calculate the posterior
            end -- last point to calculate the posterior

            Returns:

            tvec -- vector with posterior density values
            elapsed -- time elapsed
        '''


        if not self.data_fed:

            print("Data not initialized! Call feed_data.")
            return(-1)

        cdef long t, n, istart, iend, tstep, tstart, tend
        cdef double sstart, send, st, st1, dprior
        cdef np.ndarray[DTYPE_t, ndim = 1] tvec = np.repeat(-np.inf, self.N)
        cdef np.ndarray[DTYPE_t, ndim = 1] esumw2 = self.sumw2

        cdef long mlen = minlen

        self.N = len(self.wave)

        if end > self.N:
            raise ValueError("Invalid value for tend.")

        if start < 0:
            raise ValueError("Invalid value for start.")

        tstep = res

        # Sets start and end
        tstart = start
        tend = end

        # Obtains MAP estimate of the cut point
        # Parallelized

        # Bounds for start and end
        istart = tstart + 3
        iend = tend - 3
        n = int((iend-istart)/tstep)

        sstart = self.sumw2[self.tstart]
        send = self.sumw2[self.tend]

        tvec = np.repeat(-np.inf, n + 1)
        beta = self.beta


        begin = time.time()
        with nogil, parallel():
            for t in prange(n + 1, schedule = 'static'):
                st = esumw2[istart + t*tstep]
                st1 = esumw2[istart + t*tstep + 1]
                dprior = cprior_t(istart + t*tstep, tstart, tend, mlen)

                tvec[t] = cposterior_t(istart + t*tstep, tstart, tend, dprior, send, sstart, st, st1)


        end = time.time()
        elapsed = end - begin

        return tvec, elapsed

    def get_posterior_var(self, start, end, minlen = 0, res = 1):
        ''' Returns the posterior values for the changepoint using variances parameterization.

            Arguments:

            start -- first point to calculate the posterior
            end -- last point to calculate the posterior

            Returns:

            tvec -- vector with posterior density values
            elapsed -- time elapsed
        '''


        if not self.data_fed:

            print("Data not initialized! Call feed_data.")
            return(-1)

        cdef long t, n, istart, iend, tstep, tstart, tend
        cdef double sstart, send, st, st1, dprior
        cdef np.ndarray[DTYPE_t, ndim = 1] tvec = np.repeat(-np.inf, self.N)
        cdef np.ndarray[DTYPE_t, ndim = 1] esumw2 = self.sumw2

        cdef long mlen = minlen

        self.N = len(self.wave)

        if end > self.N:
            raise ValueError("Invalid value for tend.")

        if start < 0:
            raise ValueError("Invalid value for start.")

        tstep = res

        # Sets start and end
        tstart = start
        tend = end

        # Obtains MAP estimate of the cut point
        # Parallelized

        # Bounds for start and end
        istart = tstart + 3
        iend = tend - 3
        n = int((iend-istart)/tstep)

        sstart = self.sumw2[self.tstart]
        send = self.sumw2[self.tend]

        tvec = np.repeat(-np.inf, n + 1)
        beta = self.beta


        begin = time.time()
        with nogil, parallel():
            for t in prange(n + 1, schedule = 'static'):
                st = esumw2[istart + t*tstep]
                st1 = esumw2[istart + t*tstep + 1]
                dprior = cprior_t(istart + t*tstep, tstart, tend, mlen)

                tvec[t] = cposterior_t(istart + t*tstep, tstart, tend, dprior, send, sstart, st, st1)


        end = time.time()
        elapsed = end - begin

        return tvec, elapsed




    def segments(self, minlen, res, method = 'jeffreys', regularize = False, normalize = False, verbose = False, adaptive_res = False):
        ''' Applies the sequential segmentation algorithm to the current signal using variances parameterization with independent Jeffreys' priors for each variance.
        
            Arguments:
            
            minlen -- the minimum segment length accepted
            res -- resolution; how many positions to jump when optimizing the posterior for the changepoint; tstep = 1 is the maximum resolution
            method - jeffreys, to use uninformative priors, or laplace to use informative priors
            regularize -- whether to use the regularized e-value (significance value)
            normalize -- whether to normalize the signal to have first segment with variance 1
            adaptive_res -- whether to use the adaptive resolution; if true, will calculate p = res / N and keep this ratio constant throughout the execution
            
            Returns:
            
            tseg -- unordered array of accepted changepoints
            elapsed -- time elapsed in seconds
        '''

        if not self.data_fed:

            print("Data not initialized! Call feed_data.")
            return(-1, -1)

        if method not in ['jeffreys', 'laplace']:
            print("Method unknown, must be 'jeffreys' or 'laplace'.")
            return(-1, -1)
        
        begin = time.time()

        # Cannot have a minimum segment of less than 5 points for the algorithm to make sense
        minlen = max(5, minlen)

        cdef long t, tmax, tstart, tend, n, istart, iend, tstep, mlen
        cdef double maxp, posterior, sstart, send, st, st1, pres
        cdef np.ndarray[DTYPE_t, ndim = 1] tvec = np.repeat(-np.inf, self.N)
        cdef np.ndarray[DTYPE_t, ndim = 1] esumw2 = self.sumw2

        tstep = res
        
        self.tstart = 0
        self.tend = len(self.wave) - 1
        self.N = len(self.wave)
        
        if adaptive_res:
            pres = float(tstep) / self.N
        else:
            pres = 1. / self.N

        tseg = []
        # Creates index to keep track of tested segments
        # True, if the segment must be tested, False otherwise
        iseg = {(self.tstart, self.tend) : True}
        
        mlen = minlen
        # Main loop: while there are untested segments
        while sum(iseg.values()) > 0:
            # Iterates through segments to be tested
            isegold = [i for i in iseg if iseg[i] == True]
            for seg in isegold:
                # Sets start and end
                self.tstart = seg[0]
                self.tend = seg[1]
                self.N = self.tend - self.tstart + 1
                
                if adaptive_res:
                    tstep = int(pres * self.N)
                    if tstep == 0:
                        tstep = 1


                # Obtains MAP estimate of the cut point
                # Parallelized
                tstart = self.tstart
                tend = self.tend

                # Bounds for start and end
                istart = tstart + 3
                iend = tend - 3
                n = int((iend-istart)/tstep)
                
                if n > 3:

                    sstart = self.sumw2[self.tstart]
                    send = self.sumw2[self.tend]

                    tvec = np.repeat(-np.inf, n + 1)
                    with nogil, parallel():
                        for t in prange(n + 1, schedule = 'static'):
                            st = esumw2[istart + t*tstep]
                            #st1 = esumw2[istart + t*tstep + 1]
                            st1 = esumw2[istart + t*tstep + 1]
                            tvec[t] = cposterior_t(istart + t*tstep, tstart, tend, cprior_t(istart + t*tstep, tstart, tend, mlen), send, sstart, st, st1)

                    tmax, maxp = max(enumerate(tvec), key=operator.itemgetter(1))

                    # tmax is the optimum position in range(tstart+2, tend-3)
                    # but WITH STEP SIZE = TSTEP
                    tmax = istart + tmax*tstep

                    if tmax - tstart > minlen and tend - tmax > minlen:

                        # Test the segments
                        if method == 'jeffreys':
                            evidence = self.tester(tmax, normalize, regularize = False)
                        elif method == 'laplace':
                            evidence = self.tester_laplace_log(tmax, normalize, regularize = False)
                        

                        # Regularizing
                        if regularize == True:
                            evidence = 1-stats.chi2.cdf(stats.chi2.ppf(1-evidence, df = 2), 1)

                        if evidence < self.alpha:
                            if verbose:
                                print("Tcut = " + str(tmax) + ", start = " + str(self.tstart) + ", tend = " + str(self.tend) + ", N = " + str(self.N) + ", accepted: evidence = " + str(evidence))
                            #endif

                            # Different variances
                            # Update list of segments
                            tseg.append(tmax)

                            # Update dict
                            iseg[(tstart, tmax)] = True
                            iseg[(tmax + 1, self.tend)] = True
                            del iseg[seg]

                        else:

                            iseg[seg] = False
                            if verbose:
                                print("Tcut = " + str(tmax) + ", start = " + str(self.tstart) + ", tend = " + str(self.tend) + ", N = " + str(self.N) + ", rejected: evidence = " + str(evidence))
                            #endif
                        #endif

                    else:
                        # Segment has been tested, no significant cut point found
                        iseg[seg] = False
                    #endif
                else:
                    iseg[seg] = False
                #end if
            #end for
        #endwhile
        end = time.time()
        elapsed = end - begin
        if verbose:
            print("End of execution: " + str(len(tseg) + 1) + " segments found in " + str(elapsed) + " seconds.")
        #endif

        return tseg, elapsed

    
    def segments_laplace(self, minlen, res, iprior, regularize = False, normalize = False, verbose = False):
        ''' Applies the sequential segmentation algorithm to the current signal using quotient of variances parameterization
        
            Arguments:
            
            minlen -- the minimum segment length accepted
            res -- resolution; how many positions to jump when optimizing the posterior for the changepoint; tstep = 1 is the maximum resolution
            iprior -- which prior to use for the variance quotient; i = 0 for laplace, i = 1 for gaussian, i = 2 for uniform
            regularize -- whether to use the regularized e-value (significance value)
            normalize -- whether to normalize the signal to have first segment with variance 1
            
            Returns:
            
            tseg -- unordered array of accepted changepoints
            elapsed -- time elapsed in seconds
        '''

        if not self.data_fed:

            print("Data not initialized! Call feed_data.")
            return(-1)

        begin = time.time()

        # Cannot have a minimum segment of less than 5 points for the algorithm to make sense
        minlen = max(5, minlen)

        cdef long t, tmax, tstart, tend, n, istart, iend, tstep, mlen
        cdef double maxp, posterior, sstart, send, st, st1
        cdef np.ndarray[DTYPE_t, ndim = 1] tvec = np.repeat(-np.inf, self.N)
        cdef np.ndarray[DTYPE_t, ndim = 1] esumw2 = self.sumw2

        tstep = res

        self.tstart = 0
        self.tend = len(self.wave) - 1
        self.N = len(self.wave)

        #tstep = res
        
        if verbose:
            if iprior == 0:
                print('Using Laplace prior.')
            elif iprior == 1:
                print('Using Gaussian prior.')
            else:
                print('Using uniform prior.')

        tseg = []
        # Creates index to keep track of tested segments
        # True, if the segment must be tested, False otherwise
        iseg = {(self.tstart, self.tend) : True}
        
        mlen = minlen
        # Main loop: while there are untested segments
        while sum(iseg.values()) > 0:
            # Iterates through segments to be tested
            isegold = [i for i in iseg if iseg[i] == True]
            for seg in isegold:
                # Sets start and end
                self.tstart = seg[0]
                self.tend = seg[1]
                self.N = self.tend - self.tstart + 1

                # Obtains MAP estimate of the cut point
                # Parallelized
                tstart = self.tstart
                tend = self.tend

                # Bounds for start and end
                istart = tstart + 3
                iend = tend - 3
                n = int((iend-istart)/tstep)
                
                if n > 3:

                    sstart = self.sumw2[self.tstart]
                    send = self.sumw2[self.tend]

                    tvec = np.repeat(-np.inf, n + 1)
                    with nogil, parallel():
                        for t in prange(n + 1, schedule = 'static'):
                            st = esumw2[istart + t*tstep]
                            #st1 = esumw2[istart + t*tstep + 1]
                            st1 = esumw2[istart + t*tstep + 1]
                            tvec[t] = cposterior_t(istart + t*tstep, tstart, tend, cprior_t(istart + t*tstep, tstart, tend, mlen), send, sstart, st, st1)

                    tmax, maxp = max(enumerate(tvec), key=operator.itemgetter(1))

                    # tmax is the optimum position in range(tstart+2, tend-3)
                    # but WITH STEP SIZE = TSTEP
                    tmax = istart + tmax*tstep

                    if tmax - tstart > minlen and tend - tmax > minlen:

                        # Test the segments
                        evidence = self.tester_laplace(tmax, iprior, normalize, regularize = False)

                        # Regularizing
                        if regularize == True:
                            evidence = 1-stats.chi2.cdf(stats.chi2.ppf(1-evidence, df = 2), 1)

                        if evidence < self.alpha:
                            if verbose:
                                print("Tcut = " + str(tmax) + ", start = " + str(self.tstart) + ", tend = " + str(self.tend) + ", N = " + str(self.N) + ", accepted: evidence = " + str(evidence))
                            #endif

                            # Different variances
                            # Update list of segments
                            tseg.append(tmax)

                            # Update dict
                            iseg[(tstart, tmax)] = True
                            iseg[(tmax + 1, self.tend)] = True
                            del iseg[seg]

                        else:

                            iseg[seg] = False
                            if verbose:
                                print("Tcut = " + str(tmax) + ", start = " + str(self.tstart) + ", tend = " + str(self.tend) + ", N = " + str(self.N) + ", rejected: evidence = " + str(evidence))
                            #endif
                        #endif

                    else:
                        # Segment has been tested, no significant cut point found
                        iseg[seg] = False
                    #endif
                else:
                    iseg[seg] = False
                #endif
            #end for
        #endwhile
        end = time.time()
        elapsed = end - begin
        if verbose:
            print("End of execution: " + str(len(tseg) + 1) + " segments found in " + str(elapsed) + " seconds.")
        #endif

        return tseg, elapsed
    
    def segments_laplace_log(self, minlen, res, regularize = False, normalize = False, verbose = False):
        ''' Applies the sequential segmentation algorithm to the current signal using log of quotient of variances parameterization
        
            Arguments:
            
            minlen -- the minimum segment length accepted
            res -- resolution; how many positions to jump when optimizing the posterior for the changepoint; tstep = 1 is the maximum resolution
            regularize -- whether to use the regularized e-value (significance value)
            normalize -- whether to normalize the signal to have first segment with variance 1
            
            Returns:
            
            tseg -- unordered array of accepted changepoints
            elapsed -- time elapsed in seconds
        '''

        if not self.data_fed:

            print("Data not initialized! Call feed_data.")
            return(-1)

        begin = time.time()

        # Cannot have a minimum segment of less than 5 points for the algorithm to make sense
        minlen = max(5, minlen)

        cdef long t, tmax, tstart, tend, n, istart, iend, tstep, mlen
        cdef double maxp, posterior, sstart, send, st, st1
        cdef np.ndarray[DTYPE_t, ndim = 1] tvec = np.repeat(-np.inf, self.N)
        cdef np.ndarray[DTYPE_t, ndim = 1] esumw2 = self.sumw2

        tstep = res

        self.tstart = 0
        self.tend = len(self.wave) - 1
        self.N = len(self.wave)

        #tstep = res
        
        tseg = []
        # Creates index to keep track of tested segments
        # True, if the segment must be tested, False otherwise
        iseg = {(self.tstart, self.tend) : True}
        
        mlen = minlen
        # Main loop: while there are untested segments
        while sum(iseg.values()) > 0:
            # Iterates through segments to be tested
            isegold = [i for i in iseg if iseg[i] == True]
            for seg in isegold:
                # Sets start and end
                self.tstart = seg[0]
                self.tend = seg[1]
                self.N = self.tend - self.tstart + 1

                # Obtains MAP estimate of the cut point
                # Parallelized
                tstart = self.tstart
                tend = self.tend

                # Bounds for start and end
                istart = tstart + 3
                iend = tend - 3
                n = int((iend-istart)/tstep)
                
                if n > 3:

                    sstart = self.sumw2[self.tstart]
                    send = self.sumw2[self.tend]

                    tvec = np.repeat(-np.inf, n + 1)
                    with nogil, parallel():
                        for t in prange(n + 1, schedule = 'static'):
                            st = esumw2[istart + t*tstep]
                            st1 = esumw2[istart + t*tstep + 1]
                            tvec[t] = cposterior_t(istart + t*tstep, tstart, tend, cprior_t(istart + t*tstep, tstart, tend, mlen), send, sstart, st, st1)

                    tmax, maxp = max(enumerate(tvec), key=operator.itemgetter(1))

                    # tmax is the optimum position in range(tstart+3, tend-3)
                    # but WITH STEP SIZE = TSTEP
                    tmax = istart + tmax*tstep

                    if tmax - tstart > minlen and tend - tmax > minlen:

                        # Test the segments
                        evidence = self.tester_laplace_log(tmax, normalize, regularize = False)

                        # Regularizing
                        if regularize == True:
                            evidence = 1-stats.chi2.cdf(stats.chi2.ppf(1-evidence, df = 2), 1)

                        if evidence < self.alpha:
                            if verbose:
                                print("Tcut = " + str(tmax) + ", start = " + str(self.tstart) + ", tend = " + str(self.tend) + ", N = " + str(self.N) + ", accepted: evidence = " + str(evidence))
                            #endif

                            # Different variances
                            # Update list of segments
                            tseg.append(tmax)

                            # Update dict
                            iseg[(tstart, tmax)] = True
                            iseg[(tmax + 1, tend)] = True
                            del iseg[seg]

                        else:

                            iseg[seg] = False
                            if verbose:
                                print("Tcut = " + str(tmax) + ", start = " + str(self.tstart) + ", tend = " + str(self.tend) + ", N = " + str(self.N) + ", rejected: evidence = " + str(evidence))
                            #endif
                        #endif

                    else:
                        # Segment has been tested, no significant cut point found
                        iseg[seg] = False
                    #endif
                else:
                    iseg[seg] = False
                #endif
            #end for
        #endwhile
        end = time.time()
        elapsed = end - begin
        if verbose:
            print("End of execution: " + str(len(tseg) + 1) + " segments found in " + str(elapsed) + " seconds.")
        #endif

        return tseg, elapsed
    
    def bic(self, tseg):
        ''' Calculates BIC criterion for a given segmentation
        
            Arguments:
            
            tseg - numpy vector with the changepoints
            
            Returns:
            
            bic - value of BIC criterion, given by log(N)*(2*k+1) - 2 * log(L), where N is the signal length, k is the number of changepoints, and L is the likelihood
            
        '''
        tcuts = tseg.copy()

        tcuts.sort()
        if len(tseg) == 0:
            s = np.var(self.wave)
            loglik = -(len(self.wave)/2)*np.log(s) - np.sum(self.wave**2)/(2*s)
        else:
            loglik = 0
            for j in range(len(tcuts)):
                if j == 0:
                    s = np.var(self.wave[:tcuts[j]])
                    loglik = loglik - (tcuts[j]/2)*np.log(s) - np.sum(self.wave[:tcuts[j]]**2)/(2*s)
                elif j < len(tcuts)-1:
                    s = np.var(self.wave[tcuts[j-1]:tcuts[j]])
                    loglik = loglik - ((tcuts[j]-tcuts[j-1]+1)/2)*np.log(s) - np.sum(self.wave[tcuts[j-1]:tcuts[j]]**2)/(2*s)
                else:
                    s = np.var(self.wave[tcuts[j]:])
                    loglik = loglik - ((len(self.wave)-tcuts[j])/2)*np.log(s) - np.sum(self.wave[tcuts[j]:]**2)/(2*s)

        bic = np.log(len(self.wave))*(2*len(tcuts)+1) - 2*loglik
        
        return bic
       
    def mbic(self, tseg):
        ''' Calculates MBIC criterion for a given segmentation
        
            Arguments:
            
            tseg - numpy vector with the changepoints
            
            Return:
            
            mbic - value of MBIC criterion
        '''
        m = len(tseg) + 1
        T = len(self.wave)
        tcuts = tseg.copy()
        tcuts.sort()
        ybar = np.mean(self.wave)
        
        tcuts = [0] + tcuts + [T]
        
        ssall = np.sum((self.wave - ybar)**2)
        
        ssbg = 0.
        slog = 0.
        for i in range(m):
            ssbg = ssbg + (tcuts[i+1] - tcuts[i])*(np.mean(self.wave[tcuts[i]:tcuts[i+1]])-ybar)**2
            slog = slog + Ln(tcuts[i+1] - tcuts[i])
        
        sswg = ssall - ssbg
        
        
        mbic = ((T-m+1.)/2.)*Ln(1 + ssbg/sswg) + gammaln((T-m+1.)/2.) - gammaln((T+1.)/2.) + (m/2.)*Ln(ssall) - 0.5*slog + (0.5-m)*Ln(T)
        
        return mbic
            
