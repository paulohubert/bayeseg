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

# Random number generator - Mersenne Twister
cdef gsl_rng *r = gsl_rng_alloc(gsl_rng_mt19937)


# Cython pure C functions
cdef double cprior_t(long t, long tstart, long tend, long minlen = 0) nogil:
    ''' Prior distribution for the change point.
    '''
    
    # Uniform prior over [tstart + minlen, tend - minlen]
    if t > tstart + minlen and t < tend - minlen:
        return 0.
    else:
        return -1e-32

cdef double cposterior_t(long t, long tstart, long tend, double prior_v, double send, double sstart, double st, double st1) nogil:
    ''' Calculates the log-posterior distribution for t

        @args:

        t - segmentation point
        tstart - start index of current signal window
        tend - end indef of current signal window
        prior_v - prior value associated to t
        send - sum of amplitude squared from 0 to tend
        sstart - sum of amplitude squared from 0 t0 tstart
        st - sum of amplitude squared from 0 to t
        st1 - sum of amplitude squared from t+1 to tend
    '''

    cdef long adjt = t - tstart + 1
    cdef long Nw = tend - tstart + 1
    cdef double dif1 = st-sstart
    cdef double dif2 = send - st1
    cdef double arg1 = 0.5*(adjt + 6)
    cdef double arg2 = 0.5*(Nw - adjt - 2)
    cdef double post = prior_v - arg1*(Ln(dif1)) - arg2*(Ln(dif2)) + gammaln(arg1) + gammaln(arg2)

    return post

cdef double cposterior_full(double d, double s, long Nw, long N2, double beta, double sum1, double sum2, long iprior) nogil:
    ''' Full log-posterior kernel for MCMC sampling

        @args:

        d - current value for delta
        s - current value for sigma
        Nw - total signal size
        N2 - size of second segment
        beta - parameter for laplace prior
        sum1 - sum of amplitudes squared for first segment
        sum2 - sum of amplitudes squared for second segment
        iprior - prior to use; 0 = laplace, 1 = gaussian, 2 = uniform
    '''

    if d <= 0 or s <= 0:
        return -1e+308
    
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


cdef double cmcmc(int mcburn, int mciter, double p0, double beta, long N, long N2, double sum1, double sum2, long iprior) nogil:
    ''' Run MCMC

        @args:

        mcburn - burn-in period for chain
        mciter - number of points to sample
        p0 - posterior under H0
        beta - parameter for Laplace prior
        N - total signal size
        N2 - size of second segment
        sum1 - sum of amplitude squared for the first segment
        sum2 - sum of amplitude squared for the second segment

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
    pcur = cposterior_full(dcur, scur, N, N2, beta, sum1, sum2, iprior)

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
        pcand = cposterior_full(dcand, scur, N, N2, beta, sum1, sum2, iprior)

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
        pcand = cposterior_full(dcur, scand, N, N2, beta, sum1, sum2, iprior)

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
            pcand = cposterior_full(dcand, scand, N, N2, beta, sum1, sum2, iprior)

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
            pcand = cposterior_full(dcand, scand, N, N2, beta, sum1, sum2, iprior)

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
        ''' Stores the signal and updates internal variables
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


    cpdef double tester(self, long tcut, long iprior, bint normalize = False):
        ''' Tests if tcut is a significant cutpoint
            Can be called separately to test the current segment.
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
        s0 = Sqrt((sum1 + sum2)/(N + 1.))
        p0 = cposterior_full(1.0, s0, N, N2, beta, sum1, sum2, iprior)

        # Run chains
        with nogil, parallel():
            for i in prange(self.nchains, schedule = 'static'):
                vev[i] = cmcmc(nburn, npoints, p0, beta, N, N2, sum1, sum2, iprior)

        # Evidence IN FAVOR OF null hypothesis (delta = 1)
        ev = 1 - sum(vev) / self.nchains

        return ev

    def get_posterior(self, start, end, minlen = 0, res = 1):
        ''' Returns the posterior values for the changepoint.

            @args:

                start: first point to calculate the posterior
                end: last point to calculate the posterior

				@returns:

					tvec - vector with posterior density values
					elapsed - time elapsed
        '''


        if not self.data_fed:

            print("Data not initialized! Call feed_data.")
            return(-1)

        cdef long t, n, istart, iend, tstep, tstart, tend
        cdef double sstart, send, st, st1
        cdef np.ndarray[DTYPE_t, ndim = 1] tvec = np.repeat(-np.inf, self.N)
        cdef np.ndarray[DTYPE_t, ndim = 1] esumw2 = self.sumw2

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
                tvec[t] = cposterior_t(istart + t*tstep, tstart, tend, cprior_t(istart + t*tstep, tstart, tend, minlen), send, sstart, st, st1)


        end = time.time()
        elapsed = end - begin

        return tvec, elapsed





    def segments(self, minlen, res, iprior, regularize = False, normalize = False, verbose = False):
        ''' Applies the sequential segmentation algorithm to the wave,
            returns the vector with segments' index
        '''

        if not self.data_fed:

            print("Data not initialized! Call feed_data.")
            return(-1)

        begin = time.time()

        # Cannot have a minimum segment of less than 5 points for the algorithm to make sense
        minlen = max(5, minlen)

        cdef long t, tmax, tstart, tend, n, istart, iend, tstep
        cdef double maxp, posterior, sstart, send, st, st1
        cdef np.ndarray[DTYPE_t, ndim = 1] tvec = np.repeat(-np.inf, self.N)
        cdef np.ndarray[DTYPE_t, ndim = 1] esumw2 = self.sumw2



        self.tstart = 0
        self.tend = len(self.wave) - 1
        self.N = len(self.wave)

        tstep = res
        
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

                sstart = self.sumw2[self.tstart]
                send = self.sumw2[self.tend]

                tvec = np.repeat(-np.inf, n + 1)
                with nogil, parallel():
                    for t in prange(n + 1, schedule = 'static'):
                        st = esumw2[istart + t*tstep]
                        #st1 = esumw2[istart + t*tstep + 1]
                        st1 = esumw2[istart + t*tstep + 1]
                        tvec[t] = cposterior_t(istart + t*tstep, tstart, tend, cprior_t(istart + t*tstep, tstart, tend, minlen), send, sstart, st, st1)

                tmax, maxp = max(enumerate(tvec), key=operator.itemgetter(1))

                # tmax is the optimum position in range(tstart+2, tend-3)
                # but WITH STEP SIZE = TSTEP
                tmax = istart + tmax*tstep

                if tmax - tstart > minlen and tend - tmax > minlen:
                    
                    # Test the segments
                    evidence = self.tester(tmax, iprior, normalize)
                    
                    # Regularizing
                    if regularize == True:
                        evidence = 1-stats.chi2.cdf(stats.chi2.pdf(1-evidence, df = 2), 1)

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
            #end for
        #endwhile
        end = time.time()
        elapsed = end - begin
        if verbose:
            print("End of execution: " + str(len(tseg) + 1) + " segments found in " + str(elapsed) + " seconds.")
        #endif

        return tseg, elapsed
