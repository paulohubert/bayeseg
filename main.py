# -*- coding: utf-8 -*-
"""

@author: paulo.hubert@gmail.com

Please cite:

- A sequential algorithm for signal segmentation, Hubert, P., Padovese, L., Stern, J.M. Entropy 20 (55) (2018)
- Fast implementation of a Bayesian unsupervised segmentation algorithm, Hubert, P., Padovese, L., Stern, J.M. (2018)

"""

import sys
sys.path.append("/home/paulo/github/")

import time
import datetime

import numpy as np
import pandas as pd

from bayeseg.OceanPod import OceanPod
from bayeseg.SeqSeg import SeqSeg

import os
currfolder = os.path.dirname(os.path.abspath(__file__))

# Folder with wave files
wavfolder = currfolder + '/Data/'

savefolder = currfolder + '/Output/'

# Parameters
beta = 3e-5
alpha = 0.01
mciter = 10000
mcburn = 10000

# Creates object to read wave files and segments
op = OceanPod(wavfolder)
ss = SeqSeg()
ss.initialize(beta, alpha, mciter, mcburn, nchains = 1)

count = 0
nfiles = len(op.Filelist)
timerun = datetime.datetime.strftime(datetime.datetime.now(), format = '%Y-%m-%d_%H_%M_%S')
arq = open(savefolder + 'log_' + timerun + '.txt', 'w')
dados = []

before = time.time()
nevents = -1
powerlast = 0
nsegments = 0
log = []
for f in op.Filelist:
    # Reads file
    fs, wave = op.readFile(f)
    # Feeds wave
    ss.feed_data(wave)

    # Run the segmentation
    t, timeelapsed = ss.segments(minlen = 11025, res = 11025, verbose = False)
    t.sort()

    if len(t) == 0:
        t = [0, len(wave) - 1]
    elif len(t) == 1:
        t = [0, t[0], len(wave) - 1]
    else:
        t = np.concatenate([[0], t, [len(wave) - 1]])
    #endif

    nsegments = nsegments + len(t) - 1
    # Store segments
    for i in range(len(t)-1):
        starttime = op.index2date(f, t[i])
        endtime = op.index2date(f, t[i+1] + 1)
        duration = endtime - starttime
        power = sum(np.power(wave[t[i]:t[i+1]], 2)) / (t[i+1] - t[i] + 1)

        dados.append([starttime, endtime, duration, power, nevents])
    #endfor

    print("({:.2%}) File ".format(count / nfiles) + f + " processed: " + str(len(t) - 1) + " segments in this file, " + str(nsegments) + " total segments, " + str(timeelapsed) + " seconds.")
    arq.write("({:.2%}) File ".format(count / nfiles) + f + " processed: " + str(len(t) - 1) + " segments in this file, " + str(nsegments) + " total segments, " + str(timeelapsed) + " seconds.\n")
    log.append([f, len(t)-1])
    count = count + 1
#endfor

df = pd.DataFrame(dados, columns = ['start', 'end', 'duration', 'power', 'nevents'])
df.index.name = 'indice'
df.to_csv(savefolder + 'segments_' + timerun + '.csv', header = True, index = True)

dflog = pd.DataFrame(log, columns = ['file', 'segments'])
dflog.to_csv(savefolder + 'dflog_' + timerun + '.csv', header = True, index = False)

after = time.time()
print("Execution finished in " + str((after - before)/60) + " minutes. Total of " + str(len(df)) + " segments found.")
arq.write("Execution finished in " + str((after - before)/60) + " minutes. Total of " + str(len(df)) + " segments found.")
