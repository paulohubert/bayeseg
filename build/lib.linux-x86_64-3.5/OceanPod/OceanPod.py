# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 18:29:21 2017

@author: paulo
"""


import scipy.io.wavfile
import numpy as np
import os
import re
from datetime import datetime, timedelta


class OceanPod:
    ''' class OceanPod: provides an interface to read the wave files,
        obtain segments from datetime, and datetime from segment indexes
    '''
    
    def __init__(self, wav_folder):
        # Constructor
        # Folder with waveforms
        self.wav_folder = wav_folder
        self.Filelist = [f for f in os.listdir(wav_folder) if f.endswith('.wav')]
        self.Filedt = [self.index2date(f) for f in self.Filelist]
        self.Filelist = [f for _,f in sorted(zip(self.Filedt, self.Filelist))]
    
    def readFile(self, filename):
        # Reads file, return fs and wave
        fs, waveform = scipy.io.wavfile.read(self.wav_folder + filename)
        waveform = waveform / 32767 # To normalize amplitudes
        
        return fs, waveform
       
    def index2date(self, filename, seg_index = 0, fs = 11025):
        # Converts an index plus the file name in a datetime
        # if no index is given, converts filename to datetime
        date_raw = re.search('\d\d\d\d.\d\d.\d\d_\d\d.\d\d.\d\d', filename)
        date_final = datetime.strptime(date_raw.group(0), '%Y.%m.%d_%H.%M.%S')
        date_final = date_final + timedelta(seconds = seg_index / fs)
        
        return date_final

    def date2file(self, dt):
        # Converts date into filename
        y = dt.year
        m = str(dt.month)
        if dt.month < 10:
            m = '0' + m
        d = str(dt.day)
        if dt.day < 10:
            d = '0' + d
        h = str(dt.hour)
        if dt.hour < 10:
            h = '0' + h
        mi = str(dt.minute)
        if dt.minute < 10:
            mi = '0' + mi
        s = str(dt.second)
        if dt.second < 10:
            s = '0' + s
            
        filename = str(y) + '.' + m + '.' + d + '_' + h + '.' + mi + '.' + s + '.wav'
        
        return filename
       

    def getSegment(self, starttime, duration):
        # Read the segment starting at datetime start_time,
        #    with duration in seconds
        endtime = starttime + timedelta(seconds = duration)
        if max(self.Filedt) + timedelta(minutes = 15) < endtime:
            # Error: segment time span not contained in audio files list
            return None
        #endif
        
        # Finds date of file with beginning of the segment
        dtstart = max([d for d in self.Filedt if d <= starttime])

        # Reads file
        fs, wav = self.readFile(self.date2file(dtstart))
        
        # Index of segment start
        istart = (starttime - dtstart).days * 24 * (60 ** 2) + (starttime - dtstart).seconds
        istart = istart * fs
        
        # Segment duration in number of points
        idur = duration * fs
        
        segment = wav[istart:min(istart + idur, len(wav)-1)]
        dtnext = dtstart
        
        while idur > len(wav) - istart - 1:
            
            idur = idur - (len(wav) - istart) + 1
            istart = 0
            # End of segment is in posterior file
            dtnext = min([d for d in self.Filedt if d > dtnext])
            fsnext, wav = self.readFile(self.date2file(dtnext))
            
            # Adjusting to treat different sample rates in files
            idur = int(fsnext / fs) * idur
            indwav = min(idur, len(wav)-1)
            segment = np.concatenate([segment, wav[:indwav]])
        #endwhile
            
        return segment