# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 18:29:21 2017

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

Audio wave files interface class
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
        self.filelist = [f for f in os.listdir(wav_folder) if f.endswith('.wav')]
        self.filedt = [self.index2date(f) for f in self.filelist]
        self.filelist = [f for _,f in sorted(zip(self.filedt, self.filelist))]

    def read_file(self, filename):
        # Reads file, return fs and wave
        fs, waveform = scipy.io.wavfile.read(self.wav_folder + filename)
        waveform = waveform / 32767 # To normalize amplitudes

        return fs, waveform

    def index2date(self, filename, seg_index = 0, fs = 11025):
        # Converts an index plus the file name in a datetime
        # if no index is given, converts filename to datetime
        date_raw = re.search('\d\d\d\d\.\d\d\.\d\d_\d\d\.\d\d\.\d\d', filename)
        if date_raw is None:
            date_raw = re.search('\d\d\d\d_\d\d_\d\d_\d\d_\d\d_\d\d', filename)
            if date_raw is None:
                date_raw = re.search('\d\d-(\d\d\d\d\d\d_\d\d\d\d)', filename)
                date_final = datetime.strptime(date_raw.group(1), '%y%m%d_%H%M')                     
            else:
                date_final = datetime.strptime(date_raw.group(0), '%Y_%m_%d_%H_%M_%S')                     
        else:
            date_final = datetime.strptime(date_raw.group(0), '%Y.%m.%d_%H.%M.%S')
        date_final = date_final + timedelta(seconds = seg_index / fs)

        return date_final

    def date2file(self, dt):
        # Converts date into filename

        filename = datetime.strftime(dt, '%Y.%m.%d_%H.%M.%S.wav')

        return filename


    def get_segment(self, starttime, duration):
        # Read the segment starting at datetime start_time,
        #    with duration in seconds
        endtime = starttime + timedelta(seconds = duration)
        if max(self.filedt) + timedelta(minutes = 15) < endtime:
            # Error: segment time span not contained in audio files list
            return None
        #endif

        # Finds date of file with beginning of the segment
        dtstart = max([d for d in self.filedt if d <= starttime])

        # Reads file
        fs, wav = self.read_file(self.date2file(dtstart))

        # Index of segment start
        istart = (starttime-dtstart).total_seconds()
        istart = int(istart * fs)

        # Segment duration in number of points
        idur = duration * fs

        segment = wav[istart:min(istart + idur, len(wav)-1)]
        dtnext = dtstart

        while idur > len(wav) - istart - 1:

            idur = idur - (len(wav) - istart) + 1
            istart = 0
            # End of segment is in posterior file
            dtnext = min([d for d in self.filedt if d > dtnext])
            fsnext, wav = self.read_file(self.date2file(dtnext))

            # Adjusting to treat different sample rates in files
            idur = int(fsnext / fs) * idur
            indwav = min(idur, len(wav)-1)
            segment = np.concatenate([segment, wav[:indwav]])
        #endwhile

        return segment
