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
import soundfile as sf # scipy cannot read 24-bit files
from scipy.signal import welch
import numpy as np
import os
import re
from datetime import datetime, timedelta


class OceanPod:
    ''' class OceanPod: provides an interface to read the wave files,
        obtain segments from datetime, and datetime from segment indexes
    '''

    def __init__(self, wav_folder, file_format = '\d\d\d\d.\d\d.\d\d_\d\d.\d\d.\d\d', date_format = '%Y.%m.%d_%H.%M.%S'):
        # Constructor
        # Folder with waveforms
        if '(' not in file_format:
            file_format = '(' + file_format + ')'
            self.pre = ''
        else:
            self.pre = file_format[:file_format.find('(')]
        self.file_format = file_format
        self.date_format = date_format        
        self.wav_folder = wav_folder
        #self.filelist = [f for f in os.listdir(wav_folder) if f.endswith('.wav')]
        self.filelist = [f for f in os.listdir(wav_folder) if f.endswith('.wav') and re.search(file_format, f) is not None]
        self.filedt = [self.index2date(f) for f in self.filelist]
        self.filelist = [f for _,f in sorted(zip(self.filedt, self.filelist))]
        self.filedt = [f for f,_ in sorted(zip(self.filedt, self.filelist))]
        self.fs = None
        
        # Calculates max time
        filename = self.filelist[np.argmax(self.filedt)]
        #fs, waveform = scipy.io.wavfile.read(self.wav_folder + filename)
        waveform, fs = sf.read(self.wav_folder + filename)
        self.fs = fs
        self.maxtime = max(self.filedt) + timedelta(seconds = len(waveform) / fs)


    def read_file(self, filename):
        # Reads file, return fs and wave
        #fs, waveform = scipy.io.wavfile.read(self.wav_folder + filename)
        waveform, fs = sf.read(self.wav_folder + filename)
        #waveform = waveform / 32767 # To normalize amplitudes: UNNECESSARY IF USING sf.read
        self.fs = fs

        return fs, waveform

    def index2date(self, filename, seg_index = 0, fs = 11025):
        # Converts an index plus the file name in a datetime
        # if no index is given, converts filename to datetime

        date_raw = re.search(self.file_format, filename)
        date_final = datetime.strptime(date_raw.groups(0)[0], self.date_format)
        date_final = date_final + timedelta(seconds = seg_index / fs)

        return date_final

    def date2file(self, dt):
        # Converts date into filename

        filename = self.pre + datetime.strftime(dt, self.date_format + '.wav')

        return filename


    def get_segment(self, starttime, duration):
        # Read the segment starting at datetime start_time,
        #    with duration in seconds
        endtime = starttime + timedelta(seconds = duration)
        
        if self.maxtime < endtime:
            # Error: segment time span not contained in audio files list
            raise ValueError("Error: segment time span not contained in audio files list")
        #endif

        # Finds date of file with beginning of the segment
        dtstart = max([d for d in self.filedt if d <= starttime])

        # Reads file
        fs, wav = self.read_file(self.date2file(dtstart))
        self.fs = fs

        # Index of segment start
        istart = (starttime-dtstart).total_seconds()
        istart = int(np.floor(istart * fs))

        # Segment duration in number of points
        idur = int(np.floor(duration * fs))
        N = idur

        segment = wav[istart:min(istart + idur, len(wav)-1)]
        dtnext = dtstart

        while idur > len(wav) - istart - 1:
            # End of segment is in posterior file
            istart = 0            
            idur = idur - len(wav) + 1
            dtnext = min([d for d in self.filedt if d > dtnext])
            fsnext, wav = self.read_file(self.date2file(dtnext))

            # Adjusting to treat different sample rates in files
            idur = int(fsnext / fs) * idur
            indwav = min(idur, len(wav)-1)
            segment = np.concatenate([segment, wav[:indwav]])
        #endwhile

        return segment[:N]
    
    def get_spectrogram(self, starttime, duration, tdur, toverlap = 0, nwindow_welch = 3, poverlap_welch = 0, tipo_window = 'hann'):
        ''' Cria o espectrograma do sinal começando em starttime e com duração duration, em segundos.
            O espectrograma será criado usando o método de Welch.
            O sinal completo será primeiro dividido em janelas com duração de tdur segundos com overlap de toverlap segundos.
            Em cada janela será aplicado o método de Welch usando nwindow_welch janelas, com proporção de overlap
            dada por poverlap_welch.
        '''
        
        # TODO: toverlap não está sendo usado
        
        # Quanto da duração já foi rodada (em segundos)
        total_duration = 0
        
        # Marcador do tempo do sinal
        tempo_atual = starttime
        
        # Matriz vazia para guardar o espectrograma final
        espectrograma = None
        while total_duration < duration:
            # Primeiro obtenho a janela
            window = self.get_segment(tempo_atual, tdur)
            
            # TODO: tratar arquivos com fs diferentes!
            
            # Tamanho da janela
            N = len(window)
            
            # Número de pontos por segmento para cálculo do welch
            nperseg = np.floor(N / (nwindow_welch + (nwindow_welch - 1)*poverlap_welch))

            # Número de pontos de overlap do cálculo do welch
            noverlap = np.floor(nperseg * poverlap_welch)
            
            f, Pxx = welch(window, self.fs, window = tipo_window, nperseg = nperseg, noverlap = noverlap, nfft = None)
            
            if espectrograma is None:
                espectrograma = Pxx
            else:
                espectrograma = np.vstack([espectrograma, Pxx])
            
            # TODO: ele vai repetir o último segundo da janela?
            tempo_atual = tempo_atual + timedelta(seconds = tdur)
            
            total_duration = total_duration + tdur
            
        return f, espectrograma.T
        
        
    def get_spectrogram_filelist(self, data_minima, data_maxima, nwindow, poverlap = 1/3, tipo_window = 'hann', by = None, nfft = None):
        ''' Cria o espectrograma da lista de arquivos no diretório com a data entre data_minima e data_maxima. 
            Vai criar um espectrograma por período, dado por "by", onde 'by' pode ser 'year', 'month', 'day', 'hour' ou 'minute'.
            Se 'by' = 'day', por exemplo, vai concatenar os espectrogramas de todos os ARQUIVOS que tem o mesmo dia NO NOME.
            
            Cada arquivo será dividido em nwindow janelas para o método de welch.
            
            Se 'by' = None, vai gerar um único espectrograma
        '''
        
        # Pego a lista de datas dos arquivos entre data mínima e data máxima
        lista_datas = [d for d in self.filedt if d < data_maxima and d >= data_minima]
        
        espectrogramas = []
        espectrograma = None
        data_anterior = None
        for data in lista_datas:

            # Primeiro obtenho o nome do arquivo
            arquivo = self.date2file(data)

            # Leio o arquivo
            fs, y = self.read_file(arquivo)
            
            nperseg = np.floor(len(y) / nwindow)
            noverlap = np.floor(poverlap * nperseg)
            
            # Obtenho o espectrograma            
            f, Pxx = welch(y, fs, window = tipo_window, nperseg = nperseg, noverlap = noverlap, nfft = nfft)
            
            if espectrograma is None:
                if by is not None:
                    espectrograma = Pxx
                else:
                    espectrogramas.append(Pxx.T)
                data_anterior = data
                  
            else:
                # Verifica se os arquivo atual está no mesmo período ('by') do arquivo anterior
                if by is not None:
                    if by == 'year':
                        if data.year == data_anterior.year:
                            espectrograma = np.vstack([espectrograma, Pxx])
                        else:
                            espectrogramas.append(espectrograma.T)
                            espectrograma = Pxx
                    elif by == 'month':
                        if data.month == data_anterior.month:
                            espectrograma = np.vstack([espectrograma, Pxx])
                        else:
                            espectrogramas.append(espectrograma.T)
                            espectrograma = Pxx                        
                    elif by == 'day':
                        if data.day == data_anterior.day:
                            espectrograma = np.vstack([espectrograma, Pxx])
                        else:
                            espectrogramas.append(espectrograma.T)
                            espectrograma = Pxx         
                    elif by == 'hour':
                        if data.hour == data_anterior.hour:
                            espectrograma = np.vstack([espectrograma, Pxx])
                        else:
                            espectrogramas.append(espectrograma.T)
                            espectrograma = Pxx           
                    elif by == 'minute':
                        if data.minute == data_anterior.minute:
                            espectrograma = np.vstack([espectrograma, Pxx])
                        else:
                            espectrogramas.append(espectrograma.T)
                            espectrograma = Pxx               
                    else:
                        print("Erro! Valor de by não permitido ({})".format(by))
                        raise ValueError
                else:
                    espectrogramas.append(Pxx.T)
                    
                data_anterior = data

        if espectrograma is not None:
            espectrogramas.append(espectrograma.T)
           
            
        return espectrogramas