import os
import math
import numpy as np
import pandas as pd
import librosa
from scipy import signal

from LibFMP.B import read_csv

PI = math.pi


# sonficiations
plato = librosa.load(os.getcwd() + "/plato.wav")[0]
electro = librosa.load(os.getcwd() + "/electro.wav")[0]
beat = librosa.load(os.getcwd() + "/beat.wav")[0]


##########################################
def get_sound(label,fs, sinusoid=True):

    if(label==1):
        sound = plato
        sine_periods = 8
        sine_freq = 800
    elif(label==2):
        sound = beat
        sine_periods = 16
        sine_freq = 1600
    elif(label==3):
        sound = electro
        sine_periods = 32
        sine_freq = 3200
    else:
        sound = plato
        sine_periods = 8
        sine_freq = 880

    if(sinusoid==True):
        sound = np.sin(np.linspace(0, sine_periods * 2 * np.pi, sine_periods * fs//sine_freq))

    return sound


##########################################
def get_frequency(label):

    if(label==1):
        freq = 1000
    elif(label==2):
        freq = 2000
    elif(label==3):
        freq = 4000
    else:
        freq = 1000

    return freq


#####################################
def get_peaks(novelty):

    peak_idx = signal.find_peaks( novelty, prominence=0.01 )[0]
    return peak_idx


##########################################
def sonification_own( peaks, amplitudes, labels, length, featureRate, fs=44100 ):

    sonification = np.zeros( length )

    for i in range(len(labels)):

        sound = get_sound(labels[i], fs)
        sound *= amplitudes[i]

        start = int(peaks[i]*fs)
        end = start + len(sound)

        if(end>len(sonification)):
            end = len(sonification)
            sound = sound[:end-start]

        sonification[start:end] = sound



    return sonification


##########################################
def sonification_librosa( peaks, amplitudes, labels, length, featureRate, fs=44100 ):


    #applying librosa sonification
    sonification = np.zeros( length )

    for i in [1,2,3]:

        idxs = np.where(labels==i)
        current_peaks = peaks[idxs]
        freq = get_frequency(i)

        current_sonification = librosa.clicks(current_peaks, sr=fs, click_freq=freq, length=length)
        sonification += current_sonification

    #scaling the sonificiations with the amplitude of the novelty curvve
    for i in range(len(peaks)-1):
        current_peak = int(peaks[i]*fs)
        length = int(0.1*fs)
        window = np.ones(length) #length of sonification is 100ms
        window = window*amplitudes[i]
        sonification[current_peak:current_peak+length] *= window

    return sonification



##########################################
def sonification_hpss_lab(novelty, length, fs, featureRate):

    pos = np.append(novelty, novelty[-1]) > np.insert(novelty, 0, novelty[0])
    neg = np.logical_not(pos)
    peaks = np.where(np.logical_and(pos[:-1], neg[1:]))[0]

    values = novelty[peaks]
    values /= np.max(values)
    peaks = peaks[values >= 0.01]
    values = values[values >= 0.01]
    peaks_idx = np.int32(np.round(peaks / featureRate * fs))

    sine_periods = 8
    sine_freq = 880
    click = np.sin(np.linspace(0, sine_periods * 2 * np.pi, sine_periods * fs//sine_freq))
    ramp = np.linspace(1, 1/len(click), len(click)) ** 2
    click = click * ramp
#     click = (click * np.abs(np.max(x)))

    out = np.zeros(length)
    for i, start in enumerate(peaks_idx):
        idx = np.arange(start, start+len(click))
        out[idx] += (click * values[i])

#     return np.vstack((x, out))
    return out, peaks






##########################################
def save_to_csv( path, file_name, time, amplitudes, labels ):

    #creating pandas df
    data = list(zip(time, amplitudes, labels))
    df = pd.DataFrame(data, columns = ['Time (sec)', 'Amplitude', "Label"])

    #exporting the df to a csv file
    df.to_csv( path + "/"+ file_name )

    return


##########################################
def load_from_csv( path, file_name):

    df = read_csv( path + "/" + file_name )
    time = df.loc[:,"Time (sec)"].values
    amplitudes = df.loc[:,"Amplitude"].values
    labels = df.loc[:,"Label"].values

    return {
        "time":time,
        "amplitudes":amplitudes,
        "labels":labels
    }
=======
"""
Module: LibFMP.C6.S1_Sonification
Author: Angel Villar-Corrales, Meinard Müller
License: Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License

This file is part of the Python Notebooks for Fundamentals of Music Processing (https://www.audiolabs-erlangen.de/FMP).
"""

import numpy as np
import pandas as pd
import librosa, os, math, csv
from scipy import signal
from matplotlib import pyplot as plt

from ..B.annotation import read_csv

PI = math.pi


#sonficiations
plato = librosa.load(os.getcwd() + "/plato.wav")[0]
electro = librosa.load(os.getcwd() + "/electro.wav")[0]
beat = librosa.load(os.getcwd() + "/beat.wav")[0]


##########################################
def get_sound(label,fs, sinusoid=True):

    if(label==1):
        sound = plato
        sine_periods = 8
        sine_freq = 800
    elif(label==2):
        sound = beat
        sine_periods = 16
        sine_freq = 1600
    elif(label==3):
        sound = electro
        sine_periods = 32
        sine_freq = 3200
    else:
        sound = plato
        sine_periods = 8
        sine_freq = 880

    if(sinusoid==True):
        sound = np.sin(np.linspace(0, sine_periods * 2 * np.pi, sine_periods * fs//sine_freq))

    return sound


##########################################
def get_frequency(label):

    if(label==1):
        freq = 1000
    elif(label==2):
        freq = 2000
    elif(label==3):
        freq = 4000
    else:
        freq = 1000

    return freq


#####################################
def get_peaks(novelty):

    peak_idx = signal.find_peaks( novelty, prominence=0.01 )[0]
    return peak_idx


##########################################
def sonification_own( peaks, amplitudes, labels, length, featureRate, fs=44100 ):

    sonification = np.zeros( length )

    for i in range(len(labels)):

        sound = get_sound(labels[i], fs)
        sound *= amplitudes[i]

        start = int(peaks[i]*fs)
        end = start + len(sound)

        if(end>len(sonification)):
            end = len(sonification)
            sound = sound[:end-start]

        sonification[start:end] = sound



    return sonification


##########################################
def sonification_librosa( peaks, amplitudes, labels, length, featureRate, fs=44100 ):


    #applying librosa sonification
    sonification = np.zeros( length )

    for i in [1,2,3]:

        idxs = np.where(labels==i)
        current_peaks = peaks[idxs]
        freq = get_frequency(i)

        current_sonification = librosa.clicks(current_peaks, sr=fs, click_freq=freq, length=length)
        sonification += current_sonification

    #scaling the sonificiations with the amplitude of the novelty curvve
    for i in range(len(peaks)-1):
        current_peak = int(peaks[i]*fs)
        length = int(0.1*fs)
        window = np.ones(length) #length of sonification is 100ms
        window = window*amplitudes[i]
        sonification[current_peak:current_peak+length] *= window

    return sonification



##########################################
def sonification_hpss_lab(novelty, length, fs, featureRate):

    pos = np.append(novelty, novelty[-1]) > np.insert(novelty, 0, novelty[0])
    neg = np.logical_not(pos)
    peaks = np.where(np.logical_and(pos[:-1], neg[1:]))[0]

    values = novelty[peaks]
    values /= np.max(values)
    peaks = peaks[values >= 0.01]
    values = values[values >= 0.01]
    peaks_idx = np.int32(np.round(peaks / featureRate * fs))

    sine_periods = 8
    sine_freq = 880
    click = np.sin(np.linspace(0, sine_periods * 2 * np.pi, sine_periods * fs//sine_freq))
    ramp = np.linspace(1, 1/len(click), len(click)) ** 2
    click = click * ramp
#     click = (click * np.abs(np.max(x)))

    out = np.zeros(length)
    for i, start in enumerate(peaks_idx):
        idx = np.arange(start, start+len(click))
        out[idx] += (click * values[i])

#     return np.vstack((x, out))
    return out, peaks






##########################################
def save_to_csv( path, file_name, time, amplitudes, labels ):

    #creating pandas df
    data = list(zip(time, amplitudes, labels))
    df = pd.DataFrame(data, columns = ['Time (sec)', 'Amplitude', "Label"])

    #exporting the df to a csv file
    df.to_csv( path + "/"+ file_name )

    return


##########################################
def load_from_csv( path, file_name):

    df = read_csv( path + "/" + file_name )
    time = df.loc[:,"Time (sec)"].values
    amplitudes = df.loc[:,"Amplitude"].values
    labels = df.loc[:,"Label"].values

    return {
        "time":time,
        "amplitudes":amplitudes,
        "labels":labels
    }
