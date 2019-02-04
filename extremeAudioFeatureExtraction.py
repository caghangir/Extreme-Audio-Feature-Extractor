%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Extreme Audio Feature Extractor                          %
%                                                          %
% Copyright (C) 2019 Cagatay Demirel. All rights reserved. %
%                    demirelc16@itu.edu.tr                 %
%                                                          %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#import sys
import time
import os
import glob
import numpy
#import cPickle
#import aifc
import math
import numpy as np
#from numpy import NaN, Inf, arange, isscalar, array
#from scipy.fftpack import rfft
from scipy.fftpack import fft
from scipy.fftpack.realtransforms import dct
#from scipy.signal import fftconvolve
from matplotlib.mlab import find
import matplotlib.pyplot as plt
#from scipy import linalg as la
#import audioTrainTest as aT
#import audioBasicIO
#import utilities
from scipy.signal import lfilter, hamming, savgol_filter, hilbert, fftconvolve, butter, iirnotch, freqz, firwin, filtfilt, resample
import scipy as sp
from scipy import stats
#import os
#from scipy.io import wavfile
#import peakutils as pu
import librosa

eps = 0.00000001


#class audioFeatureExtraction():
# =========================== General Methods ==================================
def parabolic(f, x):
    xv = 1/2. * (f[x-1] - f[x+1]) / (f[x-1] - 2 * f[x] + f[x+1]) + x
    yv = f[x] - 1/4. * (f[x-1] - f[x+1]) * (xv - x)
    return (xv, yv)
	
def signalNorm(tempSignal, bitdepth):
	signal = np.double(tempSignal)
	signal = signal / (2.0 ** bitdepth)
	DC = signal.mean()
	MAX = np.abs(signal).max()
	signal = (signal - DC) / (MAX + 0.0000000001)
	
	return signal

def FFT(signal, Fs):
    nFFT = len(signal) / 2
    nFFT = int(nFFT)
    #Hamming Window
    w = np.hamming(len(signal))
    #FFT
    X = abs(fft(signal * w))                                  # get fft magnitude
    X = X[0:nFFT]                                    # normalize fft
    X = X / len(X)
    
    fIndexes = (Fs / (2*nFFT)) * np.r_[0:nFFT] # [1,9] 9peet üretti
    
    return X, fIndexes

def envelopeCreator(timeSignal, degree, fs):
    absoluteSignal = np.abs(hilbert(timeSignal))
    intervalLength = int(fs / 40) 
    if(intervalLength % 2 == 0):
        intervalLength -= 1
    amplitude_envelopeFiltered = savgol_filter(absoluteSignal, intervalLength, degree)
    return amplitude_envelopeFiltered    
  
def envelopeCreatorMultiple(timeSignal, intervalLength, degree):
    amplitudeEnvelopeFiltered = np.zeros((np.size(timeSignal,axis=0),np.size(timeSignal,axis=1) - intervalLength))
    gap = 250
    n = len(timeSignal)
    for i in range(n):
        amplitudeEnvelopeFiltered[i,:] = envelopeCreator(timeSignal[i,:], intervalLength, degree)
        amplitudeEnvelopeFiltered[i,:] + gap * (n - 1 - i)
  
    return amplitudeEnvelopeFiltered

def spectrogramExtraction(timeSignal, Fs, nfft):
    f, t, Sxx = sp.signal.spectrogram(timeSignal, Fs, nfft=nfft, scaling = 'density', mode = 'magnitude', window = 'hamming')
    return f, Sxx
    
def melSpectrogramExtraction(timeSignal, Fs, n_mels, hop_length, power, fmax=8000):
    melSxx = librosa.feature.melspectrogram(y=timeSignal, sr=Fs, n_mels=n_mels, fmax=fmax, power = power, hop_length = hop_length)
    return melSxx
#============== MDVP (Multi-Dimensional Voice program) features ================   
def F0UsingAutocorrelation(sig, fs):

    corr = fftconvolve(sig, sig[::-1], mode='full')
    corr = corr[len(corr)//2:]
    # Find the first low point
    d = np.diff(corr)
    start = find(d > 0)[0]

    peak = np.argmax(corr[start:]) + start
    px, py = parabolic(corr, peak)

    return fs / px

def pitchPeriod(F0s): # T0 (Mt)
    return np.mean(1/F0s)    

def highestF0(F0s): #(MT)
    return np.max(F0s)

def lowestF0(F0s): #(MT)
    return np.min(F0s)    
    
def phonatoryF0Range(F0s): #PFR (Phonatory Fundamental Frequency Range) (MT)
    return highestF0(F0s) - lowestF0(F0s)

def f0Variation(F0s): # vF0 (MT)
    return np.std(F0s)
    
def amplitudeVariation(Amps): # vAm (MT)
    return np.std(Amps)
    
def noiseToHarmonicRatio( signal, rate, time_step = 0, min_pitch = 75,  # HNR (ST and MT but ST is better)
             silence_threshold = .1, periods_per_window = 4.5 ): 

    y_harmonic, y_percussive = librosa.effects.hpss(signal, margin=(1.0,5.0))
    ratio = sum(y_percussive) / sum(y_harmonic)
   
    return ratio

def percussiveSignal(signal, Fs):
    D = librosa.stft(signal, n_fft=40)
#                     int(Fs * 0.046))
    H, P = librosa.decompose.hpss(D, margin=(1.0,5.0))
    y_percussive = librosa.core.istft(P)
    return y_percussive
    
def harmonicSignal(signal, Fs):
    D = librosa.stft(signal, n_fft=40)
#                     int(Fs * 0.046))
    H, P = librosa.decompose.hpss(D, margin=(1.0,5.0))
    y_harmonic = librosa.core.istft(H)
    return y_harmonic
 
def PQ(x,k):
    """
    Perturbation Quotient in percentage of the signal x
    input: x--> input sequence: F0 values or Amplitude values
    k--> average factor (must be an odd number)
    """
    N=len(x)
    if N<k or k%2==0:
        return 0
    m=int(0.5*(k-1))
    summ=0
    for n in range(N-k):
        dif=0
        for r in range(k):
            dif=dif+x[n+r]-x[n+m]
        dif=np.abs(dif/float(k))
        summ=summ+dif
    num=summ/(N-k)
    den=np.mean(np.abs(x))
    c=100*num/den
#    if np.sum(np.isnan(c))>0:
#        print(x)
    return c

def APQ(PAS): # input olarak f0 lar alir (MT)
    """
    Amplitude perturbation quotient (APQ)
    input:-->PAS: secuence of peak amplitudes of a signal
    """
    return PQ(PAS,11)

def PPQ(PPS): # input olarak 1/f0 lar alir (MT)
    """
    Period perturbation quotient (APQ)
    input:-->PAS: secuence of pitch periods of a signal
    """
    return PQ(PPS,5)    
    
def calculateJitterRatio(data): #Jitt (relative) input f0 (MT)
    n = len(data)
    sum1 = 0
    sum2 = 0
    for i in range(n):
        if i > 0:
           sum1 += abs(data[i-1] - data[i])
        sum2 += data[i]
    sum1 /= float(n - 1)
    sum2 /= float(n)
    return 100.0 * sum1 / sum2    
    
def calculateJitterFactor(data): # Jitta (uS) input f0 (MT)
    n = len(data)
    dataF = numpy.zeros(n)
    for i in range(n):
         # convert from F0 to period per cycle
       dataF[i] = 1.0 / data[i]
    sum1 = 0
    sum2 = 0
    for i in range(n):
        if i > 0:
            sum1 += abs(dataF[i] - dataF[i-1])
        sum2 += dataF[i]
    sum1 /= float(n - 1)
    sum2 /= float(n)
    return 1000.0 * sum1 / sum2    
    
def shimmerInDb(data): # ShdB (Shimmer in db) (MT)
    sum1 = 0
    n = len(data)
    for i in range(n - 1):
        sum1 += np.abs(np.log10(data[i+1] / data[i]))
    sum1 = sum1 * (20 / n - 1) 
    return sum1
    
def shimmerRelative(data): # Shim (Shimmer in percent) (MT)
    sum1 = 0
    sum2 = 0
    n = len(data)
    for i in range(n):
       if i > 0:
          sum1 += abs(data[i-1] - data[i])
       sum2 += data[i]
    sum1 /= float(n - 1)
    sum2 /= float(n)
    return 100.0 * sum1 / sum2    
    
 #   calculate jitter in percent
#def calculateJitterPercent(data): # Jitt (Local %) input f0
#    return calculateJitterRatio(data) / 10.0       

  
 #   calculate the relative average perturbation (Koike, 1973), also termed
 #   freuqency perturbation quotient (Takahashi & Koike, 1975)
 #   
def calculateRelativeAveragePerturbation(data): # RAP (T0s) (Relative Average Perturbation) (MT)
     n = len(data)
     if n < 3:
         raise Exception("need at least three data points")
     sum1 = 0
     sum2 = 0
     for i in range(n):
         if i > 0 and i < (n-1):
             sum1 += abs((data[i-1] + data[i] + data[i+1]) / 3 - data[i])
         sum2 += data[i]
     sum1 /= float(n-2)
     sum2 /= float(n)
     return sum1 / sum2

# band-pass between two frequency     
def butter_bandpass(lowcut, highcut, fs, order=3): # 3 ten sonra lfilter NaN degerler vermeye basliyor
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band', analog=False)
    return b, a
    
# band-pass filter between two frequency     
def butter_bandpass_filter(data, lowcut, highcut, fs, order=3):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y
    
def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=3):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def butter_lowpass(cutoff, fs, order=3):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=3):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def notchFilter(data, Fs, f0, Q):
    w0 = f0/(Fs/2)
    b, a = iirnotch(w0, Q)
    y = lfilter(b, a, data)
#    bp_stop_Hz = np.array([49.0, 51.0])
#    b, a = butter(2,bp_stop_Hz/(Fs / 2.0), 'bandstop')
#    w, h = freqz(b, a)
    return y
     
def voiceTurbulanceIndexScore(timeSignal, fs): #VTI (MT and ST)
    # it finds high-frequency noise energy
    highFreqFilteredSignal = butter_bandpass_filter(timeSignal, 2800, 5800, fs, order=2) #order 5 olsun isterdim!!!
    highFreqFilteredDenoisedSignal = savgol_filter(highFreqFilteredSignal, 25, 5) # polyorder is 5, window is 25
    enHighFreqFilteredDenoised = np.sum(highFreqFilteredDenoisedSignal ** 2)
    enHighFreqFiltered = np.sum(highFreqFilteredSignal ** 2)
    enHighFreqNoise = enHighFreqFiltered - enHighFreqFilteredDenoised
    # it finds low-frequency denoised energy
    lowFreqFilteredSignal = butter_bandpass_filter(timeSignal, 70, 4500, fs, order=2)
    lowFreqFilteredDenoisedSignal = savgol_filter(lowFreqFilteredSignal, 25, 5) # polyorder is 5, window is 25
    enLowFreqFilteredDenoised = np.sum(lowFreqFilteredDenoisedSignal ** 2)
    
    ratio = enHighFreqNoise / enLowFreqFilteredDenoised
    return ratio

def ifVoiceless(timeSignal, fs):    # DUV'a giden yol (ST)
    [HR, F0] = stHarmonicRatio(timeSignal, fs)
    if(F0 != 0):
        return 1
    return F0
    
def degreeOfVoiceBreak(timeSignal, fs, threshold): # DVB (default threshold : 0) (MT)
    envelopeSignal = envelopeCreator(timeSignal, 1, fs)
    smallerAmountThreshold = sum(i < threshold for i in envelopeSignal)  # gives amount rather than sum of values
    ratio = smallerAmountThreshold / len(timeSignal)        
    return ratio
    
def phormants(timeSignal, Fs):
     # Get Hamming window.
        N = len(timeSignal)
        w = np.hamming(N)
    
        # Apply window and high pass filter.
        x1 = timeSignal * w
        x1 = lfilter([1], [1., 0.63], x1)
    
        # Get LPC.
        ncoeff = 2 + Fs / 1000
        A, e, k = lpc(x1, ncoeff)  
     
        # Get roots.
        rts = np.roots(A)
        rts = [r for r in rts if np.imag(r) >= 0]
    
        # Get angles.
        angz = np.arctan2(np.imag(rts), np.real(rts))
    
        # Get frequencies.
    #    Fs = spf.getframerate()
        frqs = sorted(angz * (Fs / (2 * math.pi)))
        if frqs[0] == 0:
            frqs = frqs[1:5]
        else:
            frqs = frqs[0:4] 
        return frqs          

#def SPI(signal, Fs): # SPI (ST)
##    y_harmonic = librosa.effects.harmonic(signal, margin=3.0) 
#    y_harmonic, y_percussive = librosa.effects.hpss(signal, margin=(1.0,5.0))
#    f, t, Sxx = sp.signal.spectrogram(y_harmonic, Fs, nfft=512, scaling = 'density', mode = 'magnitude', window = 'hamming')
#    spigram = np.sum(Sxx[20:52,:]) / np.sum(Sxx[1:20,:]) * 250
#    return spigram           
def SPI(signal, Fs): # SPI (ST)
#    signal = chettoaudio.signalNorm(signal, bitdepth)
    signal = signal / max(np.abs(signal))
    signal = butter_lowpass_filter(signal, 6000, Fs)
    #=====
    downsample = 12500
    ratio = downsample / 44100
    size = int(len(signal) * ratio)
    signal = resample(signal, size)
    #=====
#    signal = np.abs(hilbert(signal))
    #=====
#    y_harmonic, y_percussive = librosa.effects.hpss(signal, margin=(1.0,5.0))
    f, t, Sxx = sp.signal.spectrogram(signal, downsample, nfft=512, scaling = 'density', mode = 'magnitude', window = 'hamming')
#    fft, f = chettoaudio.fftCalc(y_harmonic, downsample)
    #======
    lowest_bound_l=int(np.argwhere(f>70)[0])
    highest_bound_l=int(np.argwhere(f>1800)[0] + 1)
    highest_bound_h=int(np.argwhere(f>4500)[0] + 1)
    #======
#    spigram = np.sum(Sxx[lowest_bound_l:highest_bound_l,:]) / np.sum(Sxx[highest_bound_l:highest_bound_h,:]) * 1.63
#    spigram = np.sum(fft[lowest_bound_l:highest_bound_l]) / np.sum(fft[highest_bound_l:highest_bound_h]) * 2.45
    spigram = np.sum(Sxx[lowest_bound_l:highest_bound_l,:]) / np.sum(Sxx[highest_bound_l:highest_bound_h,:]) * 1.82
    
    return spigram

def newSPI(signal, Fs): # SPI (ST)
#    signal = chettoaudio.signalNorm(signal, bitdepth)
    signal = signal / max(np.abs(signal))
    signal = butter_lowpass_filter(signal, 6000, Fs)
    #=====
    downsample = 12500
    ratio = downsample / 44100
    size = int(len(signal) * ratio)
    signal = resample(signal, size)
    #=====
#    signal = np.abs(hilbert(signal))
    #=====
#    y_harmonic, y_percussive = librosa.effects.hpss(signal, margin=(1.0,5.0))
#    Sxx, f, t, img = plt.specgram(signal, NFFT=256, Fs=downsample, Fc=0, noverlap=128, cmap=None, xextent=None, pad_to=None)
#    plt.close()
    f, t, Sxx = sp.signal.spectrogram(signal, downsample, nfft=256, scaling = 'density', mode = 'magnitude', window = 'hamming')
#    fft, f = chettoaudio.fftCalc(y_harmonic, downsample)
    #======
    lowest_bound_l=int(np.argwhere(f>70)[0])
    highest_bound_l=int(np.argwhere(f>1800)[0] + 1)
    highest_bound_h=int(np.argwhere(f>4500)[0] + 1)
    #======
#    spigram = np.sum(Sxx[lowest_bound_l:highest_bound_l,:]) / np.sum(Sxx[highest_bound_l:highest_bound_h,:]) * 1.63
#    spigram = np.sum(fft[lowest_bound_l:highest_bound_l]) / np.sum(fft[highest_bound_l:highest_bound_h]) * 2.45
    spigram = np.sum(Sxx[highest_bound_l:highest_bound_h,:]) / np.sum(Sxx[lowest_bound_l:highest_bound_l,:])
    
    return spigram
    
def newSPI_FFT(signal, Fs): # SPI (ST)
#    signal = chettoaudio.signalNorm(signal, bitdepth)
    signal = signal / max(np.abs(signal))
    signal = butter_lowpass_filter(signal, 6000, Fs)
    #=====
    downsample = 12500
    ratio = downsample / 44100
    size = int(len(signal) * ratio)
    signal = resample(signal, size)
    #=====
    fft, frequencies = FFT(signal, downsample)
    lowest_bound_l=int(np.argwhere(frequencies>70)[0])
    highest_bound_l=int(np.argwhere(frequencies>1800)[0] + 1)
    highest_bound_h=int(np.argwhere(frequencies>4500)[0] + 1)
    
    #=====
    spigram = np.sum(fft[lowest_bound_l:highest_bound_l]) / np.sum(fft[highest_bound_l:highest_bound_h])   

    return spigram
    
def highBandSpectrogramEnergy(signal, Fs):
    f, t, Sxx = sp.signal.spectrogram(signal, Fs, nfft=256, scaling = 'density', mode = 'magnitude', window = 'hamming')
#    Sxx, f, t, img = plt.specgram(signal, NFFT=256, Fs=Fs, Fc=0, noverlap=128, cmap=None, xextent=None, pad_to=None)
    lowest_bound=int(np.argwhere(f>5000)[0])
    highest_bound=int(np.argwhere(f>7000)[0] + 1)
    
    highBandEnergy = np.exp(sum(sum(Sxx[lowest_bound:highest_bound, :])))
#    f, t, Sxx = sp.signal.spectrogram(signal, downsample, nfft=512, scaling = 'density', mode = 'magnitude', window = 'hamming')
#    fft, f = chettoaudio.fftCalc(y_harmonic, downsample)
    #======
#    lowest_bound=int(np.argwhere(f>5000)[0])
#    highest_bound=int(np.argwhere(f>7000)[0])
    
    return highBandEnergy
    
    
def mtSPI(signal, Fs, Win): #Win = ms
    Win = int(Win * (Fs / 1000))
    N = len(signal) # total number of samples
    windowAmount = int(np.floor(N / Win))
    spiVals = np.zeros((windowAmount))
    for i in range(windowAmount):
        spiVals[i] = SPI(signal[i*Win:(i+1)*Win], Fs)
    meanSPI = np.mean(spiVals)
    return meanSPI

#def ATRI    
#==================================================================================================================================    
def stSpectralLowHighRatio(dataFFT, thresholdLow, thresholdHigh, Fs): #fft degerlerinde low threshold / highthreshold
    nyquist = Fs / 2
    thresholdLowFreq = round(thresholdLow / nyquist * len(dataFFT))
    thresholdHighFreq = round(thresholdHigh / nyquist * len(dataFFT))
    SMLH = numpy.sum(dataFFT[0:thresholdLowFreq]) / numpy.sum(dataFFT[thresholdHighFreq:])    
    return SMLH
    
#def stSpectralHarmonic()
    
def stHarmonicRatio(frame, fs):
    """
    Computes harmonic ratio and pitch
    """
    eps = 0.00000001
    M = np.round(0.016 * fs) - 1
    R = np.correlate(frame, frame, mode='full')

    g = R[len(frame)-1]
    R = R[len(frame):-1]

    # estimate m0 (as the first zero crossing of R)
    [a, ] = np.nonzero(np.diff(np.sign(R)))

    if len(a) == 0:
        m0 = len(R)-1
    else:
        m0 = a[0]
    if M > len(R):
        M = len(R) - 1

#    Gamma = np.zeros((M), dtype=np.float64)
    M = int(M)
    Gamma = np.zeros((M))
    CSum = np.cumsum(frame ** 2)
#    CSum = float(CSum)
    Gamma[m0:M] = R[m0:M] / (np.sqrt((g * CSum[M:m0:-1])) + eps)

    ZCR = stZCR(Gamma)

    if ZCR > 0.15:
        HR = 0.0
        f0 = 0.0
    else:
        if len(Gamma) == 0:
            HR = 1.0
            blag = 0.0
            Gamma = np.zeros((M), dtype=np.float64)
        else:
            HR = np.max(Gamma)
            blag = np.argmax(Gamma)

        # Get fundamental frequency:
        f0 = fs / (blag + eps)
        if f0 > 5000:
            f0 = 0.0
        if HR < 0.1:
            f0 = 0.0

    return HR, f0

def stZCR(frame):
    """Computes zero crossing rate of frame"""
    count = len(frame)
    countZ = numpy.sum(numpy.abs(numpy.diff(numpy.sign(frame)))) / 2
    return (numpy.float64(countZ) / numpy.float64(count-1.0))


def stEnergy(frame):
    """Computes signal energy of frame"""
    return numpy.sum(frame ** 2) / numpy.float64(len(frame))


def stEnergyEntropy(frame, numOfShortBlocks=10):
    """Computes entropy of energy"""
    Eol = numpy.sum(frame ** 2)    # total frame energy
    L = len(frame)
    subWinLength = int(numpy.floor(L / numOfShortBlocks)) # short block uzunlugu
    if L != subWinLength * numOfShortBlocks:
            frame = frame[0:subWinLength * numOfShortBlocks]
    # subWindows is of size [numOfShortBlocks x L]
    subWindows = frame.reshape(subWinLength, numOfShortBlocks, order='F').copy()

    # Compute normalized sub-frame energies:
    s = numpy.sum(subWindows ** 2, axis=0) / (Eol + eps)

    # Compute entropy of the normalized sub-frame energies:
    Entropy = -numpy.sum(s * numpy.log2(s + eps))
    return Entropy


""" Frequency-domain audio features """


def stSpectralCentroidAndSpread(X, fs):
    """Computes spectral centroid of frame (given abs(FFT))"""
    ind = (numpy.arange(1, len(X) + 1)) * (fs/(2.0 * len(X)))

    Xt = X.copy()
    Xt = Xt / Xt.max()
    NUM = numpy.sum(ind * Xt)
    DEN = numpy.sum(Xt) + eps

    # Centroid:
    C = (NUM / DEN)

    # Spread:
    S = numpy.sqrt(numpy.sum(((ind - C) ** 2) * Xt) / DEN)

    # Normalize:
    C = C / (fs / 2.0)
    S = S / (fs / 2.0)

    return (C, S)


def stSpectralEntropy(X, numOfShortBlocks=10):
    """Computes the spectral entropy"""
    L = len(X)                         # number of frame samples
    Eol = numpy.sum(X ** 2)            # total spectral energy

    subWinLength = int(numpy.floor(L / numOfShortBlocks))   # length of sub-frame
    if L != subWinLength * numOfShortBlocks:
        X = X[0:subWinLength * numOfShortBlocks]

    subWindows = X.reshape(subWinLength, numOfShortBlocks, order='F').copy()  # define sub-frames (using matrix reshape)
    s = numpy.sum(subWindows ** 2, axis=0) / (Eol + eps)                      # compute spectral sub-energies
    En = -numpy.sum(s*numpy.log2(s + eps))                                    # compute spectral entropy

    return En


def stSpectralFlux(X, Xprev):
    """
    Computes the spectral flux feature of the current frame
    ARGUMENTS:
        X:        the abs(fft) of the current frame
        Xpre:        the abs(fft) of the previous frame
    """
    # compute the spectral flux as the sum of square distances:
    sumX = numpy.sum(X + eps)
    sumPrevX = numpy.sum(Xprev + eps)
    F = numpy.sum((X / sumX - Xprev/sumPrevX) ** 2)

    return F


def stSpectralRollOff(X, c, fs):
    """Computes spectral roll-off"""
    totalEnergy = numpy.sum(X ** 2)
    fftLength = len(X)
    Thres = c*totalEnergy
    # Ffind the spectral rolloff as the frequency position where the respective spectral energy is equal to c*totalEnergy
    CumSum = numpy.cumsum(X ** 2) + eps
    [a, ] = numpy.nonzero(CumSum > Thres)
    if len(a) > 0:
        mC = numpy.float64(a[0]) / (float(fftLength))
    else:
        mC = 0.0
    return (mC)

def mfccInitFilterBanks(fs, nfft):
    """
    Computes the triangular filterbank for MFCC computation (used in the stFeatureExtraction function before the stMFCC function call)
    This function is taken from the scikits.talkbox library (MIT Licence):
    https://pypi.python.org/pypi/scikits.talkbox
    """

    # filter bank params:
    lowfreq = 133.33
    linsc = 200/3.
    logsc = 1.0711703
    numLinFiltTotal = 13
    numLogFilt = 27

    if fs < 8000:
        nlogfil = 5

    # Total number of filters
    nFiltTotal = numLinFiltTotal + numLogFilt

    # Compute frequency points of the triangle:
    freqs = numpy.zeros(nFiltTotal+2)
    freqs[:numLinFiltTotal] = lowfreq + numpy.arange(numLinFiltTotal) * linsc
    freqs[numLinFiltTotal:] = freqs[numLinFiltTotal-1] * logsc ** numpy.arange(1, numLogFilt + 3)
    heights = 2./(freqs[2:] - freqs[0:-2])

    # Compute filterbank coeff (in fft domain, in bins)
    fbank = numpy.zeros((nFiltTotal, nfft))
    nfreqs = numpy.arange(nfft) / (1. * nfft) * fs

    for i in range(nFiltTotal):
        lowTrFreq = freqs[i]
        cenTrFreq = freqs[i+1]
        highTrFreq = freqs[i+2]

        lid = numpy.arange(numpy.floor(lowTrFreq * nfft / fs) + 1, numpy.floor(cenTrFreq * nfft / fs) + 1, dtype=numpy.int)
        lslope = heights[i] / (cenTrFreq - lowTrFreq)
        rid = numpy.arange(numpy.floor(cenTrFreq * nfft / fs) + 1, numpy.floor(highTrFreq * nfft / fs) + 1, dtype=numpy.int)
        rslope = heights[i] / (highTrFreq - cenTrFreq)
        fbank[i][lid] = lslope * (nfreqs[lid] - lowTrFreq)
        fbank[i][rid] = rslope * (highTrFreq - nfreqs[rid])

    return fbank, freqs


def stMFCC(X, fbank, nceps):
    """
    Computes the MFCCs of a frame, given the fft mag

    ARGUMENTS:
        X:        fft magnitude abs(FFT)
        fbank:    filter bank (see mfccInitFilterBanks)
    RETURN
        ceps:     MFCCs (13 element vector)

    Note:    MFCC calculation is, in general, taken from the scikits.talkbox library (MIT Licence),
    #    with a small number of modifications to make it more compact and suitable for the pyAudioAnalysis Lib
    """

    mspec = numpy.log10(numpy.dot(X, fbank.T)+eps)
    ceps = dct(mspec, type=2, norm='ortho', axis=-1)[:nceps]
    return ceps


def stChromaFeaturesInit(nfft, fs):
    """
    This function initializes the chroma matrices used in the calculation of the chroma features
    """
    freqs = numpy.array([((f + 1) * fs) / (2 * nfft) for f in range(nfft)])    
    Cp = 27.50    
    nChroma = numpy.round(12.0 * numpy.log2(freqs / Cp)).astype(int)

    nFreqsPerChroma = numpy.zeros((nChroma.shape[0], ))

    uChroma = numpy.unique(nChroma)
    for u in uChroma:
        idx = numpy.nonzero(nChroma == u)
        nFreqsPerChroma[idx] = idx[0].shape
    
    return nChroma, nFreqsPerChroma


def stChromaFeatures(X, fs, nChroma, nFreqsPerChroma):
    #TODO: 1 complexity
    #TODO: 2 bug with large windows

    chromaNames = ['A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#']
    spec = X**2    
    if nChroma.max()<nChroma.shape[0]:        
        C = numpy.zeros((nChroma.shape[0],))
        C[nChroma] = spec
        C /= nFreqsPerChroma[nChroma]
    else:        
        I = numpy.nonzero(nChroma>nChroma.shape[0])[0][0]        
        C = numpy.zeros((nChroma.shape[0],))
        C[nChroma[0:I-1]] = spec            
        C /= nFreqsPerChroma
    finalC = numpy.zeros((12, 1))
    newD = int(numpy.ceil(C.shape[0] / 12.0) * 12)
    C2 = numpy.zeros((newD, ))
    C2[0:C.shape[0]] = C
    C2 = C2.reshape(int(C2.shape[0]/12), 12) #!!!!!!!!!!!!!!!!!!edited!!!!!!!!!!!!!!!
    #for i in range(12):
    #    finalC[i] = numpy.sum(C[i:C.shape[0]:12])
    finalC = numpy.matrix(numpy.sum(C2, axis=0)).T
    finalC /= spec.sum()

#    ax = plt.gca()
#    plt.hold(False)
#    plt.plot(finalC)
#    ax.set_xticks(range(len(chromaNames)))
#    ax.set_xticklabels(chromaNames)
#    xaxis = numpy.arange(0, 0.02, 0.01);
#    ax.set_yticks(range(len(xaxis)))
#    ax.set_yticklabels(xaxis)
#    plt.show(block=False)
#    plt.draw()

    return chromaNames, np.ravel(finalC)


def stChromagram(signal, Fs, Win, Step, PLOT=False):
    """
    Short-term FFT mag for spectogram estimation:
    Returns:
        a numpy array (nFFT x numOfShortTermWindows)
    ARGUMENTS:
        signal:      the input signal samples
        Fs:          the sampling freq (in Hz)
        Win:         the short-term window size (in samples)
        Step:        the short-term window step (in samples)
        PLOT:        flag, 1 if results are to be ploted
    RETURNS:
    """
    Win = int(Win)
    Step = int(Step)
    signal = numpy.double(signal)
    signal = signal / (2.0 ** 15)
    DC = signal.mean()
    MAX = (numpy.abs(signal)).max()
    signal = (signal - DC) / (MAX - DC)

    N = len(signal)        # total number of signals
    curPos = 0
    countFrames = 0
    nfft = int(Win / 2)
    nChroma, nFreqsPerChroma = stChromaFeaturesInit(nfft, Fs)
    chromaGram = numpy.array([], dtype=numpy.float64)

    while (curPos + Win - 1 < N):
        countFrames += 1
        x = signal[curPos:curPos + Win]
        curPos = curPos + Step
        X = abs(fft(x))
        X = X[0:nfft]
        X = X / len(X)
        chromaNames, C = stChromaFeatures(X, Fs, nChroma, nFreqsPerChroma)
        C = C[:, 0]
        if countFrames == 1:
            chromaGram = C.T
        else:
            chromaGram = numpy.vstack((chromaGram, C.T))
    FreqAxis = chromaNames
    TimeAxis = [(t * Step) / Fs for t in range(chromaGram.shape[0])]

    if (PLOT):
        fig, ax = plt.subplots()
        chromaGramToPlot = chromaGram.transpose()[::-1, :]
        Ratio = chromaGramToPlot.shape[1] / (3*chromaGramToPlot.shape[0])        
        if Ratio < 1:
            Ratio = 1
        chromaGramToPlot = numpy.repeat(chromaGramToPlot, Ratio, axis=0)
        imgplot = plt.imshow(chromaGramToPlot)
        Fstep = int(nfft / 5.0)
#        FreqTicks = range(0, int(nfft) + Fstep, Fstep)
#        FreqTicksLabels = [str(Fs/2-int((f*Fs) / (2*nfft))) for f in FreqTicks]
        ax.set_yticks(range(Ratio / 2, len(FreqAxis) * Ratio, Ratio))
        ax.set_yticklabels(FreqAxis[::-1])
        TStep = countFrames / 3
        TimeTicks = range(0, countFrames, TStep)
        TimeTicksLabels = ['%.2f' % (float(t * Step) / Fs) for t in TimeTicks]
        ax.set_xticks(TimeTicks)
        ax.set_xticklabels(TimeTicksLabels)
        ax.set_xlabel('time (secs)')
        imgplot.set_cmap('jet')
        plt.colorbar()
        plt.show()

    return (chromaGram, TimeAxis, FreqAxis)

def lpc(signal, order):
    """Compute the Linear Prediction Coefficients.

    Return the order + 1 LPC coefficients for the signal. c = lpc(x, k) will
    find the k+1 coefficients of a k order linear filter:

      xp[n] = -c[1] * x[n-2] - ... - c[k-1] * x[n-k-1]

    Such as the sum of the squared-error e[i] = xp[i] - x[i] is minimized.

    Parameters
    ----------
    signal: array_like
        input signal
    order : int
        LPC order (the output will have order + 1 items)"""

    order = int(order)

    if signal.ndim > 1:
        raise ValueError("Array of rank > 1 not supported yet")
    if order > signal.size:
        raise ValueError("Input signal must have a lenght >= lpc order")

    if order > 0:
        p = order + 1
        r = np.zeros(p, signal.dtype)
        # Number of non zero values in autocorrelation one needs for p LPC
        # coefficients
        nx = np.min([p, signal.size])
        x = np.correlate(signal, signal, 'full')
        r[:nx] = x[signal.size-1:signal.size+order]
        phi = np.dot(sp.linalg.inv(sp.linalg.toeplitz(r[:-1])), -r[1:])
        return np.concatenate(([1.], phi)), None, None
    else:
        return np.ones(1, dtype = signal.dtype), None, None
    
#def phormants(x, Fs, Order):
#    N = len(x)
#    w = numpy.hamming(N)
#
#    # Apply window and high pass filter.
#    x1 = x * w   
#    x1 = lfilter([1], [1., 0.63], x1)
#    
#    # Get LPC.    
#    ncoeff = 2 + Fs / 1000
#    A, e, k = lpc(x1, ncoeff)    
#    #A, e, k = lpc(x1, 8)
#
#    # Get roots.
#    rts = numpy.roots(A)
#    rts = [r for r in rts if numpy.imag(r) >= 0]
#
#    # Get angles.
#    angz = numpy.arctan2(numpy.imag(rts), numpy.real(rts))
#
#    # Get frequencies.    
#    frqs = sorted(angz * (Fs / (2 * math.pi)))
#    return frqs[0:Order]
    
def beatExtraction(stFeatures, winSize, PLOT=False):
    """
    This function extracts an estimate of the beat rate for a musical signal.
    ARGUMENTS:
     - stFeatures:     a numpy array (numOfFeatures x numOfShortTermWindows)
     - winSize:        window size in seconds
    RETURNS:
     - BPM:            estimates of beats per minute
     - Ratio:          a confidence measure
    """

    # Features that are related to the beat tracking task:
    toWatch = [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]

    maxBeatTime = int(round(2.0 / winSize))
    HistAll = numpy.zeros((maxBeatTime,))
    for ii, i in enumerate(toWatch):                                        # for each feature
        DifThres = 2.0 * (numpy.abs(stFeatures[i, 0:-1] - stFeatures[i, 1::])).mean()    # dif threshold (3 x Mean of Difs)
        if DifThres<=0:
            DifThres = 0.0000000000000001        
        [pos1, _] = utilities.peakdet(stFeatures[i, :], DifThres)           # detect local maxima
        posDifs = []                                                        # compute histograms of local maxima changes
        for j in range(len(pos1)-1):
            posDifs.append(pos1[j+1]-pos1[j])
        [HistTimes, HistEdges] = numpy.histogram(posDifs, numpy.arange(0.5, maxBeatTime + 1.5))
        HistCenters = (HistEdges[0:-1] + HistEdges[1::]) / 2.0
        HistTimes = HistTimes.astype(float) / stFeatures.shape[1]
        HistAll += HistTimes
        if PLOT:
            plt.subplot(9, 2, ii + 1)
            plt.plot(stFeatures[i, :], 'k')
            for k in pos1:
                plt.plot(k, stFeatures[i, k], 'k*')
            f1 = plt.gca()
            f1.axes.get_xaxis().set_ticks([])
            f1.axes.get_yaxis().set_ticks([])

    if PLOT:
        plt.show(block=False)
        plt.figure()

    # Get beat as the argmax of the agregated histogram:
    I = numpy.argmax(HistAll)
    BPMs = 60 / (HistCenters * winSize)
    BPM = BPMs[I]
    # ... and the beat ratio:
    Ratio = HistAll[I] / HistAll.sum()

    if PLOT:
        # filter out >500 beats from plotting:
        HistAll = HistAll[BPMs < 500]
        BPMs = BPMs[BPMs < 500]

        plt.plot(BPMs, HistAll, 'k')
        plt.xlabel('Beats per minute')
        plt.ylabel('Freq Count')
        plt.show(block=True)

    return BPM, Ratio


def stSpectogram(signal, Fs, Win, Step, PLOT=False):
    """
    Short-term FFT mag for spectogram estimation:
    Returns:
        a numpy array (nFFT x numOfShortTermWindows)
    ARGUMENTS:
        signal:      the input signal samples
        Fs:          the sampling freq (in Hz)
        Win:         the short-term window size (in samples)
        Step:        the short-term window step (in samples)
        PLOT:        flag, 1 if results are to be ploted
    RETURNS:
    """
    Win = int(Win)
    Step = int(Step)
    signal = numpy.double(signal)
    signal = signal / (2.0 ** 15)
    DC = signal.mean()
    MAX = (numpy.abs(signal)).max()
    signal = (signal - DC) / (MAX - DC)

    N = len(signal)        # total number of signals
    curPos = 0
    countFrames = 0
    nfft = int(Win / 2)
    specgram = numpy.array([], dtype=numpy.float64)

    while (curPos + Win - 1 < N):
        countFrames += 1
        x = signal[curPos:curPos+Win]
        curPos = curPos + Step
        X = abs(fft(signal))
        X = X[0:nfft]
        X = X / len(X)

        if countFrames == 1:
            specgram = X ** 2
        else:
            specgram = numpy.vstack((specgram, X))

    FreqAxis = [((f + 1) * Fs) / (2 * nfft) for f in range(specgram.shape[1])]
    TimeAxis = [(t * Step) / Fs for t in range(specgram.shape[0])]

    if (PLOT):
        fig, ax = plt.subplots()
        imgplot = plt.imshow(specgram.transpose()[::-1, :])
        Fstep = int(nfft / 5.0)
        FreqTicks = range(0, int(nfft) + Fstep, Fstep)
        FreqTicksLabels = [str(Fs / 2 - int((f * Fs) / (2 * nfft))) for f in FreqTicks]
        ax.set_yticks(FreqTicks)
        ax.set_yticklabels(FreqTicksLabels)
        TStep = countFrames/3
        TimeTicks = range(0, countFrames, TStep)
        TimeTicksLabels = ['%.2f' % (float(t * Step) / Fs) for t in TimeTicks]
        ax.set_xticks(TimeTicks)
        ax.set_xticklabels(TimeTicksLabels)
        ax.set_xlabel('time (secs)')
        ax.set_ylabel('freq (Hz)')
        imgplot.set_cmap('jet')
        plt.colorbar()
        plt.show()

    return (specgram, TimeAxis, FreqAxis)


""" Windowing and feature extraction """

def mtMDVPAnalysis(longSignal, Fs, Win, Step, ifVoice):
    '''
    ARGUMENTS
        signal:       the input signal samples
        Fs:           the sampling freq (in Hz)
        Win:          the short-term window size (in ms)
        Step:         the short-term window step (in ms)    
    '''
    Win = int(Win * (Fs / 1000))
    if(Win % 2 == 0):
        Win -= 1

    Step = int(Step * (Fs / 1000))

    # Signal normalization
    signal = numpy.double(longSignal)

    signal = signal / (2.0 ** 31)
    DC = signal.mean()
    MAX = (numpy.abs(signal)).max()
    signal = (signal - DC) / (MAX + 0.0000000001)
    
    #========filtering========
#    if(ifSound == 1):
#        signal = savgol_filter(signal, 25, 5)
#    else:
#        signal = butter_bandpass_filter(signal, 1, 12, Fs, 2)
        
#    w = np.hamming(Win) # fft icinde carpilir (Spektral Leagea'i (sizinti) azaltir)
    #=========================

    N = len(signal) # total number of samples
    curPos = 0
#    countFrames = 0
#    nFFT = Win / 2
#    nFFT = int(nFFT)
    
    windowAmount = int(np.floor((N - Win) / Step + 1)) 
    numStFeatures = 4
      
    stFeatures = np.zeros((numStFeatures,windowAmount))
    F0s = np.zeros((windowAmount))
    Amps = np.zeros((windowAmount))
    for i in range(windowAmount):
        x = signal[curPos:curPos+Win]                    # get current window
        curPos = curPos + Step                           # update window position

        stFeatures[0,i] = ifVoiceless(x, Fs)
        stFeatures[1,i] = noiseToHarmonicRatio(x, Fs)
        stFeatures[2,i] = voiceTurbulanceIndexScore(x, Fs)
        stFeatures[3,i] = SPI(x, Fs)
        F0s[i] = F0UsingAutocorrelation(x, Fs)
        Amps[i] = max(np.abs(x)) 
        
    T0s = 1 / F0s
    
    if(ifVoice == 1):
        numFeatures = 19    
        mtMDVPFeatures = np.zeros((numFeatures)) 
        
        mtMDVPFeatures[0] = sum(a==0 for a in stFeatures[0]) / windowAmount   #DUV
        mtMDVPFeatures[1] = np.mean(stFeatures[1]) # HNR
        mtMDVPFeatures[2] = np.mean(stFeatures[2]) #VTI
        mtMDVPFeatures[3] = np.mean(stFeatures[3]) #SPI
        mtMDVPFeatures[4] = pitchPeriod(F0s) #T0
        mtMDVPFeatures[5] = highestF0(F0s) #Fhi
        mtMDVPFeatures[6] = lowestF0(F0s) #Flo
        mtMDVPFeatures[7] = phonatoryF0Range(F0s) #PFR
        mtMDVPFeatures[8] = f0Variation(F0s) #STD
        mtMDVPFeatures[9] = amplitudeVariation(Amps) #vAM
        mtMDVPFeatures[10] = APQ(F0s) # APQ
        mtMDVPFeatures[11] = PPQ(T0s) # PPQ
        mtMDVPFeatures[12] = calculateJitterRatio(F0s) #Jitt
        mtMDVPFeatures[13] = calculateJitterFactor(F0s) #Jitta
        mtMDVPFeatures[14] = shimmerInDb(Amps) #Shdb
        mtMDVPFeatures[15] = shimmerRelative(Amps) #Shim
        mtMDVPFeatures[16] = calculateRelativeAveragePerturbation(T0s) #RAP 
        mtMDVPFeatures[17] = voiceTurbulanceIndexScore(signal, Fs) #VTI
        mtMDVPFeatures[18] = degreeOfVoiceBreak(signal, Fs, 0) #DVB
    else:
        numFeatures = 5    
        mtMDVPFeatures = np.zeros((numFeatures))
        mtMDVPFeatures[0] = np.mean(stFeatures[1])
        mtMDVPFeatures[1] = np.mean(stFeatures[2])
        mtMDVPFeatures[2] = amplitudeVariation(Amps)
        mtMDVPFeatures[3] = shimmerInDb(Amps)
        mtMDVPFeatures[4] = shimmerRelative(Amps)
    
    return mtMDVPFeatures[1:18], stFeatures, F0s, Amps
        
def dynamicMDVPAnalysis(stFeatures, newSignal, windowAmount, Fs, ifVoice, F0s, Amps):
    '''
    ARGUMENTS
        signal:       the input signal samples
        Fs:           the sampling freq (in Hz)
    '''

    # Signal normalization
    signal = numpy.double(newSignal)

    signal = signal / (2.0 ** 31)
    DC = signal.mean()
    MAX = (numpy.abs(signal)).max()
    signal = (signal - DC) / (MAX + 0.0000000001)
#    Win = len(newSignal)
    #========filtering========
#    w = np.hamming(Win) # fft icinde carpilir (Spektral Leagea'i (sizinti) azaltir)
    #=========================
#    nFFT = Win / 2
#    nFFT = int(nFFT)
#
#    X = abs(fft(signal * w))                                  # get fft magnitude
#    X = X[0:nFFT]                                    # normalize fft
#    X = X / len(X) # fft of stSignal        

    tempFeatures = np.zeros((4))
    tempFeatures[0] = ifVoiceless(signal, Fs)
    tempFeatures[1] = noiseToHarmonicRatio(signal, Fs)
    tempFeatures[2] = voiceTurbulanceIndexScore(signal, Fs)
    tempFeatures[3] = SPI(signal, Fs)
    stFeatures[:,0:-1] = stFeatures[:,1:]
    stFeatures[:,-1] = tempFeatures
#   F0s[i] = stHarmonicRatio(x, Fs)[1]
#    F0s = np.append(stFeatures[4], F0UsingAutocorrelation(signal, Fs))
#    Amps = np.append(stFeatures[5], max(np.abs(signal))) 
    F0s[0:-1] = F0s[1:]
    F0s[-1] = F0UsingAutocorrelation(signal, Fs)
    Amps[0:-1] = Amps[1:]
    Amps[-1] = max(np.abs(signal))
        
    T0s = 1 / F0s
    
    if(ifVoice == 1):
        numFeatures = 19    
        mtMDVPFeatures = np.zeros((numFeatures)) 
        
        mtMDVPFeatures[0] = sum(a==0 for a in stFeatures[0]) / windowAmount   #DUV
        mtMDVPFeatures[1] = np.mean(stFeatures[1]) # HNR
        mtMDVPFeatures[2] = np.mean(stFeatures[2]) #VTI
        mtMDVPFeatures[3] = np.mean(stFeatures[3]) #SPI
        mtMDVPFeatures[4] = pitchPeriod(F0s) #T0
        mtMDVPFeatures[5] = highestF0(F0s) #Fhi
        mtMDVPFeatures[6] = lowestF0(F0s) #Flo
        mtMDVPFeatures[7] = phonatoryF0Range(F0s) #PFR
        mtMDVPFeatures[8] = f0Variation(F0s) #STD
        mtMDVPFeatures[9] = amplitudeVariation(Amps) #vAM
        mtMDVPFeatures[10] = APQ(Amps) # APQ
        mtMDVPFeatures[11] = PPQ(F0s) # PPQ
        mtMDVPFeatures[12] = calculateJitterRatio(F0s) #Jitt
        mtMDVPFeatures[13] = calculateJitterFactor(F0s) #Jitta
        mtMDVPFeatures[14] = shimmerInDb(Amps) #Shdb
        mtMDVPFeatures[15] = shimmerRelative(Amps) #Shim
        mtMDVPFeatures[16] = calculateRelativeAveragePerturbation(F0s) #RAP 
        mtMDVPFeatures[17] = voiceTurbulanceIndexScore(signal, Fs) #VTI
        mtMDVPFeatures[18] = degreeOfVoiceBreak(signal, Fs, 0) #DVB
    else:
        numFeatures = 5    
        mtMDVPFeatures = np.zeros((numFeatures))
        mtMDVPFeatures[0] = np.mean(stFeatures[1])
        mtMDVPFeatures[1] = np.mean(stFeatures[2])
        mtMDVPFeatures[2] = amplitudeVariation(Amps)
        mtMDVPFeatures[3] = shimmerInDb(Amps)
        mtMDVPFeatures[4] = shimmerRelative(Amps)
    
    return mtMDVPFeatures[1:18], stFeatures, F0s, Amps

def stFeatureExtraction(signal, Fs, Win, Step):
    """
    This function implements the shor-term windowing process. For each short-term window a set of features is extracted.
    This results to a sequence of feature vectors, stored in a numpy matrix.

    ARGUMENTS
        signal:       the input signal samples
        Fs:           the sampling freq (in Hz)
        Win:          the short-term window size (in samples)
        Step:         the short-term window step (in samples)
    RETURNS
        stFeatures:   a numpy array (numOfFeatures x numOfShortTermWindows)
    """
    Win = int(Win)
    Step = int(Step)

    # Signal normalization
    signal = numpy.double(signal)

    signal = signal / (2.0 ** 31)
    DC = signal.mean()
    MAX = (numpy.abs(signal)).max()
    signal = (signal - DC) / (MAX + 0.0000000001)
    
    #========filtering========
#    if(ifSound == 1):
#        signal = savgol_filter(signal, 25, 5)
#    else:
#        signal = butter_bandpass_filter(signal, 1, 12, Fs, 2)
        
    w = np.hamming(Win) # fft icinde carpilir (Spektral Leagea'i (sizinti) azaltir)
    #=========================
    N = len(signal)                                # total number of samples
    curPos = 0
    countFrames = 0
    nFFT = Win / 2
    nFFT = int(nFFT)

    [fbank, freqs] = mfccInitFilterBanks(Fs, nFFT)                # compute the triangular filter banks used in the mfcc calculation
    nChroma, nFreqsPerChroma = stChromaFeaturesInit(nFFT, Fs)

    numOfTimeSpectralFeatures = 10
    numOfHarmonicFeatures = 0
    nceps = 13
    numOfChromaFeatures = 12
    nphormants = 4
    totalNumOfFeatures = numOfTimeSpectralFeatures + nceps + numOfHarmonicFeatures + numOfChromaFeatures + nphormants
#    totalNumOfFeatures = nceps
#    totalNumOfFeatures = numOfTimeSpectralFeatures +  nceps
#    totalNumOfFeatures = numOfTimeSpectralFeatures +  nceps + numOfChromaFeatures + nphormants
#    totalNumOfFeatures = 40
    windowAmount = int(np.floor((N-Win) / Step + 1))
    stFeatures = np.zeros((windowAmount,39))
    for i in range(windowAmount):                        # for each short-term window until the end of signal
#        countFrames += 1
        x = signal[curPos:curPos+Win]                    # get current window
        curPos = curPos + Step                           # update window position
        X = abs(fft(x * w))                                  # get fft magnitude
        X = X[0:nFFT]                                    # normalize fft
        X = X / len(X)
        if i == 0:
            Xprev = X.copy()                             # keep previous fft mag (used in spectral flux)
        curFV = numpy.zeros((totalNumOfFeatures))
        curFV[0] = stZCR(x)                              # zero crossing rate
        curFV[1] = stEnergy(x)                           # short-term energy
        curFV[2] = stEnergyEntropy(x)                    # short-term entropy of energy
        [curFV[3], curFV[4]] = stSpectralCentroidAndSpread(X, Fs)    # spectral centroid and spread
        curFV[5] = stSpectralEntropy(X)                  # spectral entropy
        curFV[6] = stSpectralFlux(X, Xprev)              # spectral flux
        curFV[7] = stSpectralRollOff(X, 0.90, Fs)        # spectral rolloff
        curFV[8] = stSpectralLowHighRatio(X, 4000, 10000, Fs)
        curFV[9] = stHarmonicRatio(x, Fs)[0]
        curFV[10:23] = stMFCC(X, fbank, nceps).copy()
        chromaNames, chromaF = stChromaFeatures(X, Fs, nChroma, nFreqsPerChroma)
        curFV[23:35] = chromaF
        curFV[35:39] = phormants(x, Fs)

        stFeatures[i,:] = curFV
        # delta features
        '''
        if countFrames>1:
            delta = curFV - prevFV
            curFVFinal = numpy.concatenate((curFV, delta))            
        else:
            curFVFinal = numpy.concatenate((curFV, curFV))
        prevFV = curFV
        stFeatures.append(curFVFinal)        
        '''
        # end of delta
        Xprev = X.copy()

#    stFeatures = numpy.column_stack(stFeatures)
    return stFeatures, X
    
def dynamicStFeatureExtraction(stFeatures, newSignal, Xprev, windowAmount, Fs, Win, Step):
    """
    This function implements the shor-term windowing process. For each short-term window a set of features is extracted.
    This results to a sequence of feature vectors, stored in a numpy matrix.

    ARGUMENTS
        signal:       the input signal samples
        Fs:           the sampling freq (in Hz)
        Win:          the short-term window size (in samples)
        Step:         the short-term window step (in samples)
    RETURNS
        stFeatures:   a numpy array (numOfFeatures x numOfShortTermWindows)
    """
    Win = len(newSignal)

    # Signal normalization
    signal = numpy.double(newSignal)

    signal = signal / (2.0 ** 31)
    DC = signal.mean()
    MAX = (numpy.abs(signal)).max()
    signal = (signal - DC) / (MAX + 0.0000000001)
    #========filtering========
    w = np.hamming(Win) # fft icinde carpilir (Spektral Leagea'i (sizinti) azaltir)
    #=========================
    nFFT = Win / 2
    nFFT = int(nFFT)

    [fbank, freqs] = mfccInitFilterBanks(Fs, nFFT)                # compute the triangular filter banks used in the mfcc calculation
    nChroma, nFreqsPerChroma = stChromaFeaturesInit(nFFT, Fs)

    numOfTimeSpectralFeatures = 10
    numOfHarmonicFeatures = 0
    nceps = 13
    numOfChromaFeatures = 12
    nphormants = 4
    totalNumOfFeatures = numOfTimeSpectralFeatures + nceps + numOfHarmonicFeatures + numOfChromaFeatures + nphormants

    X = abs(fft(signal * w))                                  # get fft magnitude
    X = X[0:nFFT]                                    # normalize fft
    X = X / len(X)

    curFV = numpy.zeros((totalNumOfFeatures))
    curFV[0] = stZCR(signal)                              # zero crossing rate
    curFV[1] = stEnergy(signal)                           # short-term energy
    curFV[2] = stEnergyEntropy(signal)                    # short-term entropy of energy
    [curFV[3], curFV[4]] = stSpectralCentroidAndSpread(X, Fs)    # spectral centroid and spread
    curFV[5] = stSpectralEntropy(X)                  # spectral entropy
    curFV[6] = stSpectralFlux(X, Xprev)              # spectral flux
    curFV[7] = stSpectralRollOff(X, 0.90, Fs)        # spectral rolloff
    curFV[8] = stSpectralLowHighRatio(X, 4000, 10000, Fs)
    curFV[9] = stHarmonicRatio(signal, Fs)[0]
    curFV[10:23] = stMFCC(X, fbank, nceps).copy()
    chromaNames, chromaF = stChromaFeatures(X, Fs, nChroma, nFreqsPerChroma)
    curFV[23:35] = chromaF
    curFV[35:39] = phormants(signal, Fs)

#    stFeatures = np.append(stFeatures[1:,:], curFV)
    stFeatures[0:-1,:] = stFeatures[1:,:]
    stFeatures[-1,:] = curFV
#    stFeatures.append(curFV)
#    stFeatures = stFeatures[]
#    stFeatures = numpy.column_stack(stFeatures)
    return stFeatures, X
    
def mtFeatureExtraction(signal, Fs, mtWin, mtStep, stWin, stStep):
    """
    Mid-term feature extraction
    """

    mtWinRatio = int(round(mtWin / stStep)) #80
    mtStepRatio = int(round(mtStep / stStep)) #40

    mtFeatures = []

    Win = Fs * stWin
    Step = int(Fs * stStep)
    stFeatures = stFeatureExtraction(signal, Fs, Win, Step)
    numOfFeatures = len(stFeatures) # 34
    numOfStatistics = 7

    mtFeatures = []
    #for i in range(numOfStatistics * numOfFeatures + 1):
    for i in range(numOfStatistics * numOfFeatures): #170
        mtFeatures.append([])

    for i in range(numOfFeatures):        # for each of the features
        curPos = 0
        N = len(stFeatures[i]) #159
        while (curPos < N):
            N1 = curPos
            N2 = curPos + mtWinRatio
            if N2 > N: # arrayi asma diye son kontrol
                N2 = N
            curStFeatures = stFeatures[i][N1:N2] # i,80

            mtFeatures[i].append(numpy.mean(curStFeatures))
            mtFeatures[i + numOfFeatures].append(numpy.std(curStFeatures))
            mtFeatures[i + 2*numOfFeatures].append(numpy.median(curStFeatures))
            mtFeatures[i + 3*numOfFeatures].append(numpy.max(curStFeatures))
            mtFeatures[i + 4*numOfFeatures].append(numpy.min(curStFeatures))
            mtFeatures[i + 5*numOfFeatures].append(stats.skew(curStFeatures, axis=0)) 
            mtFeatures[i + 6*numOfFeatures].append(stats.kurtosis(curStFeatures, axis=0))
            #mtFeatures[i+2*numOfFeatures].append(numpy.std(curStFeatures) / (numpy.mean(curStFeatures)+0.00000010))
            curPos += mtStepRatio

    return numpy.array(mtFeatures), stFeatures


# TODO
def stFeatureSpeed(signal, Fs, Win, Step):

    signal = numpy.double(signal)
    signal = signal / (2.0 ** 15)
    DC = signal.mean()
    MAX = (numpy.abs(signal)).max()
    signal = (signal - DC) / MAX
    # print (numpy.abs(signal)).max()

    N = len(signal)        # total number of signals
    curPos = 0
    countFrames = 0

    lowfreq = 133.33
    linsc = 200/3.
    logsc = 1.0711703
    nlinfil = 13
    nlogfil = 27
    nceps = 13
    nfil = nlinfil + nlogfil
    nfft = Win / 2
    if Fs < 8000:
        nlogfil = 5
        nfil = nlinfil + nlogfil
        nfft = Win / 2

    # compute filter banks for mfcc:
    [fbank, freqs] = mfccInitFilterBanks(Fs, nfft, lowfreq, linsc, logsc, nlinfil, nlogfil)

    numOfTimeSpectralFeatures = 8
    numOfHarmonicFeatures = 1
    totalNumOfFeatures = numOfTimeSpectralFeatures + nceps + numOfHarmonicFeatures
    #stFeatures = numpy.array([], dtype=numpy.float64)
    stFeatures = []

    while (curPos + Win - 1 < N):
        countFrames += 1
        x = signal[curPos:curPos + Win]
        curPos = curPos + Step
        X = abs(fft(x))
        X = X[0:nfft]
        X = X / len(X)
        Ex = 0.0
        El = 0.0
        X[0:4] = 0
#        M = numpy.round(0.016 * fs) - 1
#        R = numpy.correlate(frame, frame, mode='full')
        stFeatures.append(stHarmonic(x, Fs))
#        for i in range(len(X)):
            #if (i < (len(X) / 8)) and (i > (len(X)/40)):
            #    Ex += X[i]*X[i]
            #El += X[i]*X[i]
#        stFeatures.append(Ex / El)
#        stFeatures.append(numpy.argmax(X))
#        if curFV[numOfTimeSpectralFeatures+nceps+1]>0:
#            print curFV[numOfTimeSpectralFeatures+nceps], curFV[numOfTimeSpectralFeatures+nceps+1]
    return numpy.array(stFeatures)


""" Feature Extraction Wrappers

 - The first two feature extraction wrappers are used to extract long-term averaged
   audio features for a list of WAV files stored in a given category.
   It is important to note that, one single feature is extracted per WAV file (not the whole sequence of feature vectors)

 """


def dirWavFeatureExtraction(dirName, mtWin, mtStep, stWin, stStep, computeBEAT=False):
    """
    This function extracts the mid-term features of the WAVE files of a particular folder.

    The resulting feature vector is extracted by long-term averaging the mid-term features.
    Therefore ONE FEATURE VECTOR is extracted for each WAV file.

    ARGUMENTS:
        - dirName:        the path of the WAVE directory
        - mtWin, mtStep:    mid-term window and step (in seconds)
        - stWin, stStep:    short-term window and step (in seconds)
    """

    allMtFeatures = numpy.array([])
    processingTimes = []

    types = ('*.wav', '*.aif',  '*.aiff', '*.mp3','*.au')
    wavFilesList = []
    for files in types:
        wavFilesList.extend(glob.glob(os.path.join(dirName, files)))

    wavFilesList = sorted(wavFilesList)    
    wavFilesList2 = []
    for i, wavFile in enumerate(wavFilesList):        
        print ("Analyzing file {0:d} of {1:d}: {2:s}".format(i+1, len(wavFilesList), wavFile.encode('utf-8')))
        if os.stat(wavFile).st_size == 0:
            print ("EMPTY FILE -- SKIPPING")
            continue        
        [Fs, x] = audioBasicIO.readAudioFile(wavFile)            # read file    
        if isinstance(x, int):
            continue        

        t1 = time.clock()        
        x = audioBasicIO.stereo2mono(x)                          # convert stereo to mono                
        if x.shape[0]<float(Fs)/10:
            print ("AUDIO FILE TOO SMALL - SKIPPING")
            continue
        wavFilesList2.append(wavFile)
        if computeBEAT:                                          # mid-term feature extraction for current file
            [MidTermFeatures, stFeatures] = mtFeatureExtraction(x, Fs, round(mtWin * Fs), round(mtStep * Fs), round(Fs * stWin), round(Fs * stStep))
            [beat, beatConf] = beatExtraction(stFeatures, stStep)
        else:
            [MidTermFeatures, _] = mtFeatureExtraction(x, Fs, round(mtWin * Fs), round(mtStep * Fs), round(Fs * stWin), round(Fs * stStep))

        MidTermFeatures = numpy.transpose(MidTermFeatures)
        MidTermFeatures = MidTermFeatures.mean(axis=0)         # long term averaging of mid-term statistics
        if (not numpy.isnan(MidTermFeatures).any()) and (not numpy.isinf(MidTermFeatures).any()):            
            if computeBEAT:
                MidTermFeatures = numpy.append(MidTermFeatures, beat)
                MidTermFeatures = numpy.append(MidTermFeatures, beatConf)
            if len(allMtFeatures) == 0:                              # append feature vector
                allMtFeatures = MidTermFeatures
            else:
                allMtFeatures = numpy.vstack((allMtFeatures, MidTermFeatures))
            t2 = time.clock()
            duration = float(len(x)) / Fs
            processingTimes.append((t2 - t1) / duration)
    if len(processingTimes) > 0:
        print ("Feature extraction complexity ratio: {0:.1f} x realtime".format((1.0 / numpy.mean(numpy.array(processingTimes)))))
    return (allMtFeatures, wavFilesList2)


def dirsWavFeatureExtraction(dirNames, mtWin, mtStep, stWin, stStep, computeBEAT=False):
    '''
    Same as dirWavFeatureExtraction, but instead of a single dir it takes a list of paths as input and returns a list of feature matrices.
    EXAMPLE:
    [features, classNames] =
           a.dirsWavFeatureExtraction(['audioData/classSegmentsRec/noise','audioData/classSegmentsRec/speech',
                                       'audioData/classSegmentsRec/brush-teeth','audioData/classSegmentsRec/shower'], 1, 1, 0.02, 0.02);

    It can be used during the training process of a classification model ,
    in order to get feature matrices from various audio classes (each stored in a seperate path)
    '''

    # feature extraction for each class:
    features = []
    classNames = []
    fileNames = []
    for i, d in enumerate(dirNames):
        [f, fn] = dirWavFeatureExtraction(d, mtWin, mtStep, stWin, stStep, computeBEAT=computeBEAT)
        if f.shape[0] > 0:       # if at least one audio file has been found in the provided folder:
            features.append(f)
            fileNames.append(fn)
            if d[-1] == "/":
                classNames.append(d.split(os.sep)[-2])
            else:
                classNames.append(d.split(os.sep)[-1])
    return features, classNames, fileNames


def dirWavFeatureExtractionNoAveraging(dirName, mtWin, mtStep, stWin, stStep):
    """
    This function extracts the mid-term features of the WAVE files of a particular folder without averaging each file.

    ARGUMENTS:
        - dirName:          the path of the WAVE directory
        - mtWin, mtStep:    mid-term window and step (in seconds)
        - stWin, stStep:    short-term window and step (in seconds)
    RETURNS:
        - X:                A feature matrix
        - Y:                A matrix of file labels
        - filenames:
    """

    allMtFeatures = numpy.array([])
    signalIndices = numpy.array([])
    processingTimes = []

    types = ('*.wav', '*.aif',  '*.aiff')
    wavFilesList = []
    for files in types:
        wavFilesList.extend(glob.glob(os.path.join(dirName, files)))

    wavFilesList = sorted(wavFilesList)

    for i, wavFile in enumerate(wavFilesList):
        [Fs, x] = audioBasicIO.readAudioFile(wavFile)            # read file
        if isinstance(x, int):
            continue        
        
        x = audioBasicIO.stereo2mono(x)                          # convert stereo to mono
        [MidTermFeatures, _] = mtFeatureExtraction(x, Fs, round(mtWin * Fs), round(mtStep * Fs), round(Fs * stWin), round(Fs * stStep))  # mid-term feature

        MidTermFeatures = numpy.transpose(MidTermFeatures)
#        MidTermFeatures = MidTermFeatures.mean(axis=0)        # long term averaging of mid-term statistics
        if len(allMtFeatures) == 0:                # append feature vector
            allMtFeatures = MidTermFeatures
            signalIndices = numpy.zeros((MidTermFeatures.shape[0], ))
        else:
            allMtFeatures = numpy.vstack((allMtFeatures, MidTermFeatures))
            signalIndices = numpy.append(signalIndices, i * numpy.ones((MidTermFeatures.shape[0], )))

    return (allMtFeatures, signalIndices, wavFilesList)


# The following two feature extraction wrappers extract features for given audio files, however
# NO LONG-TERM AVERAGING is performed. Therefore, the output for each audio file is NOT A SINGLE FEATURE VECTOR
# but a whole feature matrix.
#
# Also, another difference between the following two wrappers and the previous is that they NO LONG-TERM AVERAGING IS PERFORMED.
# In other words, the WAV files in these functions are not used as uniform samples that need to be averaged but as sequences

def mtFeatureExtractionToFile(fileName, midTermSize, midTermStep, shortTermSize, shortTermStep, outPutFile,
                              storeStFeatures=False, storeToCSV=False, PLOT=False):
    """
    This function is used as a wrapper to:
    a) read the content of a WAV file
    b) perform mid-term feature extraction on that signal
    c) write the mid-term feature sequences to a numpy file
    """
    [Fs, x] = audioBasicIO.readAudioFile(fileName)            # read the wav file
    x = audioBasicIO.stereo2mono(x)                           # convert to MONO if required
    if storeStFeatures:
        [mtF, stF] = mtFeatureExtraction(x, Fs, round(Fs * midTermSize), round(Fs * midTermStep), round(Fs * shortTermSize), round(Fs * shortTermStep))
    else:
        [mtF, _] = mtFeatureExtraction(x, Fs, round(Fs*midTermSize), round(Fs * midTermStep), round(Fs * shortTermSize), round(Fs * shortTermStep))

    numpy.save(outPutFile, mtF)                              # save mt features to numpy file
    if PLOT:
        print ("Mid-term numpy file: " + outPutFile + ".npy saved")
    if storeToCSV:
        numpy.savetxt(outPutFile+".csv", mtF.T, delimiter=",")
        if PLOT:
            print ("Mid-term CSV file: " + outPutFile + ".csv saved")

    if storeStFeatures:
        numpy.save(outPutFile+"_st", stF)                    # save st features to numpy file
        if PLOT:
            print ("Short-term numpy file: " + outPutFile + "_st.npy saved")
        if storeToCSV:
            numpy.savetxt(outPutFile+"_st.csv", stF.T, delimiter=",")    # store st features to CSV file
            if PLOT:
                print ("Short-term CSV file: " + outPutFile + "_st.csv saved")


def mtFeatureExtractionToFileDir(dirName, midTermSize, midTermStep, shortTermSize, shortTermStep, storeStFeatures=False, storeToCSV=False, PLOT=False):
    types = (dirName + os.sep + '*.wav', )
    filesToProcess = []
    for files in types:
        filesToProcess.extend(glob.glob(files))
    for f in filesToProcess:
        outPath = f
        mtFeatureExtractionToFile(f, midTermSize, midTermStep, shortTermSize, shortTermStep, outPath, storeStFeatures, storeToCSV, PLOT)
