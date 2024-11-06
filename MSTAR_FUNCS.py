import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy import signal
import h5py
import sys
import PLL_FUNCS as PLL
from tabulate import tabulate
import PLOT_FUNCS as PLOT
from moku.instruments import SpectrumAnalyzer
import OPTICS_FUNCS as OF
import pandas as pd
import allantools as allan

def print_attrs(name, obj):
    print(f"\nObject Name: {name}")
    if isinstance(obj, h5py.Dataset):
        print(f"Dataset: {name}, shape: {obj.shape}, dtype: {obj.dtype}")
    elif isinstance(obj, h5py.Group):
        print(f"Group: {name}")
    for key, val in obj.attrs.items():
        print(f"    Attribute: {key} = {val}")


def debias_phase(phase, t):
    # Take gradient, remove frequency bias. Integrate from zero
    # fit linear line
    p = np.polyfit(t, phase, 1)
    # remove gradient
    phase = phase - p[0]*t
    print(f"Removed gradient: {p[0]}")
    return phase

def openSweepFile(file_name, print_vals = False):
    with h5py.File(file_name, 'r') as f:
        if(print_vals):
            print("File Attributes:")
            for key, val in f.attrs.items():
                print(f"    {key} = {val}")
            # Print all datasets and attributes
            print("\nDatasets and Attributes:")
            f.visititems(print_attrs)
            
    return tx_lo_frequencies

def returnSidebandPhase(fn, holdOverride = None, skipWLs = None, startTime = 50e-3):
    with h5py.File(fn, 'r') as saveFile:
        tx_lo_frequencies = saveFile['TX_LOs'][:]
        fs = saveFile.attrs['sample_rate']
        hold_time = saveFile.attrs['hold_time']
        
        PD1_LSB_ARRAY = []
        PD1_USB_ARRAY = []
        PD2_LSB_ARRAY = []
        PD2_USB_ARRAY = []
        
        frequencies = []
        
        for i, freq in enumerate(tx_lo_frequencies):
            if skipWLs is not None:
                if i in skipWLs:
                    continue
            dataGroup = saveFile[f'tx_lo_{freq}']
            frequencies.append(freq)
            PM0 = dataGroup['PM0'][:]
            PM1 = dataGroup['PM1'][:]
            PM2 = dataGroup['PM2'][:]
            PM3 = dataGroup['PM3'][:]
            PM4 = dataGroup['PM4'][:]
            PM5 = dataGroup['PM5'][:]
            if holdOverride is not None:
                hold_time = holdOverride
                Ns = int(hold_time*fs)
                PM0 = PM0[:Ns]
                PM1 = PM1[:Ns]
                PM2 = PM2[:Ns]
                PM3 = PM3[:Ns]
                PM4 = PM4[:Ns]
                PM5 = PM5[:Ns]
            Nstart = int(startTime*fs)
            PM0 = PM0[Nstart:]
            PM1 = PM1[Nstart:]
            PM2 = PM2[Nstart:]
            PM3 = PM3[Nstart:]
            PM4 = PM4[Nstart:]
            PM5 = PM5[Nstart:]
            PD1_USB = PM5
            PD1_LSB = PM4
            PD2_USB = PM1
            PD2_LSB = PM2
            PD1_LSB_ARRAY.append(PD1_LSB)
            PD1_USB_ARRAY.append(PD1_USB)
            PD2_LSB_ARRAY.append(PD2_LSB)
            PD2_USB_ARRAY.append(PD2_USB)
            
        
        return PD1_LSB_ARRAY, PD1_USB_ARRAY, PD2_LSB_ARRAY, PD2_USB_ARRAY, fs, frequencies
            # if i == freqIdx:
            #     return PD1_LSB, PD1_USB, PD2_LSB, PD2_USB, fs, freq

def SweepFileAnalysis(fn, zeroValue, filt = False, holdOverride = None, skipWLs = None, fc = 100, showPlots = True, neg = -1, startTime = 50e-3, timeStamp = False, carrier = False, resample = None):
    if(showPlots):
        fig, axs = plt.subplots()
        figPSD, axsPSD = PLOT.CreatePSDPlot()
    nperseg = 2**12
    zerothAlign = [zeroValue]
    distances = []
    carrierDistUSBs = []
    carrierDistLSBs = []
    wls = []
    averages = []
    sigmas = []
    LO_frequencies = []
    
    with h5py.File(fn, 'r') as saveFile:
        tx_lo_frequencies = saveFile['TX_LOs'][:]
        fs = saveFile.attrs['sample_rate']
        hold_time = saveFile.attrs['hold_time']
        
        if resample is not None:
            print("This is broken at the moment. Need to fix")
            fs_old = fs
            fs = fs/resample
            print(f"Resampled to {fs}Hz from {fs_old}Hz")
        
        if(timeStamp):
            time_stamp = saveFile.attrs['time_stamp']
        for i, freq in enumerate(tx_lo_frequencies):
            if skipWLs is not None:
                if i in skipWLs:
                    continue
            LO_frequencies.append(freq)
            # print(f"TX_LO: {freq}")
            dataGroup = saveFile[f'tx_lo_{freq}']
            PM0 = dataGroup['PM0'][:]
            PM1 = dataGroup['PM1'][:]
            PM2 = dataGroup['PM2'][:]
            PM3 = dataGroup['PM3'][:]
            PM4 = dataGroup['PM4'][:]
            PM5 = dataGroup['PM5'][:]
            
            if resample is not None:
                # resample is power of 2
                PM0 = signal.decimate(PM0, resample)
                PM1 = signal.decimate(PM1, resample)
                PM2 = signal.decimate(PM2, resample)
                PM3 = signal.decimate(PM3, resample)
                PM4 = signal.decimate(PM4, resample)
                PM5 = signal.decimate(PM5, resample)
                
            
            if holdOverride is not None:
                hold_time = holdOverride
                Ns = int(hold_time*fs)
                PM0 = PM0[:Ns]
                PM1 = PM1[:Ns]
                PM2 = PM2[:Ns]
                PM3 = PM3[:Ns]
                PM4 = PM4[:Ns]
                PM5 = PM5[:Ns]
            
            Nstart = int(startTime*fs)
            PM0 = PM0[Nstart:]
            PM1 = PM1[Nstart:]
            PM2 = PM2[Nstart:]
            PM3 = PM3[Nstart:]
            PM4 = PM4[Nstart:]
            PM5 = PM5[Nstart:]
            
            PD1_USB = PM5
            PD1_LSB = PM4
            PD2_USB = PM1
            PD2_LSB = PM2
            
            if showPlots:
                plt.figure()
                plt.plot(PM1, label = "PM1")
                plt.plot(PM2, label = "PM2")
                plt.plot(PM3, label = "PM3")
                plt.plot(PM4, label = "PM4")
                plt.plot(PM5, label = "PM5")
                plt.legend(loc = "upper right")
            
            if filt:
                b, a = signal.butter(4, fc, 'low', fs = fs)
                # Filter all PD signals
                PD1_USB = signal.filtfilt(b, a, PD1_USB)
                PD1_LSB = signal.filtfilt(b, a, PD1_LSB)
                PD2_USB = signal.filtfilt(b, a, PD2_USB)
                PD2_LSB = signal.filtfilt(b, a, PD2_LSB)
            
            PD1 = PM0
            PD2 = PM3
            t = np.arange(0, len(PD1_USB)/fs, 1/fs) + hold_time*i
            FM = dataGroup.attrs['FM']
            FL = dataGroup.attrs['FL']
            
            sum_diif, dist, wl = synthetic_dist(PD1_USB, PD1_LSB, PD2_USB, PD2_LSB, FL, FM, neg = neg)
            if i == 0:
                n = matchSyntheticDist(np.array(zerothAlign), dist, 0.5*wl)
                sum_diif, dist, wl = synthetic_dist(PD1_USB, PD1_LSB, PD2_USB, PD2_LSB, FL, FM, n = n*-neg, neg = neg)
            else:
                n = matchSyntheticDist(distances[-1], dist, 0.5*wl)
                sum_diif, dist, wl = synthetic_dist(PD1_USB, PD1_LSB, PD2_USB, PD2_LSB, FL, FM, n = n*-neg, neg = neg)
            
            if(carrier):
                lasWL = 1542e-9
                carrierPhase, carrierDist = carrier_dist(PD1_USB, PD2_USB, lasWL)
                carrierPhase, carrierDist2 = carrier_dist(PD1_LSB, PD2_LSB, lasWL)
                carrierDistLSBs.append(carrierDist2)
                carrierDistUSBs.append(carrierDist)
            
            distances.append(dist)
            wls.append(wl) 
            averages.append(np.mean(dist))
            sigmas.append(np.std(dist))
            
            if i == 0:
            
                try:
                    PSDCH1 = dataGroup[f'MOKU_PSD_{i}_CH1']
                    PSDCH2 = dataGroup[f'MOKU_PSD_{i}_CH2']
                    # print all attributes
                    for key, val in PSDCH1.attrs.items():
                        print(f"    {key} = {val}")
                    for key, val in PSDCH2.attrs.items():
                        print(f"    {key} = {val}")
                except:
                    print("No MOKU PSD data")
            
            
            if(showPlots):
                axs.plot(t, dist, label = f"TX_LO: {freq}, Wavelength: {wl*1000:.2f}mm")
                f, Pxx_den = signal.welch(sum_diif, fs, nperseg=nperseg)
                axsPSD.plot(f, Pxx_den, label = f"TX_LO: {freq}, Wavelength: {wl*1000:.2f}mm")
                # check if moku data exists
                try:
                    figPS, axsPS = PLOT.CreatePSDPlot()
                    freqData = dataGroup[f'MOKU_PSD_{i}_freq'][:]
                    PSDCH1 = dataGroup[f'MOKU_PSD_{i}_CH1'][:]
                    PSDCH2 = dataGroup[f'MOKU_PSD_{i}_CH2'][:]
                    set = dataGroup[f'MOKU_PSD_{i}_freq']
                    RBW = set.attrs['MOKU_RBW']
                    axsPS.plot(freqData, PSDCH1, label = 'CH1')
                    axsPS.plot(freqData, PSDCH2, label = 'CH2')
                    figPS, axsPS = PLOT.formatPSDPlotdBm(figPS, axsPS, title = f"TX_LO: {freq}, Wavelength: {wl*1000:.2f}mm", density = False, PSD_View = True, RBW = RBW)
                except:
                    print("No MOKU PSD data")
    if showPlots:
        axs.legend()
        figPSD, axsPSD = PLOT.formatPSDPlot(figPSD, axsPSD, linear= False)
    if(timeStamp):
        return distances, averages, sigmas, LO_frequencies, wls, fs, time_stamp
    
    
    
    if(carrier):
        return distances, averages, sigmas, LO_frequencies, wls, fs, carrierPhase, carrierDistUSBs, carrierDistLSBs
    return distances, averages, sigmas, LO_frequencies, wls, fs

def openFile(file_name, print_vals = False, ts = 0.01, debias = False, filter = False,fc = 100, unit = 'rad'):
    if(print_vals):
        with h5py.File(file_name, 'r') as f:
            print("File Attributes:")
            for key, val in f.attrs.items():
                print(f"    {key} = {val}")
            
            # Print all datasets and attributes
            print("\nDatasets and Attributes:")
            f.visititems(print_attrs)
    if(unit == 'rad'):
        scale = 2*np.pi
    else:
        scale = 1
    with h5py.File(file_name, 'r') as f:
        # Read channel 0 dataset
        PM0 = f['PM0'][:] * scale
        PM1 = f['PM1'][:] * scale
        PM2 = f['PM2'][:] * scale
        PM3 = f['PM3'][:] * scale
        PM4 = f['PM4'][:] * scale
        PM5 = f['PM5'][:] * scale
        # Read channel 1 dataset
        fs = f.attrs['sample_rate']
        fL = f.attrs['FL']
        fM = f.attrs['FM']

    plt.figure()
    plt.plot(PM1, label = "PM1")
    plt.plot(PM2, label = "PM2")
    plt.plot(PM3, label = "PM3")
    plt.plot(PM4, label = "PM4")
    plt.plot(PM5, label = "PM5")
    plt.legend(loc = "upper right")
    #plt.title(f"Raw Phase Data fL: {fL}, fM: {fM}")
    plt.ylabel("Phase [cyc]")
    plt.xlabel("Sample")

    wavelength = 299792458/(fL+fM)
    print(f"Wavelength: {wavelength}")

    fig, axs = PLOT.CreatePSDPlot()
    nperseg = 1024*64
    f1, Pxx1 = signal.welch(PM1, fs, nperseg = nperseg)
    f2, Pxx2 = signal.welch(PM2, fs, nperseg = nperseg)
    f4, Pxx4 = signal.welch(PM4, fs, nperseg = nperseg)
    f5, Pxx5 = signal.welch(PM5, fs, nperseg = nperseg)
    axs.plot(f1, Pxx1, label = "PM1")
    axs.plot(f2, Pxx2, label = "PM2")
    axs.plot(f4, Pxx4, label = "PM4")
    axs.plot(f5, Pxx5, label = "PM5")
    fig, axs = PLOT.formatPSDPlot(fig, axs, title = f'Raw Phase Noise - Wavelength {wavelength}', unit = unit, phase = True, dB = False, density = True, linear = False)

    PD1_USB = PM5
    PD1_LSB = PM4
    PD2_USB = PM1
    PD2_LSB = PM2
    PD1 = PM0
    PD2 = PM3
    settle = 0.1
    Nsettle = int(settle*fs)
    if(ts is not None):
        Ns = int(ts*fs)
        PD1_USB = PD1_USB[Nsettle:Nsettle+Ns]
        PD1_LSB = PD1_LSB[Nsettle:Nsettle+Ns]
        PD2_USB = PD2_USB[Nsettle:Nsettle+Ns]
        PD2_LSB = PD2_LSB[Nsettle:Nsettle+Ns]
        PD1 = PD1[Nsettle:Nsettle+Ns]
        PD2 = PD2[Nsettle:Nsettle+Ns]
    t = np.arange(len(PD1_USB))/fs
    if(debias):
        PD1_USB = debias_phase(PD1_USB, t)
        PD1_LSB = debias_phase(PD1_LSB, t)
        PD2_USB = debias_phase(PD2_USB, t)
        PD2_LSB = debias_phase(PD2_LSB, t)
        PD1 = debias_phase(PD1, t)
        PD2 = debias_phase(PD2, t)
    if(filter):
        b, a = signal.butter(3, fc, 'low', fs = fs)
        # Filter all PD signals
        PD1_USB = signal.filtfilt(b, a, PD1_USB)
        PD1_LSB = signal.filtfilt(b, a, PD1_LSB)
        PD2_USB = signal.filtfilt(b, a, PD2_USB)
        PD2_LSB = signal.filtfilt(b, a, PD2_LSB)
        PD1 = signal.filtfilt(b, a, PD1)
        PD2 = signal.filtfilt(b, a, PD2)

    return PD1, PD1_LSB, PD1_USB, PD2, PD2_LSB, PD2_USB, fs, fL, fM

def synthetic_dist(PD1_USB, PD1_LSB, PD2_USB, PD2_LSB, FL, FM,  n = None, neg = -1, sidebandMult = 1):
    ''' Calculates synthetic wavelength distance. Adds number of wavelengths until it is within bound. '''
    c = 299792458
    #c = c/n_ref_index

    USB_SUM = PD1_USB + PD2_USB
    LSB_SUM = PD1_LSB + PD2_LSB
    SUM_DIFF = USB_SUM - LSB_SUM

    USB1_LSB1 = PD1_USB - PD2_LSB
    USB2_LSB1 = PD1_LSB - PD2_USB
    SUM_DIFF2 = USB1_LSB1 - USB2_LSB1
    if n is not None:
        SUM_DIFF += n
        SUM_DIFF2 += n
    WAVELENGTH = c/(FL+FM)/sidebandMult
    DISTANCE = neg*0.5*SUM_DIFF2*WAVELENGTH

    return SUM_DIFF, DISTANCE, WAVELENGTH

def carrier_dist(SB1, SB2, wavelength):
    ''' Calculates carrier distance from sidebands '''
    c = 299792458
    n_ref_index = 1
    c = c/n_ref_index
    PHASE = SB1 + SB2
    DISTANCE = PHASE*wavelength/2
    return PHASE, DISTANCE

def find_synthetic_dist(PD1_USB, PD1_LSB, PD2_USB, PD2_LSB, bound_high, bound_low, FL, FM, fs,  debias = False, filter = True, n = None):
    ''' Calculates synthetic wavelength distance. Adds number of wavelengths until it is within bound. '''
    n_ref_index = 1.4606
    c = 299792458
    c = c/n_ref_index
    t = np.arange(len(PD1_USB))/fs
    if(debias):
        PD1_USB = debias_phase(PD1_USB, t)
        PD1_LSB = debias_phase(PD1_LSB, t)
        PD2_USB = debias_phase(PD2_USB, t)
        PD2_LSB = debias_phase(PD2_LSB, t)
    if(filter):
        b, a = signal.butter(3, 1000, 'low', fs = fs)
        # Filter all PD signals
        PD1_USB = signal.filtfilt(b, a, PD1_USB)
        PD1_LSB = signal.filtfilt(b, a, PD1_LSB)
        PD2_USB = signal.filtfilt(b, a, PD2_USB)
        PD2_LSB = signal.filtfilt(b, a, PD2_LSB)
    
    # USB_SUM = PD1_USB + PD2_USB
    # LSB_SUM = PD1_LSB + PD2_LSB
    # SUM_DIFF = USB_SUM - LSB_SUM
    USB_LSB1 = PD1_USB - PD2_LSB
    USB2_LSB1 = PD1_LSB - PD2_USB
    SUM_DIFF = USB_LSB1 - USB2_LSB1
    INTEGER = int(np.floor(SUM_DIFF[0]))
    SUM_DIFF = SUM_DIFF - INTEGER

    SYNTHETIC_WAVELENGTH = c/(FL+FM)
    # print(f"Synthetic Wavelength: {SYNTHETIC_WAVELENGTH}")
    # print(f"F_L: {FL}, F_M: {FM}")
    SYNTHETIC_DISTANCE = -0.5*SUM_DIFF*SYNTHETIC_WAVELENGTH
    # Find number of wavelengths to add

    max = np.max(SYNTHETIC_DISTANCE)
    min = np.min(SYNTHETIC_DISTANCE)
    # get initial integer guess based on bound_low
    if(n == None):
        n_guess = 2 * int((bound_low - min)/SYNTHETIC_WAVELENGTH)
        print(f"Initial guess: {n_guess}")
        n_test = np.arange(n_guess, 4+n_guess, 1) # Half wavelengths 
        for n in n_test:
            SUM_DIFF_NEW = SUM_DIFF - n
            SYNTHETIC_DISTANCE_NEW = -0.5*SUM_DIFF_NEW*SYNTHETIC_WAVELENGTH
            max = np.max(SYNTHETIC_DISTANCE_NEW)
            min = np.min(SYNTHETIC_DISTANCE_NEW)
            print(f"Max: {max}, Min: {min}")
            if( min > bound_low or max > bound_low):
                print(f"Adding {n} wavelengths to SYNTHETIC_DISTANCE")
                # stop for loop
                break
    else:
        SYNTHETIC_DISTANCE_NEW = SYNTHETIC_DISTANCE + n*SYNTHETIC_WAVELENGTH
        print(f"Adding {n} wavelengths to SYNTHETIC_DISTANCE")
    max = np.max(SYNTHETIC_DISTANCE_NEW)
    min = np.min(SYNTHETIC_DISTANCE_NEW)
    max = np.max(SYNTHETIC_DISTANCE_NEW)
    min = np.min(SYNTHETIC_DISTANCE_NEW)
    max_min = max-min
    print(f"Max-Min uncertainty is {max_min}, or {max_min/SYNTHETIC_WAVELENGTH} wavelengths")
    print(f"Max-Min uncertainty is {max_min}, or {max_min/SYNTHETIC_WAVELENGTH} wavelengths")
    print(f"Standard deviation of distance: {np.std(SYNTHETIC_DISTANCE_NEW)}, and mean: {np.mean(SYNTHETIC_DISTANCE_NEW)}")
    return SYNTHETIC_DISTANCE_NEW, max, min


def matchSyntheticDist(previousDist, newDist, newWaveLen):
    ''' Match start of new synthetic distance to old synthetic distance by adding half wavelengths. '''
    endPrevious = np.average(previousDist)#previousDist[-1]
    startNew = np.average(newDist)
    diff = startNew - endPrevious
    # print(f"Diff: {diff}")  
    n = diff/newWaveLen
    # Round to nearest 0.5
    n = round(n)#round(n*2)/2
    # print(f"Adding {n} wavelengths to new synthetic distance")
    return n

def calcDiffSection(PD1_USB, PD1_LSB, PD2_USB, PD2_LSB, t, ts, te, fL, fM, n = None, neg = 1):
    indexes = np.where((t >= ts) & (t <= te))
    PD1_USB = PD1_USB[indexes]
    PD1_LSB = PD1_LSB[indexes]
    PD2_USB = PD2_USB[indexes]
    PD2_LSB = PD2_LSB[indexes]
    t = t[indexes]
    USB_SUM = PD1_USB + PD2_USB
    LSB_SUM = PD1_LSB + PD2_LSB
    DIFF = USB_SUM - LSB_SUM
    if n is not None:
        DIFF += n
    wavelength = 299792458/(fM + fL)
    DIST = neg*0.5*DIFF*wavelength
    return wavelength, DIST, DIFF, t

def findSteps(t, data, numSteps, threshold = 800, holdTime = 1, offset = 0.05, debug = False):
    ''' Find the times at which steps occur in the data. Correlate with step function, or peak find gradient. '''
    
    dataGradient = np.gradient(data, t)
    dataGradient = np.abs(dataGradient)
    idxs = np.where(dataGradient > threshold)
    current_t = 0
    old_t = 0
    popIdxs = []
    print(f"Times of steps Before Trim: {t[idxs]}")
    for i, idx in enumerate(idxs[0]):
        current_t = t[idx]
        if(current_t - old_t < holdTime):
            popIdxs.append(i)
        old_t = current_t
    idxs = np.delete(idxs, popIdxs)
        
    print(f"Times of steps: {t[idxs]}")
    tRanges = []
    for idx in idxs:
        startTime = t[idx] - holdTime
        endTime = t[idx] - offset
        
        tRanges.append((startTime, endTime))
    endTimeLast = tRanges[-1][1]
    tRanges.append((endTimeLast+2*offset, t[-1]))
    
    # Plot tRanfes with data to check. Dashed horizontal lines with vertical bars at end
    if debug:
        plt.figure()
        plt.plot(t, dataGradient)
        
        plt.figure()
        plt.plot(t, data)
        for tRange in tRanges:
            plt.axvline(tRange[0], color = 'r', linestyle = '--')
            plt.axvline(tRange[1], color = 'r', linestyle = '--')
            plt.axvline(tRange[1], color = 'r', linestyle = '--')
            plt.axvspan(tRange[0], tRange[1], alpha=0.5, color='red')
        #plt.show()
    return tRanges
def plutoTXGain(LO, amp = "daisy_chain"):
    ''' Returns optimum gain of TX based on LO frequency. The gain of the TX is not flat over frequency. Depends on amplifiers. This is empirical. Looking at spectrum analyser. '''
    # 70MHz to 500MHz range
    if(amp == "ZX60-14012L-S+"):
        if( LO <= 300e6):
            return 0   
        # 500 MHz to 1 GHz range
        elif(LO <= 1.5e9):
            return -3
        else:
            return 0
    
    if(amp == "daisy_chain"):
        # Two amplifiers in series
        
        freqs = [70e6, 250e6, 500e6, 1e9, 1.5e9, 2e9, 2.5e9, 3e9, 3.5e9, 4e9, 4.5e9, 5e9, 5.5e9, 6e9]
        amps = [0,     -10,   -14,  -14, -13,   -12, -10,   -8,  -7,    -6,  -3, 0, 0, 0]
        
        # find nearest two frequencies to tx_lo
        if(LO in freqs):
            idx = np.where(np.array(freqs) == LO)[0][0]
            return amps[idx]
        else:
            idx2 = np.where(np.array(freqs) > LO)[0][0]
            if(idx2 == 0):
                return amps[0]
            idx1 = idx2 - 1
            
            f1 = freqs[idx1]
            f2 = freqs[idx2]
            a1 = amps[idx1]
            a2 = amps[idx2]
            # linear interpolation
            gain = a1 + (a2 - a1)/(f2 - f1)*(LO - f1)
            return gain
            
def plutoTXSamples(FM, FL, fs, scale = 2**14, N = 128, modPhase = False, stdDev1 = 0.1, stdDev2 = 0.1):
    ''' Generate frequency offset for MSTAR modulation. Can also add things like phase modulation or noise in here eventually. '''
    
    # calculate period of modulation
    T_FM = 1/FM
    # calculate number of samples per period
    N = int(T_FM*fs) * N
    
    
    ts = N/FM
    Ns = int(ts * fs)
    print(f"Number of samples: {Ns}")
    t = np.arange(Ns)/fs
    
    scale = 2**14
    samples = np.exp(2.0j*np.pi*FM*t) # Simulate a sinusoid of 100 kHz, so it should show up at 915.1 MHz at the receiver
    samples *= scale # The PlutoSDR expects samples to be between -2^14 and +2^14, not -1 and +1 like some SDRs
    samples2 = np.exp(2.0j*np.pi*FL*t) # Simulate a sinusoid of 100 kHz, so it should show up at 915.1 MHz at the receiver
    samples2 *= scale # The PlutoSDR expects samples to be between -2^14 and +2^14, not -1 and +1 like some SDRs
    
    # generate white noise
    # noise1 = np.random.normal(0, stdDev1, Ns)
    # noise2 = np.random.normal(0, stdDev2, Ns)
    # samples *= np.exp(1j*noise1)
    # samples2 *= np.exp(1j*noise2)
    
    return samples, samples2

def getMOKUSpec(IP, AOM_POW_dBm, connect = False, show = True, hold = False, freqCenter = 105e6, freqSpan = 25e6, externalClk = True):
    if(connect):
        SpecAmp = SpectrumAnalyzer(IP, force_connect=True)
        SpecAmp.set_external_clock(enable=externalClk) # Use external clock
        AOM1_POW = AOM_POW_dBm # dBm
        AOM2_POW = AOM_POW_dBm # dBm
        AOM1_VOLTS, _, _ = OF.dBm2Volts(AOM1_POW)
        AOM2_VOLTS, _, _ = OF.dBm2Volts(AOM2_POW)
        print(f"AOM1 Voltage: {AOM1_VOLTS}, AOM2 Voltage: {AOM2_VOLTS}")

        SpecAmp.sa_output(1, AOM1_VOLTS, 80e6)
        SpecAmp.sa_output(2, AOM2_VOLTS, 75e6)

        # freq_center = 115e6
        # freq_span = 15e6
        freq_up = freqCenter + freqSpan/2
        freq_low = freqCenter - freqSpan/2 

        SpecAmp.set_span(frequency1=freq_low, frequency2=freq_up)
        SpecAmp.set_frontend(1, impedance='50Ohm', coupling='DC', range='1Vpp')
        SpecAmp.set_frontend(2, impedance='50Ohm', coupling='DC', range='1Vpp')
        SpecAmp.set_rbw('Minimum')  # Auto-mode

        SpecData = SpecAmp.get_data()
        freqData = SpecData['frequency']
        ch1Data = SpecData['ch1']
        ch2Data = SpecData['ch2']
        RBW = SpecAmp.get_rbw()['value']
        
        if(show):
            fig, axs = PLOT.CreatePSDPlot()
            axs.plot(freqData, ch1Data, label = "Channel 1")
            axs.plot(freqData, ch2Data, label = "Channel 2")
            print(f"RBW: {RBW}")
            fig, axs = PLOT.formatPSDPlotdBm(fig, axs, density=False, PSD_View=True, RBW=RBW)
        if(hold):
            plt.show()
        return freqData, ch1Data, ch2Data, RBW
    else:
        return None, None, None, None