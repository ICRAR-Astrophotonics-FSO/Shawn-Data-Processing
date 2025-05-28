import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import PLL_FUNCS as PLL
from tabulate import tabulate
import PLOT_FUNCS as PLOT
import OPTICS_FUNCS as OF
import allantools as allan
from numba import jit
import MSTAR_FUNCS as MF
import os
import h5py
from scipy.optimize import curve_fit

def linear(x, m, c):
    return m*x + c

def readData(fn, holdOverride = None, skipWLs = None, startTime = 50e-3):
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

def LinkDIstance(PD1_USB, PD1_LSB, PD2_USB, PD2_LSB, EOMfrequency, N=0):
    SYNTHETIC_PHASE = PD1_USB + PD2_USB - PD1_LSB - PD2_LSB # Phase always measured in cycles
    EOM_SUM_FREQUENCY = 2 * EOMfrequency # Modulation is always performed such that omega_1+omega_2 = 2*omega
    c = 299792458
    SYNTETIC_WAVELENGTH = c / 2 / EOM_SUM_FREQUENCY
    LINK_ESTIMATE = SYNTETIC_WAVELENGTH * (SYNTHETIC_PHASE + N)
    return LINK_ESTIMATE, SYNTHETIC_PHASE

def guessN(PHASE, wl, guess):
    N = guess/wl - np.average(PHASE)
    # print(f" Guessing N to be {N}")
    # print(f'N = {N}')
    return np.round(N)

def estimateNextN(PHASE1, PHASE2, wl1, wl2, N1 = 0):
    N = (wl1/wl2)*(np.average(PHASE1) + N1) - np.average(PHASE2)
    # If our measurement time is small enough we know that using an average is okay
    fracN = N
    # print(f'N = {N}')
    # round N to nearest integer
    N = np.round(N)
    # print(f'rounded N = {N}')
    fracN = fracN - N # See what the fractional part of N is. In the ideal world it would be zero right? But useful to compare it to powers of 2. Maybe there's a quarter wavelength offset is something silly.
    # print(f'fracN = {fracN}')
    return N

def calculateLinkEstimate(PD1_USB_ARRAY, PD1_LSB_ARRAY, PD2_USB_ARRAY, PD2_LSB_ARRAY, frequencies, guess = None):
    LINK_ESTIMATES = []
    SYNTHETIC_PHASES = []
    c = 299792458
    Ns = []
    wls = c / 2 / ( 2*np.array(frequencies))
    for i, freq in enumerate(frequencies):
        if i == 0:
            LINK_ESTIMATE, SYNTHETIC_PHASE = LinkDIstance(PD1_USB_ARRAY[i], PD1_LSB_ARRAY[i], PD2_USB_ARRAY[i], PD2_LSB_ARRAY[i], freq, N = 0)
            N1 = guessN(SYNTHETIC_PHASE, wls[0], guess)
            Ns.append(N1)
            LINK_ESTIMATE, SYNTHETIC_PHASE = LinkDIstance(PD1_USB_ARRAY[i], PD1_LSB_ARRAY[i], PD2_USB_ARRAY[i], PD2_LSB_ARRAY[i], freq, N = N1)
        else:
            LINK_ESTIMATE, SYNTHETIC_PHASE = LinkDIstance(PD1_USB_ARRAY[i], PD1_LSB_ARRAY[i], PD2_USB_ARRAY[i], PD2_LSB_ARRAY[i], freq, N = 0)
            nextN = estimateNextN(SYNTHETIC_PHASES[-1], SYNTHETIC_PHASE, wls[i-1], wls[i], N1 = Ns[-1])
            Ns.append(nextN)
            LINK_ESTIMATE, SYNTHETIC_PHASE = LinkDIstance(PD1_USB_ARRAY[i], PD1_LSB_ARRAY[i], PD2_USB_ARRAY[i], PD2_LSB_ARRAY[i], freq, N = nextN)
        LINK_ESTIMATES.append(LINK_ESTIMATE)
        SYNTHETIC_PHASES.append(SYNTHETIC_PHASE)
    return LINK_ESTIMATES, SYNTHETIC_PHASES, wls

basefn = "/home/shawn/Documents/PhD-Vault/1.2.Projects/MS2/MSTAR-V2/System-Analysis-V2/MSTAR-V2-Measurements/SWEEP_TESTS_PAPER/fiber"
def readFolder(modefn, DIST, POWER, D1_cal=None, D2_cal=None, plot=True, basefn = basefn, num = 100, guess = 8, rejection = 3 ):
    wl1_averages = []
    wl2_averages = []
    for i in range(1, num):
        fn = f"{basefn}/{modefn}/mirror_RETURN_POWER_{POWER}_{DIST}um/MSTAR_SWEEP_TEST_{DIST}um_pluto2_{i}.hdf5"
        PD1_LSB_ARRAY, PD1_USB_ARRAY, PD2_LSB_ARRAY, PD2_USB_ARRAY, fs, frequencies = readData(fn)
        LINK_ESTIMATES, SYNTHETIC_PHASES, wls = calculateLinkEstimate(PD1_USB_ARRAY, PD1_LSB_ARRAY, PD2_USB_ARRAY, PD2_LSB_ARRAY, frequencies, guess=guess)
        wl1_averages.append(np.average(LINK_ESTIMATES[0]))
        wl2_averages.append(np.average(LINK_ESTIMATES[1]))
    # wl1_averages = np.array(wl1_averages)
    # wl2_averages = np.array(wl2_averages)
    wl1_averages, _ = reject_average_outliers(np.array(wl1_averages), m = rejection)
    wl2_averages, _ = reject_average_outliers(np.array(wl2_averages), m = rejection)  
    if (D1_cal is not None) and (D2_cal is not None):
        print("Using Calibration")
        wl1_averages = wl1_averages - D1_cal
        wl2_averages = wl2_averages - D2_cal
    else:
        print("Calibrating")
        D1_mean = np.average(wl1_averages)
        D2_mean = np.average(wl2_averages)
        D1_cal = D1_mean
        D2_cal = D2_mean
        wl1_averages = wl1_averages - D1_cal
        wl2_averages = wl2_averages - D2_cal
    
    D2_mean = np.average(wl2_averages)
    D2_std = np.std(wl2_averages)
    print(f'D2 Mean = {D2_mean*1e6:.1f}um, D2 STD = {D2_std*1e6:.1f}um')

    D1_mean = np.average(wl1_averages)
    D1_std = np.std(wl1_averages)
    print(f'D1 Mean = {D1_mean*1e6:.1f}um, D1 STD = {D1_std*1e6:.1f}um')
    if(plot):
        title = f"DIST {DIST} POWER {POWER}"
        plt.figure()
        plt.hist(wl1_averages*1e6, bins = 5, label = f'$\lambda=${wls[0]*1e3:.1f}mm')
        plt.hist(wl2_averages*1e6, bins = 5, label = f'$\lambda=${wls[1]*1e3:.1f}mm')
        plt.xlabel('Link Estimate [um]')
        plt.ylabel('Frequency')
        plt.legend(loc = 'upper right')
        plt.title(title)
        
        
    return wl1_averages, wl2_averages, wls, D1_cal, D2_cal

def readFolderFit(modefn, DIST, POWER, D1_cal=None, D2_cal=None, plot=True, basefn = basefn, num = 100, guess = 8, rejection = 3 ):
    PD1_LSB_fits = np.array([])
    PD2_LSB_fits = np.array([])
    PD1_USB_fits = np.array([])
    PD2_USB_fits = np.array([])

    for i in range(1, num):
        fn = f"{basefn}/{modefn}/mirror_RETURN_POWER_{POWER}_{DIST}um/MSTAR_SWEEP_TEST_{DIST}um_pluto2_{i}.hdf5"
        PD1_LSB_ARRAY, PD1_USB_ARRAY, PD2_LSB_ARRAY, PD2_USB_ARRAY, fs, frequencies = readData(fn)
        LINK_ESTIMATES, SYNTHETIC_PHASES, wls = calculateLinkEstimate(PD1_USB_ARRAY, PD1_LSB_ARRAY, PD2_USB_ARRAY, PD2_LSB_ARRAY, frequencies, guess=guess)

        # fit each array to a line
        t = np.arange(0, len(PD1_LSB_ARRAY[-1])) / fs
        popt1, pcov1 = curve_fit(linear, t, PD1_LSB_ARRAY[-1])
        popt2, pcov2 = curve_fit(linear, t, PD1_USB_ARRAY[-1])
        popt3, pcov3 = curve_fit(linear, t, PD2_LSB_ARRAY[-1])
        popt4, pcov4 = curve_fit(linear, t, PD2_USB_ARRAY[-1])
        m1, c1 = popt1
        m2, c2 = popt2
        m3, c3 = popt3
        m4, c4 = popt4
        PD1_LSB_fits = np.append(PD1_LSB_fits, m1)
        PD1_USB_fits = np.append(PD1_USB_fits, m2)
        PD2_LSB_fits = np.append(PD2_LSB_fits, m3)
        PD2_USB_fits = np.append(PD2_USB_fits, m4)
    if(plot):
        mean1 = np.average(PD1_LSB_fits)
        std1 = np.std(PD1_LSB_fits)
        print(f'PD1_LSB Mean = {mean1*1e3:.3f}mHz, PD1_LSB STD = {std1*1e3:.1f}mHz')
        mean2 = np.average(PD1_USB_fits)
        std2 = np.std(PD1_USB_fits)
        print(f'PD1_USB Mean = {mean2*1e3:.3f}mHz, PD1_USB STD = {std2*1e3:.1f}mHz')
        mean3 = np.average(PD2_LSB_fits)
        std3 = np.std(PD2_LSB_fits)
        print(f'PD2_LSB Mean = {mean3*1e3:.3f}mHz, PD2_LSB STD = {std3*1e3:.1f}mHz')
        mean4 = np.average(PD2_USB_fits)
        std4 = np.std(PD2_USB_fits)
        print(f'PD2_USB Mean = {mean4*1e3:.3f}mHz, PD2_USB STD = {std4*1e3:.1f}mHz')

        title = f"Data Fits"
        plt.figure()
        plt.hist(PD1_LSB_fits*1e3, bins = 5, label = f'PD1 LSB')
        plt.hist(PD1_USB_fits*1e3, bins = 5, label = f'PD1 USB')
        plt.hist(PD2_LSB_fits*1e3, bins = 5, label = f'PD2 LSB')
        plt.hist(PD2_USB_fits*1e3, bins = 5, label = f'PD2 USB')
        plt.xlabel('Phase gradient [mHz/s]')
        plt.ylabel('Frequency')
        plt.legend(loc = 'upper right')
        plt.title(title)



def calc_oadev(dt, input_data, taus = None):
    ''' This function is intended for phase timeseries. I.e. distance or cycles. Its output will be an ADEV that is in units of its input, Ex. distance => m or phase => cycles. '''
    if np.isscalar(dt):
        samp_freq=1/dt
    else:
        dt=np.array(dt)
        samp_freq=1/np.average(np.diff(dt))
    (ad_taus, adev, ad_errors, ad_ns)=allan.oadev(input_data,rate=samp_freq,data_type="freq",taus=taus)
    (_,_,errors,ns)=allan.adev(input_data,rate=samp_freq,data_type="freq",taus=taus)

    if len(ns) < len(ad_ns):
        ad_taus = ad_taus[:len(ns)]
        adev = adev[:len(ns)]
    errors = [d/np.sqrt(n) for [d,n] in zip(adev,ns)]
    return ad_taus, adev, errors
    

def calc_mdev(dt, input_data, taus = None, data_type="freq", usePhase = False, laser_freq=1):
    # check if dt is a scalar, if so samp_freq=1/dt. If array, then it is time array. fs = 1/np.average(np.diff(dt))
    
    if data_type == "phase":
        usePhase = True
    if np.isscalar(dt):
        samp_freq=1/dt
    else:
        dt=np.array(dt)
        samp_freq=1/np.average(np.diff(dt))
    if data_type=="phase":
        phase=[p/laser_freq for p in input_data]  #convert phase into seconds
    elif data_type=="freq":
        data=input_data-np.mean(input_data) # convert phase into seconds
        phase = np.cumsum(data) / samp_freq /laser_freq# convert frequency into seconds
 
    # The md_errors from allantools seem to be wrong. This fixes that problem.
    if usePhase:
        (md_taus, mdev, errors, ns)=allan.mdev(phase,rate=samp_freq,data_type="phase",taus=taus)
        (ad_taus, adev, ad_errors, ad_ns)=allan.adev(phase,rate=samp_freq,data_type="phase",taus=taus)
    else:
        (md_taus, mdev, errors, ns)=allan.mdev(data,rate=samp_freq,data_type="freq",taus=taus)
        (ad_taus, adev, ad_errors, ad_ns)=allan.adev(data,rate=samp_freq,data_type="freq",taus=taus)
    md_errors=[d/np.sqrt(n) for [d,n] in zip(mdev, ad_ns)]
    return md_taus, mdev, md_errors

def calc_adev(dt, input_data, taus = None, data_type="freq", usePhase = False, laser_freq=1):
    # check if dt is a scalar, if so samp_freq=1/dt. If array, then it is time array. fs = 1/np.average(np.diff(dt))
    if data_type == "phase":
        usePhase = True
    if np.isscalar(dt):
        samp_freq=1/dt
    else:
        dt=np.array(dt)
        samp_freq=1/np.average(np.diff(dt))
    if data_type=="phase":
        phase=[p/laser_freq for p in input_data]  #convert phase into seconds
    elif data_type=="freq":
        data=input_data-np.mean(input_data) # convert phase into seconds
        phase = np.cumsum(data) / samp_freq /laser_freq# convert frequency into seconds
    
    
    # The md_errors from allantools seem to be wrong. This fixes that problem.
    if usePhase:
        (md_taus, mdev, errors, ns)=allan.mdev(phase,rate=samp_freq,data_type="phase",taus=taus)
        (ad_taus, adev, ad_errors, ad_ns)=allan.adev(phase,rate=samp_freq,data_type="phase",taus=taus)
    else:
        (md_taus, mdev, errors, ns)=allan.mdev(data,rate=samp_freq,data_type="freq",taus=taus)
        (ad_taus, adev, ad_errors, ad_ns)=allan.adev(data,rate=samp_freq,data_type="freq",taus=taus)
    md_errors=[d/np.sqrt(n) for [d,n] in zip(mdev, ad_ns)]
    return ad_taus, adev, ad_errors

@jit(nopython=True, cache=True)
def movingAverageFilter(data, N):
    averageData = np.zeros(len(data) - N + 1)
    stdDeviations = np.zeros(len(data) - N + 1)
    
    for endPoint in range(N, len(data) + 1):
        startPoint = endPoint - N
        averageData[startPoint] = np.average(data[startPoint:endPoint])
        stdDeviations[startPoint] = np.std(data[startPoint:endPoint])
    
    return averageData, stdDeviations

def gaussian(x, mean, sigma):
    return (1/(sigma*np.sqrt(2*np.pi))) * np.exp(-0.5*((x-mean)/sigma)**2)

def reject_average_outliers(data, m = None):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else np.zeros(len(d))
    if(m is not None):
        # print (f"Rejecting {np.sum(s<m)} outliers")
        print(f"Rejected {np.sum(s>=m)} outliers")
        # get index of outliers
        idxs = np.where(s>=m)
        # return array of indexes
        idxs = idxs[0]
        return data[s<m], idxs
    else:
        return data
    

def timeseriesHistogram(data, bins = 100):
    # calculate histogram of data
    hist, bin_edges = np.histogram(data, bins = bins)
    return hist, bin_edges
    
def bulkTimeseriesHistograms(dataArray, bins = 100):
    ''' dataArray is a list of timeseries arrays. Idea is to calculate a histogram for each time series. Each histogram can be plotted/overllaped. Also want an average histogram (average each bin value etc). '''
    return

def windowedHistogram(data, windowTime, bins = 100):
    ''' Plot histogram for time segments in a data run!!! Huge. For example, if ADEV tells me that I'm white noise limited up to 200ms, I should look at histograms for 200ms segments! Can plot mean of each hisotgram (this is what we really care about anyway! Histogram at each time point.)'''
    return

def MSTAR_DataRunADEVs(linkLength = 'fiber', holdTime = 1, mode = "bulk_runs_surface", linkDist = None, SNR = None, root_folder = "/media/shawn/PhD/MS2-Measurements/MSTAR-Measurements/SWEEP_TESTS_PAPER", 
                       overwrite = False, holdOverride = None, V2 = False, nperseg = 2**14, hostName = 'pluto2'):
    ''' Will iterate through root folder and perform ADEVs on first data runs. If linkDist is specified, will only analyze that link distance. If SNR is specified, will only analyze that SNR. '''
    # Get folders to analyse
    fn = f"{root_folder}/{linkLength}/{holdTime}s_hold_{mode}"
    folders = os.listdir(fn)
    print(folders)
    foldersToAnalyze = []
    SNRs = []
    linkDists = []
    
    if linkLength == 'fiber':
        n = 1.46
        calibration =  0.04 + 6/n
    elif linkLength == '1km_spool':
        n = 1.46
        calibration = 1050/n + 0.04 + 6/n
    for folder in folders:
        # if folder is 'results', skip
        if folder == 'results':
            continue
        # folder name format is 'mirror_SNR_distance'
        if not V2:
            parts = folder.split('_')
            SNR_FOLDER = parts[1]
            linkDist_FOLDER = parts[2]
            print (f"SNR: {SNR_FOLDER} linkDist: {linkDist_FOLDER}")
        else:
            parts = folder.split('_')
            SNR_FOLDER = "_".join(parts[1:5])
            linkDist_FOLDER = parts[5]
        if SNR_FOLDER == SNR or SNR is None:
            if linkDist_FOLDER == linkDist or linkDist is None:
                foldersToAnalyze.append(folder)
                if SNR_FOLDER not in SNRs:
                    SNRs.append(SNR_FOLDER)
                if linkDist_FOLDER not in linkDists:
                    linkDists.append(linkDist_FOLDER)
    
    print(foldersToAnalyze)
    print(SNRs)
    print(linkDists)
    
    # create folder in fn to store results
    resultsFolder = f"{fn}/results/"
    if not os.path.exists(resultsFolder):
        os.makedirs(resultsFolder)
    
    # Can write a function in MSTAR to fetch data needed from a file once to save opening files multiple times.
    
    # Bulk run plots: Histogram of mean results. Will use window time to segment runs.
    
    figADEV, axsADEV = PLOT.CreatePlot()
    figPSD, axsPSD = PLOT.CreatePlot()
    figTimeSeries, axsTimeSeries = PLOT.CreatePlot()
    figUSBPSD, axsUSBPSD = PLOT.CreatePlot()
    figLSBPSD, axsLSBPSD = PLOT.CreatePlot()
    freqIDX = -1
    
    for s, SNR in enumerate(SNRs):
        for l, linkDist in enumerate(linkDists):
            
            data_folder = f"{fn}/mirror_{SNR}_{linkDist}"
            print(f"Analyzing {data_folder}")
                
            data_fn = f"{fn}/mirror_{SNR}_{linkDist}/MSTAR_SWEEP_TEST_{linkDist}_{hostName}_1.hdf5"
            print(f"Analyzing {data_fn}")
        
            
            distances, averages, sigmas, LO_frequencies, wls, fs, carrierPhase, carrierDistUSBs, carrierDistLSBs = MF.SweepFileAnalysis(data_fn, calibration, holdOverride=holdOverride, showPlots = False, startTime = 50e-3, carrier = True)
            PD1_LSB_ARRAY, PD1_USB_ARRAY, PD2_LSB_ARRAY, PD2_USB_ARRAY, fs, frequencies = MF.returnSidebandPhase(data_fn, holdOverride=holdOverride, startTime=50e-3 )
            LSB_SUM = PD1_LSB_ARRAY + PD2_LSB_ARRAY
            USB_SUM = PD1_USB_ARRAY + PD2_USB_ARRAY
            SYNTHETIC_SUM = LSB_SUM + USB_SUM
            
            # calc mdev for distances[freqIDX]
            dt = 1/fs
            ad_taus, adev, ad_errors = calc_adev(dt, distances[freqIDX], data_type="freq")
            axsADEV.errorbar(ad_taus, adev, yerr = ad_errors, label = f"SNR: {SNR} linkDist: {linkDist}")
            
            # calc PSD
            f, Pxx = signal.welch(distances[freqIDX], fs, nperseg = nperseg)
            axsPSD.plot(f, Pxx, label = f"SNR: {SNR} linkDist: {linkDist}")
            fUSB, PxxUSB = signal.welch(carrierDistUSBs[freqIDX], fs, nperseg = nperseg)
            fLSB, PxxLSB = signal.welch(carrierDistLSBs[freqIDX], fs, nperseg = nperseg)
            axsUSBPSD.plot(fUSB, PxxUSB, label = f"SNR: {SNR} linkDist: {linkDist}")
            axsLSBPSD.plot(fLSB, PxxLSB, label = f"SNR: {SNR} linkDist: {linkDist}")
            
            axsTimeSeries.plot(np.arange(0, len(distances[freqIDX])/fs, 1/fs), distances[freqIDX], label = f"SNR: {SNR} linkDist: {linkDist}")
            
            
            
            
    figADEV, axsADEV = PLOT.formatMDEVPlot(figADEV, axsADEV)
    figPSD, axsPSD = PLOT.formatPSDPlot(figPSD, axsPSD, linear=False, unit='m')
    figADEV.tight_layout()
    figPSD.tight_layout()
    figADEV.savefig(f"{resultsFolder}/adev.png")
    figPSD.savefig(f"{resultsFolder}/psd.png")
    figTimeSeries, axsTimeSeries = PLOT.formatPlot(figTimeSeries, axsTimeSeries, title = "Time Series", xlabel = "Time (s)", ylabel = "Distance (m)")
    figTimeSeries.tight_layout()
    figTimeSeries.savefig(f"{resultsFolder}/timeseries.png")
    figUSBPSD, axsUSBPSD = PLOT.formatPSDPlot(figUSBPSD, axsUSBPSD, linear=False, unit='m')
    figUSBPSD.tight_layout()
    figUSBPSD.savefig(f"{resultsFolder}/usb_psd.png")
    figLSBPSD, axsLSBPSD = PLOT.formatPSDPlot(figLSBPSD, axsLSBPSD, linear=False, unit='m')
    figLSBPSD.tight_layout()
    figLSBPSD.savefig(f"{resultsFolder}/lsb_psd.png")
    return

def MSTAR_DataRunAnalysis(linkLength = 'fiber', holdTime = 1, mode = "bulk_runs_surface", linkDist = None, SNR = None, root_folder = "/media/shawn/PhD/MS2-Measurements/MSTAR-Measurements/SWEEP_TESTS_PAPER", overwrite = False, windowTime = None,
                          plotPhase = True, plotADEV = True, plotDistances = True, plotHistograms = True, V2 = False, hostName = 'pluto2'):
    ''' Will iterate through root folder and perform data analysis. If linkDist is specified, will only analyze that link distance. If SNR is specified, will only analyze that SNR. '''
    # Get folders to analyse
    fn = f"{root_folder}/{linkLength}/{holdTime}s_hold_{mode}"
    folders = os.listdir(fn)
    print(folders)
    foldersToAnalyze = []
    SNRs = []
    linkDists = []
    
    if linkLength == 'fiber':
        n = 1.46
        calibration =  0.04 + 6/n
    elif linkLength == '1km_spool':
        n = 1.46
        calibration = 1050/n + 0.04 + 6/n
    for folder in folders:
        # if folder is 'results', skip
        if folder == 'results':
            continue
        # folder name format is 'mirror_SNR_distance'
        if not V2:
            parts = folder.split('_')
            SNR_FOLDER = parts[1]
            linkDist_FOLDER = parts[2]
            print (f"SNR: {SNR_FOLDER} linkDist: {linkDist_FOLDER}")
        else:
            parts = folder.split('_')
            SNR_FOLDER = "_".join(parts[1:5])
            linkDist_FOLDER = parts[5]
        if SNR_FOLDER == SNR or SNR is None:
            if linkDist_FOLDER == linkDist or linkDist is None:
                foldersToAnalyze.append(folder)
                if SNR_FOLDER not in SNRs:
                    SNRs.append(SNR_FOLDER)
                if linkDist_FOLDER not in linkDists:
                    linkDists.append(linkDist_FOLDER)
    
    print(foldersToAnalyze)
    print(SNRs)
    print(linkDists)
    
    # create folder in fn to store results
    resultsFolder = f"{fn}/results/"
    if not os.path.exists(resultsFolder):
        os.makedirs(resultsFolder)
    
    # Can write a function in MSTAR to fetch data needed from a file once to save opening files multiple times.
    
    # Bulk run plots: Histogram of mean results. Will use window time to segment runs.
    
    meanMeasurementResults = np.array([])
    sigmaMeasurementResults = np.array([])
    meanMeasurementResults = np.zeros((len(SNRs), len(linkDists), 100))
    sigmaMeasurementResults = np.zeros((len(SNRs), len(linkDists), 100))
    holdOverride = None
    
    for s, SNR in enumerate(SNRs):
        for l, linkDist in enumerate(linkDists):
            
            data_folder = f"{fn}/mirror_{SNR}_{linkDist}"
            resultsFolderCurrent = f"{resultsFolder}mirror_{SNR}_{linkDist}"
            if not os.path.exists(resultsFolderCurrent):
                os.makedirs(resultsFolderCurrent)
            
            phaseFolder = f"{resultsFolderCurrent}/phase"
            if not os.path.exists(phaseFolder):
                os.makedirs(phaseFolder)
            
            adevFolder = f"{resultsFolderCurrent}/adev"
            if not os.path.exists(adevFolder):
                os.makedirs(adevFolder)
                
            distFolder = f"{resultsFolderCurrent}/distances"
            if not os.path.exists(distFolder):
                os.makedirs(distFolder)
                
            histFolder = f"{resultsFolderCurrent}/histograms"
            if not os.path.exists(histFolder):
                os.makedirs(histFolder)
            
            print(f"Analyzing {data_folder}")
            numberOfRuns = len(os.listdir(data_folder))
            for k in range(1, numberOfRuns+1):
                data_fn = f"{fn}/mirror_{SNR}_{linkDist}/MSTAR_SWEEP_TEST_{linkDist}_{hostName}_{k}.hdf5"
                print(f"Analyzing {data_fn}")
                
                
                
                if k == 1:
                    # Create text file of config
                    PD1_LSB_ARRAY, PD1_USB_ARRAY, PD2_LSB_ARRAY, PD2_USB_ARRAY, fs, frequencies = MF.returnSidebandPhase(data_fn)
                    numFrequencies = len(frequencies)
                    configTxt = ""
                    AOM1_POW, AOM2_POW = MF.getMOKUData(data_fn)
                    configTxt += "AOM1 Power (dBm): " + str(AOM1_POW) + "\n"
                    configTxt += "AOM2 Power (dBm): " + str(AOM2_POW) + "\n"
                    configTxt += "linkDist: " + str(linkDist) + "\n"
                    configTxt += "SNR: " + str(SNR) + "\n"
                    for i, freq in enumerate(frequencies):
                        runTime = len(PD1_LSB_ARRAY[i])/fs
                        configTxt += f"Frequency {i}: {freq}Hz, Run Time: {runTime}s\n"
                    with open(f"{resultsFolderCurrent}/config.txt", "w") as f:
                        f.write(configTxt)
                        
                if plotPhase or plotADEV or plotDistances or plotHistograms:
                    PD1_LSB_ARRAY, PD1_USB_ARRAY, PD2_LSB_ARRAY, PD2_USB_ARRAY, fs, frequencies = MF.returnSidebandPhase(data_fn)
                    numFrequencies = len(frequencies)
                
                # check for phase folder in results folder
                if plotPhase:
                    # check if phase data has already been analyzed
                    if os.path.exists(f"{phaseFolder}/phase_{k}.png") and not overwrite:
                        print(f"Phase data already analyzed for {data_fn}")
                    else:
                        # create figure with numRows = numFrequencies
                        fig, axs = PLOT.CreatePlotRowCol(numFrequencies, 1, figsize = (20, 20))
                        for i in range(numFrequencies):
                            timeArray = np.arange(0, len(PD1_LSB_ARRAY[i])/fs, 1/fs)
                            axs[i].plot(timeArray, PD1_LSB_ARRAY[i], label = f"PD1 LSB")
                            axs[i].plot(timeArray, PD1_USB_ARRAY[i], label = f"PD1 USB")
                            axs[i].plot(timeArray, PD2_LSB_ARRAY[i], label = f"PD2 LSB")
                            axs[i].plot(timeArray, PD2_USB_ARRAY[i], label = f"PD2 USB")
                            axs[i].set_title(f"Frequency: {frequencies[i]}Hz")
                            axs[i].set_xlabel("Time (s)")
                            axs[i].set_ylabel("Phase (cycles)")
                            axs[i].legend()
                            # save fig as phase_{k}.png
                        fig.tight_layout()
                        fig.savefig(f"{phaseFolder}/phase_{k}.png")
                        plt.close(fig)
                
                distances, averages, sigmas, LO_frequencies, wls, fs, carrierPhase, carrierDistUSBs, carrierDistLSBs = MF.SweepFileAnalysis(data_fn, calibration, holdOverride=holdOverride, showPlots = False, startTime = 50e-3, carrier = True)
                
                meanDistance = np.mean(distances[-1])
                sigmaDistance = np.std(distances[-1])
                meanMeasurementResults[s, l, k-1] = meanDistance
                sigmaMeasurementResults[s, l, k-1] = sigmaDistance
                print(f"Mean: {meanDistance*1e3}mm Sigma: {sigmaDistance*1e6}um")
                
                
                # check for adev folder in results folder
                if plotADEV:
                    if os.path.exists(f"{adevFolder}/adev_{k}.png") and not overwrite:
                        print(f"ADEV data already analyzed for {data_fn}")
                    else:
                        # create figure with numRows = numFrequencies
                        fig, axs = PLOT.CreatePlotRowCol(numFrequencies, 1, figsize = (20, 20))
                        for i in range(numFrequencies):
                            dt = 1/fs
                            md_taus, mdev, md_errors = calc_mdev(dt, distances[i], data_type="freq")
                            ad_taus, adev, ad_errors = calc_adev(dt, distances[i], data_type="freq")
                            # USB and LSB
                            md_taus_USB, mdev_USB, md_errors_USB = calc_mdev(dt, carrierDistUSBs[i], data_type="freq")
                            ad_taus_USB, adev_USB, ad_errors_USB = calc_adev(dt, carrierDistUSBs[i], data_type="freq")
                            md_taus_LSB, mdev_LSB, md_errors_LSB = calc_mdev(dt, carrierDistLSBs[i], data_type="freq")
                            ad_taus_LSB, adev_LSB, ad_errors_LSB = calc_adev(dt, carrierDistLSBs[i], data_type="freq")
                            
                            # axs[i].plot(md_taus, mdev, label = "Synthetic Distance MDEV")
                            axs[i].plot(ad_taus, adev, label = "Synthetic Distance ADEV")
                            # axs[i].plot(md_taus_USB, mdev_USB, label = "USB Carrier MDEV")
                            axs[i].plot(ad_taus_USB, adev_USB, label = "USB Carrier ADEV")
                            # axs[i].plot(md_taus_LSB, mdev_LSB, label = "LSB Carrier MDEV")
                            axs[i].plot(ad_taus_LSB, adev_LSB, label = "LSB Carrier ADEV")
                            fig, axs[i] = PLOT.formatMDEVPlot(fig, axs[i], title = f"Frequency: {frequencies[i]}Hz", xlabel = "Integration Time (s)", ylabel = "Std. Deviation (m)")
                        fig.tight_layout()
                        fig.savefig(f"{adevFolder}/adev_{k}.png") 
                        plt.close(fig)
                if plotDistances:
                    if os.path.exists(f"{distFolder}/distances_{k}.png") and not overwrite:
                        print(f"Distance data already analyzed for {data_fn}")
                    else:
                        fig, axs = PLOT.CreatePlotRowCol(numFrequencies, 1, figsize = (20, 20))
                        for i in range(numFrequencies):
                            timeArray = np.arange(0, len(PD1_LSB_ARRAY[i])/fs, 1/fs)
                            axs[i].plot(timeArray, distances[i] - distances[i][0], label = f"Synthetic Distance")
                            axs[i].plot(timeArray, carrierDistUSBs[i] - carrierDistUSBs[i][0], label = f"USB Carrier Distance")
                            axs[i].plot(timeArray, carrierDistLSBs[i] - carrierDistLSBs[i][0], label = f"LSB Carrier Distance")
                            axs[i].set_title(f"Frequency: {frequencies[i]}Hz")
                            axs[i].set_xlabel("Time (s)")
                            axs[i].set_ylabel("Distance (m)")
                            axs[i].legend()
                        fig.tight_layout()
                        fig.savefig(f"{distFolder}/distances_{k}.png")
                        plt.close(fig)
                if plotHistograms:
                    if os.path.exists(f"{histFolder}/histograms_{k}.png") and not overwrite:
                        print(f"Histogram data already analyzed for {data_fn}")
                    fig, axs = PLOT.CreatePlotRowCol(1, numFrequencies, figsize = (20, 20))
                    for i in range(numFrequencies):
                        hist, bin_edges = timeseriesHistogram(distances[i] - distances[i][0], bins = 100)
                        axs[i].plot(bin_edges[:-1], hist)
                        axs[i].set_title(f"Frequency: {frequencies[i]}Hz")
                        axs[i].set_xlabel("Distance (m)")
                        axs[i].set_ylabel("Frequency")
                    fig.tight_layout()
                    fig.savefig(f"{histFolder}/histograms_{k}.png")
                    plt.close(fig)
    return

def MSTAR_BulkDataRunAnalysisData(linkLength = 'fiber', holdTime = 1, mode = "bulk_runs_surface", linkDist = None, SNR = None, root_folder = "/media/shawn/PhD/MS2-Measurements/MSTAR-Measurements/SWEEP_TESTS_PAPER", overwrite = False, windowTime = None,
                              numRunsOverride = None, startTime = 50e-3, V2 = False, hostName = 'pluto2'):
    ''' Will return mean measurement data. '''
    fn = f"{root_folder}/{linkLength}/{holdTime}s_hold_{mode}"
    folders = os.listdir(fn)
    print(folders)
    foldersToAnalyze = []
    SNRs = []
    linkDists = []
    
    if linkLength == 'fiber':
        n = 1.46
        calibration =  0.04 + 6/n
    elif linkLength == '1km_spool':
        n = 1.46
        calibration = 1050/n + 0.04 + 6/n
    
    for folder in folders:
        # if folder is 'results', skip
        if folder == 'results':
            continue
        # folder name format is 'mirror_SNR_distance'
        parts = folder.split('_')
        SNR_FOLDER = parts[1]
        linkDist_FOLDER = parts[2]
        if V2:
            SNR_FOLDER = "_".join(parts[1:5])
            linkDist_FOLDER = parts[5]
        
        print (f"SNR: {SNR_FOLDER} linkDist: {linkDist_FOLDER}")
        if SNR_FOLDER == SNR or SNR is None:
            if linkDist_FOLDER == linkDist or linkDist is None:
                foldersToAnalyze.append(folder)
                if SNR_FOLDER not in SNRs:
                    SNRs.append(SNR_FOLDER)
                if linkDist_FOLDER not in linkDists:
                    linkDists.append(linkDist_FOLDER)
    
    print(foldersToAnalyze)
    print(SNRs)
    print(linkDists)
    # create folder in fn to store results
    resultsFolder = f"{fn}/results/"
    if not os.path.exists(resultsFolder):
        os.makedirs(resultsFolder)
    # Can write a function in MSTAR to fetch data needed from a file once to save opening files multiple times.
    # Bulk run plots: Histogram of mean results. Will use window time to segment runs.
    
    meanMeasurementResults = np.array([])
    sigmaMeasurementResults = np.array([])
    meanMeasurementResults = np.zeros((len(SNRs), len(linkDists), 100))
    sigmaMeasurementResults = np.zeros((len(SNRs), len(linkDists), 100))
    startTime = startTime
    if windowTime is not None:
        holdOverride = windowTime + startTime
    else:
        holdOverride = None
    
    
    skips = []
    for s, SNR in enumerate(SNRs):
        for l, linkDist in enumerate(linkDists):
            data_folder = f"{fn}/mirror_{SNR}_{linkDist}"
            resultsFolderCurrent = f"{resultsFolder}mirror_{SNR}_{linkDist}"
            print(f"Analyzing {data_folder}")
            # check if data folder exists, if not skip
            if not os.path.exists(data_folder):
                print(f"Data folder {data_folder} does not exist. Skipping.")
                skips.append((SNR, linkDist))
                continue
            numberOfRuns = len(os.listdir(data_folder))
            if numRunsOverride is not None:
                numberOfRuns = numRunsOverride
            # trim array to number of runs
            for k in range(1, numberOfRuns+1):
                data_fn = f"{fn}/mirror_{SNR}_{linkDist}/MSTAR_SWEEP_TEST_{linkDist}_{hostName}_{k}.hdf5"
                print(f"Analyzing {data_fn}")
                
                distances, averages, sigmas, LO_frequencies, wls, fs, carrierPhase, carrierDistUSBs, carrierDistLSBs = MF.SweepFileAnalysis(data_fn, calibration, holdOverride=holdOverride, showPlots = False, startTime = startTime, carrier = True)
                
                meanDistance = np.mean(distances[-1])
                sigmaDistance = np.std(distances[-1])
                meanMeasurementResults[s, l, k-1] = meanDistance
                sigmaMeasurementResults[s, l, k-1] = sigmaDistance
                print(f"Mean: {meanDistance*1e3}mm Sigma: {sigmaDistance*1e6}um")
    meanMeasurementResults = meanMeasurementResults[:, :, :numberOfRuns]        
    # return snrs, linkDists, meanMeasurementResults, sigmaMeasurementResults
    return SNRs, linkDists, meanMeasurementResults, sigmaMeasurementResults

def MSTAR_BulkDataRunAnalysis(linkLength = 'fiber', holdTime = 1, mode = "bulk_runs_surface", linkDist = None, SNR = None, root_folder = "/media/shawn/PhD/MS2-Measurements/MSTAR-Measurements/SWEEP_TESTS_PAPER", overwrite = False, windowTime = None,
                              numRunsOverride = None, startTime = 50e-3, V2 = False, hostName = 'pluto2'):
    ''' Will iterate through root folder and perform data analysis. If linkDist is specified, will only analyze that link distance. If SNR is specified, will only analyze that SNR. '''
    # Get folders to analyse
    fn = f"{root_folder}/{linkLength}/{holdTime}s_hold_{mode}"
    folders = os.listdir(fn)
    print(folders)
    foldersToAnalyze = []
    SNRs = []
    linkDists = []
    
    if linkLength == 'fiber':
        n = 1.46
        calibration =  0.04 + 6/n
    elif linkLength == '1km_spool':
        n = 1.46
        calibration = 1050/n + 0.04 + 6/n
    
    for folder in folders:
        # if folder is 'results', skip
        if folder == 'results':
            continue
        # folder name format is 'mirror_SNR_distance'
        parts = folder.split('_')
        SNR_FOLDER = parts[1]
        linkDist_FOLDER = parts[2]
        if V2:
            SNR_FOLDER = "_".join(parts[1:5])
            linkDist_FOLDER = parts[5]
        
        print (f"SNR: {SNR_FOLDER} linkDist: {linkDist_FOLDER}")
        if SNR_FOLDER == SNR or SNR is None:
            if linkDist_FOLDER == linkDist or linkDist is None:
                foldersToAnalyze.append(folder)
                if SNR_FOLDER not in SNRs:
                    SNRs.append(SNR_FOLDER)
                if linkDist_FOLDER not in linkDists:
                    linkDists.append(linkDist_FOLDER)
    
    print(foldersToAnalyze)
    print(SNRs)
    print(linkDists)
    # create folder in fn to store results
    resultsFolder = f"{fn}/results/"
    if not os.path.exists(resultsFolder):
        os.makedirs(resultsFolder)
    # Can write a function in MSTAR to fetch data needed from a file once to save opening files multiple times.
    # Bulk run plots: Histogram of mean results. Will use window time to segment runs.
    
    meanMeasurementResults = np.array([])
    sigmaMeasurementResults = np.array([])
    meanMeasurementResults = np.zeros((len(SNRs), len(linkDists), 100))
    sigmaMeasurementResults = np.zeros((len(SNRs), len(linkDists), 100))
    startTime = startTime
    if windowTime is not None:
        holdOverride = windowTime + startTime
    else:
        holdOverride = None
    
    
    skips = []
    for s, SNR in enumerate(SNRs):
        for l, linkDist in enumerate(linkDists):
            data_folder = f"{fn}/mirror_{SNR}_{linkDist}"
            resultsFolderCurrent = f"{resultsFolder}mirror_{SNR}_{linkDist}"
            print(f"Analyzing {data_folder}")
            # check if data folder exists, if not skip
            if not os.path.exists(data_folder):
                print(f"Data folder {data_folder} does not exist. Skipping.")
                skips.append((SNR, linkDist))
                continue
            numberOfRuns = len(os.listdir(data_folder))
            if numRunsOverride is not None:
                numberOfRuns = numRunsOverride
            # trim array to number of runs
            for k in range(1, numberOfRuns+1):
                data_fn = f"{fn}/mirror_{SNR}_{linkDist}/MSTAR_SWEEP_TEST_{linkDist}_{hostName}_{k}.hdf5"
                print(f"Analyzing {data_fn}")
                
                distances, averages, sigmas, LO_frequencies, wls, fs, carrierPhase, carrierDistUSBs, carrierDistLSBs = MF.SweepFileAnalysis(data_fn, calibration, holdOverride=holdOverride, showPlots = False, startTime = startTime, carrier = True)
                
                meanDistance = np.mean(distances[-1])
                sigmaDistance = np.std(distances[-1])
                meanMeasurementResults[s, l, k-1] = meanDistance
                sigmaMeasurementResults[s, l, k-1] = sigmaDistance
                print(f"Mean: {meanDistance*1e3}mm Sigma: {sigmaDistance*1e6}um")
    # trim array to number of runs
    meanMeasurementResults = meanMeasurementResults[:, :, :numberOfRuns]            
    # Create plot with number of SNR columns
    # figHist, axsHist = PLOT.CreatePlotRowCol(1, len(SNRs), figsize = (20, 10))
    # if len(SNRs) == 1:
    #     axsHist = [axsHist]
    # histogram each mean array
    
    for s, SNR in enumerate(SNRs):
        figScatter, axsScatter = plt.subplots() 
        figHist, axsHist = plt.subplots()
        for l, linkDist in enumerate(linkDists):
            if (SNR, linkDist) in skips:
                continue
            rejectMeans, idx = reject_average_outliers(meanMeasurementResults[s, l], m = 4)
            rejectMeans -= calibration
            meanofMeans = np.mean(rejectMeans)
            axsHist.hist(rejectMeans*1e3, bins = 25, alpha = 0.5, label = f"SNR: {SNR} linkDist: {linkDist} mean: {meanofMeans*1e3:.2f}mm")
            
            axsScatter.scatter(np.arange(len(rejectMeans)), rejectMeans*1e3, label = f"linkDist: {linkDist}")
            print(f"SNR: {SNR} linkDist: {linkDist} mean: {meanofMeans*1e3:.2f}mm")
        axsScatter.set_title(f"SNR: {SNR} linkDist: {linkDist}")
        axsScatter.set_xlabel("Run Number")
        axsScatter.set_ylabel("Mean Distance (mm)")
        axsScatter.legend()
        figScatter.tight_layout()
        figScatter.savefig(f"{resultsFolder}/scatter_{SNR}.png")
        plt.close(figScatter)
            
        axsHist.set_title(f"SNR: {SNR}")
        axsHist.set_xlabel("Mean Distance (mm)")
        axsHist.set_ylabel("Frequency")
        axsHist.legend()
        figHist.tight_layout()
        figHist.savefig(f"{resultsFolder}/histogram_{SNR}.png")
    
    # save figure to results folder
    figHist.tight_layout()
    figHist.savefig(f"{resultsFolder}/mean_histograms.png")        
    plt.show()
    
    return

if __name__ == "__main__":
    # Example usage of calc_mdev and calc_adev
    debug = "MSTAR"
    
    if debug == "MSTAR":
        # Example usage of MSTAR_DataRunAnalysis
        MSTAR_BulkDataRunAnalysis(linkLength = '1km_spool', holdTime = 1, mode = "micro_delta", linkDist = None, SNR = None, root_folder = "/media/shawn/PhD/MS2-Measurements/MSTAR-Measurements/SWEEP_TESTS_PAPER", windowTime=None, startTime=0.4)
        # MSTAR_DataRunAnalysis(linkLength = 'fiber', holdTime = 5, mode = "micro_delta_check", linkDist = None, SNR = None, root_folder = "/media/shawn/PhD/MS2-Measurements/MSTAR-Measurements/SWEEP_TESTS_PAPER", numRunsOverride=None, windowTime=10e-2)
    
    if debug == "DEVIATIONS":
        # Play around with kickup frequency. You will see that a moving average becomes useless for certain noise conditions. This is why we have an ADEV/MDEV!
        fs = 1000
        N = 1000*fs
        T = N/fs
        tArray = np.arange(0, T, 1/fs)
        print(T)
        floor = 1e-2
        kickup = 1
        
        # find frequency range < 1Hz, and range >1Hz
        freqs = PLL.get_single_freqs(N, fs)
        fRange1 = np.where(freqs < kickup)[0] #1Hz kickup
        fRange2 = np.where(freqs >= kickup)[0]
        PSD = np.ones_like(freqs) 
        PSD[fRange1] = floor / freqs[fRange1]**2
        PSD[fRange2] = floor

        noise = PLL.psdnoise(PSD, N, fs)
        windowTau = [ 0.001, 0.01, 0.05, 0.1, 0.5, 1 ] # length of subsequent average window in seconds
        windowSize = [int(ws*fs) for ws in windowTau]
        deviations = []

        for Ns in windowSize:
            averageData, stdDeviations = movingAverageFilter(noise, Ns)
            times = np.arange(0, len(averageData)/fs, 1/fs)
            deviations.append(np.std(averageData))
            
        fig, axs = PLOT.CreatePlot()
        axs.plot(windowTau, deviations, label = "Moving Average Filter")
        dt = 1/fs
        md_taus, mdev, md_errors = calc_mdev(dt, noise, data_type="freq")
        axs.plot(md_taus, mdev, label = "Mod. Allan Deviation (Freq Data)")
        ad_taus, adev, ad_errors = calc_adev(dt, noise, data_type="freq")
        axs.plot(ad_taus, adev, label = "Allan Deviation (Freq Data)")
        ad_taus, adev, ad_errors = calc_adev(dt, noise, data_type="freq", usePhase = True)
        axs.plot(ad_taus, adev, label = "Allan Deviation (Phase Data)")
        md_taus, mdev, md_errors = calc_mdev(dt, noise, data_type="freq", usePhase = True)
        axs.plot(md_taus, mdev, label = "Mod. Allan Deviation (Phase Data)")
        
        fig, axs = PLOT.formatMDEVPlot(fig, axs, title = "Debugging", xlabel = "Integration Time (s)", ylabel = "Std. Deviation (m)")
        
        plt.show()
        
        