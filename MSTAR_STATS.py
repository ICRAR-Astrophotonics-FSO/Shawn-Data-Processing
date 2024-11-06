import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import PLL_FUNCS as PLL
from tabulate import tabulate
import PLOT_FUNCS as PLOT
import OPTICS_FUNCS as OF
import allantools as allan
from numba import jit

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


if __name__ == "__main__":
    # Example usage of calc_mdev and calc_adev
    debug = "DEVIATIONS"
    
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