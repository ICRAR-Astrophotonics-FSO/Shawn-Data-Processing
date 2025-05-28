import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
from scipy import integrate
import shutil
import os
from scipy.fft import fft, fftfreq
from numba import jit

##### To-Dos: #####
# 1. Compare the output of phaseToIQ with the output of basebandToIQ. Looks good.

def IQToBaseband(IQdata, f0 = 0):
    ''' Function to convert complex IQ data to baseband data. I.e. data = I(t) + jQ(t). f(t) = I cos(2*pi*f0*t) + Q sin(2*pi*f0*t)'''
    return np.real(IQdata) * np.cos(2*np.pi*f0*np.arange(len(IQdata))) + np.imag(IQdata) * np.sin(2*np.pi*f0*np.arange(len(IQdata)))

def basebandToIQ(data, f0 = 0):
    ''' Function to convert baseband data to IQ data. I.e. data = A cos(2*pi*f0*t + phase) = I cos(2*pi*f0*t) + Q sin(2*pi*f0*t).'''
    # Analytic signal has the same properties as IQ data
    # taking the conjugate of the analytic signal gives the proper IQ data, otherwise I saw a phase shift
    # sig = np.real(signal_iq)*np.cos(2*np.pi*f*t) + np.imag(signal_iq) * np.sin(2*np.pi*f*t)
    # Note this is still valid IQ data if f0 = 0. As we essentially only care about the real I part then. as Q*sin(0) = 0
    return np.conjugate(signal.hilbert(data) * np.exp(-1j*2*np.pi*f0*np.arange(len(data)))) #f0 is the normalised center frequency

@jit(nopython=True, cache=True)
def compute_IQ(tVals, f0, scale, data, amp_scale):
    return amp_scale * np.exp(1j * (2 * np.pi * f0 * tVals + scale * data))

def phaseToIQ(data, tVals, f0, A, additiveNoise=0, unit='cycles'):
    ''' Function to convert phase data to IQ data. f0 is the carrier frequency. Default unit for phase data is cycles.'''
    
    if unit == 'cycles':
        scale = 2 * np.pi
    else:
        scale = 1
    print('Scale: ', scale)
    amp_scale = A  # Length of phasor is amplitude A

    # if additiveNoise is scalar
    if np.isscalar(additiveNoise):
        IQ = compute_IQ(tVals, f0, scale, data, amp_scale)
    else:
        IQ = compute_IQ(tVals, f0, scale, data, amp_scale) + signal.hilbert(additiveNoise)
    
    return IQ

def libreGainTable(GAIN = 10000):
    ''' Controller gain table for libre IQ PLLs. This has been tuned on Pluto 2 with slow_attack to get the ADC scale to be around -12dBFS/0.25x. For MSTAR it will reduce slightly if sidebands aren't balanced. '''
    KP = 0
    KI = 0
    KII = 0
    if(GAIN <= 1):
        KP = -1
        KI = -27
        KII = -58
        print("PLL BW 1Hz")
    elif(GAIN <= 10):
        KP = 2
        KI = -21
        KII = -48
        print("PLL BW 10Hz")
    elif(GAIN <= 100):
        KP = 6
        KI = -12
        KII = -34
        print("PLL BW 100Hz")
        
    elif(GAIN <= 500):
        KP = 8
        KI = -9
        KII = -30
        print("PLL BW 500Hz")
    elif(GAIN <= 1000):
        KP = 9
        KI = -6
        KII = -24
        print("PLL BW 1kHz")
    elif(GAIN <= 10000):
        KP = 13
        KI = -1
        KII = -20
        print("PLL BW 10kHz")
    else:
        KP = 15
        KI = 6
        KII = -5
        print("PLL BW HIGH")
    return KP, KI, KII


def plutoGainTable(GAIN = 10000):
    ''' Controller gain table for pluto IQ PLLs. This has been tuned on Pluto 2 with slow_attack to get the ADC scale to be around -12dBFS/0.25x. For MSTAR it will reduce slightly if sidebands aren't balanced. '''
    KP = 0
    KI = 0
    KII = 0
    if(GAIN <= 1):
        KP = -1
        KI = -27
        KII = -58
        print("PLL BW 1Hz")
    elif(GAIN <= 10):
        KP = 2
        KI = -21
        KII = -48
        print("PLL BW 10Hz")
    elif(GAIN <= 100):
        KP = 6
        KI = -12
        KII = -34
        print("PLL BW 100Hz")
        
    elif(GAIN <= 500):
        KP = 8
        KI = -9
        KII = -30
        print("PLL BW 500Hz")
    elif(GAIN <= 1000):
        KP = 9
        KI = -6
        KII = -24
        print("PLL BW 1kHz")
    elif(GAIN <= 10000):
        KP = 13
        KI = 1
        KII = -14
        print("PLL BW 10kHz")
    else:
        KP = 15
        KI = 6
        KII = -5
        print("PLL BW HIGH")
    return KP, KI, KII

def PLL_FPGA_TF(A = 2**10, B = 2**11, P = 2**14, I = 2**2, I2 = 0, fs = 30.72e6, L = 12, AB = 32, R = 32, f = None):
    if(f is None):
        f = np.linspace(0.00001, fs/2, 100000)
    z = np.exp(2j*np.pi*f/fs)
    den = 1 - z**-1

    N = 2
    R = 32
    T = int(N*np.log2(R))

    integrator_stages = (z**-1)/den
    integrator_stages = (z**-1) * (z**-R) * integrator_stages**N

    sub_stages = z**-R * 2**-T * (1-z**-R)**N
    myCIC = integrator_stages * sub_stages

    D = z**-1
    P_STAGE = D * P * D
    I_STAGE = D * (1 / (1-D)) * D  * I * D
    II_STAGE = D * (1 / (1-D)) * D * (1 / (1-D)) * D * I2 * D
    PI = D * (P_STAGE + I_STAGE+ II_STAGE)
    MIXER_GAIN = A * B * D * 2 ** -12 * D # Extra D is for the Q_sum register
    #L = 12
    # AB = 32
    D = z**-1

    LUT_SCALE = 1 # or 2pi
    myNCO = D**3 * 2 ** (L-AB) * LUT_SCALE /(1-D)/2**L # Removed 2pi, as analysis may be in cycles!!!
    LOOP_GAIN = MIXER_GAIN * myCIC * PI * myNCO 
    P_GAIN = MIXER_GAIN * myCIC * P_STAGE * D * myNCO
    I_GAIN = MIXER_GAIN * myCIC * I_STAGE * D * myNCO
    I2_GAIN = MIXER_GAIN * myCIC * II_STAGE * D * myNCO

    OOL = (1/2**AB) * 1/(1-D)
    FORWARD_LOOP_TF = MIXER_GAIN * myCIC * PI * OOL / (1+LOOP_GAIN)
    ERROR_TF = 1 / (1+LOOP_GAIN)
    return f, P_GAIN, I_GAIN, I2_GAIN, LOOP_GAIN, FORWARD_LOOP_TF, ERROR_TF

def PLL_TF(A = 1, P = 0.2, I = 0.05, I2 = 0.005, fs = 1):
    ''' Function to calculate the transfer function of a PLL. '''

    amp_scale = A

    # Phase out = phase_in * (P + I/(1-z^-1) + I2/(1-z^-1)^2)
    # Transfer function = (P + I/(1-z^-1) + I2/(1-z^-1)^2)
    f = np.linspace(0.00001, fs/2, 100000)
    z = np.exp(2j*np.pi*f/fs)
    den = 1 - z**-1

    ## PII2 Controller
    PII2_TF = P + I/den + I2/den**2
    NCO = 2*np.pi/den

    LOOP_GAIN = amp_scale * PII2_TF * NCO
    FORWARD_LOOP_TF = amp_scale * PII2_TF * NCO/ (1 + LOOP_GAIN) # Phase_out / Phase_in (1 is ideal phase lock)
    ERROR_TF = 1 / (1 + LOOP_GAIN) # Error / Phase_in (0 is ideal phase lock)

    ## Controller Tuning Functions

    P_GAIN = amp_scale * P * NCO
    I_GAIN = amp_scale * I/den * NCO
    I2_GAIN = amp_scale * (I2/den**2) * NCO


    return f, P_GAIN, I_GAIN, I2_GAIN, LOOP_GAIN, FORWARD_LOOP_TF, ERROR_TF

@jit(nopython=True, cache=True)
def PLL(dataIQ, P, I, I2):
    ''' Function for Phase Locked Loop (PLL) for IQ data (in complex form, i.e. I+jQ) '''
    N = len(dataIQ)
    adjustment = np.zeros(N, dtype=np.complex64)
    phase_error = np.zeros(N)
    freq_error = np.zeros(N)
    phase = 0
    sum_error = 0
    sum_sum_error = 0  
    for i in range(N):
        adjustment[i] = dataIQ[i] * np.exp(-1j*phase)
        error = np.real(adjustment[i])
        sum_error += error
        sum_sum_error += sum_error
        freq_error[i]  = P * error + I * sum_error + I2 * sum_sum_error
        phase += 2*np.pi*freq_error[i]
        phase_error[i] = phase/(2*np.pi)
    return adjustment, phase_error

def PLL_Q(dataIQ, P, I, I2):
    ''' Function for Phase Locked Loop (PLL) for IQ data (in complex form, i.e. I+jQ) '''
    N = len(dataIQ)
    adjustment = np.zeros(N, dtype=np.complex64)
    phase_error = np.zeros(N)
    freq_error = np.zeros(N)
    phase = 0
    sum_error = 0
    sum_sum_error = 0  
    for i in range(N):
        adjustment[i] = dataIQ[i] * np.exp(-1j*phase)
        error = np.imag(adjustment[i])
        sum_error += error
        sum_sum_error += sum_error
        freq_error[i]  = P * error + I * sum_error + I2 * sum_sum_error
        phase += 2*np.pi*freq_error[i]
        phase_error[i] = phase/(2*np.pi)
    return adjustment, phase_error

@jit(nopython=True, cache=True)
def movingAverageFilter(data, N):
    print("Deprecated. Using function in MSTAR_STATS.py")
    averageData = np.zeros(len(data) - N + 1)
    stdDeviations = np.zeros(len(data) - N + 1)
    
    for endPoint in range(N, len(data) + 1):
        startPoint = endPoint - N
        averageData[startPoint] = np.average(data[startPoint:endPoint])
        stdDeviations[startPoint] = np.std(data[startPoint:endPoint])
    
    return averageData, stdDeviations

def findMinSTDWindow(data, threshold, N, fs, filt = True, fc = 50):
    # filter data
    print("Fix me. Im forward average")
    if filt:
        b, a = signal.butter(10, fc, 'low', fs = fs)
        data = signal.filtfilt(b, a, data)
    averageData = np.zeros(len(data) - N + 1)
    stdDeviations = np.zeros(len(data) - N + 1)
    
    for startPoint in range(len(data) - N + 1):
        endPoint = startPoint + N
        averageData[startPoint] = np.average(data[startPoint:endPoint])
        stdDeviations[startPoint] = np.std(data[startPoint:endPoint])
        if stdDeviations[startPoint] <= threshold:
            return averageData[startPoint], stdDeviations[startPoint], startPoint        
    meanSTD = np.mean(stdDeviations)
    print(f"No statistical certainty!, mean std deviation: {meanSTD*1e6}um")
    plt.figure()
    plt.hist(stdDeviations*1e6)
    plt.show()
    
    return averageData, stdDeviations, -1

def movingAverageFilterNoJIT(data, N):
    print("Fix me. Im forward average")
    averageData = np.zeros(len(data) - N + 1)
    stdDeviations = np.zeros(len(data) - N + 1)
    
    for startPoint in range(len(data) - N + 1):
        endPoint = startPoint + N
        averageData[startPoint] = np.average(data[startPoint:endPoint])
        stdDeviations[startPoint] = np.std(data[startPoint:endPoint])
    
    return averageData, stdDeviations

def movingAverageFilterConvolve(data, N):
    averageData = np.zeros(len(data) - N + 1)
    stdDeviations = np.zeros(len(data) - N + 1)
    
    # convolve data with 1/N array
    averageData = np.convolve(data, np.ones(N)/N, mode='valid')
    stdDeviations = np.convolve(data**2, np.ones(N)/N, mode='valid') - averageData**2
    # calculate std deviation
    return averageData, np.sqrt(stdDeviations)
    
    
@jit(nopython=True, cache=True)
def FUNC_DELAY(timeseries, time_delay, fs):
    ''' Function to delay a time series f(t) of length N by the instantaneous time delay T(t) of length N (i.e. f(t-T(t))). '''
    # Run if timeseries is a real vector    
    N = len(timeseries)
    t = np.arange(N)/fs
    delayed_timeseries = np.zeros(N)  # Create a new array to store the delayed time series
    
    for i in range(N):
        if time_delay[i] > 0:
            t_delayed = t[i] - time_delay[i]
            if t_delayed < 0 or t_delayed > t[-1]:  # Handle boundary conditions
                delayed_timeseries[i] = 0
            else:
                delayed_timeseries[i] = np.interp(t_delayed, t, timeseries)
        else:
            delayed_timeseries[i] = timeseries[i]

    return delayed_timeseries
    
def FUNC_DELAY_COMPLEX(timeseries, time_delay, fs):
    ''' Function to delay a time series f(t) of length N by the instantaneous time delay T(t) of length N (i.e. f(t-T(t))). '''
    # Run if timeseries is a complex vector
    real_delay = FUNC_DELAY(np.real(timeseries), time_delay, fs)
    imag_delay = FUNC_DELAY(np.imag(timeseries), time_delay, fs)
    delayed_timeseries = real_delay + 1j*imag_delay
    return delayed_timeseries



@jit(nopython=True, cache=True)
def PLL_WL(dataIQ, addData, P, I, I2):
    ''' Function for Phase Locked Loop (PLL) for IQ data (in complex form, i.e. I+jQ). AddData is additive noise added to error signal. '''
    N = len(dataIQ)
    adjustment = np.zeros(N, dtype=np.complex64)
    phase_error = np.zeros(N)
    freq_error = np.zeros(N)
    phaseDiff = np.zeros(N)
    phase = 0
    sum_error = 0
    sum_sum_error = 0  
    for i in range(N):
        adjustment[i] = dataIQ[i] * np.exp(-1j*phase)
        error = np.imag(adjustment[i]) 
        error += addData[i]# Likely a better way of doing this
        phaseDiff[i] = error
        sum_error += error
        sum_sum_error += sum_error
        freq_error[i]  = P * error + I * sum_error + I2 * sum_sum_error
        phase += 2*np.pi*freq_error[i]
        phase_error[i] = phase/(2*np.pi)
    return adjustment, phase_error, phaseDiff
    
def fftnoise(f):
    ''' Function to generate noise with a given frequency spectrum. '''
    f = np.array(f, dtype='complex')
    Np = (len(f) - 1) // 2
    phases = np.random.rand(Np) * 2 * np.pi
    phases = np.cos(phases) + 1j * np.sin(phases)
    f[1:Np+1] *= phases
    f[-1:-1-Np:-1] = np.conj(f[1:Np+1])
    return np.fft.ifft(f).real

def complex_fftnoise(f):
    ''' Function to generate noise with a given frequency spectrum. '''
    f = np.array(f, dtype='complex')
    Np = (len(f) - 1) // 2
    phases = np.random.rand(Np) * 2 * np.pi
    phases = np.cos(phases) + 1j * np.sin(phases)
    f[1:Np+1] *= phases
    f[-1:-1-Np:-1] = f[1:Np+1]
    return np.fft.ifft(f) * 2

def get_single_freqs(samples, fs):
    ''' Function to get the single sided frequency array. Does not include 0 Hz.'''
    return np.fft.fftfreq(samples, 1/fs)[1:samples//2]

def band_limit_freqs(fs, psd, cut_off, debug = False):
    ''' Function to return the band limited noise frequencies. For now just setting to zero. Though will eventually try apply a roll off or something. '''
    #idx = np.where(np.logical_or(freqs<min_freq, freqs>max_freq))[0]
    #psd[idx] = 0
    
    b, a = signal.butter(3, cut_off, 'low', fs = fs)
    w, h = signal.freqz(b, a, worN = len(psd))
    if(debug):
        # convert w to frequency
        f = w * fs / (2*np.pi)
        plt.figure()
        plt.plot(f, 20*np.log10(np.abs(h)))
        plt.xscale('log')
        # plt.show()
    adjusted_psd = psd * np.abs(h)**2
    return adjusted_psd
    
def psdnoise(psd, samples, fs = 1, complex = False):
    ''' Function to generate noise with a given single sided power spectral density. See get_single_freqs for freqs.'''
    # psd = (1/(fs*N))*abs(fft)**2
    single_fft = np.sqrt(fs*samples/2 * psd) # single sided fft
    #single_fft = fs*samples/2 * np.sqrt(freqs) # single sided fft
    double_fft = np.fft.fftfreq(samples, 1/fs) # frequency array for all frequencies
    double_fft[1:samples//2] = single_fft
    double_fft[samples//2+1:] = single_fft[::-1]
    # Set 0 Hz to 0 and fs/2 to 0
    #double_fft[0] = 0
    double_fft[samples//2] = 0
    # divide double_fft by 2 as it is not a single sided fft
    if complex:
        return complex_fftnoise(double_fft)
    else:
        return fftnoise(double_fft)

def white_psd_noise(level, samples, fs = 1):
    ''' Function to generate white noise with a given power spectral density. '''
    f = get_single_freqs(samples, fs)
    psd = np.ones_like(f) * level
    return psdnoise(psd, samples, fs), psd

def band_limited_noise(min_freq, max_freq, samples=1024, samplerate=1):
    ''' Function to generate band-limited noise. Parameters: min_freq, max_freq, samples, samplerate. '''
    freqs = np.abs(np.fft.fftfreq(samples, 1/samplerate))
    f = np.zeros(samples)
    idx = np.where(np.logical_and(freqs>=min_freq, freqs<=max_freq))[0]
    f[idx] = 1
    return fftnoise(f)

def IQData_AdditiveNoise(PhaseSig, PhaseNoisePSD, AdditiveNoisePSD, A, f0, N, fs):
    ''' Constructs IQ Test signal of the form. V(t) = A * cos(2 * pi * f0 * t + phi(t)) + sqrt(2) * AdditiveNoise(t). Sqrt(2) factor was found to be needed for correct scaling. 
        Additive noise and phase noise psd are in rad^2/Hz, the additive noise psd taken in is the final phase measurement! Not the psd of the actual additive noise. The additive 
        noise is scaled by A^2 to get it to be additive. '''
    # if AdditiveNoisePSD is a scaler create white noise psd
    if np.isscalar(AdditiveNoisePSD):
        freqs = get_single_freqs(N, fs)
        AdditiveNoisePSD = np.ones_like(freqs) * AdditiveNoisePSD
    if np.isscalar(PhaseNoisePSD):
        freqs = get_single_freqs(N, fs)
        PhaseNoisePSD = np.ones_like(freqs) * PhaseNoisePSD
    AdditiveNoiseRealPSD = A ** 2 * AdditiveNoisePSD / 2
    AdditiveNoise = psdnoise(AdditiveNoiseRealPSD, N, fs)
    PhaseNoise = psdnoise(PhaseNoisePSD, N, fs)
    PhaseSig += PhaseNoise
    tsim = np.arange(N) / fs
    IQData = phaseToIQ(PhaseSig, tsim, f0, A, additiveNoise = AdditiveNoise * np.sqrt(2), unit='rad')


    return tsim, IQData, AdditiveNoise, PhaseNoise

# Code to run if main
if __name__ == '__main__':

    # Test the delay function
    fs = 10000  # Sampling frequency
    t = 1000
    N = fs*t
    # create frequency array
    f = get_single_freqs(N, fs)
    psd = np.ones(len(f))/f
    noise = psdnoise(psd, N, fs)
    plt.figure()
    plt.plot(noise)
    plt.show()

    from scipy.signal import welch
    f, Pxx = welch(noise, fs, nperseg=1024)
    plt.figure()
    plt.plot(f, Pxx)
    plt.xscale('log')
    plt.yscale('log')
    plt.show()

    # timeseries = np.sin(2 * np.pi * 0.5 * np.arange(fs) / 100)  # Example sine wave
    # time_delay = 0.01 * np.sin(2 * np.pi * 0.1 * np.arange(fs) / 100)  # Example delay
    

    # delayed_timeseries = FUNC_DELAY(timeseries, -time_delay, fs)
    # plt.figure()
    # plt.plot(timeseries, label='Original')
    # plt.plot(delayed_timeseries, label='Delayed')
    # plt.show()
    # # Test the PLL TF function
    # P = 2**-8
    # I = 2**-17
    # I2 = 2**-30
    # fs = 250e6
    # f, P_GAIN, I_GAIN, I2_GAIN, LOOP_GAIN, FORWARD_LOOP_TF, ERROR_TF = PLL_TF({'P':P, 'I':I, 'I2':I2}, fs)
    
    # fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True)
    # ax0.plot(f, 20*np.log10(np.abs(P_GAIN)), label = 'P')
    # ax0.plot(f, 20*np.log10(np.abs(I_GAIN)), label = 'I')
    # ax0.plot(f, 20*np.log10(np.abs(I2_GAIN)), label = 'I2')

    # unity_gain_freq = f[np.argmin(np.abs(np.abs(P_GAIN) - 1))]
    # ax0.axvline(unity_gain_freq, color='r', linestyle='--')

    # ax0.legend()
    # ax0.set_title('LOOP GAINs Magnitude Response')
    # ax0.set_xscale('log')
    # ax0.set_yscale('linear')
    # ax0.set_ylabel('Magnitude (dB)')
    # ax0.grid()
    # ax1.plot(f, np.angle(P_GAIN))
    # ax1.plot(f, np.angle(I_GAIN))
    # ax1.plot(f, np.angle(I2_GAIN))
    # ax1.set_title('LOOP GAIN Phase Response')
    

    # # Plot LOOP GAIN magnitude and phase, and show unity gain frequency and phase margin
    # fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True)
    # ax0.plot(f, 20*np.log10(np.abs(LOOP_GAIN)))
    # ax0.set_title('LOOP GAIN Magnitude Response')
    # ax0.set_xscale('log')
    # ax0.set_yscale('linear')
    # ax0.set_ylabel('Magnitude (dB)')
    # ax1.plot(f, np.angle(LOOP_GAIN))
    # ax1.set_title('LOOP GAIN Phase Response')
    
    # # Find unity gain frequency and phase margin
    # unity_gain_freq = f[np.argmin(np.abs(np.abs(LOOP_GAIN) - 1))]
    # phase_margin = np.angle(LOOP_GAIN[np.argmin(np.abs(f - unity_gain_freq))])
    # # Draw unity gain frequency line and phase margin line
    # ax0.axvline(unity_gain_freq, color='r', linestyle='--')
    # ax1.axvline(unity_gain_freq, color='r', linestyle='--')
    # ax1.axhline(phase_margin, color='r', linestyle='--')
    # ax1.axhline(- np.pi, color='r', linestyle='--')
    # # print phase margin in degrees
    # print('Unity Gain Frequency: ', unity_gain_freq)
    # print('Phase Margin: ', 180 + phase_margin*180/np.pi)

    # # Plot Magnitude and Phase response in subplots vertically
    # fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True)
    # ax0.plot(f, 20*np.log10(np.abs(FORWARD_LOOP_TF)))
    # ax0.set_title('Forward Loop Magnitude Response')
    # ax0.set_xscale('log')
    # ax0.set_yscale('linear')
    # ax0.set_ylabel('Magnitude (dB)')
    # ax1.plot(f, np.angle(FORWARD_LOOP_TF))
    # ax1.set_title('Phase Response')
    # ax1.set_xscale('log')
    # ax1.set_yscale('linear')
    # ax1.set_xlabel('Frequency (Hz)')
    # ax0.grid()
    # ax1.grid()

    # fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True)
    # ax0.plot(f, 20*np.log10(np.abs(ERROR_TF)))
    # ax0.set_title('Error Loop Magnitude Response')
    # ax0.set_xscale('log')
    # ax0.set_yscale('linear')
    # ax0.set_ylabel('Magnitude (dB)')
    # ax1.plot(f, np.angle(ERROR_TF))
    # ax1.set_title('Phase Response')
    # ax1.set_xscale('log')
    # ax1.set_yscale('linear')
    # ax1.set_xlabel('Frequency (Hz)')
    # ax0.grid()
    # ax1.grid()

    # # plot the error response

    # plt.show()

