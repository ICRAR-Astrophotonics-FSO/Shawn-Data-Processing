import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
from scipy import integrate
import shutil
import os
from scipy.fft import fft, fftfreq
from numba import jit

def prbs3(code):
    next_bit = ~((code>>2) ^ (code>>1))&0x01
    code = ((code<<1) | next_bit) & 0xFFFFFFFF
    return code, next_bit

def prbs7(code):
    next_bit = ~((code>>6) ^ (code>>5))&0x01
    code = ((code<<1) | next_bit) & 0xFFFFFFFF
    return code, next_bit

def prbs15(code):
    next_bit = ~((code>>14) ^ (code>>13))&0x01
    code = ((code<<1) | next_bit) & 0xFFFFFFFF
    return code, next_bit

@jit(nopython=True)
def PRBS15_GEN_JIT(sample_rate, duration, k, init_freq, freq_var, samp_delay = 0):
    """Generate a PRBS-k waveform using a NCO. 

        Args:
            sample_rate (float): Sample rate (in samples/second)
            duration (float): Duration of the transmission/reception (in seconds)
            k (int): PRBS polynomial order
            init_freq (float): Initial frequency (in Hz)
            freq_var (float): Frequency variation (in Hz)
        returns:
            np.array: PRBS waveform
    """
    NUM_SAMPS = int(sample_rate * duration)

    PRBS_WAVEFORM = np.zeros(NUM_SAMPS)
    prn_bit = 0
    code = 1
    accumulator = 0
    clock_samples = np.zeros(NUM_SAMPS)
    prev_clock = 0
    current_clock = 0
    # Software NCO
    for i in range(NUM_SAMPS):
        prev_clock = current_clock
        clock_samples[i] = 0.5 * (np.sign(np.sin(accumulator)) + 1)
        current_clock = clock_samples[i]
        if(current_clock - prev_clock > 0):
            prn_bit = ~((code>>14) ^ (code>>13))&0x01
            code = ((code<<1) | prn_bit) & 0xFFFFFFFF
        PRBS_WAVEFORM[i] = prn_bit
        accumulator = (accumulator + 2 * np.pi * init_freq / sample_rate + 2 * np.pi * freq_var / sample_rate ) % (2 * np.pi)
    
    t = np.arange(0, NUM_SAMPS) / sample_rate
    return t, clock_samples, PRBS_WAVEFORM

def PRBS7_GEN_JIT(sample_rate, duration, k, init_freq, freq_var, samp_delay = 0):
    """Generate a PRBS-k waveform using a NCO. 

        Args:
            sample_rate (float): Sample rate (in samples/second)
            duration (float): Duration of the transmission/reception (in seconds)
            k (int): PRBS polynomial order
            init_freq (float): Initial frequency (in Hz)
            freq_var (float): Frequency variation (in Hz)
        returns:
            np.array: PRBS waveform
    """
    NUM_SAMPS = int(sample_rate * duration)

    PRBS_WAVEFORM = np.zeros(NUM_SAMPS)
    prn_bit = 0
    code = 1
    accumulator = 0
    clock_samples = np.zeros(NUM_SAMPS)
    prev_clock = 0
    current_clock = 0
    # Software NCO
    for i in range(NUM_SAMPS):
        prev_clock = current_clock
        clock_samples[i] = 0.5 * (np.sign(np.sin(accumulator)) + 1)
        current_clock = clock_samples[i]
        if(current_clock - prev_clock > 0):
            prn_bit = ~((code>>6) ^ (code>>5))&0x01
            code = ((code<<1) | prn_bit) & 0xFFFFFFFF
        PRBS_WAVEFORM[i] = prn_bit
        accumulator = (accumulator + 2 * np.pi * init_freq / sample_rate + 2 * np.pi * freq_var / sample_rate ) % (2 * np.pi)
    
    t = np.arange(0, NUM_SAMPS) / sample_rate
    return t, clock_samples, PRBS_WAVEFORM

def PRBS_GEN(sample_rate, duration, k, init_freq, freq_var, samp_delay = 0):
    """Generate a PRBS-k waveform using a NCO. 

        Args:
            sample_rate (float): Sample rate (in samples/second)
            duration (float): Duration of the transmission/reception (in seconds)
            k (int): PRBS polynomial order
            init_freq (float): Initial frequency (in Hz)
            freq_var (float): Frequency variation (in Hz)
        returns:
            np.array: PRBS waveform
    """
    NUM_SAMPS = int(sample_rate * duration)

    PRBS_WAVEFORM = np.zeros(NUM_SAMPS)
    prn_bit = 0
    code = 1
    accumulator = 0
    clock_samples = np.zeros(NUM_SAMPS)
    prev_clock = 0
    current_clock = 0
    # Software NCO
    for i in range(NUM_SAMPS):
        prev_clock = current_clock
        clock_samples[i] = signal.square(accumulator)
        current_clock = clock_samples[i]

        if(i > samp_delay):
            if(current_clock - prev_clock > 0):
                #print("rising edge")
                if k == 3:
                    code, prn_bit = prbs3(code)
                elif k == 7:
                    code, prn_bit = prbs7(code)
                elif k == 15:
                    code, prn_bit = prbs15(code)
                else:
                    code, prn_bit = prbs15(code)
                #code, prn_bit = prbs15(code)
        PRBS_WAVEFORM[i] = prn_bit

        if isinstance(freq_var, np.ndarray):
            accumulator = (accumulator + 2 * np.pi * init_freq / sample_rate + 2 * np.pi * freq_var[i] / sample_rate ) % (2 * np.pi)
        else:
            accumulator = (accumulator + 2 * np.pi * init_freq / sample_rate + 2 * np.pi * freq_var / sample_rate ) % (2 * np.pi)
    t = np.arange(0, NUM_SAMPS) / sample_rate
    return t, clock_samples, PRBS_WAVEFORM

@jit(nopython=True)
def PFD(REF, NCO, QREF_PREV, QNCO_PREV, REF_PREV, NCO_PREV):
    ###Phase frequency detector see GARDNER
    QREF = 0
    QNCO = 0
    CLR = QREF_PREV & QNCO_PREV
    if(CLR == True):
        QREF = False
    elif(REF == True and REF_PREV == False):
        #print("Rising edge")
        QREF = True
    else:
        QREF = QREF_PREV
    if(CLR == True):
        QNCO = False
    elif(NCO == True and NCO_PREV == False):
        QNCO = True
    else:
        QNCO = QNCO_PREV
        
    return QREF, QNCO ## UP, DOWN

@jit(nopython=True)
def FULL_PRBS_CORRELATE(PRBS_RX, PRBS_TX, corr_step = 1, min_lag = None, max_lag = None):
    ''' Make sure PRBS_RX is the delayed PRBS. PRBS_TX is the reference PRBS.'''
    if(max_lag == None):
        max_lag = len(PRBS_RX)
    if(min_lag == None):
        min_lag = 0
    lags = []
    corrs = []
    for lag in range(min_lag, max_lag, corr_step):
        #print(lag)
        #TX_ROLLED = np.zeros(length)
        TX_ROLLED = np.roll(PRBS_TX, lag)
        #TX_ROLLED[0:lag] = 0
        lags.append(lag)
        corrs.append(np.sum(PRBS_RX * TX_ROLLED))
    
    return lags, corrs


def PRBS_CORRELATE_PARABOLIC_negDelay(fs, fc, PRBS_RX, PRBS_TX, k, delay_guess = 0, delay_range = (-5, 5), CH = "", window = 0.05, corr_step = 1, debug = False, folder_path = None):
    """Correlate a PRBS-k waveform.
        Args:
            PRBS_RX (np.array): PRBS receieved. To be delay matched.
            PRBS_TX (np.array): PRBS transmitted. This is the reference PRBS transmitted.
            window (np.array): Window to apply to the correlation. If None, the entire time series is used for one sliding correlation. Otherwise it breaks the RX PRBS into segments, where each is windowed and correlated with the TX PRBS.
            Want range < delay_guess ideally I reckon, though should be able to work around it
        returns:
            np.array: Correlation
    
    """
    # Pre roll PRBS_TX by delay_guess
    
    samp_length = np.size(PRBS_RX)
    window_length = int(np.floor((2**k-1)*(fs/fc)*window))
    print("Window length ", window_length)
    delay_offset = 0
    if(np.abs(delay_guess) > window_length):
        print(f"Delay guess is greater than window length, pre rolling. No zero filling at the moment, otherwise NaNs. Rolling by {delay_guess}")
        PRBS_TX = np.roll(PRBS_TX, delay_guess)
        PRBS_TX[-delay_guess:] = 0
        delay_offset = delay_guess
        delay_guess = 0
        
    plt.figure()
    plt.plot(PRBS_RX, label = "Rx")
    plt.plot(PRBS_TX, label = "Rolled Tx")
    plt.legend(loc = "upper right")
    plt.show()
            
    return 

def PRBS_CORRELATE_PARABOLIC(fs, fc, PRBS_RX, PRBS_TX, k, delay_guess = 0, delay_range = (-5, 5), CH = "", window = 0.05, corr_step = 1, debug = False, folder_path = None):
    """Correlate a PRBS-k waveform.
        Args:
            PRBS_RX (np.array): PRBS receieved. To be delay matched.
            PRBS_TX (np.array): PRBS transmitted. This is the reference PRBS transmitted.
            window (np.array): Window to apply to the correlation. If None, the entire time series is used for one sliding correlation. Otherwise it breaks the RX PRBS into segments, where each is windowed and correlated with the TX PRBS.
            Want range < delay_guess ideally I reckon, though should be able to work around it
        returns:
            np.array: Correlation
    
    """
    # Pre roll PRBS_TX by delay_guess
    
    samp_length = np.size(PRBS_RX)
    window_length = int(np.floor((2**k-1)*(fs/fc)*window))
    delay_offset = 0
    if(delay_guess > window_length):
        print("Delay guess is greater than window length, pre rolling. No zero filling at the moment, otherwise NaNs")
        PRBS_TX = np.roll(PRBS_TX, delay_guess)
        #PRBS_TX[0:delay_guess] = 0
        delay_offset = delay_guess
        delay_guess = 0
        
    print("Window length ", window_length)
    RX_CORR_SAMPS = np.zeros(window_length)
    TX_CORR_SAMPS = np.zeros(window_length)
    NUM_CORRELATIONS = int(np.floor(samp_length/window_length))
    CORR = np.zeros(delay_range[1]-delay_range[0]+1)
    CORRS_INDEX = np.zeros(NUM_CORRELATIONS)
    CORR2 = np.zeros(delay_range[1]-delay_range[0]+1)

    window_num = []    

    for i in range(NUM_CORRELATIONS):
        window_num.append(i)
        RX_CORR_SAMPS = PRBS_RX[i*window_length:(i+1)*window_length]
        #num_shifts = delay_range[1] - delay_range[0] + 1
        for k in range(delay_range[0], delay_range[1]+1):   
            low = i*window_length-delay_guess-k
            if(low < 0 ):
                TX_CORR_SAMPS[:] = np.concatenate((np.zeros(np.abs(low)),PRBS_TX[:(i+1)*window_length-delay_guess-k]))
                low = 0
            else:
                high = (i+1)*window_length-delay_guess-k
                if(high > samp_length):
                    TX_CORR_SAMPS[:] = np.concatenate((PRBS_TX[low:], np.zeros(high-samp_length)))
                    high = samp_length
                else:
                    TX_CORR_SAMPS[:] = PRBS_TX[(i)*window_length-delay_guess-k:high]
            
            CORR[k-delay_range[0]] = np.abs(np.sum(RX_CORR_SAMPS*TX_CORR_SAMPS)) # Doubel check this
            #print(i, k, CORR[k-delay_range[0]])
            
        # CORR = CORR - np.mean(CORR)
        # CORR = np.abs(CORR)
        MAX_CORR = np.max(CORR)
        MAX_INDEX = np.argmax(CORR)
        #print('Max Correlation: ', MAX_CORR, 'at index ', MAX_INDEX, ' Num chips ', MAX_INDEX/125)

        x1 = MAX_INDEX - 1
        x2 = MAX_INDEX
        x3 = MAX_INDEX + 1
        try:
            y1 = CORR[x1]
            y2 = CORR[x2]
            y3 = CORR[x3]
        except:
            # go to next iteration
            print("Ruh Roh")
            continue
        x_mat = np.array([[x1**2, x1, 1],[x2**2, x2, 1],[x3**2, x3, 1]])
        y_mat = np.array([y1,y2,y3])
        a = np.linalg.solve(x_mat, y_mat)

        x_max = -a[1]/(2*a[0])
        actual_delay = (delay_offset + delay_guess + delay_range[0]) + x_max
        print('Parabolic Fit x_max: ', x_max, actual_delay)
        CORRS_INDEX[i] = actual_delay
        
        if(debug):
            fig, axs = plt.subplots(1, 2, figsize = (10,5))
            RX_CORR_SAMPS = PRBS_RX[i*window_length:(i+1)*window_length]
            
            low = i*window_length-delay_guess
            if(low < 0 ):
                TX_CORR_SAMPS[:] = np.concatenate((np.zeros(np.abs(low)),PRBS_TX[:(i+1)*window_length-delay_guess]))
                low = 0
            else:
                high = (i+1)*window_length-delay_guess
                if(high > samp_length):
                    TX_CORR_SAMPS[:] = np.concatenate((PRBS_TX[low:], np.zeros(high-samp_length)))
                    high = samp_length
                else:
                    TX_CORR_SAMPS[:] = PRBS_TX[(i)*window_length-delay_guess:high]
            
            n = np.array(range(window_length))
            # Plot on subplot 0.0
            axs[0].plot(n[-100:], RX_CORR_SAMPS[-100:], label = 'RX')
            axs[0].plot(n[-100:], TX_CORR_SAMPS[-100:], label = f'TX Guess 1 Delayed {delay_guess}')
            # Also plot delay guess 2 PRN if it exists
            plt.legend()
            k = np.array(range(delay_range[0], delay_range[1]+1))
            # offset k array by delay_offset
            k = k + delay_guess + delay_offset
            print(k)
            axs[1].plot(k, CORR, label = 'Coarse Correlation')
            axs[1].set_title("Correlation for window {}".format(i))
            k = np.array(range(delay_range[0], delay_range[1]+1))
            
            # create fractional delay array in delay_range
            NUM_POINTS = 1000
            fine_lags = np.linspace(x1, x3, NUM_POINTS)
            parabolicFit = a[0] * fine_lags ** 2 + a[1] * fine_lags + a[2]
            fine_lags += delay_offset + delay_guess + delay_range[0]
            axs[1].plot(fine_lags, parabolicFit, label = f'Parabolic Fit: Max @ lag {actual_delay}')
            axs[1].legend()
            axs[0].set_title("RX and TX PRBS")
            fig.tight_layout()
            if folder_path is not None:
                fig.savefig(os.path.join(folder_path, f'{CH}_correlation_window_{i}.png'))
                plt.close(fig)
            
    return window_num, CORRS_INDEX

def PRBS_CORRELATE(fs, fc, PRBS_RX, PRBS_TX, k, delay_guess = 0, delay_guess2 = None, XOR = None, delay_range = (-5, 5), window = 0.05, corr_step = 1, debug = False, folder_path = None):
    """Correlate a PRBS-k waveform.
        Args:
            PRBS_RX (np.array): PRBS receieved. To be delay matched.
            PRBS_TX (np.array): PRBS transmitted. This is the reference PRBS transmitted.
            window (np.array): Window to apply to the correlation. If None, the entire time series is used for one sliding correlation. Otherwise it breaks the RX PRBS into segments, where each is windowed and correlated with the TX PRBS.
            Want range < delay_guess ideally I reckon, though should be able to work around it
        returns:
            np.array: Correlation
    
    """
    if(window == None):
        if(folder_path == None):
            folder_path = 'Ettus//.Data//CORR_DUMP'
        try:
            shutil.rmtree(folder_path)
            os.mkdir(folder_path)
        except:
            os.mkdir(folder_path)
        length = np.size(PRBS_RX)
        alen = int(length/corr_step)
        lags = []
        corrs = []
        for lag in range(0, length, corr_step):
            #print(lag)
            #TX_ROLLED = np.zeros(length)
            TX_ROLLED = np.roll(PRBS_TX, lag)
            #TX_ROLLED[0:lag] = 0
            lags.append(lag)
            corrs.append(np.sum(PRBS_RX * TX_ROLLED))
            if(debug):
                plt.figure()
                plt.plot(TX_ROLLED, label = 'tx Rolled')
                plt.plot(PRBS_RX - 0.5, label = 'Rx')
                plt.xlim([length-250, length])
                plt.legend()
                plt.title("Lag {} Corr {}".format(lag, corrs[-1]))
                plt.savefig(folder_path + '//{}_corr.png'.format(lag), dpi=300)
                plt.legend()
                plt.close('all')
            #plt.show()
        return lags, corrs
    else:
        # Pre roll PRBS_TX by delay_guess
        
        samp_length = np.size(PRBS_RX)
        window_length = int(np.floor((2**k-1)*(fs/fc)*window))
        delay_offset = 0
        if(delay_guess > window_length):
            print("Delay guess is greater than window length, pre rolling. No zero filling at the moment, otherwise NaNs")
            PRBS_TX = np.roll(PRBS_TX, delay_guess)
            #PRBS_TX[0:delay_guess] = 0
            delay_offset = delay_guess
            delay_guess = 0

        print("Window length ", window_length)
        RX_CORR_SAMPS = np.zeros(window_length)
        TX_CORR_SAMPS = np.zeros(window_length)
        NUM_CORRELATIONS = int(np.floor(samp_length/window_length))
        CORR = np.zeros(delay_range[1]-delay_range[0]+1)
        CORRS_INDEX = np.zeros(NUM_CORRELATIONS)
        CORR2 = np.zeros(delay_range[1]-delay_range[0]+1)
        CORRS2_INDEX = np.zeros(NUM_CORRELATIONS)

        

        for i in range(NUM_CORRELATIONS):
            RX_CORR_SAMPS = PRBS_RX[i*window_length:(i+1)*window_length]
            #num_shifts = delay_range[1] - delay_range[0] + 1
            for k in range(delay_range[0], delay_range[1]+1):   
                low = i*window_length-delay_guess-k
                if(low < 0 ):
                    TX_CORR_SAMPS[:] = np.concatenate((np.zeros(np.abs(low)),PRBS_TX[:(i+1)*window_length-delay_guess-k]))
                    low = 0
                else:
                    high = (i+1)*window_length-delay_guess-k
                    if(high > samp_length):
                        TX_CORR_SAMPS[:] = np.concatenate((PRBS_TX[low:], np.zeros(high-samp_length)))
                        high = samp_length
                    else:
                        TX_CORR_SAMPS[:] = PRBS_TX[(i)*window_length-delay_guess-k:high]
                
                CORR[k-delay_range[0]] = np.abs(np.sum(RX_CORR_SAMPS*TX_CORR_SAMPS)) # Doubel check this
                #print(i, k, CORR[k-delay_range[0]])

            if(delay_guess2 is not None):
                for k in range(delay_range[0], delay_range[1]+1):
                    low = i*window_length-delay_guess2-k
                    if(low < 0 ):
                        TX_CORR_SAMPS[:] = np.concatenate((np.zeros(np.abs(low)),PRBS_TX[:(i+1)*window_length-delay_guess2-k]))
                        low = 0
                    else:
                        high = (i+1)*window_length-delay_guess2-k
                        if(high > samp_length):
                            TX_CORR_SAMPS[:] = np.concatenate((PRBS_TX[low:], np.zeros(high-samp_length)))
                            high = samp_length
                        else:
                            TX_CORR_SAMPS[:] = PRBS_TX[(i)*window_length-delay_guess2-k:high]
                    CORR2[k-delay_range[0]] = np.abs(np.sum(RX_CORR_SAMPS*TX_CORR_SAMPS))
            if(debug):
                ## Plot RX and TX PRBS
                # Create four subplots
                fig, axs = plt.subplots(2, 2)
                RX_CORR_SAMPS = PRBS_RX[i*window_length:(i+1)*window_length]
                
                low = i*window_length-delay_guess
                if(low < 0 ):
                    TX_CORR_SAMPS[:] = np.concatenate((np.zeros(np.abs(low)),PRBS_TX[:(i+1)*window_length-delay_guess]))
                    low = 0
                else:
                    high = (i+1)*window_length-delay_guess
                    if(high > samp_length):
                        TX_CORR_SAMPS[:] = np.concatenate((PRBS_TX[low:], np.zeros(high-samp_length)))
                        high = samp_length
                    else:
                        TX_CORR_SAMPS[:] = PRBS_TX[(i)*window_length-delay_guess:high]
                


                n = np.array(range(window_length))
                # Plot on subplot 0.0
                axs[0, 0].plot(n, RX_CORR_SAMPS, label = 'RX')
                axs[0, 0].plot(n, TX_CORR_SAMPS, label = f'TX Guess 1 Delayed {delay_guess}')
                # Also plot delay guess 2 PRN if it exists
                if(delay_guess2 is not None):
                    low = i*window_length-delay_guess2
                    if(low < 0 ):
                        TX_CORR_SAMPS[:] = np.concatenate((np.zeros(np.abs(low)),PRBS_TX[:(i+1)*window_length-delay_guess2]))
                        low = 0
                    else:
                        high = (i+1)*window_length-delay_guess2
                        if(high > samp_length):
                            TX_CORR_SAMPS[:] = np.concatenate((PRBS_TX[low:], np.zeros(high-samp_length)))
                            high = samp_length
                        else:
                            TX_CORR_SAMPS[:] = PRBS_TX[(i)*window_length-delay_guess2:high]
                    axs[0,0].plot( TX_CORR_SAMPS, label = f'TX Guess 2 Delayed {delay_guess2}')
                
                plt.legend()
                k = np.array(range(delay_range[0], delay_range[1]+1)) 
                # offset k array by delay_offset
                k = k + delay_guess + delay_offset
                print(k)
                axs[0, 1].plot(k, CORR)
                axs[0, 1].set_title("Correlation for window {}".format(i))
                k = np.array(range(delay_range[0], delay_range[1]+1))
                if(delay_guess2 is not None):
                    k = k + delay_guess2 + delay_offset
                    axs[1, 1].plot(k, CORR2)
                    axs[1, 1].set_title("Correlation for window {}".format(i))

                # Normalise PRBS_RX
                # RX_CORR_SAMPS = np.sign(RX_CORR_SAMPS) + 1
                # XOR RX_CORR_SAMPS with TX_CORR_SAMPS
                # Make RX and TX boolean values for XOR
                # RX_CORR_SAMPS = RX_CORR_SAMPS.astype(bool)
                # TX_CORR_SAMPS = TX_CORR_SAMPS.astype(bool)
                # NEW_PRN = RX_CORR_SAMPS ^ TX_CORR_SAMPS
                # axs[1, 0].plot(NEW_PRN, label = f'Reveres XOR with TX Delayed {delay_guess2}')
                axs[0,0].legend()
                # Share x axis between 0,0 and 1,0
                axs[0, 0].set_title("RX and TX PRBS")
                axs[1, 0].set_title("Reversed XOR")
                axs[1,0].sharex(axs[0,0])
                plt.tight_layout()

                plt.show()
            # CORR = CORR - np.mean(CORR)
            # CORR = np.abs(CORR)
            MAX_CORR = np.max(CORR)
            MAX_INDEX = np.argmax(CORR)
            #print('Max Correlation: ', MAX_CORR, 'at index ', MAX_INDEX, ' Num chips ', MAX_INDEX/125)

            x1 = MAX_INDEX - 1
            x2 = MAX_INDEX
            x3 = MAX_INDEX + 1
            try:
                y1 = CORR[x1]
                y2 = CORR[x2]
                y3 = CORR[x3]
            except:
                
                # go to next iteration
                continue
            x_mat = np.array([[x1**2, x1, 1],[x2**2, x2, 1],[x3**2, x3, 1]])
            y_mat = np.array([y1,y2,y3])
            a = np.linalg.solve(x_mat, y_mat)

            x_max = -a[1]/(2*a[0])
            actual_delay = (delay_offset + delay_guess + delay_range[0]) + x_max
            print('Parabolic Fit x_max: ', x_max, actual_delay)
            CORRS_INDEX[i] = actual_delay
        return CORRS_INDEX

@jit(nopython=True)
def PRBS15_DLL_JIT(PRBS_IN, sample_rate, duration, k, init_freq, loop_delay = 0, prn_delay = 0, pGain = 1, iGain = 0.1):
    """Lock a PRBS-k waveform using a NCO. Starting with a lead-lag phase detector.

        Args:
            sample_rate (float): Sample rate (in samples/second)
            duration (float): Duration of the transmission/reception (in seconds)
            k (int): PRBS polynomial order
            init_freq (float): Initial frequency (in Hz)
            freq_var (float): Frequency variation (in Hz)
            loop_delay : delays loop action by N samples
            prn_delay : delays PRN output by N samples
        returns:
            np.array: PRBS waveform
    """
    NUM_SAMPS = int(sample_rate * duration)
    PRBS_WAVEFORM = np.zeros(NUM_SAMPS)
    LEAD_SAMPLES = np.zeros(NUM_SAMPS)
    LAG_SAMPLES = np.zeros(NUM_SAMPS)
    PULSE_SAMPLES = np.zeros(NUM_SAMPS)
    FREQ_SAMPLES = np.zeros(NUM_SAMPS)

    PD_RST = 0

    prn_bit = 0
    code = 1
    accumulator = 0
    clock_samples = np.zeros(NUM_SAMPS)
    prev_clock = 0
    current_clock = 0

    # Software NCO
    QR_P = False
    QN_P = False
    R_P = False
    N_P = False
    pulse_accumulator = 0
    pulse_accumulator2 = 0
    freq_var = 0
    freq_var_avg = 0
    for i in range(NUM_SAMPS):
        prev_clock = current_clock
        clock_samples[i] = (np.sign(np.sin(accumulator)) + 1)
        current_clock = clock_samples[i]
        if(i > prn_delay and (i+prn_delay)<NUM_SAMPS):
            if(current_clock - prev_clock > 0):
                prn_bit = ~((code>>14) ^ (code>>13))&0x01
                code = ((code<<1) | prn_bit) & 0xFFFFFFFF
            PRBS_WAVEFORM[i+prn_delay] = prn_bit
        if(i > loop_delay):
            # Phase detector
            QREF, QNCO = PFD(PRBS_IN[i], PRBS_WAVEFORM[i], QR_P, QN_P, R_P, N_P)
            QR_P = QREF
            QN_P = QNCO
            R_P = PRBS_IN[i]
            N_P = PRBS_WAVEFORM[i]
            LEAD_SAMPLES[i] = QREF
            LAG_SAMPLES[i] = QNCO
            PULSE = (QREF or QNCO) 
            if(QNCO):
                PULSE_SAMPLES[i] = -PULSE
            else:
                PULSE_SAMPLES[i] = PULSE
            pulse_accumulator = pulse_accumulator + PULSE_SAMPLES[i]

            freq_var = pGain * PULSE_SAMPLES[i] + iGain * pulse_accumulator
            freq_var_avg = freq_var_avg + (freq_var)
            FREQ_SAMPLES[i] = freq_var

        accumulator = (accumulator + 2 * np.pi * init_freq / sample_rate + 2 * np.pi * freq_var / sample_rate ) % (2 * np.pi)

    return PRBS_WAVEFORM, LEAD_SAMPLES, LAG_SAMPLES, PULSE_SAMPLES, FREQ_SAMPLES

def PRBS_DLL(PRBS_IN, pi_param, sample_rate, duration, k, init_freq, samp_delay = 0, prn_delay = 0):
    """Lock a PRBS-k waveform using a NCO. Starting with a lead-lag phase detector.

        Args:
            sample_rate (float): Sample rate (in samples/second)
            duration (float): Duration of the transmission/reception (in seconds)
            k (int): PRBS polynomial order
            init_freq (float): Initial frequency (in Hz)
            freq_var (float): Frequency variation (in Hz)
            samp_delay : delays loop action by N samples
            prn_delay : delays PRN output by N samples
        returns:
            np.array: PRBS waveform
    """
    NUM_SAMPS = int(sample_rate * duration)
    PRBS_WAVEFORM = np.zeros(NUM_SAMPS)
    LEAD_SAMPLES = np.zeros(NUM_SAMPS)
    LAG_SAMPLES = np.zeros(NUM_SAMPS)
    PULSE_SAMPLES = np.zeros(NUM_SAMPS)
    FREQ_SAMPLES = np.zeros(NUM_SAMPS)

    PD_RST = 0

    prn_bit = 0
    code = 1
    accumulator = 0
    clock_samples = np.zeros(NUM_SAMPS)
    prev_clock = 0
    current_clock = 0

    # Software NCO
    QR_P = False
    QN_P = False
    R_P = False
    N_P = False
    pulse_accumulator = 0
    pulse_accumulator2 = 0
    freq_var = 0
    freq_var_avg = 0
    for i in range(NUM_SAMPS):
        prev_clock = current_clock
        clock_samples[i] = signal.square(accumulator)
        current_clock = clock_samples[i]

        if(i > samp_delay):
            if(current_clock - prev_clock > 0):
                #print("rising edge")
                if k == 3:
                    code, prn_bit = prbs3(code)
                elif k == 7:
                    code, prn_bit = prbs7(code)
                elif k == 15:
                    code, prn_bit = prbs15(code)
                else:
                    code, prn_bit = prbs15(code)
                #code, prn_bit = prbs15(code)
            if(i > prn_delay and (i+prn_delay)<NUM_SAMPS):
                PRBS_WAVEFORM[i+prn_delay] = prn_bit

            # Phase detector
            
            QREF, QNCO = PFD(PRBS_IN[i], PRBS_WAVEFORM[i], QR_P, QN_P, R_P, N_P)
            QR_P = QREF
            QN_P = QNCO
            R_P = PRBS_IN[i]
            N_P = PRBS_WAVEFORM[i]
            LEAD_SAMPLES[i] = QREF
            LAG_SAMPLES[i] = QNCO
            PULSE = (QREF or QNCO) 
            if(QNCO):
                PULSE_SAMPLES[i] = -PULSE
            else:
                PULSE_SAMPLES[i] = PULSE
            pulse_accumulator = pulse_accumulator + PULSE_SAMPLES[i]

            freq_var = pi_param[0] * PULSE_SAMPLES[i] + pi_param[1]*pulse_accumulator
            freq_var_avg = freq_var_avg + (freq_var)
            FREQ_SAMPLES[i] = freq_var

        accumulator = (accumulator + 2 * np.pi * init_freq / sample_rate + 2 * np.pi * freq_var / sample_rate ) % (2 * np.pi)

    return PRBS_WAVEFORM, LEAD_SAMPLES, LAG_SAMPLES, PULSE_SAMPLES, FREQ_SAMPLES

@jit(nopython=True)
def COSTAS_JIT(rx_data, sample_rate, P=0.2, I = 0.01, mode='BPSK', saturate=False):
    """Costas Loop for BPSK (order 2) or QPSK (order 4). Cleans up RX IQ data.
    
    Args:
        rx_data (np.array): IQ data
        pi_param (dict): PI parameters {'P': float, 'I': float}
        sample_rate (float): Sample rate (in samples/second)
        mode (str): Modulation mode ('BPSK' or 'QPSK')
        saturate (bool): Flag to saturate the output PRN using np.sign
    Returns:
        tuple: (np.array, np.array) - Adjusted IQ data and PRN signal
    """
    sam_len = len(rx_data)
    adjustment = np.zeros(sam_len, dtype=np.complex64)
    phase_error = np.zeros(sam_len)
    phase = 0
    sum_error = 0
    
    for i in range(sam_len):
        if mode == 'BPSK':
            # BPSK specific processing
            adjustment[i] = rx_data[i] * np.exp(-1j * phase)
            error = np.real(adjustment[i]) * np.imag(adjustment[i])
        elif mode == 'QPSK':
            # QPSK specific processing
            adjustment[i] = (np.real(rx_data[i]) * np.cos(phase) + np.imag(rx_data[i]) * np.sin(phase)) + \
                            1j * (np.imag(rx_data[i]) * np.cos(phase) - np.real(rx_data[i]) * np.sin(phase))
            error = np.imag(adjustment[i]) * np.sign(np.real(adjustment[i])) - np.real(adjustment[i]) * np.sign(np.imag(adjustment[i]))
        
        sum_error += error
        phase = P * error + I * sum_error
        phase_error[i] = phase
    
    # Generate PRN from the adjusted data
    if mode == 'BPSK':
        PRN = np.real(adjustment)
    elif mode == 'QPSK':
        PRN = np.angle(adjustment)
    if saturate:
        PRN = np.sign(PRN)
    
    return adjustment, phase_error, PRN

def COSTAS(rx_data, sample_rate, pi_param = {'P':0.6, 'I':0.2}, mode='BPSK', saturate=False):
    """Costas Loop for BPSK (order 2) or QPSK (order 4). Cleans up RX IQ data.
    
    Args:
        rx_data (np.array): IQ data
        pi_param (dict): PI parameters {'P': float, 'I': float}
        sample_rate (float): Sample rate (in samples/second)
        mode (str): Modulation mode ('BPSK' or 'QPSK')
        saturate (bool): Flag to saturate the output PRN using np.sign
    Returns:
        tuple: (np.array, np.array) - Adjusted IQ data and PRN signal
    """
    P = pi_param['P']
    I = pi_param['I']
    sam_len = len(rx_data)
    adjustment = np.zeros(sam_len, dtype=np.complex64)
    phase_error = np.zeros(sam_len)
    phase = 0
    sum_error = 0
    
    for i in range(sam_len):
        if mode == 'BPSK':
            # BPSK specific processing
            adjustment[i] = rx_data[i] * np.exp(-1j * phase)
            error = np.real(adjustment[i]) * np.imag(adjustment[i])
        elif mode == 'QPSK':
            # QPSK specific processing
            adjustment[i] = (np.real(rx_data[i]) * np.cos(phase) + np.imag(rx_data[i]) * np.sin(phase)) + \
                            1j * (np.imag(rx_data[i]) * np.cos(phase) - np.real(rx_data[i]) * np.sin(phase))
            error = np.imag(adjustment[i]) * np.sign(np.real(adjustment[i])) - np.real(adjustment[i]) * np.sign(np.imag(adjustment[i]))
        
        sum_error += error
        phase = P * error + I * sum_error
        phase_error[i] = phase
    
    # Generate PRN from the adjusted data
    if mode == 'BPSK':
        PRN = np.real(adjustment)
    elif mode == 'QPSK':
        PRN = np.angle(adjustment)
    if saturate:
        PRN = np.sign(PRN)
    
    return adjustment, phase_error, PRN

if __name__ == "__main__":
    sr = 16e6
    k = 15
    init_freq = 4e6

    prn_length = 2**k-1
    prn_samp_len = int(prn_length * sr/init_freq)
    # Frequency chirp
    freq_var = 0 #np.linspace(0, 1000, num_samps)

    d =  0.25 * prn_samp_len / sr

    in_delay = 0
    t, clock, prn = PRBS_GEN(sr, d, k, init_freq, freq_var, in_delay)

    in_delay2 = 100
    t, clock, prn2 = PRBS_GEN(sr, d, k, init_freq, freq_var, in_delay2)

    # plt.figure()
    # plt.plot(PRBS_CORRELATE(sr, init_freq, prn2, prn, k, delay_guess=100, delay_range = (-10, 10)))
    


    # plt.figure()    
    # lags, corr = PRBS_CORRELATE(sr,init_freq, prn2, prn, k = 7, window = None)
    # plt.plot(lags, corr)
    
    # plt.figure()
    # plt.plot(prn2)
    # plt.plot(np.roll(prn, 100))

    plt.figure()
    plt.step(t, prn, label = 'prn')
    plt.step(t, clock/2, label = 'clock')
    plt.legend()

    plt.figure()
    plt.step(t*sr, prn)

    T = 1/sr
    N = len(clock)
    yf = fft(prn)
    xf = fftfreq(N, T)[:N//2]

    plt.figure()
    plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
    plt.yscale('log')
    plt.grid()

    plt.show()
    plt.show()

    # Take FFT




    # plt.figure()
    # plt.plot(lags, correlation)

    # plt.show()
    
    # pi_param = [50, 0.1]

    # loop_delay = 110
    # lock_prn, lead_samples, lag_samples, pulse_samples, freq_samples = PRBS_DLL(prn, pi_param, sr, d, k, init_freq, loop_delay)
    
    # chip_period = 1/init_freq
    # chip_period_samps = chip_period * sr
    # print("Expected delay {} chips".format((loop_delay-in_delay)/chip_period_samps))
    # plt.plot(t, prn, label = 'input PRN')
    # plt.plot(t, lock_prn*2, label = 'lock PRN')
    # plt.plot(t, lead_samples/3, label = 'lead')
    # plt.plot(t, lag_samples/4, label = 'lag')
    # plt.plot(t, pulse_samples/2, label = 'pulse')
    # plt.legend()
    # plt.show()

    # fig, ax = plt.subplots(5, sharex = True)

    # ax[0].set_title("Simulated Phase Detection For Delayed PRBS")
    # ax[0].plot(t,prn)
    # ax[0].set_ylabel("PRBS")
    # ax[1].step(t,lock_prn)
    # ax[1].set_ylabel("Locked PRBS")
    # ax[2].step(t,lead_samples)
    # ax[2].set_ylabel("Lead")
    # ax[3].step(t,lag_samples)
    # ax[3].set_ylabel("lag")
    # ax[3].set_xticks([])
    # ax[4].step(t,pulse_samples)
    # ax[4].set_ylabel("lag")
    # ax[4].set_xticks([])
    # plt.tight_layout()

    # fig, ax = plt.subplots(2, sharex = True)
    # ax[0].step(t,freq_samples)
    # ax[0].set_ylabel("Freq_Error")
    # ax[0].set_xticks([])

    # dec = 50
    # ydem = signal.decimate(freq_samples, dec)
    # xnew = np.linspace(0, d, int(num_samps/dec), endpoint=False)
    # ax[0].step(xnew,ydem)
    # ax[0].set_ylabel("Freq_Error")
    # ax[0].set_xticks([])

    # ax[1].step(t,np.cumsum(freq_samples)/sr)
    # ax[1].set_ylabel("Phase_Error")
    # ax[1].set_xticks([])
    
    # # Take spectrum of prn
    # prn_fft = np.fft.fft(prn)
    # prn_fft = np.fft.fftshift(prn_fft)
    # f = np.fft.fftfreq(len(prn), 1/sr)
    # f = np.fft.fftshift(f)
    # plt.plot(f, np.abs(prn_fft))
    # plt.show()

