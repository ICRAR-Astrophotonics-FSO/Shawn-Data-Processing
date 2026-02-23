import numpy as np
import json
import datetime
import matplotlib.pyplot as plt
import allantools as AT
from scipy import signal

SMALL_SIZE = 12
MEDIUM_SIZE = 12
BIGGER_SIZE = 12
TITLE_PAD = 12
linewidth = 1
linewidthCarrier = 0.5

plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)  # legend fontsize

plt.rcParams["font.family"] = "serif"

docWidth = 379 * 1.5 # pt
docHeight = 200 * 1.5 # pt

# convert to inches for matplotlib
docWidthInches = docWidth / 72
docHeightInches = docHeight / 72
print(f"Document size: {docWidthInches} x {docHeightInches} inches")



def calculatePropDelay(distance, refIndex = 1):
    # Calculate the propagation delay based on the distance and reference index
    c = 299792458 / refIndex
    propDelay = distance / c
    return propDelay

def calculateDistance(delay, refIndex = 1):
    # Calculate the distance based on the delay and reference index
    c = 299792458 / refIndex
    distance = delay * c
    return distance

def estimateNFromApriori(syntheticDelay, frequency, prior):
    delta = np.average(prior) - np.average(syntheticDelay)
    cycles = delta * frequency
    print("Number of cycles: ", cycles)
    n = int(np.round(cycles))
    # print("Estimated n: ", n)
    return n

def estimateNextNFromPriorSynthetic(PHASE1, N1, PHASE2, F1, F2, floor = 1):
    N2 = (F2/F1) * (np.median(PHASE1)+N1) - np.median(PHASE2)
    # print("N2 before rounding: ", N2)
    N2 = int(np.round(N2 * floor) / floor)
    # print("Estimated N2: ", N2)
    return N2

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
    
def getDeltas(t, measurementID, phase, settleSamps=0, minJump = 1, max_valid=None):
    oldPhase = 0
    phase = phase - phase[-1]
    _, start_idxs = np.unique(measurementID, return_index=True)
    start_idxs = np.append(start_idxs, len(measurementID))  # for slicing
    if max_valid is None:
        max_valid = len(start_idxs) - 1  # prevents out-of-bounds on start_idxs[mID + 1]
    deltas = np.zeros(max_valid-2)
    timeArray = np.zeros(max_valid-2)
    for mID in range(0, max_valid):
        start, end = start_idxs[mID], start_idxs[mID + 1]
        start += settleSamps  # skip initial samples to avoid settling time
        length = end - start
        if length == 0:
            print(f"Measurement ID {mID} has no data, skipping.")
            continue
        # cPhase = phase[start + length // 2]
        timeArray[mID-2] = t[start ]
        cPhase = np.median(phase[start:end])
        # check cPhase is not NaN
        if np.isnan(cPhase):
            print(f"Measurement ID {mID} has NaN phase, skipping.")
            cPhase = oldPhase
        delta = cPhase - oldPhase
        # print(f"Measurement ID: {mID}, Delta: {delta}")
        oldPhase = cPhase
        deltas[mID-2] = delta
        # print(f"Measurement ID: {mID}, Delta: {delta}, Fractional Delta: {fracDelta}, Scale: {scale}")
    # deltas, outlier_idxs = CL.reject_average_outliers(deltas, m=5)
    deltas_carrier = np.round(deltas / minJump) * minJump
    accumulated_carrier = np.cumsum(deltas_carrier)
    phase_copy = np.copy(phase)
    phase_averaged = np.zeros(max_valid-2)
    for mID in range(0, max_valid):
        # correct phase for accumulated carrier
        start, end = start_idxs[mID], start_idxs[mID + 1]
        length = end - start
        if length == 0:
            print(f"Measurement ID {mID} has no data, skipping correction.")
            continue
        phase_copy[start:end] -= accumulated_carrier[mID-2]
        avg = np.median(phase_copy[start:end])
        if np.isnan(avg):
            print(f"Measurement ID {mID} has NaN phase, skipping averaging.")
            avg = phase_averaged[mID-3]
            phase_copy[start:end] = avg  # fill with previous average
        phase_averaged[mID-2] = avg
        # fill settling samps with average of the last samples
        phase_copy[start:start + settleSamps] = np.mean(phase_copy[start+settleSamps:end])    

    return deltas, timeArray, phase_copy, phase_averaged 

def readBinFile(filename):
    f = open(filename, "rb")
    data = np.fromfile(f, dtype=np.int64)
    f.close()

    json_file = filename.replace(".bin", ".json")
    with open(json_file, "r") as f:
        params = json.load(f)

    # Access values
    down_sample = params["down_sample"]
    hold_time = params["holdTimes"]
    EOM_frequency = np.array(params["freqs"])
    print(f"Down sample: {down_sample}")
    print(f"Hold time: {hold_time}")
    Fs = 307.2e6 / 2**(down_sample+1)
    print(f"Sampling Frequency: {Fs} Hz")
    timestamp = params["timestamp"] 
    # Convert UTC timestamp to local time
    utc_dt = datetime.datetime.fromisoformat(timestamp)
    local_dt = utc_dt.astimezone()
    print(f"UTC Timestamp: {utc_dt}")
    print(f"Local Timestamp: {local_dt}")

    NUM_CHANNELS = 16

    measurementCombination = data[8::NUM_CHANNELS]
    measurementID = measurementCombination & 0xFFFFFFFF
    rstFlag = (measurementCombination >> 32) & 0xFFFFFFFF
    numMeasurements = measurementID[-1]
    print(f"Number of measurements: {numMeasurements}")

    dataTrim = 0
    COUNTER = data[0::NUM_CHANNELS]
    phase0 = data[1::NUM_CHANNELS] / 2 ** 32 * 2**5 # CH1 LSB
    phase1 = data[2::NUM_CHANNELS] / 2 ** 32 * 2**5 # CH1 CENT
    phase2 = data[3::NUM_CHANNELS] / 2 ** 32 * 2**5# CH1 USB
    phase3 = data[4::NUM_CHANNELS] / 2 ** 32 * 2**5 # CH2 LSB
    phase4 = data[5::NUM_CHANNELS] / 2 ** 32 * 2**5 # CH2 CENT
    phase5 = data[6::NUM_CHANNELS] / 2 ** 32 * 2**5 # CH2 USB

    COUNTER = COUNTER[dataTrim:]
    phase0 = phase0[dataTrim:]
    phase1 = phase1[dataTrim:]
    phase2 = phase2[dataTrim:]
    phase3 = phase3[dataTrim:]
    phase4 = phase4[dataTrim:]
    phase5 = phase5[dataTrim:]
    measurementID = measurementID[dataTrim:]  

    CH1_LSB = phase0
    CH1_CENT = phase1
    CH1_USB = phase2
    CH2_LSB = phase5 # Swapped as per theory
    CH2_CENT = phase4
    CH2_USB = phase3 # Swapped as per theory
    t = np.arange(len(measurementID)) / Fs
    SWI_COMBINATION = (CH2_USB - CH2_LSB) + (CH1_USB - CH1_LSB)
    return t, Fs, measurementID, rstFlag, numMeasurements, \
        COUNTER, CH1_LSB, CH1_CENT, CH1_USB, CH2_LSB, CH2_CENT, CH2_USB, SWI_COMBINATION, EOM_frequency

def continuous_phase(fn, tLow, tHigh, slips = None, deltas = None, verbose=False, returnAll = False):
    t, Fs, measurementID, rstFlag, numMeasurements, \
            COUNTER, CH1_LSB, CH1_CENT, CH1_USB, CH2_LSB, CH2_CENT, CH2_USB, SWI_COMBINATION, EOM_frequency = readBinFile(fn)

    
    CH1_LSB = CH1_LSB[(t >= tLow) & (t <= tHigh)]
    CH1_CENT = CH1_CENT[(t >= tLow) & (t <= tHigh)]
    CH1_USB = CH1_USB[(t >= tLow) & (t <= tHigh)]
    CH2_LSB = CH2_LSB[(t >= tLow) & (t <= tHigh)]
    CH2_CENT = CH2_CENT[(t >= tLow) & (t <= tHigh)]
    CH2_USB = CH2_USB[(t >= tLow)   & (t <= tHigh)]
    SWI_COMBINATION = SWI_COMBINATION[(t >= tLow) & (t <= tHigh)]
    t = t[(t >= tLow) & (t <= tHigh)]

    if slips is not None and deltas is not None:
        for slip, delta in zip(slips, deltas):
            if verbose:
                print(f"Slip: {slip}, Delta: {delta}")
            tIdx = np.where((t >= slip))[0]
            SWI_COMBINATION[tIdx] -= delta
            # fill 0.02s from the slip with average of the previous and next 0.02s
            slipStart = tIdx[0]
            slipEnd = slipStart + int(0.04 * Fs)
            slip_prev_002s = SWI_COMBINATION[slipStart - int(0.02 * Fs): slipStart]
            slip_next_002s = SWI_COMBINATION[slipEnd: slipEnd + int(0.02 * Fs)]
            SWI_COMBINATION[slipStart: slipEnd] = (np.mean(slip_prev_002s) + np.mean(slip_next_002s)) / 2

    if verbose:
        
        plt.figure(figsize=(docWidthInches, docHeightInches))
        plt.plot(t, CH1_LSB, label='CH1 LSB')
        plt.plot(t, CH2_LSB, label='CH2 LSB')
        plt.plot(t, CH1_USB, label='CH1 USB')
        plt.plot(t, CH2_USB, label='CH2 USB')
        plt.plot(t, SWI_COMBINATION, label='SWI COMBINATION')
        plt.xlabel('Time (s)')
        plt.ylabel('Phase (cyc)')
        plt.legend(loc = 'upper right')
        plt.title('Phase Measurements with Window Open')
        
        plt.figure(figsize=(docWidthInches, docHeightInches))
        nperseg = 1024 * 1024 * 32
        f, Pxx_phase0 = signal.welch(CH1_LSB, fs=Fs, nperseg=nperseg)
        f, Pxx_phase1 = signal.welch(CH2_LSB, fs=Fs, nperseg=nperseg)
        f, Pxx_phase2 = signal.welch(CH1_USB, fs=Fs, nperseg=nperseg)
        f, Pxx_phase3 = signal.welch(CH2_USB, fs=Fs, nperseg=nperseg)
        fSWI, Pxx_SWI = signal.welch(SWI_COMBINATION, fs=Fs, nperseg=nperseg)
        plt.plot(f, Pxx_phase0, label="Phase 0 (CH1 LSB)")
        plt.plot(f, Pxx_phase1, label="Phase 2 (CH2 LSB)")
        plt.plot(f, Pxx_phase2, label="Phase 3 (CH1 USB)")
        plt.plot(f, Pxx_phase3, label="Phase 5 (CH2 USB)")
        plt.plot(fSWI, Pxx_SWI, label="SWI Combination", linewidth=linewidth)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Power Spectral Density (cyc$^2$/Hz)")
        plt.legend(loc='upper right')
        plt.xscale('log')
        plt.yscale('log')
        # show every y tick from 1e-6 to 1e3
        yticks = 10.0**np.arange(-9, 4, 1)
        plt.yticks(yticks)
        # plt.ylim(1e-9, 1e3)

        plt.minorticks_on()
        plt.grid(which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.show()
    if not returnAll:
        return t, Fs, SWI_COMBINATION
    else:
        return t, Fs, measurementID, rstFlag, numMeasurements, \
            COUNTER, CH1_LSB, CH1_CENT, CH1_USB, CH2_LSB, CH2_CENT, CH2_USB, SWI_COMBINATION, EOM_frequency

def reset_phase(fn, settleSamps = 0, verbose = False):
    t, Fs, measurementID, rstFlag, numMeasurements, \
            COUNTER, CH1_LSB, CH1_CENT, CH1_USB, CH2_LSB, CH2_CENT, CH2_USB, SWI_COMBINATION, EOM_FREQUENCY = readBinFile(fn)
    minJump = 1
    diff = SWI_COMBINATION
    deltas, mT, corrected, averaged = getDeltas(t, measurementID, diff, settleSamps=settleSamps, minJump=minJump)
    print(f"Number of measurements: {len(deltas)}")

    carrier_distance = - 1542e-9 * (CH2_CENT + CH1_CENT) / 2
    # Get the start indices of each segment
    _, start_idxs = np.unique(measurementID, return_index=True)

    # Take the first value of each segment
    carrier_estimates = carrier_distance[start_idxs]
    carrier_estimates = carrier_estimates[2:]  # align with deltas

    c = 299792458
    F = 4 * np.array(EOM_FREQUENCY)[0]  # 2 * (wE1 + wE2)
    swi_dist = averaged * c / F
    if verbose:
        plt.figure(figsize=(docWidthInches, docHeightInches))
        plt.plot(t, diff, label='SWI COMBINATION')
        plt.plot(mT, deltas, label='Deltas')
        plt.plot(t, corrected, label='Corrected SWI COMBINATION')
        plt.plot(mT, averaged, label='Averaged SWI COMBINATION')
        plt.xlabel('Time (s)')
        plt.ylabel('Phase (cyc)')
        plt.legend(loc='upper right')
        plt.show()
    residual = swi_dist - carrier_estimates
    return t, mT, Fs, diff, corrected, averaged, swi_dist, carrier_estimates, residual

def reset_ranging(fn, fiberLength = 12, freeSpaceLength = 1, beSmart = False, floor = 1, neg = 1, settleSammps = 10):
    """ Applies iterative ranging algorithm (integer cycle calculations). Be smart. Can use multiple wavelength steps or just one. 
    Takes in apriori. Has checks. Review code for MSTAR paper 2, use that as inspo. But can try make it better."""
    c = 299792458
    fiberDelay = calculatePropDelay(fiberLength, refIndex=1.467)
    freeSpaceDelay = calculatePropDelay(freeSpaceLength, refIndex=1)
    totDelay = fiberDelay + freeSpaceDelay
    freeSpaceDistance = calculateDistance(totDelay, refIndex = 1)
    print("Total Free Space Distance: ", freeSpaceDistance)
    t, Fs, measurementID, rstFlag, numMeasurements, \
            COUNTER, CH1_LSB, CH1_CENT, CH1_USB, CH2_LSB, CH2_CENT, CH2_USB, SWI_COMBINATION, EOM_FREQUENCY = readBinFile(fn)
    

    carrierEstimate = -1542e-9 * (CH2_CENT + CH1_CENT) / 2

    _, start_idxs = np.unique(measurementID, return_index=True)
    start_idxs = np.append(start_idxs, len(measurementID))  # for slicing
    max_valid = len(start_idxs)  # prevents out-of-bounds on start_idxs[mID + 1]
    numFrequencies = len(EOM_FREQUENCY)
    max_valid = (max_valid // numFrequencies) - 2
    deltas = np.zeros((numFrequencies, max_valid))

    timeArray = np.zeros((numFrequencies, max_valid))
    estimates = np.zeros((numFrequencies, max_valid))
    residuals = np.zeros((numFrequencies, max_valid))
    estimateSTD = np.zeros((numFrequencies, max_valid))
    N = estimates.copy()
    Nc = 0 # current integer
    Np = 0 # previous integer
    Fc = 0
    Fp = 0
    synthFrequencies = 4 * np.array(EOM_FREQUENCY) # 2 * (wE1 + wE2)
    measurementLength = int(start_idxs[1] - start_idxs[0]) - settleSammps
    currentSynthetic = np.zeros(measurementLength)
    previousSynthetic = np.zeros(measurementLength)
    prevDelay = 0

    Nc0 = 0
    Np0 = 0
    Nc1 = 0
    Np1 = 0
    Fc = 0
    Fp = 0

    for mID in range(0, max_valid):
        try:
            start, end = start_idxs[mID]+settleSammps, start_idxs[mID + 1]
        except IndexError:
            start = start_idxs[mID]
            end = len(measurementID)
            
        currentSynthetic = SWI_COMBINATION[start:end]
        if mID % 2 == 0:
            Nc0 = int(np.round(np.median(currentSynthetic[0] - Np0)))
            if beSmart:
                if np.abs(Nc0 + Np0) > 2:
                    Nc0 = Np0
            print(f"Measurement {mID}, Initial Nc0: {Nc0}, Np0: {Np0}")
            currentSynthetic = currentSynthetic - Nc0
            Np0 = np.median(currentSynthetic)
        else:
            Nc1 = int(np.round(np.median(currentSynthetic[0] - Np1)))
            print(f"Measurement {mID}, Initial Nc1: {Nc1}, Np1: {Np1}")
            currentSynthetic = currentSynthetic - Nc1
            Np1 = np.median(currentSynthetic)
            if beSmart:
                if np.abs(Nc0 + Np0) > 2:
                    Nc0 = Np0
        
        currentFrequency = mID % numFrequencies
        Fp = Fc
        Fc = synthFrequencies[currentFrequency]
        
        print(" Frequency, ", Fc)
        
        SYNTHETIC_DELAY = neg * currentSynthetic / Fc
        print("Estimating run ", mID)
        
        if mID == 0:
            Nc = estimateNFromApriori(SYNTHETIC_DELAY, Fc, prior=totDelay)
            print("Initial Nc: ", Nc)
            N[currentFrequency, mID] = Nc
        else:
            Nc = estimateNextNFromPriorSynthetic(previousSynthetic, Np, currentSynthetic, Fp, Fc, floor = floor)
            N[currentFrequency, mID] = Nc
        
        SYNTHETIC_DELAY += Nc / Fc
        prevDelay = SYNTHETIC_DELAY
        SYNTHETIC_DISTANCE = SYNTHETIC_DELAY * c
        carrierDist = np.average(carrierEstimate[start:end])
        residuals[currentFrequency, mID] = np.average(SYNTHETIC_DISTANCE) - carrierDist
         # Store values
        timeArray[currentFrequency, mID] = t[start]
        estimates[currentFrequency, mID] = np.mean(SYNTHETIC_DISTANCE)
        print("Estimated ", np.mean(SYNTHETIC_DISTANCE), " at ", t[start])
        estimateSTD[currentFrequency, mID] = np.std(SYNTHETIC_DISTANCE)
        Np = Nc
        previousSynthetic = currentSynthetic
        Fp = Fc
        # check if latest estimate is Nan, if so stop
        if np.isnan(estimates[currentFrequency, mID]):
            print("Latest estimate is NaN, stopping.")
            break
    return timeArray, estimates, estimateSTD, N, t, carrierEstimate, residuals

def reset_ranging_fast(fn, fiberLength=12, freeSpaceLength=1,
                       beSmart=False, floor=1, neg=1, settleSammps=10):
    c = 299792458
    fiberDelay = calculatePropDelay(fiberLength, refIndex=1.467)
    freeSpaceDelay = calculatePropDelay(freeSpaceLength, refIndex=1)
    totDelay = fiberDelay + freeSpaceDelay
    freeSpaceDistance = calculateDistance(totDelay, refIndex=1)
    print("Total Free Space Distance: ", freeSpaceDistance)

    t, Fs, measurementID, rstFlag, numMeasurements, \
        COUNTER, CH1_LSB, CH1_CENT, CH1_USB, CH2_LSB, CH2_CENT, \
        CH2_USB, SWI_COMBINATION, EOM_FREQUENCY = readBinFile(fn)

    carrierEstimate = -1542e-9 * (CH2_CENT + CH1_CENT) / 2

    _, start_idxs = np.unique(measurementID, return_index=True)
    start_idxs = np.append(start_idxs, len(measurementID))
    max_valid = (len(start_idxs) // len(EOM_FREQUENCY)) - 2

    numFrequencies = len(EOM_FREQUENCY)
    synthFrequencies =  np.array(EOM_FREQUENCY)
    print("Synthesis Frequencies: ", synthFrequencies)
    measurementLength = int(start_idxs[1] - start_idxs[0]) - settleSammps

    # Pre-slice all segments into array: [max_valid, seg_len]
    segments = [
        SWI_COMBINATION[start_idxs[i]+settleSammps:start_idxs[i+1]]
        for i in range(max_valid)
    ]


    # Stats arrays
    timeArray   = np.zeros((numFrequencies, max_valid))
    estimates   = np.zeros((numFrequencies, max_valid))
    estimateSTD = np.zeros((numFrequencies, max_valid))
    residuals   = np.zeros((numFrequencies, max_valid))
    N           = np.zeros((numFrequencies, max_valid))

    Np = 0
    previousSynthetic = None
    Fp = 0

    prevMean = 0
    cMean = 0

    WL0 = c / (4 * synthFrequencies[0])
    WL1 = c / (4 * synthFrequencies[1]) if numFrequencies > 1 else WL0
    print(f"Wavelengths: {WL0*1e3} mm, {WL1*1e3} mm")
    for mID in range(max_valid):
        currentSynthetic = segments[mID]
        currentFrequency = mID % numFrequencies
        Fc = 4*synthFrequencies[currentFrequency]
        # print(f"T is {t[start_idxs[mID] + settleSammps]}")
        # history-dependent integer estimate
        if mID == 0:
            Nc = estimateNFromApriori(neg * currentSynthetic / Fc,
                                      Fc, prior=totDelay)
            # Np = Nc
        else:
            try:
                Nc = estimateNextNFromPriorSynthetic(
                    previousSynthetic, Np, currentSynthetic, Fp, Fc, floor=floor
                )
                
            except:
                Nc = Np
        N[currentFrequency, mID] = Nc
        
        # Vectorized delay & distance
        SYNTHETIC_DELAY = neg * currentSynthetic / Fc + Nc / Fc            
        SYNTHETIC_DISTANCE = SYNTHETIC_DELAY * c
        if beSmart:
            if mID % 2:
                cMean = np.mean(SYNTHETIC_DISTANCE)
                if mID > 4:
                    diff = cMean - prevMean
                    if np.abs(diff) > WL0/2:
                        n = int(np.round(diff / WL0))
                        SYNTHETIC_DISTANCE -= n * WL0
                        # print(f"Correcting {n} cycles on WL0")
                    cMean = np.mean(SYNTHETIC_DISTANCE)
                    newDiff = cMean - prevMean
                    if np.abs(newDiff) > WL1/2:
                        n = int(np.round(newDiff / WL1))
                        SYNTHETIC_DISTANCE -= n * WL1
                        # print(f"Correcting {n} cycles on WL1")
                    cMean = np.mean(SYNTHETIC_DISTANCE)
                prevMean = cMean

        carrierDist = np.mean(carrierEstimate[start_idxs[mID] + settleSammps:
                                             start_idxs[mID+1]])

        residuals[currentFrequency, mID] = np.mean(SYNTHETIC_DISTANCE) - carrierDist
        timeArray[currentFrequency, mID] = t[start_idxs[mID] + settleSammps]
        estimates[currentFrequency, mID] = np.mean(SYNTHETIC_DISTANCE)
        estimateSTD[currentFrequency, mID] = np.std(SYNTHETIC_DISTANCE)

        # update priors
        Np, previousSynthetic, Fp = Nc, currentSynthetic, Fc

    return timeArray, estimates, estimateSTD, N, t, carrierEstimate, residuals

def reset_phase_carrier(fn, settleSamps = 0, verbose = False, max_valid=None):
    t, Fs, measurementID, rstFlag, numMeasurements, \
            COUNTER, CH1_LSB, CH1_CENT, CH1_USB, CH2_LSB, CH2_CENT, CH2_USB, SWI_COMBINATION, EOM_FREQUENCY = readBinFile(fn)
    minJump = 1
    diff = SWI_COMBINATION
    deltas, mT, corrected, averaged = getDeltas(t, measurementID, diff, settleSamps=settleSamps, minJump=minJump, max_valid=max_valid)
    print(f"Number of measurements: {len(deltas)}")

    carrier_phase = CH2_CENT + CH1_CENT
    # Get the start indices of each segment
    _, start_idxs = np.unique(measurementID, return_index=True)

    # Take the first value of each segment
    carrier_estimates = carrier_phase[start_idxs]
    carrier_estimates = carrier_estimates[2:]  # align with deltas

    c = 299792458
    F = 4 * np.array(EOM_FREQUENCY)[0]  # 2 * (wE1 + wE2)
    swi_dist = averaged * c / F
    if verbose:
        plt.figure(figsize=(docWidthInches, docHeightInches))
        plt.plot(t, diff, label='SWI COMBINATION')
        plt.plot(mT, deltas, label='Deltas')
        plt.plot(t, corrected, label='Corrected SWI COMBINATION')
        plt.plot(mT, averaged, label='Averaged SWI COMBINATION')
        plt.xlabel('Time (s)')
        plt.ylabel('Phase (cyc)')
        plt.legend(loc='upper right')
        plt.show()
    
    residual = 0#swi_dist - carrier_estimates

    carrier_sum = CH1_USB + CH2_USB + CH1_LSB + CH2_LSB #CH2_USB - CH1_LSB + CH1_USB - CH2_LSB  #+ CH1_USB #+ CH1_USB - CH1_LSB #- CH2_LSB + CH2_USB
    # carrier_sum = CH1_USB + CH2_LSB
    # carrier_sum = CH1_LSB + CH2_USB
    # carrier_sum = CH1_USB + CH2_USB
    carrier_sum = CH1_LSB 
    carrier_sum *= 30.4/307.2  # scale to match SWI combination amplitude
    return measurementID, t, mT, Fs, diff, corrected, averaged, swi_dist, carrier_estimates, residual, carrier_sum