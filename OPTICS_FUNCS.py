import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

import PLL_FUNCS as PLL
from scipy.special import jv
import scipy.integrate as integrate

#######################
# This python file contains functions for optical calculations I do regularly. This includes shot noise analysis and the sort.
#######################

# GLOBALS
#RPD = 0.95
Wavelength = 1550e-9

def CalcAdditiveShotPSD(Psig, Wavelength, eta = 1, RPD = None, unit = 'rad'):
    ''' Calculate the additive shot noise phase PSD in cyc^2/Hz or rad^2/Hz (default). Uses either hbar*c/(2pi*eta*lambda*Psig) or qe/(2pi*RPD*Psig).'''
    hbar = 6.62607015e-34
    c = 299792458
    qe = 1.60217663e-19
    scale = 1
    if unit not in ['cyc']:
        scale = (2 * np.pi) ** 2
        print('Using rad^2/Hz')
    if RPD is None:
        return scale * hbar * c / (2 * np.pi * eta * Wavelength * Psig)
    else:
        return scale * qe / (2 * np.pi * RPD * Psig)

def CalcPhotonCurrent(P, Wavelength, eta = 1, RPD = None):
    ''' Calculate the current due to photons in A. Uses either eta*q*lambda/(2pi*hbar*c) or RPD*Psig.'''
    hbar = 6.62607015e-34
    c = 299792458
    qe = 1.60217663e-19
    if RPD is None:
        return eta * qe * Wavelength * P / (2 * np.pi * hbar * c)
    else:
        return RPD * P

def CalcShotPSD(P, Wavelength, eta = 1, RPD = None):
    ''' Calculate the shot noise phase PSD in A^2/Hz. Uses 2qI.'''
    hbar = 6.62607015e-34
    c = 299792458
    qe = 1.60217663e-19
    return 2 * qe * CalcPhotonCurrent(P, Wavelength, eta = eta, RPD = RPD) 

def optdBm2Watts(dBm):
    ''' Convert dBm to Watts.'''
    return 10 ** (dBm / 10) / 1000

def heterodyneBeat(LO_POW, SIG_POW, het_freq, N, fs, RPD = 0.95):
    ''' Generate a heterodyne beat signal with AC term and LO shot noise. POW in dBm. Remember shot noise is from LO (approximately). '''
    PD_GAIN = 4e4/RPD # FPD510 Current to voltage gain
    freqs = PLL.get_single_freqs(N, fs)
    AdditiveNoisePSD = np.ones_like(freqs) * CalcShotPSD(optdBm2Watts(LO_POW), Wavelength, RPD = RPD)
    AdditiveNoise = PLL.psdnoise(AdditiveNoisePSD, N, fs)
    LO_CURRENT = CalcPhotonCurrent(optdBm2Watts(LO_POW), Wavelength, RPD = RPD)
    SIG_CURRENT = CalcPhotonCurrent(optdBm2Watts(SIG_POW), Wavelength, RPD = RPD)
    SignalPowers = 4 * LO_CURRENT * SIG_CURRENT
    amplitude = np.sqrt(SignalPowers)
    het_signal = PD_GAIN * (amplitude * np.sin(2 * np.pi * het_freq * np.arange(N) / fs) + np.sqrt(2) * AdditiveNoise) 
    return het_signal

def EOM_LINE_POWER(N, Vpi, V):
    ''' Calculate the optical power of a line in a phase modulator EOM. Also gives dB reduction.'''
    K = np.pi/Vpi
    amp = jv(N, K * V)
    return amp ** 2, 20 * np.log10(amp)

def dBm2Volts(dBm, Z = 50):
    ''' Convert dBm to volts p-p, volts rms and volts peak value.'''
    Vrms = np.sqrt(10**(dBm/10) * Z / 1000)
    Vpeak = np.sqrt(2) * Vrms
    Vpp = 2 * Vpeak
    return Vpp, Vrms, Vpeak
    
def PD_Calibration(freq, PSD, LO_POW, PSD_PREV = None, LO_POW_PREV = None):
    ''' For calibrating PD shot noise power. Compare integrated dark noise power to shot noise. '''
    NOISE_POW = 10*np.log10(integrate.trapz(1e-3*10**(PSD/10), freq)*1000)
    print(f'Noise Power: {NOISE_POW} W for {LO_POW} dBm')
    MARGIN = 0
    NOISE_DIFF = np.zeros_like(PSD)
    if(PSD_PREV is not None and LO_POW_PREV is not None):
        NOISE_POW_PREV = 10*np.log10(integrate.trapz(1e-3*10**(PSD_PREV/10), freq)*1000)
        MARGIN = NOISE_POW - NOISE_POW_PREV
        print(f'Margin: {MARGIN} dB')
        NOISE_DIFF = PSD - PSD_PREV
    return NOISE_POW, MARGIN, NOISE_DIFF


if __name__ == '__main__':
    # Test the heterodyne beat
    SIG_POW = -100
    wavelength = 1542e-9
    RPD = 0.95
    Psig = optdBm2Watts(SIG_POW)
    Psig = 1e-14
    additiveLevel = CalcAdditiveShotPSD(Psig, wavelength, RPD = RPD)
    print(f'Additive Shot Noise PSD: {additiveLevel} rad^2/Hz')

    LO_POW = -16.8
    Plo = optdBm2Watts(LO_POW)
    shot_psd = CalcShotPSD(Plo, wavelength, RPD = RPD)
    Gain = 4e4/RPD
    shot_psd *= Gain ** 2
    print(f'LO Shot Noise PSD: {shot_psd} V^2/Hz')
    shot_psd = 10 * np.log10(shot_psd*1000/50)
    print(f'LO Shot Noise PSD: {shot_psd} dBm/Hz')

    POWER = -27
    print(f'Power: {optdBm2Watts(POWER)} W')

    LO_current = CalcPhotonCurrent(Plo, wavelength, RPD = RPD)
    print(f'LO Current: {LO_current} A')
    SIG_current = CalcPhotonCurrent(Psig, wavelength, RPD = RPD)
    print(f'SIG Current: {SIG_current} A')
    het_current = 2 * np.sqrt(LO_current * SIG_current)
    print(f'Heterodyne Current: {het_current} A')


    shot_psd = CalcShotPSD(Plo, wavelength, RPD = RPD) * 50
    shot_psd = 10 * np.log10(shot_psd*1000)
    print(f'LO Shot Noise PSD: {shot_psd} dBm/Hz')

    N = 0
    Vpi = 4.4
    V_dBm = -2+16
    Vpp, Vrms, Vpeak = dBm2Volts(V_dBm)
    print(f'Vpp: {Vpp}, Vrms: {Vrms}, Vpeak: {Vpeak}')
    pow, gain = EOM_LINE_POWER(N, Vpi, Vpeak)
    print(f'Power: {pow}, Gain: {gain}')

    # volts_dBm =  # dBm
    # Vpp, Vrms, Vpeak = dBm2Volts(volts_dBm)
    Vpeak = np.linspace(0, 10, 100)
    # Vpp, Vrms, Vpeak = dBm2Volts(Vpeak)
    line0_pow, line0_gain = EOM_LINE_POWER(0, Vpi, Vpeak)
    line1_pow, line1_gain = EOM_LINE_POWER(1, Vpi, Vpeak)
    line2_pow, line2_gain = EOM_LINE_POWER(-1, Vpi, Vpeak)
    line3_pow, line3_gain = EOM_LINE_POWER(2, Vpi, Vpeak)
    line4_pow, line4_gain = EOM_LINE_POWER(-2, Vpi, Vpeak)

    line12_pow, line12_gain = EOM_LINE_POWER(3, Vpi, Vpeak)

    plt.figure()
    plt.plot(Vpeak, 10*np.log10(line0_pow), label = 'N = 0')
    plt.plot(Vpeak, 10*np.log10(line1_pow), label = 'N = 1')
    # plt.plot(Vpeak, line2_pow, label = 'N = -1')
    plt.plot(Vpeak, 10*np.log10(line3_pow), label = 'N = 2')
    plt.plot(Vpeak, 10*np.log10(line12_pow), label = 'N = 3')
    # plt.plot(Vpeak, line4_pow, label = 'N = -2')
    plt.ylabel('Bessel Power')
    plt.xlabel('Vpeak')
    plt.grid()  
    plt.legend()


    V0 = 1.5
    Vpi = 4.4
    pow, gain = EOM_LINE_POWER(1, Vpi, V0)
    print(f'Power Fraction: {pow}, Gain: {gain}') 
    RPD = 0.95
    Psig1 = -35.3 + 2*gain - 3 # dBm
    Psig2 = -40.1 + 2*gain - 3 # dBm. -3 for 50/50 splitter
    Psig1 = optdBm2Watts(Psig1)
    Psig2 = optdBm2Watts(Psig2)

    additiveLevel1 = CalcAdditiveShotPSD(Psig1, Wavelength, RPD = RPD)
    additiveLevel2 = CalcAdditiveShotPSD(Psig2, Wavelength, RPD = RPD)

    print(f'Additive Shot Noise PSD1: {additiveLevel1} rad^2/Hz')
    print(f'Additive Shot Noise PSD2: {additiveLevel2} rad^2/Hz')


    plt.show()
    
