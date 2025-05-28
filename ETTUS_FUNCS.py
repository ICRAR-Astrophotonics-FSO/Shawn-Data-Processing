import numpy as np
import threading
import sys
import matplotlib.pyplot as plt
import h5py

import OPTICS_FUNCS as OF
import MSTAR_FUNCS as MF
from moku.instruments import SpectrumAnalyzer
import PLOT_FUNCS as PLOT
import PLL_FUNCS as PLL
import uhd
import numpy as np
import threading
from scipy import signal

def twos_complement(value, bits):
    return value & ((1 << bits) - 1)  # Mask the value to `bits` bits

def NCO_WORD(freq, sample_rate, BITS = 32):
    """ Converts frquency command to NCO word. Uses 2^BITS to represent the frequency."""
    WORD = int(2**BITS * freq / sample_rate)
    WORD = twos_complement(WORD, BITS)  # Ensure the word is within the range of BITS bits
    return WORD

def cleanup_usrp(usrp):
    # Stop all streams
    try:
        # Set gains to 0
        usrp.set_tx_gain(0, 0)
        usrp.set_tx_gain(0, 1)
        usrp.set_rx_gain(0, 0)
        usrp.set_rx_gain(0, 1)
        
    except Exception as e:
        print(f"Cleanup error: {e}")
    finally:
        del usrp  # Explicitly delete USRP object

def NCO_TX_CONFIG(usrp, regs_ch0, regs_ch1, WORD0, WORD1):
    """ Sets some default flags for my custom registers in the FPGA to transmit from the NCO."""

    FREQ_REG = 4
    CFG_REG = 12
    RST_REG = 16

    TX_CH0_SEL = 0 # NCO_tx0
    DEC = 0
    TX_CH0_CONFIG = 0 | (TX_CH0_SEL << 1) | (DEC << 5) 
    print(bin(TX_CH0_CONFIG))

    regs_ch0.poke32(CFG_REG, TX_CH0_CONFIG)
    regs_ch1.poke32(CFG_REG, TX_CH0_CONFIG)

    regs_ch0.poke32(FREQ_REG,WORD0)
    regs_ch0.poke32(RST_REG,1) # rst
    regs_ch0.poke32(RST_REG, 0)
    regs_ch1.poke32(FREQ_REG,WORD1)
    regs_ch1.poke32(RST_REG,1) # rst
    regs_ch1.poke32(RST_REG, 0)
    return

def PLL_CONFIG(usrp, regs_ch0, regs_ch1, ch0_FREQ0, ch0_FREQ1, ch0_FREQ2, ch1_FREQ0, 
               ch1_FREQ1, ch1_FREQ2, KP, KI, KII, USE_CIC, DEC, sample_rate):
    """ Configures PLL in the FPGA. This is a custom register that I have set up to control the PLL."""

    ch0_WORD0 = NCO_WORD(ch0_FREQ0, sample_rate)
    ch0_WORD1 = NCO_WORD(ch0_FREQ1, sample_rate)
    ch0_WORD2 = NCO_WORD(ch0_FREQ2, sample_rate)
    ch1_WORD0 = NCO_WORD(ch1_FREQ0, sample_rate)
    ch1_WORD1 = NCO_WORD(ch1_FREQ1, sample_rate)
    ch1_WORD2 = NCO_WORD(ch1_FREQ2, sample_rate)

    KP_enc = twos_complement(KP, 8)
    KI_enc = twos_complement(KI, 8)
    KII_enc = twos_complement(KII, 8)

    RX_CH0_CONFIG = (USE_CIC << 1) | (KP_enc << 5) | (KI_enc << 13) | (KII_enc << 21) 
    RX_CH1_CONFIG = (USE_CIC << 1) | (KP_enc << 5) | (KI_enc << 13) | (KII_enc << 21)


    print("Decimal config: ", RX_CH0_CONFIG)
    print("Binary config 0: ", bin(RX_CH0_CONFIG))
    print("Binary config 1: ", bin(RX_CH1_CONFIG))

    RX_SEL_REG = 8
    WORD0_REG = 20
    WORD1_REG = 24
    WORD2_REG = 28
    CIC_DEC_PHASE_REG = 32
    CIC_DEC_AMP_REG = 36
    CIC_DEC_PHASE = DEC * 4
    CIC_DEC_amp = DEC * 512
    print(f"Decimated Phase fs: {30.72e6/CIC_DEC_PHASE:.2f} Hz")
    print(f"Decimated Amp fs: {30.72e6/CIC_DEC_amp:.2f} Hz")

    regs_ch0.poke32(0,DEC)
    regs_ch0.poke32(WORD0_REG,ch0_WORD0)
    regs_ch0.poke32(WORD1_REG,ch0_WORD1)
    regs_ch0.poke32(WORD2_REG,ch0_WORD2)
    regs_ch0.poke32(CIC_DEC_PHASE_REG,CIC_DEC_PHASE)
    regs_ch0.poke32(CIC_DEC_AMP_REG,CIC_DEC_amp)


    regs_ch1.poke32(0,DEC)
    regs_ch1.poke32(WORD0_REG,ch1_WORD0)
    regs_ch1.poke32(WORD1_REG,ch1_WORD1)
    regs_ch1.poke32(WORD2_REG,ch1_WORD2)
    regs_ch1.poke32(CIC_DEC_PHASE_REG,CIC_DEC_PHASE)
    regs_ch1.poke32(CIC_DEC_AMP_REG,CIC_DEC_amp)

    return RX_CH0_CONFIG, RX_CH1_CONFIG



def get_usrp(sample_rate):
    """Initialize the USRP device with external clock source and bandwidth settings. 
    This function sets up the USRP device for transmission and reception with a bandwidth of 30.72 MHz.
    It also sets the clock source to external, which is useful for synchronization with other devices.

    returns:
        usrp: UHD USRP device object
    """

    usrp = uhd.usrp.MultiUSRP("type=b200,enable_user_regs")
    usrp.set_clock_source("external")
    usrp.set_tx_bandwidth(sample_rate)
    usrp.set_rx_bandwidth(sample_rate)
    usrp.set_rx_rate(sample_rate, 0)
    usrp.set_rx_rate(sample_rate, 1)
    usrp.set_tx_rate(sample_rate, 0)
    usrp.set_tx_rate(sample_rate, 1)

    # Disable DC_offset and IQ balance
    # usrp.set_rx_dc_offset(False, 0)
    # usrp.set_rx_dc_offset(False, 1)
    # usrp.set_rx_iq_balance(False, 0)
    # usrp.set_rx_iq_balance(False, 1)

    regs_ch0 = usrp.get_user_settings_iface(0)
    regs_ch1 = usrp.get_user_settings_iface(1)
    return usrp, regs_ch0, regs_ch1
