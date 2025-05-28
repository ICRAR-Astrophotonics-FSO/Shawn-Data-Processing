import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import scipy.spatial as spatial
import mplcursors
###############################
# This file contains functions for plotting data in several common formats.
###############################

# This file can create axs, fig for main program. Or just the formatter function can be used.

### PSD, most used.

SMALL_SIZE = 14
MEDIUM_SIZE = 14
BIGGER_SIZE = 14
TITLE_PAD = 14

plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)  # legend fontsize

plt.rcParams["font.family"] = "serif"

def read_spec_csv_2Chan(fn):
    data = np.genfromtxt(fn, delimiter=',', skip_header=11)
    freq = data[:,0]
    input1 = data[:,1]
    input2 = data[:,2]
    return freq, input1, input2

def read_spec_csv_1Chan(fn):
    data = np.genfromtxt(fn, delimiter=',', skip_header=11)
    freq = data[:,0]
    input1 = data[:,1]
    return freq, input1

def convert_spec_to_psd(SPEC, RBW, unit = 'dBm'):
    ''' Convert a power spectrum to a power spectral density. PSD is approximated by dividing by the RBW.'''
    if unit == 'dBm':
        PSD = 10*np.log10(10**(SPEC / 10) / RBW)
    else:
        PSD = SPEC / RBW
    return PSD


def CreatePSDPlot(**params):
    ''' Create a plot for PSD data.'''
    fig, axs = plt.subplots(**params)
    return fig, axs

def CreatePlot(**params):
    ''' Create a plot for PSD data.'''
    fig, axs = plt.subplots(**params)
    return fig, axs

def CreatePlotRowCol(numRow, numCol, **params):
    fig, axs = plt.subplots(numRow, numCol, **params)
    return fig, axs

def on_legend_click(event, axs, fig):
    legend_text = event.artist
    label = legend_text.get_text()
    for line in axs.get_lines():
        if line.get_label() == label:
            line.set_visible(not line.get_visible())
    fig.canvas.draw()

def connect_legend(legend, axs, fig):
    for legend_text in legend.get_texts():
        legend_text.set_picker(True)
    fig.canvas.mpl_connect('pick_event', lambda event: on_legend_click(event, axs, fig))

def hide_except_labels(legend, labels_to_keep, axs, fig):
    """
    Hide all legend entries except for the specified labels.
    
    Parameters:
    legend (matplotlib.legend.Legend): The legend object.
    labels_to_keep (list or str): The labels to keep visible.
    axs (matplotlib.axes.Axes): The axes object.
    fig (matplotlib.figure.Figure): The figure object.
    """
    if isinstance(labels_to_keep, str):
        labels_to_keep = [labels_to_keep]
    
    for line in axs.get_lines():
        if line.get_label() not in labels_to_keep:
            line.set_visible(False)
        else:
            line.set_visible(True)
    
    fig.canvas.draw()

def formatMDEVPlot(fig, axs, title = 'Modified Allan Deviation', xlabel = 'Tau (s)', ylabel = 'MDEV (m)', paper = False, laserAmbiguity = False, laserAmbiguityWl = 1.5e-6/4, hideAll = False):
    ''' Format a MDEV plot.'''
    if not paper:
        axs.set_title(title)
    axs.set_xlabel(xlabel)
    axs.set_ylabel(ylabel)
    axs.set_xscale('log')
    axs.set_yscale('log')
    
    if laserAmbiguity:
        axs.axhline(y=laserAmbiguityWl, color='r', linestyle='--', label = 'Laser Ambiguity Wavelength $\lambda/4$')
    
    legend = axs.legend(loc = 'upper right')
    axs.grid(which = 'both')
    axs.minorticks_on()
    if not paper:
        cursor = mplcursors.cursor(axs, hover=True)
        def on_add(sel):
            x, y = sel.target
            sel.annotation.set(text=f'({x:.2f}, {y:.2f})')
            sel.annotation.get_bbox_patch().set(fc="white", alpha=0.8)
        cursor.connect("add", on_add)
        connect_legend(legend, axs, fig)
    else:
        axs.tick_params(axis='both', which='major', labelsize=12)
        # get axis label
        axs.xaxis.get_offset_text().set_fontsize(12) # set axis label font
        # set axis label font
    fig.tight_layout()
    if hideAll:
        for line in axs.get_lines():
            line.set_visible(False)
    else:
        for line in axs.get_lines():
            line.set_visible(True)
    return fig, axs

def formatPSDPlotdBm(fig, axs, title = 'Power Spectral Density', density = True, linear = True, xlabel = None, PSD_View = False, RBW = None, showLabels = None, legendLoc = 'upper right', hideAll = False ):
    ''' Format a PSD plot with data in dBm or dBm/Hz.'''
    axs.set_title(title)
    if xlabel is not None:
        axs.set_xlabel(xlabel)
    else:
        axs.set_xlabel('Frequency (Hz)')
    if density:
        axs.set_ylabel(f'Power Density (dBm/Hz)')
    else:
        axs.set_ylabel(f'Power (dBm)')
    if not linear:
        axs.set_xscale('log')
    axs2 = None
    if not density and PSD_View:
        if RBW is not None:
            axs2 = axs.twinx()
            axs2.set_ylabel('Power (dBm/Hz)')
            # Get the y limits
            ymin, ymax = axs.get_ylim()
            ymin = convert_spec_to_psd(ymin, RBW)
            ymax = convert_spec_to_psd(ymax, RBW)
            axs2.set_ylim(ymin, ymax)
            axs2.set_yscale('linear')
            axs2.set_zorder(-100)

    
    legend = axs.legend(loc = legendLoc)
    axs.grid()
    axs.minorticks_on()

    cursor = mplcursors.cursor(axs, hover=True)

    # Customize the cursor to show the axis values
    @cursor.connect("add")
    def on_add(sel):
        x, y = sel.target
        annotation_text = f'axs: ({x:.2f}, {y:.2f})'
        if axs2 is not None:
            y2 = axs2.transData.inverted().transform(axs.transData.transform([x, y]))[1]
            annotation_text += f'\naxs2: ({x:.2f}, {y2:.2f})'
        sel.annotation.set(text=annotation_text)
        sel.annotation.get_bbox_patch().set(fc="white", alpha=0.8)

    
    connect_legend(legend, axs, fig)
    if showLabels is not None:
        hide_except_labels(legend, showLabels, axs, fig)
    
    if hideAll:
        for line in axs.get_lines():
            line.set_visible(False)
    else:
        for line in axs.get_lines():
            line.set_visible(True)

    return fig, axs

def formatPlot(fig, axs, title = 'Plot', xlabel = 'X', ylabel = 'Y'):
    ''' Format a plot with title, xlabel, ylabel, grid, and minor ticks.'''
    axs.set_title(title)
    axs.set_xlabel(xlabel)
    axs.set_ylabel(ylabel)
    axs.grid()
    legend = axs.legend(loc = 'upper right')
    connect_legend(legend, axs, fig)
    return fig, axs

def formatPSDPlot(fig, axs, title = 'Power Spectral Density', unit = 'cyc', phase = True, dB = False, density = True, linear = True, hideAll = False):
    ''' Format a PSD plot. If phase, cyc or rad^2/Hz and dB is used. If not, A^2/Hz or V^2/Hz and dBm is used for 50 ohm.'''
    axs.set_title(title)
    axs.set_xlabel('Frequency (Hz)')
    if density:
        axs.set_ylabel(f'Power Density ({unit}^2/Hz)')
    else:
        axs.set_ylabel(f'Power ({unit}^2)')
    if not linear:
        axs.set_xscale('log')
    axs.set_yscale('log')
    legend = axs.legend(loc = 'upper right')
    if dB:
        axs2 = axs.twinx()
        axs2.set_ylabel(f'Power (dB{unit}^2/Hz)')
        # Get the y limits
        ymin, ymax = axs.get_ylim()
        if phase:
            axs2.set_ylim(10 * np.log10(ymin), 10 * np.log10(ymax))
        else:
            axs2.set_ylim(10 * np.log10(ymin * 1000 / 50), 10 * np.log10(ymax * 1000 / 50)) # 50 ohm dBm conversion
            axs.set_ylabel('Power Density (dBm/Hz)')
    # grid and tickers
    axs.grid()
    axs.minorticks_on()
    connect_legend(legend, axs, fig)

    if hideAll:
        for line in axs.get_lines():
            line.set_visible(False)
    else:
        for line in axs.get_lines():
            line.set_visible(True)


    return fig, axs


### Transfer Function, most used.

def plotPII2(P, I, I2, freqs):
    fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True, figsize=(10, 6))
    ax0.plot(freqs, 20*np.log10(np.abs(P)), label = 'P')
    ax0.plot(freqs, 20*np.log10(np.abs(I)), label = 'I')
    ax0.plot(freqs, 20*np.log10(np.abs(I2)), label = 'I2')

    unity_gain_freq = freqs[np.argmin(np.abs(np.abs(P) - 1))]
    ax0.axvline(unity_gain_freq, color='r', linestyle='--')

    ax0.legend()
    ax0.set_title('LOOP GAINs Magnitude Response')
    ax0.set_xscale('log')
    ax0.set_yscale('linear')
    # show every power of 10 on xscale
    ax0.set_xticks([10**i for i in range(-2, int(np.log10(freqs[-1]))+1)])
    ax0.set_ylabel('Magnitude (dB)')
    ax0.set_xlim([1, freqs[-1]])
    ax0.grid()
    ax1.plot(freqs, np.angle(P))
    ax1.plot(freqs, np.angle(I))
    ax1.plot(freqs, np.angle(I))
    ax1.set_title('LOOP GAIN Phase Response')

def plotLoopGain(LOOP_GAIN, freqs):
    fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True, figsize=(10, 6))
    ax0.plot(freqs, 20*np.log10(np.abs(LOOP_GAIN)))
    ax0.set_title('LOOP GAIN Magnitude Response')
    ax0.set_xscale('log')
    ax0.set_yscale('linear')
    ax0.set_ylabel('Magnitude (dB)')
    scale = 360/(2*np.pi)
    ax1.plot(freqs, np.angle(LOOP_GAIN)*scale)
    ax1.set_title('LOOP GAIN Phase Response')
    ax1.set_xscale('log')
    ax1.set_yscale('linear')
    ax1.set_xlabel('Frequency (Hz)')
    ax0.grid()
    ax1.grid()
    unity_gain_freq = freqs[np.argmin(np.abs(np.abs(LOOP_GAIN) - 1))]
    phase_margin = np.angle(LOOP_GAIN[np.argmin(np.abs(freqs - unity_gain_freq))])
    ax0.axvline(unity_gain_freq, color='r', linestyle='--')
    ax1.axvline(unity_gain_freq, color='r', linestyle='--')
    ax1.axhline(phase_margin*scale, color='r', linestyle='--')
    ax1.axhline(- np.pi*scale, color='r', linestyle='--')
    print('Unity Gain Frequency: ', unity_gain_freq)
    print('Phase Margin: ', 180 + phase_margin*180/np.pi)

def plotTF(TF, freqs):
    fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True, figsize=(10, 6))
    ax0.plot(freqs, 20*np.log10(np.abs(TF)))
    ax0.set_title('Magnitude Response')
    ax0.set_xscale('log')
    ax0.set_yscale('linear')
    ax0.set_ylabel('Magnitude (dB)')
    ax1.plot(freqs, np.angle(TF))
    ax1.set_title('Phase Response')
    ax1.set_xscale('log')
    ax1.set_yscale('linear')
    ax1.set_xlabel('Frequency (Hz)')
    ax0.grid()
    ax1.grid()
    