import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

import tools.spectral_functions as spec
import tools.data_i0 as data_io

"""
Here we train a collection of gaussian-like functions to UV-Vis absorption 
spectra collected on solutions of known bromine and sodium tribromide concentration
over the wavelength range 360 to 650 nm. 

    *The pure bromine spectrum is represented by a sum of three gaussians.
    *The sodium tribromide spectrum is represented by a sum of two gaussians.

Once the individual spectral shapes are determined, the concentration of both
species in an unknown solution can be determined by finding the optimum
intensity contribution of each peakshape to the observed absorption spectrum.
"""

# load training data as pandas DataFrames and measurement time as arrays
train1, train1_t = data_io.clean_agilent_uvvis_data('data/TRAIN1.TXT')
train2, train2_t = data_io.clean_agilent_uvvis_data('data/TRAIN2.TXT')

# We are working with Bromine and Sodium Tribromide, so label accordingly
HALOGEN_LABEL = '[Br2] (M)'
TRIHALIDE_LABEL = '[NaBr3] (M)'
TIME_LABEL = 'Time (min)'
ABSORBANCE_LABEL = 'Absorbance (Arb.)'
WAVELENGTH_LABEL = 'Wavelength (nm)'

# get wavelength values from first column
WLs = train1.index.values[1:].astype(float)

"""#############FIT RAW BROMINE SPECTRUM###############"""

# set upper and lower bounds for first training set
low, high = 320.0, 650.0
L1 = np.argwhere(WLs > low)[0][0]
L2 = np.argwhere(WLs > high)[0][0]

xvals, yvals = WLs[L1:L2], train1[0][1:].values[L1:L2]

# Initial values for curve_fit
p0_Br2 = (3.0, 280.0, 30.0,
          0.7, 390.0, 20.0,
          0.2, 460.0, 20.0, 0.02)

# optimize parameters
Br2opt, Br2cov = curve_fit(spec.gaussian3, xvals, yvals, p0_Br2)

# compute the fitted spectrum
Br2_fit = spec.gaussian3(xvals, Br2opt[0],
                         Br2opt[1], Br2opt[2], Br2opt[3],
                         Br2opt[4], Br2opt[5], Br2opt[6],
                         Br2opt[7], Br2opt[8], Br2opt[9])

# show each individual peak
g1, g2, g3 = (spec.gaussian(xvals, Br2opt[0], Br2opt[1], Br2opt[2]),
              spec.gaussian(xvals, Br2opt[3], Br2opt[4], Br2opt[5]),
              spec.gaussian(xvals, Br2opt[6], Br2opt[7], Br2opt[8]))

# plot results
plt.figure()
plt.title('Pure Br$_2$ in Acetonitrile')
plt.xlabel(WAVELENGTH_LABEL)
plt.ylabel(ABSORBANCE_LABEL)
plt.plot(xvals, yvals, 'x', label='data')
plt.plot(xvals, Br2_fit, color='r', label='fit')
plt.plot(xvals, g1, xvals, g2, xvals, g3)
plt.legend()


def Br2_plus_2gaussians(WL,
                        h1,
                        h2, l2, w2,
                        h3, l3, w3):
    """
    gauss5(x, h1, h2, l2, w2, h3, l3, w3)
    
    computes the predicted absorbance value for given wavelength and Br2 height,
    with two additional Gaussian-like peaks to fit the NaBr3 peak contribtution.
    
    Parameters
    ----------
    WL: scalar
        Wavelength in nm
        
    h1: scalar
        Height of total Br2 spectrum
        
    h2, h3: scalar
        NaBr3 peak heights in absorbance units
        
    w1, w2: scalar
        NaBr3 peak widths in nm
        
    l1, l2: scalar
        NaBr3 peak centers in nm
    
    Returns
    -------
    Predicted Absorbance: float
    """
    return (h1 * (Br2opt[0] * np.exp(-((WL - Br2opt[1]) ** 2) / (2 * Br2opt[2] ** 2)) +
                  Br2opt[3] * np.exp(-((WL - Br2opt[4]) ** 2) / (2 * Br2opt[5] ** 2)) +
                  Br2opt[6] * np.exp(-((WL - Br2opt[7]) ** 2) / (2 * Br2opt[8] ** 2))) +
            h2 * np.exp(-((WL - l2) ** 2) / (2 * w2 ** 2)) +
            h3 * np.exp(-((WL - l3) ** 2) / (2 * w3 ** 2)))


"""##########FIT SODIUM TRIBROMIDE SPECTRUM############"""

# set upper and lower bounds
low, high = 360.0, 650.0
L1 = np.argwhere(WLs > low)[0][0]
L2 = np.argwhere(WLs > high)[0][0]

last_line = len(train1_t) - 1
xvals, yvals = WLs[L1:L2], train1[last_line][1:].values[L1:L2]

# Initial values for curve_fit
NaBr3_p0 = (0.3,
            1.0, 400, 20,
            2.5, 340, 20)

# optimize NaBr3 parameters
g5opt, g5cov = curve_fit(Br2_plus_2gaussians, xvals, yvals, NaBr3_p0)

# compute the predicted spectrum
NaBr3_fit = Br2_plus_2gaussians(xvals,
                                g5opt[0],
                                g5opt[1], g5opt[2], g5opt[3],
                                g5opt[4], g5opt[5], g5opt[6])

# show each individual peak contributing to the spectrum
g1 = (spec.gaussian(xvals, g5opt[1], g5opt[2], g5opt[3]))
g2 = (spec.gaussian(xvals, g5opt[4], g5opt[5], g5opt[6]))

# plot results
plt.figure()
plt.title('1-x Br$_2$ + x NaBr3 in Acetonitrile')
plt.xlabel(WAVELENGTH_LABEL)
plt.ylabel(ABSORBANCE_LABEL)
plt.plot(xvals, yvals, 'x', label='data')
plt.plot(xvals, NaBr3_fit, color='r', label='fit')
plt.plot(xvals, g1, xvals, g2)
plt.legend()


def variable_Br2_NaBr3_spectrum(WL, hBr2, hNaBr3):
    """
    Computes the predicted absorbance value for given wavelength, 
    Br2 height, and NaBr3 height.
    
    Parameters
    ----------
    WL: scalar
        Wavelength in nm
        
    hBr2: scalar
        height of Br2 absorption spectrum
    
    hNaBr3: scalar:
        height of NaBr3 absorption spectrum
    
    Returns
    -------
    Predicted Absorbance: float
    """
    return (hBr2 * (Br2opt[0] * np.exp(-((WL - Br2opt[1]) ** 2) / (2 * Br2opt[2] ** 2)) +
                    Br2opt[3] * np.exp(-((WL - Br2opt[4]) ** 2) / (2 * Br2opt[5] ** 2)) +
                    Br2opt[6] * np.exp(-((WL - Br2opt[7]) ** 2) / (2 * Br2opt[8] ** 2))) +
            hNaBr3 * (g5opt[1] * np.exp(-((WL - g5opt[2]) ** 2) / (2 * g5opt[3] ** 2)) +
                      g5opt[4] * np.exp(-((WL - g5opt[5]) ** 2) / (2 * g5opt[6] ** 2))))


"""###########TRAIN###############"""

# set upper and lower bounds
low, high = 360.0, 650.0
L1 = np.argwhere(WLs > low)[0][0]
L2 = np.argwhere(WLs > high)[0][0]

Br2_conc_i = 0.00342  # Initial Br2 concentration in Mol/L.

(hBr2, hNaBr3) = np.zeros((2, len(train2_t)))

f, (ax1, ax2) = plt.subplots(1, 2)
ax1.set_xlabel(WAVELENGTH_LABEL)
ax1.set_ylabel(ABSORBANCE_LABEL)
ax2.set_xlabel(TRIHALIDE_LABEL)
ax2.set_ylabel('Peakheight (arb.)')

for i in range(len(train2_t)):
    xvals, yvals = WLs[L1:L2], train2[i][1:].values[L1:L2]

    # optimize parameters
    train2_opt, train2_cov = curve_fit(variable_Br2_NaBr3_spectrum,
                                       xvals, yvals)

    # add values to appropriate arrays
    (hBr2[i], hNaBr3[i]) = train2_opt

    # compute the fitted spectrum
    fit = variable_Br2_NaBr3_spectrum(xvals, train2_opt[0], train2_opt[1])

    # plot data and fit for each spectrum
    ax1.plot(xvals, yvals, 'x')
    ax1.plot(xvals, fit, color='r')

# compute [NaBr3] for all spectra
NaBr3_conc = Br2_conc_i * (1.0 - hBr2/hbr2[0])

# fit calibration line to data
line, cov = curve_fit(spec.linear, NaBr3_conc, hNaBr3)

# plot calibration points and line of best fit
ax2.plot(NaBr3_conc, hNaBr3, 'x')
ax2.plot(NaBr3_conc, spec.linear(NaBr3_conc, line[0], line[1]))

"""##############PREDICT##################"""


def predict_concs_from_spectra(dataset, time):
    """
    get_concs(dataset)
    
    Computes the concentrations of Br2 and NaBr3 in acetonitrile 
    from trained peak profiles and calibration spectra.
    
    Parameters
    ----------
    dataset: Pandas DataFrame containing wavelengths in the index column,
            time values in the first row, and absorbance readings in all cells.
    
    Returns
    _______
    (time, Br2 concentration (M), NaBr3 concentration (M))
    	(tuple of three NumPy arrays)
    """
    WLs = dataset.index.values[1:].astype(float)

    # lower and upper wavelengths for fit
    low, high = 360, 650

    # set upper and lower bounds
    L1 = np.argwhere(WLs > low)[0][0]
    L2 = np.argwhere(WLs > high)[0][0]

    hbr2, hnabr3 = np.zeros((2, len(time)))  # empty arrays for fitted parameters

    plt.figure()
    plt.axis([low, high, 0, 2])
    plt.title('All spectra + fits')
    plt.xlabel(WAVELENGTH_LABEL)
    plt.ylabel(ABSORBANCE_LABEL)

    for i in range(len(time)):
        xvals, yvals = WLs[L1:L2], dataset[i][1:].values[L1:L2]

        # optimize parameters
        g5Sopt, g5Scov = curve_fit(variable_Br2_NaBr3_spectrum, xvals, yvals)

        # compute the fitted spectrum
        fit = variable_Br2_NaBr3_spectrum(xvals, g5Sopt[0], g5Sopt[1])

        # add values to appropriate arrays
        (hbr2[i], hnabr3[i]) = g5Sopt

        # plot data and fit for each spectrum
        plt.plot(xvals, yvals, 'x', color='black', alpha=0.4)
        plt.plot(xvals, fit, color='r')

    return (time, Br2_conc_i * (hbr2/hbr2[0]), hnabr3 / line[0])


# Predict Br2 and NaBr3 concentrations for all spectra in calibration
(time, cBr2, cNaBr3) = predict_concs_from_spectra(train1)

# plot Conc. vs Time for the calibration run
FIGTITLE = 'Predicted %s and %s Concentrations' % (HALOGEN_LABEL,
                                                   TRIHALIDE_LABEL)
plt.figure()
plt.title(FIGTITLE)
plt.xlabel(TIME_LABEL)
plt.ylabel('[X] (M)')
plt.plot(time, cBr2, 'x', color='red', label=HALOGEN_LABEL)
plt.plot(time, cNaBr3, 'x', color='blue', label=TRIHALIDE_LABEL)
plt.legend()

# SHOW ALL RESULT$
plt.show()


if __name__ == '__main__':
    # Predict concentrations from other datasets
    run1_17C, run1_17C_t = \
        data_io.clean_agilent_uvvis_data('data/run1_17C.TXT')
    run2_25C, run2_25C_t = \
        data_io.clean_agilent_uvvis_data('data/run2_25C.TXT')
    run3_35C, run2_35C_t = \
        data_io.clean_agilent_uvvis_data('data/run3_35C.TXT')

    run1_17C_concs = predict_concs_from_spectra(run1_17C, run1_17C_t)
    run2_25C_concs = predict_concs_from_spectra(run2_25C, run2_25C_t)
    run3_35C_concs = predict_concs_from_spectra(run3_35C, run2_35C_t)

    data_io.write_concs(run1_17C_concs, TIME_LABEL, HALOGEN_LABEL,
                        TRIHALIDE_LABEL'data/run1_17C_concs')
    data_io.write_concs(run2_25C_concs, TIME_LABEL, HALOGEN_LABEL,
                        TRIHALIDE_LABEL'data/run2_25C_concs')
    data_io.write_concs(run3_35C_concs, TIME_LABEL, HALOGEN_LABEL,
                        TRIHALIDE_LABEL'data/run3_35C_concs')
