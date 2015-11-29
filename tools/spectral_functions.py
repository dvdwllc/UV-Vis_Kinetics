import numpy as np


def gaussian1(WL,
              h1, l1, w1,
              c):
    """
    gaussian(WL, h1, l1, w1)

    computes the predicted absorbance value for given wavelength and Br2 height,
    with two additional Gaussian-like peaks to fit the NaBr3 peak contribtution.

    Parameters
    ----------
    WL: scalar
        Wavelength in nm

    h1: scalar
        Height of total Br2 spectrum

    w1: scalar
        NaBr3 peak widths in nm

    l1: scalar
        NaBr3 peak centers in nm

    Returns
    -------
    Predicted Absorbance: float
    """
    return h1 * np.exp(-((WL - l1) ** 2) / (2 * w1 ** 2)) + c


def gaussian2(WL,
              h1, l1, w1,
              h2, l2, w2,
              c):
    """
    gaussian(WL, h1, l1, w1)

    computes the predicted absorbance value for given wavelength and Br2 height,
    with two additional Gaussian-like peaks to fit the NaBr3 peak contribtution.

    Parameters
    ----------
    WL: scalar
        Wavelength in nm

    h1, h2 : scalar
        Height of total Br2 spectrum

    w1, w2: scalar
        NaBr3 peak widths in nm

    l1, l2: scalar
        NaBr3 peak centers in nm

    Returns
    -------
    Predicted Absorbance: float
    """
    return (
        h1 * np.exp(-((WL - l1) ** 2) / (2 * w1 ** 2)) +
        h2 * np.exp(-((WL - l2) ** 2) / (2 * w2 ** 2)) +
        c
    )


def gaussian3(WL,
              h1, l1, w1,
              h2, l2, w2,
              h3, l3, w3,
              c):
    """
    gaussian3(x, h1, l1, w1, h2, l2, w2, h3, l3, w3, c)
    
    computes the predicted absorbance value for given wavelength and Br2 height,
    with two additional Gaussian-like peaks to fit the NaBr3 peak contribution.
    
    Parameters
    ----------
    WL: scalar
        Wavelength in nm
        
    h1, h2, h3: scalar
        Peak heights in absorbance units
        
    w1, w2, w3: scalar
        Peak widths in nm
        
    l1, l2, l3: scalar
        Peak centers in nm
    
    c: scalar
        vertical offset to account for light scattering by powder
    
    Returns
    -------
    Predicted Absorbance: float
    """
    return (
        h1 * np.exp(-((WL - l1) ** 2) / (2 * w1 ** 2)) +
        h2 * np.exp(-((WL - l2) ** 2) / (2 * w2 ** 2)) +
        h3 * np.exp(-((WL - l3) ** 2) / (2 * w3 ** 2)) +
        c
    )


def gaussian4(WL,
              h1, l1, w1,
              h2, l2, w2,
              h3, l3, w3,
              h4, l4, w4,
              c):
    """
    gaussian3(x, h1, l1, w1, h2, l2, w2, h3, l3, w3, c)

    computes the predicted absorbance value for given wavelength and Br2 height,
    with two additional Gaussian-like peaks to fit the NaBr3 peak contribtution.

    Parameters
    ----------
    WL: scalar
        Wavelength in nm

    h1, h2, h3, h4: scalar
        Peak heights in absorbance units

    w1, w2, w3, w4: scalar
        Peak widths in nm

    l1, l2, l3, l4: scalar
        Peak centers in nm

    c: scalar
        vertical offset to account for light scattering by powder

    Returns
    -------
    Predicted Absorbance: float
    """
    return (
        h1 * np.exp(-((WL - l1) ** 2) / (2 * w1 ** 2)) +
        h2 * np.exp(-((WL - l2) ** 2) / (2 * w2 ** 2)) +
        h3 * np.exp(-((WL - l3) ** 2) / (2 * w3 ** 2)) +
        h4 * np.exp(-((WL - l4) ** 2) / (2 * w4 ** 2)) +
        c
    )


def gaussian5(WL,
              h1, l1, w1,
              h2, l2, w2,
              h3, l3, w3,
              h4, l4, w4,
              h5, l5, w5, c):
    """
    gaussian3(x, h1, l1, w1, h2, l2, w2, h3, l3, w3, c)

    computes the predicted absorbance value for given wavelength and Br2 height,
    with two additional Gaussian-like peaks to fit the NaBr3 peak contribtution.

    Parameters
    ----------
    WL: scalar
        Wavelength in nm

    h1, h2, h3, h4, h5: scalar
        Peak heights in absorbance units

    w1, w2, w3, w4, w5: scalar
        Peak widths in nm

    l1, l2, l3, l4, l5: scalar
        Peak centers in nm

    c: scalar
        vertical offset to account for light scattering by powder

    Returns
    -------
    Predicted Absorbance: float
    """
    return (
        h1 * np.exp(-((WL - l1) ** 2) / (2 * w1 ** 2)) +
        h2 * np.exp(-((WL - l2) ** 2) / (2 * w2 ** 2)) +
        h3 * np.exp(-((WL - l3) ** 2) / (2 * w3 ** 2)) +
        h4 * np.exp(-((WL - l4) ** 2) / (2 * w4 ** 2)) +
        h5 * np.exp(-((WL - l5) ** 2) / (2 * w5 ** 2)) +
        c
    )


def linear(x, a, b):
    return a * x + b
