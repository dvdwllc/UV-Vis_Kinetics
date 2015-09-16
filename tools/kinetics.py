import numpy as np
from matplotlib import pyplot as plt
from scipy import stats


def linear(x, a, b):
    return a * x + b


TIME_LABEL = 'Time (min)'
BR2_LABEL = '[Br2] (M)'
I2_LABEL = '[I2] (M)'
NABR3_LABEL = '[NaBr3] (M)'
NAI3_LABEL = '[NaI3] (M)'

ZERO_ORDER_YLABEL = '[X] (M)'
FIRST_ORDER_YLABEL = 'ln[X]'
SECOND_ORDER_YLABEL = r'[X]$^{\rm -1}$ (M$^{\rm -1}$)'


def get_time_and_conc_in_range(dataset, t0, t1, include_trihalides=False):
    """
    returns time and concentration values in the interval 
    t0 < x < t1 from a Pandas dataframe containing 
    time in the first column and bromine or iodine concentrations 
    ('CBr2' or 'CI2') in the second column. A third column may be present, 
    which will contain either sodium tribromide or sodium triiodide
    ('CNaBr3' or 'CNaI3'). If the third column is present, the 
    total concentration is calculated as [X] = (2*[X2]+3*[NaX3])/2.
    
    Parameters
    ----------
        dataset: a Pandas dataframe containing time, halogen concentration, 
                    and trihalide concentration as columns.
        t0, t1: start and end times for the linear fit (floats)
        
    Returns 
    -------
    	xvals, yvals as arrays.
    """

    # Find indices of lower and upper time bounds
    # get time between lower and upper time bounds as xvals
    try:
        tf0 = np.argwhere(dataset[TIME_LABEL].values > t0)[0][0]
        tf1 = np.argwhere(dataset[TIME_LABEL].values > t1)[0][0]
        xvals = dataset[TIME_LABEL].values[tf0:tf1]
    except IndexError as e:
        print e  # if either bound is invalid
        return 0

    try:
        if not np.isnan(dataset[NABR3_LABEL].values[0]):
            include_trihalides = True
    except KeyError:
        pass

    try:
        if not np.isnan(dataset[NAI3_LABEL].values[0]):
            include_trihalides = True
    except KeyError:
        pass

    # Get total concentration based on data format and labels
    if include_trihalides:
        try:
            yvals = (
                        2 * dataset[BR2_LABEL].values[tf0:tf1] +
                        3 * dataset[NABR3_LABEL].values[tf0:tf1]
                    ) / 2.0
        except KeyError:
            yvals = (
                        2 * dataset[I2_LABEL].values[tf0:tf1] +
                        3 * dataset[NAI3_LABEL].values[tf0:tf1]
                    ) / 2.0
    else:
        try:
            yvals = (dataset[BR2_LABEL].values[tf0:tf1])
        except KeyError:
            yvals = (dataset[I2_LABEL].values[tf0:tf1])

    if len(yvals) == 0:
        print 'Incorrect data format!'
        return 0

    return xvals, yvals


def calc_rate_constant(xvals, yvals, order):
    """
    Performs a linear fit over a specified time range (t0:t1) to 
    extract rate constants from zero, first-, or second-order kinetics plot.
    
    Parameters
    ----------
        dataset: a Pandas dataframe containing time, halogen concentration, 
                    and trihalide concentration as columns.
                    
        t0, t1: start and end times for the linear fit (floats)
        
        include_trihalides: Boolean. Whether or not to account for trihalide 
                    concentration in the fit.
                    
        order: Int. specifies the desired kinetics plots 
                    (zero, first, or second).
    
    Returns
    -------
    	tuple: (slope, intercept, r-squared)
    """

    # Perform the fit to the appropriate data.
    if order == 0:
        (slope, intercept, r_value, p_value, std_err) = (
            stats.linregress(xvals, yvals))
        return (slope, intercept, r_value ** 2)

    elif order == 1:
        (slope, intercept, r_value, p_value, std_err) = (
            stats.linregress(xvals, np.log(yvals)))
        return (slope, intercept, r_value ** 2)

    elif order == 2:
        (slope, intercept, r_value, p_value, std_err) = (
            stats.linregress(xvals, 1 / yvals))
        return (slope, intercept, r_value ** 2)

else:
print 'Invalid reaction order!'
return 0


def compare_ratek_vs_timerange(dataset, starttime,
                               timerange, order, plot=True):
    """
    Calculates the rate constant for a zero, first-, or second-order
    kinetics plot over multiple different time ranges. Plots the full
    dataset and all fitted lines vs. time, and ln(k) vs. fit range.
    
    Parameters
    ----------
        dataset: a Pandas dataframe containing time, halogen concentration, 
                    and trihalide concentration as columns.
                    
        starttime: how early in the run to start the fit (defaults to 0).
        
        timerange: a range of times over which to perform the fit.
        
    Returns
    -------
    	Array of the natural logs of the extracted rate constants.
    """

    # create empty lists for fitted values
    slopes, intercepts, r2s = np.zeros((3, len(timerange)))

    # get raw data
    xvals, yvals = get_time_and_conc_in_range(dataset, 0, len(dataset) - 1)

    # find where to start fit
    start_index = np.argwhere(xvals > starttime)[0][0]

    # for each end time in timerange, compute the rate constant
    # from the data between start time and end time
    for i in range(len(timerange)):
        # find where to end fit
        end_index = np.argwhere(xvals > timerange[i])[0][0]

        # compute fit
        (slopes[i], intercepts[i], r2s[i]) = (
            get_rate_constant(xvals[start_index:end_index],
                              yvals[start_index:end_index], order))

    if plot:
        f, (ax1, ax2) = plt.subplots(1, 2)
        ax1.set_xlabel(TIME_LABEL)
        ax2.set_xlabel('End time of fit (min)')
        ax2.set_ylabel('ln(k)')

    if order == 0:
        if plot:
            ax1.plot(xvals, yvals, 'x')
            ax1.set_ylabel(ZERO_ORDER_YLABEL)

        for i in range(len(slopes)):
            if plot:
                ax1.plot(
                    xvals,
                    linear(xvals, slopes[i], intercepts[i]),
                    color='r',
                    alpha=0.4
                )
                ax2.plot(timerange[i], np.log(-slopes[i]), 'o-', color='r')

        return np.log(-slopes)

    elif order == 1:
        if plot:
            ax1.plot(xvals, np.log(yvals), 'x')
            ax1.set_ylabel(FIRST_ORDER_YLABEL)

        for i in range(len(slopes)):
            if plot:
                ax1.plot(
                    xvals,
                    linear(xvals, slopes[i], intercepts[i]),
                    color='r',
                    alpha=0.4
                )
                ax2.plot(timerange[i], np.log(-slopes[i]), 'o-', color='r')

        return np.log(-slopes)

    elif order == 2:
        if plot:
            ax1.plot(xvals, 1 / (yvals), 'x')
            ax1.set_ylabel(SECOND_ORDER_YLABEL)

        for i in range(len(slopes)):
            if plot:
                ax1.plot(
                    xvals,
                    linear(xvals, slopes[i], intercepts[i]),
                    color='r',
                    alpha=0.4
                )
                ax2.plot(timerange[i], np.log(slopes[i]), 'o-', color='r')

        return np.log(slopes)


def arrhenius_plot(
        datasets, temperatures, starttime, timerange, order, plotfits=False
):
    """
    Constructs a plot of the natural log of the rate constant versus 
    inverse temperature (ln(k) vs 1/T), commonly known as an Arrhenius plot. 
    For most reactions, such a plot yields a linear relationship with a 
    downward slope, which is related to the activation energy of the reaction
    as slope = -Ea/RT.
    
    Parameters
    ----------
    	datasets: list/tuple/array of Pandas DataFrames.
    		Each DataFrame contains a set kinetics data collected at 
    		a constant, known temperature
    	
    	temperatures: list/tuple/array of floats
    		The temperatures (in Kelvin) at which the datasets were
    		collected. (must be in the same order as datasets).
    		
    	starttime: float. The time at which to begin the linear fit to the kinetics
    		data.
    	
    	timerange: list/tuple/array containing times at which to end the 
    		linear fit to the kinetics data. 
    		
    	order: integer (0, 1 or 2). 
    		The order of the kinetics plot from which the rate constant is
    		to be determined (zero, first, or second order).
    		
    	plotfits: Boolean. Whether or not to plot the data and fits for each 
    		dataset and timerange. defaults to False.
    """

    ARRHENIUS_X_LABEL = '1/T (1/K)'
    ARRHENIUS_Y_LABEL = 'ln(k)'

    f, ax0 = plt.subplots(1)
    ax0.set_xlabel(ARRHENIUS_X_LABEL)
    ax0.set_ylabel(ARRHENIUS_Y_LABEL)

    for i in range(len(datasets)):
        logk = compare_ratek_vs_timerange(
            datasets[i],
            starttime,
            timerange,
            order,
            plotfits
        )
        ax0.plot(
            1 / (temperatures[i] * np.ones(len(timerange))),
            logk,
            'o-'
        )

        plt.show()


def arrhenius_plot_multi(
        datasets, colors, temperatures, starttime, timerange, order
):
    """
    Same as arrhenius_plot, but does not initiate a new figure. This allows
    for multiple different sets of reactions to be plotted on the same graph
    with different colors.

    See documentation for arrhenius_plot for more details.

    Parameters
    ----------
        datasets: list/tuple/array of Pandas DataFrames.
            Each DataFrame contains a set kinetics data collected at
            a constant, known temperature

        colors: any valid color in matplotlib.

        temperatures: list/tuple/array of floats
            The temperatures (in Kelvin) at which the datasets were
            collected. (must be in the same order as datasets).

        starttime: float. The time at which to begin the linear fit to the kinetics
            data.

        timerange: list/tuple/array containing times at which to end the
            linear fit to the kinetics data.

        order: integer (0, 1 or 2).
            The order of the kinetics plot from which the rate constant is
            to be determined (zero, first, or second order).

        plotfits: Boolean. Whether or not to plot the data and fits for each
            dataset and timerange. defaults to False.
    """


for i in range(len(datasets)):
    logk = compare_ratek_vs_timerange(
        datasets[i],
        starttime,
        timerange,
        order,
        False
    )
    plt.plot(
        1 / (
            (temperatures[i]) * np.ones(len(timerange))
        ),
        logk,
        'o-',
        color=colors
    )


def plot_all(datasets, order):
    """
    Plots concentration data and time for each dataset on a single graph
    according to the specified reaction order. This allows for
    direct visual comparison of multiple datasets.

    Parameters
    ----------
    datasets: list/tuple/array of Pandas DataFrames.
            Each DataFrame contains a set kinetics data collected at
            a constant, known temperature

    order: integer (0, 1 or 2).
            The order of the kinetics plot from which the rate constant is
            to be determined (zero, first, or second order).
    """


t0 = 0

if order == 0:
    plt.figure()
    plt.xlabel(TIME_LABEL)
    plt.ylabel(ZERO_ORDER_YLABEL)
elif order == 1:
    plt.figure()
    plt.xlabel(TIME_LABEL)
    plt.ylabel(FIRST_ORDER_YLABEL)
elif order == 2:
    plt.figure()
    plt.xlabel(TIME_LABEL)
    plt.ylabel(SECOND_ORDER_YLABEL
    else:
    print 'Invalid reaction order!'
    return 0

for i in datasets:
    t1 = len(i[TIME_LABEL])
    x, y = get_time_and_conc_in_range(i, t0, t1 - 1)
    if order == 0:
        plt.plot(x, y, 'x')
    elif order == 1:
        plt.plot(x, np.log(y), 'x')
    elif order == 2:
        plt.plot(x, 1 / y, 'x')

plt.show()
