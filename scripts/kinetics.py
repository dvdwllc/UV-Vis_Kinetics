import numpy as np

from matplotlib import pyplot as plt
from scipy import stats


def linear(x, a, b):
    return a*x + b


def get_range(dataset, t0, t1, include_trihalides=False):
    """
    returns x and y values in the interval t0 < x < t1 from a Pandas dataframe
    containing time in the first column and bromine or iodine concentrations 
    ('CBr2' or 'CI2') in the second column.
    If trihalide concentrations are listed in the third column of the dataset, 
    they will be included in the calculation as y = 2*[X2]+3*[NaX3].
    
    Args:
        dataset: a Pandas dataframe containing time, halogen concentration, 
                    and trihalide concentration as columns.
        t0, t1: start and end times for the linear fit (floats)
        
    returns xvals, yvals as arrays.
    """

    # ADD your 'magic strings' here. Keeps errors from misspelling from
    # happening and explains what's going on with the names.
    TIME_INDEX = 'time (min)'
    COMPOUND_A = 'CNaBr3'
    COMPOUND_B = 'CNaI3'
    # etc...
    
    #Find indices of lower and upper time bounds
    #get time between lower and upper time bounds as xvals
    try:
        # If this is a big data set (I'm sure it could be) it's probably a
        # bad idea to look through it so many times when all you want is the
        #  largest / smallest times in some range. Also IDK what 'values'
        # does.
        tf0 = np.argwhere(dataset[TIME_INDEX].values > t0)[0][0]
        tf1 = np.argwhere(dataset[TIME_INDEX].values > t1)[0][0]
        xvals = dataset[TIME_INDEX].values[tf0:tf1]
    except IndexError as e:
        # you could use 'e' here to figure out which bound is too high
        print 'Invalid t0 or t1 value!' #if either bound is too high
        return 0
        
    try:
        if not np.isnan(dataset[COMPOUND_A].values[0]):
            include_trihalides = True
    except KeyError:
        pass
    
    try:
        if not np.isnan(dataset[COMPOUND_B].values[0]):
            include_trihalides = True
    except KeyError:
        pass
    
    #Get y values to fit
    if include_trihalides:
        try:
            # It's sorta unclear here what formula you're using. Formatting
            # it up like this will help out, but it'd be sweet to see the
            # formula. Also unclear why you use different data sets.
            yvals = (
                2*dataset['CBr2'].values[tf0:tf1] +
                3*dataset['CNaBr3'].values[tf0:tf1]
            )/2.0
        except KeyError:
            yvals = (
                2*dataset['CI2'].values[tf0:tf1] +
                3*dataset['CNaI3'].values[tf0:tf1]
            )/2.0
    else:
        try:
            yvals = (dataset['CBr2'].values[tf0:tf1])
        except KeyError:
            yvals = (dataset['CI2'].values[tf0:tf1])
    
    if len(yvals) == 0:
        print 'Incorrect data format!'
        return 0
    
    return xvals, yvals


def get_rate_constant(dataset, t0, t1, order):
    """
    Performs a linear fit over a specified time range (t0:t1) to 
    extract rate constants from zero, first-, or second-order kinetics plot.
    
    Args:
    
        dataset: a Pandas dataframe containing time, halogen concentration, 
                    and trihalide concentration as columns.
                    
        t0, t1: start and end times for the linear fit (floats)
        
        include_trihalides: Boolean. Whether or not to account for trihalide 
                    concentration in the fit.

        order: Int. specifies the desired kinetics plots (zero, first,
        or second).
    
    Returns the slope and intercept, and r-squared.
    """
    
    xvals, yvals = get_range(dataset, t0, t1)
    
    #Perform the fit to the appropriate data.
    if order == 0: 
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            xvals, yvals
        )

        return (slope, intercept, r_value**2)
    elif order == 1:
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            xvals, np.log(yvals)
        )
        # Establish a max line length and stick to it. Mine is 80 chars,
        # line above is 93. Your editor should have an option to do this.
        return (slope, intercept, r_value**2)
    elif order == 2:
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            xvals, 1/yvals
        )

        return (slope, intercept, r_value**2)
    else:
        print 'Invalid reaction order!'
        return 0

        
def compare_timerange(dataset, starttime, timerange, order, plot=True):
    """
    Calculates the rate constant for a zero, first-, or second-order
    kinetics plot over multiple different time ranges. Plots the full
    dataset and all fitted lines vs. time, and ln(k) vs. fit range.
    
    Args:
        dataset: a Pandas dataframe containing time, halogen concentration, 
                    and trihalide concentration as columns.
                    
        starttime: how early in the run to start the fit (defaults to 0).
        
        timerange: a range of times over which to perform the fit.
        
    returns an array of the log of the extracted rate constants.
    """
    
    slopes, intercepts, r2s = np.zeros((3, len(timerange)))
    xvals, yvals = get_range(dataset, 0, len(dataset)-1) 
    
    for i in range(len(timerange)):
        (slopes[i], intercepts[i], r2s[i]) = (
            get_rate_constant(dataset, starttime, timerange[i], order))
    
    if plot:
      f, (ax1, ax2) = plt.subplots(1, 2)
      # label these strings
      ax1.set_xlabel('Time (min)')
      ax2.set_xlabel('End time of fit (min)')
      ax2.set_ylabel('ln(k)')
    
    if order == 0:
        if plot:
            ax1.plot(xvals, yvals, 'x')
            ax1.set_ylabel('[X] (M)')

        for i in range(len(slopes)):
            if plot:
                ax1.plot(
                    xvals,
                    linear(xvals, slopes[i], intercepts[i]),
                    color = 'r',
                    alpha=0.4
                )
                ax2.plot(timerange[i], np.log(-slopes[i]), 'o-', color='r')

        return np.log(-slopes)
    
    elif order == 1:
        if plot:
            ax1.plot(xvals, np.log(yvals), 'x')
            ax1.set_ylabel('ln[X]')

        for i in range(len(slopes)):
            if plot:
                ax1.plot(
                    xvals,
                    linear(xvals, slopes[i], intercepts[i]),
                    color = 'r',
                    alpha=0.4
                )
                ax2.plot(timerange[i], np.log(-slopes[i]), 'o-', color='r')

        return np.log(-slopes)
        
    elif order == 2:
        if plot:
            ax1.plot(xvals, 1/(yvals), 'x')
            # label this string
            ax1.set_ylabel(r'1/[X] (M$^{\rm -1}$)')

        for i in range(len(slopes)):
            if plot:
               ax1.plot(
                   xvals,
                   linear(xvals, slopes[i], intercepts[i]),
                   color = 'r',
                   alpha=0.4
               )
               ax2.plot(timerange[i], np.log(slopes[i]), 'o-', color='r')

        return np.log(slopes)


def arrhenius_plot(
        datasets, temperatures, starttime, timerange, order, plotfits=False
):
    # Define important constants for the whole file outside all the
    # functions, and ones only important to the function inside the function
    # e.g. X_LABEL = '1/T (1/K)'
    f, ax0 = plt.subplots(1)
    ax0.set_xlabel('1/T (1/K)')
    ax0.set_ylabel('ln(k)')

    for i in range(len(datasets)):
       logk = compare_timerange(
           datasets[i],
           starttime,
           timerange,
           order,
           plotfits
       )
       ax0.plot(
           1/(temperatures[i]*np.ones(len(timerange))),
           logk,
           'o-'
       )


def arrhenius_plot_multi(
        datasets, colors, temperatures, starttime, timerange, order
):
   for i in range(len(datasets)):
       logk = compare_timerange(
           datasets[i],
           starttime,
           timerange,
           order,
           False
       )
       plt.plot(
           1/(
               (temperatures[i])*np.ones(len(timerange))
           ),
           logk,
           'o-',
           color=colors
       )


def plot_all(datasets, order):
     # label these strings
    t0 = 0

    if order == 0:
        plt.figure()
        plt.xlabel('Time (min)')
        plt.ylabel('[X] (M)')
    elif order == 1:
        plt.figure()
        plt.xlabel('Time (min)')
        plt.ylabel('ln[X]')
    elif order == 2:
        plt.figure()
        plt.xlabel('Time (min)')
        plt.ylabel('1/[X] (1/M)')
    else:
        print 'Invalid reaction order!'
        return 0

    for i in datasets:
        t1 = len(i['time (min)'])
        x, y = get_range(i, t0, t1-1)
        if order == 0:
            plt.plot(x, y, 'x')
        elif order == 1:
            plt.plot(x, np.log(y), 'x')
        elif order == 2:
            plt.plot(x, 1/y, 'x')