import pandas as pd
import numpy as np

import tools.kinetics as kinetics



# Read in concentration vs. time data for three different
# temperature runs of the same reaction.
run1_17C = pd.read_csv('data/run1_17C_concs.txt')
run2_25C = pd.read_csv('data/run2_25C_concs.txt')
run3_35C = pd.read_csv('data/run3_35C_concs.txt')

runs = (run1_17C, run2_25C, run3_35C)
temperatures = np.array((17.5, 25.0, 35.0))+273.15 # run temperatures in Kelvin

kinetics.plot_all(runs, 0) # Plot the raw concentration vs. time data


# Construct an Arrhenius plot from the raw data by performing several
# linear fits to the data over specified time intervals.

start_time = 2.0 # Time at which to begin the fits to each dataset
end_time_range = range(20,30,2) # range of times at which to end each fit
kinetics_plot_order = 1 # fit lines to a plot of ln[X] vs t.
plot_fits = True # see results of each linear fit.

kinetics.arrhenius_plot(runs,
                        temperatures,
                        start_time,
                        end_time_range,
                        kinetics_plot_order,
                        plot_fits)