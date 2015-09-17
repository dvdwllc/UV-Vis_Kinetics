import pandas as pd
import numpy as np

import tools.kinetics as kinetics



# define strings here. Also what does this file do?

run1_17C = pd.read_csv('data/run1_17C_concs.txt')
run2_25C = pd.read_csv('data/run2_25C_concs.txt')
run3_35C = pd.read_csv('data/run3_35C_concs.txt')

runs = (run1_17C, run2_25C, run3_35C)
temperatures = np.array((17.5, 25.0, 35.0))+273.15

kinetics.plot_all(runs, 0)

kinetics.arrhenius_plot(runs, temperatures, 2, range(20,30,2), 1, True)