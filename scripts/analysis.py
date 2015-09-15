import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import kinetics


run1_17C    = pd.read_csv('17C_concs.txt', sep=' ', names=('time (min)','CBr2','CNaBr3'))
run2_25C    = pd.read_csv('25C_concs.txt', sep=' ', names=('time (min)','CBr2','CNaBr3'))
run3_35C    = pd.read_csv('35C_concs.txt', sep=' ', names=('time (min)','CBr2','CNaBr3'))

runs = (run1_17C, run2_25C, run3_35C)
temperatures = np.array((17.5, 25.0, 35.0))+273.15

kinetics.plot_all(runs, 0)

kinetics.arrhenius_plot(runs, temperatures, 2, range(20,30,2), 1, True)