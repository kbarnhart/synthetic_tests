import glob
import os
import numpy as np

from yaml import load

import pandas as pd
import matplotlib.pyplot as plt


pattern = 'RESULTS/create_truth.*/MPS_results.dat'

size = 11

files = glob.glob(pattern)
for file in files:
    folder = os.path.split(file)[0]
    
    with open(folder+os.path.sep+'inputs.txt', 'r') as f:
        inputs = load(f)
    df = pd.read_csv(file, sep='\s+')

    X = df.process_parameter.values.reshape((size,size))
    Y = df.faulting_duration.values.reshape((size,size))
    dx = np.diff(X).mean()
    dy = np.diff(Y, axis=0).mean()

    Y = Y-(0.5*dy) # shift by half a grid cell, so center of each square plots in correct place
    X = X-(0.5*dx) # shift by half a grid cell, so center of each square plots in correct place


    OF = df['objective_function'].values.reshape((size,size))
    plt.figure()
    plt.pcolormesh(X,Y, np.log10(OF), cmap='viridis_r')
    plt.colorbar()
    plt.plot(inputs['process_parameter'], inputs['faulting_duration'], 'r*')
    plt.xlabel('Process Parameter')
    plt.ylabel('Faulting Duration (ma)')
    plt.show()
