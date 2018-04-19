import glob
import os
import numpy as np

from yaml import load

import pandas as pd

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt


pattern = 'RESULTS/create_truth.*/MPS_results.dat'

out = 'figures'

size = 11

if os.path.exists(out) == False:
    os.mkdir(out)


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

    Xplot = np.empty((size+1, size+1))
    Yplot = np.empty((size+1, size+1))
    Yplot[:size, :size] = Y-(0.5*dy) # shift by half a grid cell, so center of each square plots in correct place
    Xplot[:size, :size] = X-(0.5*dx) # shift by half a grid cell, so center of each square plots in correct place
    Yplot[:size,-1] = Yplot[:size,-2]
    Xplot[-1, :size] = Xplot[-2, :size]
    Yplot[size,:] = Y.max() + 0.5*dy
    Xplot[:, size] = X.max() + 0.5*dx

    OF = df['objective_function'].values.reshape((size,size))

    fs = (8, 6)
    fig, ax = plt.subplots(figsize=fs, dpi=300)
    plt.pcolormesh(Xplot,Yplot, np.log10(OF), cmap='viridis_r')
    plt.colorbar()
    plt.plot(inputs['process_parameter'], inputs['faulting_duration'], 'r*')
    plt.xlabel('Process Parameter')
    plt.ylabel('Faulting Duration (ma)')

    figname = 'OF_fit.Pp_'+str(inputs['process_parameter'])+'.Fd_'+ str(inputs['faulting_duration'])+ '.png'
    plt.savefig(out+os.path.sep+figname)
    plt.close()
