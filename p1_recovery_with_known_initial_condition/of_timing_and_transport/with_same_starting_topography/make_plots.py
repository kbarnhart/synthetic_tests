import glob
import os
import numpy as np

from yaml import load

import pandas as pd

import matplotlib
matplotlib.use('Agg')


import matplotlib.pyplot as plt


pattern = 'RESULTS/create_truth.*/calib_results.csv'

out = 'figures'

size = 41

if os.path.exists(out) == False:
    os.mkdir(out)


files = np.sort(glob.glob(pattern))
for file in files:
    folder = os.path.split(file)[0]
    print(folder)
    with open(folder+os.path.sep+'inputs.txt', 'r') as f:
        inputs = load(f)
    df = pd.read_csv(file)

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
    
    Pp = np.round(inputs['process_parameter'], 2)
    Fd = np.round(inputs['faulting_duration'], 2)
    
    fig, ax = plt.subplots(figsize=fs, dpi=300)
    plt.pcolormesh(Xplot,Yplot, OF, cmap='plasma_r', vmin=0, vmax=.3)
    plt.colorbar(label='Objective Function')
    plt.plot(inputs['process_parameter'], inputs['faulting_duration'], 'r*', label='Correct Value')
    plt.xlabel('$\\mathit{Transport}$ $\\mathit{Limited}\\longleftarrow$   Process Parameter   $\\longrightarrow \\mathit{Detachment}$ $\\mathit{Limited}$')
    plt.ylabel('Faulting Duration (MA)')
    plt.title('Same initital topography random seed')
    figname = 'Pp_'+str(Pp)+'.Fd_'+ str(Fd)+ '.OF_abs.png'
    plt.savefig(out+os.path.sep+figname)
    plt.close()

    fig, ax = plt.subplots(figsize=fs, dpi=300)
    plt.pcolormesh(Xplot,Yplot, np.log10(OF), cmap='viridis_r', vmin=-6, vmax=1)
    plt.colorbar(label='Log10(Objective Function)')
    plt.plot(inputs['process_parameter'], inputs['faulting_duration'], 'r*', label='Correct Value')
    plt.xlabel('$\\mathit{Transport}$ $\\mathit{Limited}\\longleftarrow$   Process Parameter   $\\longrightarrow \\mathit{Detachment}$ $\\mathit{Limited}$')
    plt.ylabel('Faulting Duration (MA)')
    plt.title('Same initital topography random seed')
    figname = 'Pp_'+str(Pp)+'.Fd_'+ str(Fd)+ '.OF_log.png'
    plt.savefig(out+os.path.sep+figname)
    plt.close()