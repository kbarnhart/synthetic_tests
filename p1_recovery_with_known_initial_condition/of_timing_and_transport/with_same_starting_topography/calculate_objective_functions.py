#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 19:13:39 2018

@author: barnhark
"""

import glob
import os
import numpy as np

from joblib import Parallel, delayed

from yaml import load

import pandas as pd

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

from scipy.interpolate import interp1d

pattern = 'RESULTS/create_truth.*/truth_profile.csv'
files = glob.glob(pattern)
    
def calculate_objective_function(file):
    print(file)
    
    pattern = 'RESULTS/create_truth.*/truth_profile.csv'
    files = glob.glob(pattern)

    folder = os.path.split(file)[0]
    run = int(folder.split('.')[-1])

    if os.path.exists(folder+os.path.sep + 'CALIB') == False:
        os.mkdir(folder+os.path.sep + 'CALIB')

    with open(folder+os.path.sep + 'inputs.txt', 'r') as f:
        inputs = load(f)
        Pp_truth = inputs['process_parameter']
        Fd_truth = inputs['faulting_duration']
        
    df_truth = pd.read_csv(file)
    
    df_list = []
       
    for data_file in files:
        
        data_folder = os.path.split(data_file)[0]
        data_run = int(data_folder.split('.')[-1])
        
        with open(data_folder + os.path.sep + 'inputs.txt', 'r') as f:
            data_inputs = load(f)
            Pp_data = data_inputs['process_parameter']
            Fd_data = data_inputs['faulting_duration']
        
        data_frame = pd.read_csv(data_file)
    
    
        interp_ob = interp1d(np.log10(data_frame.area), np.log10(data_frame.slope), bounds_error=False, fill_value=np.nan)
        log_interpolated_slope = interp_ob(np.log10(df_truth.area))
        
        ssd = np.nansum((np.log10(df_truth.slope) - log_interpolated_slope)**2.0)
        
        
        df_list.append({'run': data_run,
                        'process_parameter': Pp_data,
                        'faulting_duration': Fd_data,
                        'objective_function': ssd})
    
 
#        fs = (8, 6)
#        fig, ax = plt.subplots(figsize=fs, dpi=300)
#        plt.plot(df_truth.area, df_truth.slope, label='Truth')
#
#        plt.plot(data_frame.area, data_frame.slope, label='Run')
#        plt.legend()
#        ax.set_xscale('log')
#        ax.set_yscale('log')
#        plt.xlabel('log 10 Area')
#        plt.ylabel('log 10 Slope')
#        
#        plt.text(4e5, 1e-2, ('Truth:\nPp = ' + str(np.round(Pp_truth, 2)) + '\n'
#                             'Fd = ' + str(np.round(Fd_truth, 2)) + '\n\n'
#                             'Run :\nPp = ' + str(np.round(Pp_data, 2)) + '\n'
#                             'Fd = ' + str(np.round(Fd_data, 2)) + '\n'))
#                             
#        plt.savefig(folder+os.path.sep+'CALIB/obj_func.' + str(data_run) + '.png')
#        plt.close()
      
    df_out = pd.DataFrame(df_list)
    df_out.set_index('run', inplace=True)
    df_out.sort_index(axis=0, inplace=True)
    
    df_out.to_csv(folder + os.path.sep + 'calib_results.csv')
#%%
#output = Parallel(n_jobs=20)(delayed(calculate_objective_function)(file) for file in files)
for file in files:
    calculate_objective_function(file)