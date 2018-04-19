# -*- coding: utf-8 -*-
"""
Driver model for Landlab Model 410 BasicHySa

Katy Barnhart March 2017
"""
import os
import sys
import shutil
from subprocess import call
from yaml import load

import numpy as np
import matplotlib
matplotlib.use('Agg')

import matplotlib.pylab as plt
import pandas as pd
from scipy.optimize import newton

from landlab import imshow_grid
from landlab.plot.channel_profile import analyze_channel_network_and_plot, plot_channels_in_map_view

from terrainbento import BasicHySa as Model

from matplotlib.collections import LineCollection
from matplotlib import cm

def solve_for_Ks(logKs, Kr, P0, U, V, r, Ff, C):
    """ """
    Ks = 10.0**(logKs)
    a = Ks * Kr
    b = (P0 * Ks) - ((Kr * U * V * (1.0 - Ff)) / r)
    c = - U * P0 * (1.0 - (V * (1 - Ff)/r))

    # if (1.0 - (V * (1 - Ff)/r)) > 1 then no real root exists....

    f = (-b + (b**2.0 - 4.0 * a * c)**0.5)/(2 * a) - C

    return f

class ChannelPlotter(object):
    """ChannelPlotter... """
    def __init__(self, model):
        self._model = model
        self.xnormalized_segments = []
        self.channel_segments = []
        self.relative_times = []

    def run_one_step(self):
        """ """
        # find the faulted node with the largest drainage area.
        largest_da = np.max(self._model.grid.at_node['drainage_area'][self._model.boundary_handler['NormalFault'].faulted_nodes==True])
        largest_da_ind = np.where(self._model.grid.at_node['drainage_area'] == largest_da)[0][0]

        start_node = self._model.grid.at_node['flow__receiver_node'][largest_da_ind]

        (profile_IDs, dists_upstr) = analyze_channel_network_and_plot(self._model.grid, number_of_channels=1, starting_nodes=[start_node], create_plot=False)
        elevs = model.z[profile_IDs]

        self.relative_times.append(self._model.model_time/model.params['run_duration'])

        offset = np.min(elevs[0])
        max_distance = np.max(dists_upstr[0][0])
        self.channel_segments.append(np.array((dists_upstr[0][0], elevs[0] - offset)).T)
        self.xnormalized_segments.append(np.array((dists_upstr[0][0]/max_distance, elevs[0] - offset)).T)

        self.relative_times.append(self._model.model_time/model.params['run_duration'])

        colors = cm.viridis_r(self.relative_times)

        xmin = [xy.min(axis=0)[0] for xy in self.channel_segments]
        ymin = [xy.min(axis=0)[1] for xy in self.channel_segments]
        xmax = [xy.max(axis=0)[0] for xy in self.channel_segments]
        ymax = [xy.max(axis=0)[1] for xy in self.channel_segments]

        fs = (8, 6)
        fig, ax = plt.subplots(figsize=fs, dpi=300)
        ax.set_xlim(0, max(xmax))
        ax.set_ylim(0, max(ymax))
        line_segments = LineCollection(self.channel_segments, colors=colors, linewidth=0.5)
        ax.add_collection(line_segments)
        yr = str(self._model.model_time/(1e6)).zfill(4)
        plt.savefig('profile_' + yr + '.png')
        plt.close()

        fig, ax = plt.subplots(figsize=fs, dpi=300)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, max(ymax))
        line_segments = LineCollection(self.xnormalized_segments, colors=colors, linewidth=0.5)
        ax.add_collection(line_segments)
        yr = str(self._model.model_time/(1e6)).zfill(4)
        plt.savefig('normalized_profile_' + yr + '.png')
        plt.close()

        plt.figure()
        plot_channels_in_map_view(self._model.grid, profile_IDs)
        plt.savefig('topography_' + yr + '.png')
        plt.close()

        plt.figure()
        imshow_grid(model.grid, model.grid.at_node['soil__depth'], cmap='viridis', limits = (0, 15))
        plt.savefig('soil_' + yr + '.png')
        plt.close()

        plt.figure()
        imshow_grid(self._model.grid, self._model.grid.at_node['sediment__flux'], cmap='viridis')
        plt.savefig('sediment_flux_' + yr + '.png')
        plt.close()

        U_eff = U_fast + U_back
        U_eff_slow = U_slow + U_back

        area = np.sort(self._model.grid.at_node['drainage_area'][self._model.boundary_handler['NormalFault'].faulted_nodes==True])
        area = area[area>0]
        little_q = (area * self._model.params['runoff_rate']) ** self._model.params['m_sp']
        #area_to_the_m = area ** self._model.params['m_sp']

        detachment_prediction = ((U_eff / (self._model.params['K_rock_sp'])) ** (1.0/self._model.params['n_sp'])
                                 * (1.0/little_q) ** (1.0/self._model.params['n_sp']))

        transport_prediction =  (((U_eff * self._model.params['v_sc']) / (self._model.params['K_sed_sp'] * self._model.params['runoff_rate'])) +
                                 ((U_eff) / (self._model.params['K_sed_sp']))
                                 ) **(1.0/self._model.params['n_sp']) * ((1.0/little_q) ** (1.0/self._model.params['n_sp']))

        space_prediction =  (((U_eff * self._model.params['v_sc']) * (1.0 - Ff) / (self._model.params['K_sed_sp'] * self._model.params['runoff_rate'])) +
                             ((U_eff) / (self._model.params['K_rock_sp'] ))
                             )**(1.0/self._model.params['n_sp']) * ((1.0/little_q) ** (1.0/self._model.params['n_sp']))

        detachment_prediction_slow = ((U_eff_slow / (self._model.params['K_rock_sp'])) ** (1.0/self._model.params['n_sp'])
                                     * (1.0/little_q) ** (1.0/self._model.params['n_sp']))

        transport_prediction_slow =  (((U_eff_slow * self._model.params['v_sc']) / (self._model.params['K_sed_sp'] * self._model.params['runoff_rate'])) +
                                      ((U_eff_slow) / (self._model.params['K_sed_sp']))
                                      ) **(1.0/self._model.params['n_sp']) * ((1.0/little_q) ** (1.0/self._model.params['n_sp']))

        space_prediction_slow =  (((U_eff_slow * self._model.params['v_sc']) * (1.0 - Ff) / (self._model.params['K_sed_sp'] * self._model.params['runoff_rate'])) +
                                  ((U_eff_slow) / (self._model.params['K_rock_sp'] ))
                                  )**(1.0/self._model.params['n_sp']) * ((1.0/little_q) ** (1.0/self._model.params['n_sp']))

# TODO need to fix space predictions here to include new soil thickness.

        fs = (8, 6)
        fig, ax = plt.subplots(figsize=fs, dpi=300)
        plt.plot(area, detachment_prediction, 'c', lw=5, label='Detachment Prediction')
        plt.plot(area, transport_prediction, 'b', label='Transport Prediction')
        plt.plot(area, space_prediction, 'm', label='Space Prediction')

        plt.plot(area, detachment_prediction_slow, 'c', lw=5, alpha=0.3)
        plt.plot(area, transport_prediction_slow, 'b', alpha=0.3)
        plt.plot(area, space_prediction_slow, 'm', alpha=0.3)

        plt.plot(self._model.grid.at_node['drainage_area'][self._model.boundary_handler['NormalFault'].faulted_nodes==True],
                              self._model.grid.at_node['topographic__steepest_slope'][self._model.boundary_handler['NormalFault'].faulted_nodes==True],
                              'k.', label='Fault Block Nodes')
        plt.plot(self._model.grid.at_node['drainage_area'][self._model.boundary_handler['NormalFault'].faulted_nodes==False],
                                  self._model.grid.at_node['topographic__steepest_slope'][self._model.boundary_handler['NormalFault'].faulted_nodes==False],
                                  'r.', label='Unfaulted Nodes')
        plt.plot(self._model.grid.at_node['drainage_area'][profile_IDs],
                                   self._model.grid.at_node['topographic__steepest_slope'][profile_IDs],
                                   'g.', label='Main Channel Nodes')
        plt.legend()

        ax.set_xscale('log')
        ax.set_yscale('log')
        plt.xlabel('log 10 Area')
        plt.ylabel('log 10 Slope')
        plt.savefig('slope_area_' + yr + '.png')
        plt.close()

# set files and directories used to set input templates.
# Files and directories.
input_file = 'inputs.txt'

# V/r = 1 means equal contribution of transport and detachment
# r = runoff, such that Q = Ar
# V = effective settling velocity

#Two terms to consider are
#v/(Ks r)

# 1/Kr

# 15 Km and ~600-700 ft (like a random B&R block)
input_file = 'inputs.txt'
input_template = 'inputs_template.txt'

# Use `dprepro` (from $DAKOTA_DIR/bin) to substitute parameter
# values from Dakota into the SWASH input template, creating a new
# inputs.txt file.

#
call(['dprepro', sys.argv[-2], input_template, input_file])
call(['rm', input_template])

with open(input_file, 'r') as f:
    inputs = load(f)

    Pp = float(inputs['process_parameter']) # this varies from 0 to 1, 0 = transport limtited, 1 = detachment limited
    faulting_duration = float(inputs['faulting_duration']) * 1e6
    topography_seed = int(inputs['topography_seed'])
fault_start = 8 * 1e6

total_time = faulting_duration + fault_start

U_back = 0.00001
U_fast = 0.001
U_slow = 0.0003

Ff = Pp

v = 1
r = 0.05
phi = 0.0
H_star = 1
n = 1.0
m = 0.5

# set such that at H = H_p = 10 * H_star the production rate is equal to the uplift
# rate in the transport limited case.
H_p = 10.0 * H_star
W_dot_min = 0.000001
Wdot_max = W_dot_min + (np.e * U_fast - W_dot_min) * (1.0-Pp) # transition from making soil when Pp = 0 to not making soil when Pp = 1

# we want the Slope-Area scaleing parameter set (U/Kr, or (U/Ks + UV/rKs)) to be
# valued such that drainage areas of 10**6 m2 have slopes of 0.2. (Taiwan would be slope of 0.4)
# S = (P_sa * 1/(A^m))^(1/n)
# P_sa = S^n * (A)^m

P_sa = 0.2*((1e6)**0.5)

# However, when we make our actual model runs, we are using q = Ar to drive
# erosion.
# thus we need to make an adjustment to P_sa to acknowlege that
# for detachment S*n = (U / Kr (rA)^m)

Ueff = U_fast + U_back
Kr = (Ueff) / P_sa / (r**m)

Ks = Ueff * (v/r + 1.0) / P_sa / (r**m)

#print(Ks)
#
#Ks = 10.0**(newton(solve_for_Ks, np.log10(Kr), args=(Kr, Wdot_max, Ueff, v, r, Ff, P_sa)))/ (r**m)
#print(Ks)
#
##%%
#KSs = np.linspace(-6, 0)
#root = []
#
#for Ff in np.arange(0, 1.1, 0.2):
#    vals = []
#    root.append(10.0**(newton(solve_for_Ks, np.log10(Kr), args=(Kr, Wdot_max, Ueff, v, r, Ff, P_sa)))/ (r**m))
#
#    for kss in KSs:
#        vals.append(solve_for_Ks(kss, Kr, Wdot_max, Ueff, v, r, Ff, P_sa))
#    plt.plot(KSs, vals, label=str(Ff))
#plt.ylim(-500, 500)
#plt.legend()
#%%
# this needs an adjustment for when P_0, Ff, etc is in between zero and one.
# this adjustment should acknowlege that the effective erodability will be
# increasing as bedrock starts to be felt. And that sediment thickness will change these dynamics.

params = {'output_filename': 'test_',
          'flow_director' :'FlowDirectorSteepest',
          'depression_finder' : 'DepressionFinderAndRouter',
          'runoff_rate': r,
          'model_grid': 'HexModelGrid',
          'shape': 'rect',
          'random_seed': topography_seed,
          'initial_elevation': 10,
          'initial_noise_std': 3,
          'run_duration' :   total_time,
          'output_interval' : total_time,
          'dt' : 1000.0,
          'number_of_node_rows': 20,
          'number_of_node_columns': 60,
          'node_spacing': 500,
          'solver': 'adaptive',
          'discharge_method': 'discharge_field',
          'linear_diffusivity': 0.00000000001,
          'K_rock_sp': Kr,
          'K_sed_sp':  Ks,
          'F_f': Ff,
          'soil_production__decay_depth': H_p,
          'm_sp': m,
          'soil_production__maximum_rate': Wdot_max,
          'phi': 0,
          'H_star': H_star,
          'n_sp': n,
          'soil_transport_decay_depth': 0.0000001,
          'v_sc': v,
          'initial_soil_thickness': 1.0,
          'BoundaryHandlers': ['NormalFault', 'ClosedNodeBaselevelHandler'],
          'ClosedNodeBaselevelHandler': {'lowering_rate': -U_back},
          'NormalFault': {'faulted_surface' : ['topographic__elevation', 'bedrock__elevation'],
                          'include_boundaries': True,
                          'fault_throw_rate_through_time' : {'time': [0, fault_start-50, fault_start],
                                                             'rate': [U_slow, U_slow, U_fast]},
                          'fault_trace': {'y1': 1500,
                                                'x1': 0,
                                               'y2': 1500,
                                               'x2': 15000}
                          }
        }



#%%
#plan for output files
output_fields =['topographic__elevation', 'soil__depth']

# initialized the model, giving it the correct base level class handle
model = Model(params=params, OutputWriters=ChannelPlotter)

model.run(output_fields=output_fields)
#%%

# find the faulted node with the largest drainage area.
largest_da = np.max(model.grid.at_node['drainage_area'][model.boundary_handler['NormalFault'].faulted_nodes==True])
largest_da_ind = np.where(model.grid.at_node['drainage_area'] == largest_da)[0][0]

#plt.figure()
#imshow_grid(model.grid, model.grid.at_node['drainage_area'] == largest_da, cmap='viridis')
#plt.show()

(profile_IDs, dists_upstr) = analyze_channel_network_and_plot(model.grid, number_of_channels=1, starting_nodes=[largest_da_ind])
elevs = model.z[profile_IDs]

data_frame = pd.DataFrame.from_dict(data={'distance':dists_upstr[0][0],
                                          'elevation': elevs[0],
                                          'area': model.grid.at_node['drainage_area'][profile_IDs][0],
                                          'slope': model.grid.at_node['topographic__steepest_slope'][profile_IDs][0]})
data_frame.to_csv('truth_profile.csv')

with open(sys.argv[-1], 'w') as f:
    relief = model.z[model.grid.core_nodes].max() - model.z[model.grid.core_nodes].min()
    f.write(str(relief))
