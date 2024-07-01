r"""
Metro Regression
================

Data imputing example in a  Santiago Metro station.

Compute the number of exits in a random metro station. Then, compute the
regression over the whole network and plot the error of the regression. Here,
Tikhonov regession is used.  Lastly, the average of neighboring nodes is also
used to compare the error of the regression.

To run this example, you need to download three files and place them in the
same directory as this script.

1. Download the file `Tablas de subidas y bajadas nov23.zip` from this link:

https://www.dtpm.cl/descargas/modelos_y_matrices/Tablas%20de%20subidas%20y%20bajadas%20nov23.zip

Then, uncompress the zip file and copy `2023.11 Matriz_baj_SS_MH.xlsb` to the
same location as this script.

2. Download the file `santiago_metro_stations_coords.geojson` from this link:

https://zenodo.org/records/11637462/files/santiago_metro_stations_coords.geojson

3. Download the file `santiago_metro_stations_connections.txt` from this link:

https://zenodo.org/records/11637462/files/santiago_metro_stations_connections.txt
"""
# %%
import sys
import os
import pandas as pd
import numpy as np
from unidecode import unidecode
import networkx as nx
import matplotlib.pyplot as plt
from pygsp2.utils_examples import (
    make_metro_graph, 
    plot_signal_in_graph, 
    metro_database_preprocessing,
    fetch_data
)
from pygsp2 import graphs, learning

current_dir = os.getcwd()
os.chdir(current_dir)

# %% Load data
assets_dir = os.path.join(current_dir, 'data')
fetch_data(assets_dir, "metro")

try:
    commutes = pd.read_excel(
        os.path.join(assets_dir, '2023.11 Matriz_baj_SS_MH.xlsb'),
        header=1,
        sheet_name='bajadas_prom_laboral')
except FileNotFoundError:
    print(f'Data file was not found in:\n {os.getcwd()}')
    print('Download it from:\n' +
          r'https://www.dtpm.cl/descargas/modelos_y_matrices/Tablas%20de%20subidas%20y%20bajadas%20nov23.zip')
    sys.exit(1)

# %% Load graph
G, pos = make_metro_graph(
    edgesfile=os.path.join(assets_dir, 'santiago_metro_stations_connections.txt'),
    coordsfile=os.path.join(assets_dir, 'santiago_metro_stations_coords.geojson')
)
pos_list = [(G.nodes[node]['y'], G.nodes[node]['x']) for node in G.nodes]

# Extract adjacency matrix
W = nx.adjacency_matrix(G).toarray()
G_pygsp = graphs.Graph(W)

# Node degree matrix
D = np.diag(G_pygsp.d)
# Compute inverse for later
D_inv = np.linalg.inv(D)

# Convert to uppercase
stations = [name.upper() for name in list(G.nodes)]
stations = [unidecode(station) for station in stations]

# %% Use only metro commutes in database
metro_commutes, signal = metro_database_preprocessing(commutes, stations)

# %% Choose a point in the network and delete its value to regress it

signal2 = signal.copy()
station_idx = np.random.randint(0, len(signal2))
# station_idx = 24

print(f'Deleted Station: {stations[station_idx]}')
signal2[station_idx] = np.nan
mask = np.ones(len(signal)).astype(bool)
mask[station_idx] = False

# Use tikhonov regression to recover the signal
recovered_signal = learning.regression_tikhonov(
    G_pygsp, signal2, mask, tau=0.5)

# Compute the average of the nodes around the missing value
average = (W@D_inv@signal)[station_idx]

print(f'Estimated: {recovered_signal[station_idx]:.2f}')
print(f'One-hop Average: {average:.2f}')
print(f'Real: {signal[station_idx]:.2f}')

# %% Compute error with each station
tikhonov_estimation = np.zeros_like(signal)
average_estimation = W@D_inv@signal

for i, s in enumerate(signal):
    # Allocate new signal
    signal2 = signal.copy()
    print(f'Deleted Station: {stations[i]}')

    # Delete value in the signal
    signal2[i] = np.nan
    mask = np.ones(len(signal)).astype(bool)
    mask[i] = False

    # Use tikhonov regression to recover the signal
    recovered_signal = learning.regression_tikhonov(
        G_pygsp, signal2, mask, tau=0.5)
    tikhonov_estimation[i] = recovered_signal[i]

abs_err = np.abs(tikhonov_estimation - signal)

# %% Plot Tikhonov error in graph
fig, ax = plot_signal_in_graph(G, abs_err,
                     title='Error of Tikhonov Regression',
                     label='Error absoluto')
#fig.savefig('metro_regression_tikhonov_error.png', dpi=300)
# %% Plot Average error in graph
# Change variable to error with average estimation
abs_err = np.abs(average_estimation - signal)

fig, ax = plot_signal_in_graph(G, abs_err,
                     title=r'Error of $y = AD^{-1}x$',
                     label='Error absoluto')
#fig.savefig('metro_regression_error_abs.png', dpi=300)
plt.show()