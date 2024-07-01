r"""
Metro simulation
================

Example of a simulation of the evolution of a graph signal over the Santiago Metro
network.

The initial condition is a graph signal with only one positive integer value
bigger.The plots or animation show how this signal distributes over the network
using:

.. math:: y = AD^-1x

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
import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import animation
from pygsp2.utils_examples import make_metro_graph, fetch_data
import pygsp2 as pg

current_dir = os.getcwd()
os.chdir(current_dir)


# If set to true make animation,
# otherwise store each frame as png
MAKE_ANIMATION = True

# %% Download data
assets_dir = os.path.join(current_dir, 'data')
fetch_data(assets_dir, "metro")


# %% Load graph and compute adjacency and node degree
G, pos = make_metro_graph(
    edgesfile=os.path.join(assets_dir, 'santiago_metro_stations_connections.txt'),
    coordsfile=os.path.join(assets_dir, 'santiago_metro_stations_coords.geojson')
)

W = nx.adjacency_matrix(G).toarray()
D = np.diag(W.sum(1))
D_inv = np.linalg.inv(D)

# %% Initialize parameters
NSTEPS = 30  # Arbitrary units, steps
INIT_VALUE = 5000  # Initial conditions
signal = np.zeros([NSTEPS, len(W)])
signal[0, np.random.randint(0, len(W), 1)] = INIT_VALUE

# %% Simulate advancing through the graph
# As a physical distribution network

print('Prerforming simulation...')

for i in np.arange(1, NSTEPS):
    signal[i, :] = W@D_inv@signal[i-1, :]

print('Finished.')

# %% Make animation
if MAKE_ANIMATION:
    # Initiate figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 9))
    cmap = plt.get_cmap('viridis')

    # Draw edges and nodes
    im = nx.draw_networkx_edges(G, pos, node_size=20, ax=ax)
    im = nx.draw_networkx_nodes(G, pos, node_color='gray', node_size=20, ax=ax)
    ax.set_title('T = 0')
    cbar = plt.colorbar(im, ax=ax, ticks=[0, 0.5, 1], label='Signal')
    cbar.set_ticklabels([0, np.amax(signal)/2, np.amax(signal)])

    # Define function for animation

    def update(frame):
        """Function that defines what happens in each frame."""

        idxs = ((np.where(signal[frame, :] > 0)[0]).astype(int))
        nodelist = list(np.array(G)[idxs])
        colors = cmap(signal[frame, idxs] / INIT_VALUE)

        nx.draw_networkx_nodes(
            G, pos, node_color='gray', node_size=20, ax=ax)
        nx.draw_networkx_nodes(G, pos, node_color=colors, nodelist=nodelist,
                               node_size=20, ax=ax)
        ax.set_title(f'T = {frame}')

    anim = animation.FuncAnimation(fig, update,
                                   frames=np.arange(0, len(signal)),
                                   interval=50)

    # saving to gif using ffmpeg writer
    writervideo = animation.PillowWriter(fps=5)

    anim.save('metro_simulation.gif', writer=writervideo)
    plt.show()

else:
    try:
        os.mkdir('metro_simulation/')
    except FileExistsError:
        print(
            'Warning: It seems like this folder already exists. Overwritting...')

    # Initiate figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 9))
    cmap = plt.get_cmap('viridis')

    # Draw edges and nodes
    im = nx.draw_networkx_edges(G, pos, node_size=20, ax=ax)
    im = nx.draw_networkx_nodes(G, pos, node_color='gray', node_size=20, ax=ax)
    ax.set_title('T = 0')
    cbar = plt.colorbar(im, ax=ax, ticks=[0, 0.5, 1], label='Signal')
    cbar.set_ticklabels([0, np.amax(signal)/2, np.amax(signal)])

    # Iterate through each graph signal
    for i, s in enumerate(signal):

        idxs = (np.where(s > 0)[0]).astype(int)
        nodelist = list(np.array(G)[idxs])
        colors = cmap(s[idxs] / INIT_VALUE)

        im = nx.draw_networkx_nodes(
            G, pos, node_color='gray', node_size=20, ax=ax)
        im = nx.draw_networkx_nodes(G, pos, node_color=colors, nodelist=nodelist,
                                    node_size=20, ax=ax)
        ax.set_title(f'T = {i}')

        fig.savefig(f'metro_simulation/{i}.png')
