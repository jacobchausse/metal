import numpy as np
import numba
import matplotlib.pyplot as plt
from utilities import *
from tqdm import tqdm
from scipy import signal as spsig
from matplotlib.image import imread
import cv2

image = imread(f'images/github_pfp.png')[:, :, 0]

Nx = image.shape[0]
Ny = image.shape[1]

print(image.shape[1])

agent_fraction = 0.2

N_agents = int(agent_fraction*Nx*Ny)

xmax = 1
xmin = -1
ymax = 1*Ny/Nx
ymin = -1*Ny/Nx

ds = np.sqrt(2*((xmax - xmin)/Nx)**2)

trail = TrailGrid(Nx, Ny, xmax, xmin, ymax, ymin, dissipation=0.5)

trail.set_bias(100*image)
trail.show_bias()

trail.set_kernel_gaussian(31, 5)
trail.show_kernel()

sense_rate = 2
agents = Agents(N_agents, Nx, Ny, xmax, xmin, ymax, ymin, trail, None, sense_rate)

agents.cull_positions(image)
agents.show()

for i in tqdm(range(100)):
    agents.step_seek(5*ds, 10*ds, np.pi/2.5)

    trail_grid = trail.grid / np.max(trail.grid)

    positions_grid = agents.positions_grid / np.max(agents.positions_grid)

    cv2.imshow('animation', np.stack([trail_grid, trail_grid, positions_grid], axis=2))

    cv2.waitKey(1)

image = 255*np.stack([trail_grid, positions_grid, trail_grid, np.ones((Nx, Ny))], axis=2)

cv2.imwrite(f'results/github_pfp.png', image)
