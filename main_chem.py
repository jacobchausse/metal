import numpy as np
import numba
import matplotlib.pyplot as plt
from utilities import *
from tqdm import tqdm
from scipy import signal as spsig
from matplotlib.image import imread
import cv2

number = 12

image = imread(f'images/chem{number}.png')[:, :, 0]

Nx = image.shape[0]
Ny = image.shape[1]

print(image.shape[1])

agent_fraction = 0.1

N_agents = int(agent_fraction*Nx*Ny)

xmax = 1
xmin = -1
ymax = 1*Ny/Nx
ymin = -1*Ny/Nx

ds = np.sqrt(2*((xmax - xmin)/Nx)**2)

trail = TrailGrid(Nx, Ny, xmax, xmin, ymax, ymin, dissipation=0.8)

trail.set_bias(100*image)
trail.show_bias()

trail.set_kernel_gaussian(31, 5)
trail.show_kernel()

sense_rate = 2
agents = Agents(N_agents, Nx, Ny, xmax, xmin, ymax, ymin, trail, None, sense_rate)

agents.cull_positions(image)
agents.show()

for i in tqdm(range(500)):
    agents.step_seek(5*ds, 10*ds, np.pi/2.5)

    trail_grid = trail.grid / np.max(trail.grid)

    positions_grid = agents.positions_grid / np.max(agents.positions_grid)

    #cv2.imshow('animation', np.stack([trail_grid, trail_grid, positions_grid], axis=2))
    mask = 1.0*(trail_grid > 0.4)
    cv2.imshow('animation', np.hstack([mask, trail_grid]))

    cv2.waitKey(1)

trail.show()
agents.show()

colour = mask*1000
image = np.stack([colour, colour, colour, colour], axis=2)

cv2.imwrite(f'results/chem{number}.png', image)
