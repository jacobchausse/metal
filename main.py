import numpy as np
import numba
import matplotlib.pyplot as plt
from utilities import *
from tqdm import tqdm
from scipy import signal as spsig
from matplotlib.image import imread
import cv2

image = imread('chem7.png')[:, :, 0]

Nx = image.shape[0]
Ny = image.shape[1]

agent_fraction = 0.1

N_agents = int(agent_fraction*Nx*Ny)

xmax = 1
xmin = -1
ymax = 1
ymin = -1

ds = np.sqrt(2*((xmax - xmin)/Nx)**2)

dissipation = 0.8

trail = TrailGrid(Nx, Ny, xmax, xmin, ymax, ymin, dissipation)

xrange = np.linspace(xmin, xmax, Nx)
yrange = np.linspace(ymin, ymax, Ny)

image *= 1000

trail.set_bias(image)
trail.show_bias()

krange = np.linspace(-1, 1, 11)
X, Y = np.meshgrid(krange, krange)

kernel = np.exp(-(X**2 + Y**2)/0.3)

plt.imshow(kernel)
plt.colorbar()
plt.show()

trail.set_kernel_custom(kernel)

#trail.set_kernel_square(5, 1)
#trail.set_kernel_gaussian(ds)

sense_rate = 3
agents = Agents(N_agents, Nx, Ny, xmax, xmin, ymax, ymin, trail, None, sense_rate)

agents.cull_positions(image)

agents.show()

for i in tqdm(range(50000)):
    agents.step_seek(2*ds, 10*ds, np.pi/4)
    # Display the updated image

    trail_grid = trail.grid / np.max(trail.grid)

    positions_grid = agents.positions_grid / np.max(agents.positions_grid)

    cv2.imshow('animation', np.hstack([trail_grid, positions_grid]))

    cv2.waitKey(10)

trail.show()