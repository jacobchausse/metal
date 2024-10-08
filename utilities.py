from __future__ import annotations
import numpy as np
import numba
import matplotlib.pyplot as plt
from scipy import signal as spsig

class TrailGrid:

    def __init__(self, Nx: int, Ny: int, xmax: float, xmin: float, ymax: float, ymin: float, dissipation: float):
        self.Nx=Nx
        self.Ny=Ny
        self.xmax=xmax
        self.xmin=xmin
        self.ymax=ymax
        self.ymin=ymin
        self.dissipation = dissipation

        self.dNxdx = (Nx - 1)/(xmax - xmin)
        self.dNydy = (Ny - 1)/(ymax - ymin)

        self.grid = np.zeros((Nx, Ny))

        self.kernel = None
        self.bias = None

    def spawn_trail(self, agents: Agents, unmoved):

        indicesx = (agents.get_positions()[:, 0] - self.xmin) * self.dNxdx + 1
        indicesy = (agents.get_positions()[:, 1] - self.ymin) * self.dNydy + 1

        indicesx = indicesx.astype(int)
        indicesy = indicesy.astype(int)

        indicesx = indicesx[~unmoved]
        indicesy = indicesy[~unmoved]

        trail_grid = np.zeros_like(self.grid)
        trail_grid[indicesx, indicesy] += 1

        if (self.kernel is None):
            raise Exception('Did not set kernel')

        self.grid += spsig.fftconvolve(trail_grid, self.kernel, mode='same')

    def get_values_from_positions(self, positions: np.ndarray):
        indicesx = (positions[:, 0] - self.xmin) * self.dNxdx + 1
        indicesy = (positions[:, 1] - self.ymin) * self.dNydy + 1

        indicesx = indicesx.astype(int)
        indicesy = indicesy.astype(int)

        if self.bias is not None:
            return self.grid[indicesx, indicesy] + self.bias[indicesx, indicesy]

        return self.grid[indicesx, indicesy]

    def set_kernel_gaussian(self, sigma: float):
        xrange = np.linspace(self.xmin, self.xmax, self.Nx)
        yrange = np.linspace(self.ymin, self.ymax, self.Ny)

        kernel = np.exp(-((xrange[:, None]**2 + yrange[None, :]**2) / (2 * sigma**2)))
        kernel /= np.sum(kernel)
        self.kernel = kernel

    def set_kernel_square(self, size: int, value: int=1):
        self.kernel = value*np.ones((size, size))

    def set_kernel_custom(self, kernel: np.ndarray):
        self.kernel = kernel

    def set_bias(self, bias: np.ndarray):
        self.bias = bias

    def dissipate_trail(self):
        self.grid *= self.dissipation
    
    def show(self, threshold: float = None):
        if threshold is not None:
            plt.imshow(self.grid > threshold)
        else:
            plt.imshow(self.grid)
        plt.show()

    def show_bias(self):
        if self.bias is not None:
            plt.imshow(self.bias)
            plt.show()

@numba.njit()
def update_positions(positions: np.ndarray, position_grid: np.ndarray, angles: np.ndarray, ds: float, N_agents: int, dNxdx: float, dNydy: float, xmax: float, xmin: float, ymax: float, ymin: float):
    minxy = np.array([xmin, ymin])
    maxxy = np.array([xmax, ymax])
    dNdxy = np.array([dNxdx, dNydy])

    unmoved = np.zeros(N_agents) > 0

    for i in range(N_agents):
        dxy = np.array([np.cos(angles[i]), np.sin(angles[i])])*ds

        curr_pos = positions[i, :]
        new_pos = curr_pos + dxy

        new_pos += (maxxy - minxy) * (1 * (new_pos < minxy) - 1 * (new_pos > maxxy))

        grid_pos = ((curr_pos - minxy) * dNdxy + 1).astype(np.int64)
        new_grid_pos = ((new_pos - minxy) * dNdxy + 1).astype(np.int64)

        if position_grid[new_grid_pos[0], new_grid_pos[1]]:
            unmoved[i] = True
            angles[i] += np.random.random()*np.pi*2# - np.pi/2
            continue
        
        position_grid[grid_pos[0], grid_pos[1]] = 0
        position_grid[new_grid_pos[0], new_grid_pos[1]] = 1

        positions[i, :] = new_pos

    return positions, position_grid, unmoved

@numba.njit()
def update_angles(values_middle: np.ndarray, values_left: np.ndarray, values_right: np.ndarray, middle_angles: np.ndarray, left_angles: np.ndarray, right_angles: np.ndarray, N_agents: int):

    angles = np.zeros(N_agents)

    for i in range(N_agents):
        middle = values_middle[i]
        left = values_left[i]
        right = values_right[i]

        if (middle > left) and (middle > right):
            angles[i] = middle_angles[i]
        elif (middle < left) and (middle < right):
            random = np.random.random() > 0.5
            angles[i] = random*left_angles[i] + (not random)*right_angles[i]
        elif left < right:
            angles[i] = right_angles[i]
        elif left > right:
            angles[i] = left_angles[i]
        else:
            angles[i] = middle_angles[i]

    return angles

class Agents:

    def __init__(self, N_agents, Nx: int, Ny: int, xmax: float, xmin: float, ymax: float, ymin: float, trail: TrailGrid, positions_radius: float=None, sense_rate: int=1):
        self.N = N_agents
        self.Nx = Nx
        self.Ny = Ny
        self.xmax = xmax
        self.xmin = xmin
        self.ymax = ymax
        self.ymin = ymin
        self.dNxdx = (Nx - 1)/(xmax - xmin)
        self.dNydy = (Ny - 1)/(ymax - ymin)
        self.sense_rate = sense_rate
        self.sense_counter = 1

        self.trail = trail

        self.positions = self._initialize_random_positions(positions_radius)
        self.positions_grid = self._initialize_positions_grid()
        self.angles = self._initialize_random_angles()

        self.trail.spawn_trail(self, np.zeros(N_agents, dtype=int))

    def _initialize_random_angles(self):
        return np.random.random(self.N)*2*np.pi

    def _initialize_random_positions(self, positions_radius: float=None):
        
        if positions_radius is None:
            positions = np.random.random((self.N, 2))

            positions[:, 0] = positions[:, 0] * (self.xmax - self.xmin) + self.xmin
            positions[:, 1] = positions[:, 1] * (self.ymax - self.ymin) + self.ymin

        else:
            theta = np.random.random(self.N)*2*np.pi
            r = positions_radius*np.sqrt(np.random.random(self.N))

            positions = np.zeros((self.N, 2))
            positions[:, 0] = r*np.cos(theta)
            positions[:, 1] = r*np.sin(theta)

        return positions

    def _initialize_positions_grid(self):
        indicesx = (self.get_positions()[:, 0] - self.xmin) * self.dNxdx + 1
        indicesy = (self.get_positions()[:, 1] - self.ymin) * self.dNydy + 1

        indicesx = indicesx.astype(int)
        indicesy = indicesy.astype(int)

        positions_grid = np.zeros_like(self.trail.grid, dtype=int)
        positions_grid[indicesx, indicesy] = 1
        
        return positions_grid

    def cull_positions(self, mask: np.ndarray):
        indicesx = (self.get_positions()[:, 0] - self.xmin) * self.dNxdx + 1
        indicesy = (self.get_positions()[:, 1] - self.ymin) * self.dNydy + 1

        indicesx = indicesx.astype(int)
        indicesy = indicesy.astype(int)

        pos_mask = mask[indicesx, indicesy] > 0

        self.positions = self.positions[pos_mask, :]
        self.angles = self.angles[pos_mask]

        self.N = self.angles.size

        self.positions_grid = self._initialize_positions_grid()

    def get_positions(self) -> np.ndarray:
        return self.positions
    
    def step_straight(self, ds: float) -> np.ndarray:
        self.positions[:, 0] += np.cos(self.angles)*ds
        self.positions[:, 1] += np.sin(self.angles)*ds

        self.positions = self.wrap_positions(self.positions)

    def step_seek(self, ds: float, dsense: float, dtheta: float):

        self.positions, self.positions_grid, unmoved = update_positions(self.positions, self.positions_grid, self.angles, ds, self.N, self.trail.dNxdx, self.trail.dNydy, self.xmax, self.xmin, self.ymax, self.ymin)

        if (self.sense_counter % self.sense_rate) == 0:
            self.sense(dsense, dtheta)
            self.sense_counter = 1
        else:
            self.sense_counter += 1

        self.trail.spawn_trail(self, unmoved)
        self.trail.dissipate_trail()

    def sense(self, dsense: float, dtheta: float):
        middle_positions = np.array(self.positions)
        left_positions = np.array(self.positions)
        right_positions = np.array(self.positions)

        middle_angles = self.angles
        left_angles = self.angles - dtheta
        right_angles = self.angles + dtheta

        middle_positions[:, 0] += np.cos(middle_angles)*dsense
        middle_positions[:, 1] += np.sin(middle_angles)*dsense

        left_positions[:, 0] += np.cos(left_angles)*dsense
        left_positions[:, 1] += np.sin(left_angles)*dsense
        
        right_positions[:, 0] += np.cos(right_angles)*dsense
        right_positions[:, 1] += np.sin(right_angles)*dsense

        middle_positions = self.wrap_positions(middle_positions)    
        left_positions = self.wrap_positions(left_positions)    
        right_positions = self.wrap_positions(right_positions)

        values_middle = self.trail.get_values_from_positions(middle_positions)     
        values_left = self.trail.get_values_from_positions(left_positions)     
        values_right = self.trail.get_values_from_positions(right_positions)

        self.angles = update_angles(values_middle, values_left, values_right, middle_angles, left_angles, right_angles, self.N)

    def wrap_positions(self, positions: np.ndarray):

        positions[:, 0] += (self.xmax - self.xmin) * (1 * (positions[:, 0] < self.xmin) - 1 * (positions[:, 0] > self.xmax))
        positions[:, 1] += (self.ymax - self.ymin) * (1 * (positions[:, 1] < self.ymin) - 1 * (positions[:, 1] > self.ymax))
        
        return positions
    
    def show(self):
        plt.imshow(self.positions_grid)
        plt.show()
