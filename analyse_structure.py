"""
Scripts to read in a RADMC-3D structure and plot them.
"""

import numpy as np
import matplotlib.pyplot as plt

# shorthand constants
AU = 1.49597871e13
Msun = 1.98847542e33
mu_gas = 2.37
m_H = 1.67353284e-24
G = 6.67408e-8
kB = 1.38064852e-16
PI = np.pi


class RADMC3Dinput:
    """
    Class to read in and plot the calculated disk grid.
    """

    def __init__(self, modelname):

        self.modelname = modelname
        self.mdir = modelname + '/'

        # Assume for now just a standard cylindrical grid.
        self._read_grid()
        self._read_model()

    def _read_grid(self):
        """Reads the input grid walls. Assumes cylindrical coordiantes."""
        with open(self.mdir + 'amr_grid.inp') as f:
            _ = [f.readline() for _ in range(5)]
            Nr, Nt, Np = f.readline().replace('\n', '').split(' ')
            self.Nr, self.Nt, self.Np = int(Nr), int(Nt), int(Np)
            self.r_walls = np.array([float(f.readline().replace('\n', ''))
                                     for _ in range(self.Nr)])
            self.t_walls = np.array([float(f.readline().replace('\n', ''))
                                     for _ in range(self.Nt)])
            self.p_walls = np.array([float(f.readline().replace('\n', ''))
                                     for _ in range(self.Np)])
        self.rvals = 0.5 * (self.r_walls[1:] + self.r_walls[:-1])
        self.tvals = 0.5 * (self.t_walls[1:] + self.t_walls[:-1])
        self.pvals = 0.5 * (self.p_walls[1:] + self.p_walls[:-1])

    @property
    def vrad(self):
        return self.vgas[0]

    @property
    def vazi(self):
        return self.vgas[1]

    @property
    def vphi(self):
        return self.vgas[2]

    def _read_model(self):
        """Reads the model data from various `*.inp` files."""
        self.ngas = self._read_model_file('gas_density.inp')[0]
        self.Tgas = self._read_model_file('gas_temperature.inp')[0]
        self.vgas = self._read_model_file('gas_velocity.inp')
        return

    def _read_model_file(self, fn):
        """Reads a specific model file and returns a reshaped data array."""
        data = np.atleast_2d(np.loadtxt(self.mdir + fn, skiprows=2).T)
        return data.reshape((data.shape[0], self.Nr, self.Nt, self.Np))

    def plot_grid(self, fig=None):
        """Plot the grids."""
        if fig is None:
            fig, ax = plt.subplots()
        return
