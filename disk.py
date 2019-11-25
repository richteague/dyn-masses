"""
Based on code by Sean Andrews (CfA) and Jane Huang (CfA).

TODO:

    - Check the pressure correction and self-gravity velocities.
"""

import yaml
import numpy as np
import scipy.constants as sc
import matplotlib.pyplot as plt
from scipy.integrate import cumtrapz
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable


class disk:
    """
    Standard protoplanetary disk model.

    Args:
        modelname (str): Name of the model with parameters in ``'modelname.yaml'``.
    """

    msun = 1.988e30  # stellar mass [kg]
    mu = 2.37  # mean molecular weight [m_p]

    min_dens = 1e3  # minimum gas density in [H2/ccm]
    max_dens = 1e20  # maximum gas density in [H2/ccm]
    min_temp = 5e0  # minimum temperature in [K]
    max_temp = 5e2  # maximum temperature in [K]

    def __init__(self, modelname):

        # open configuration file
        conf = open(modelname + ".yaml")
        config = yaml.load(conf, Loader=yaml.FullLoader)
        self.host_params = config["host_params"]
        self.disk_params = config["disk_params"]
        conf.close()

        # define the host parameters
        self.mstar = self.host_params["M_star"] * self.msun

        # define grids - 'disk_grid'.
        self._init_grids(self.disk_params["disk_grid"])
        assert self.zvals.size == self.nz
        assert self.rvals.size == self.nr

        # calculate the gas and dust surface densities
        self.sigma_g = self._parse_function("surface_density", self.disk_params["gas_surface_density"])
        assert self.sigma_g.size == self.rvals.size
        self.sigma_d = self._parse_function("surface_density", self.disk_params["dust_surface_density"])
        assert self.sigma_d.size == self.rvals.size

        # calculate the (coupled) gas and dust temperatures
        self.temperature = self._parse_function("temperature", self.disk_params["temperature"])
        self.temperature = np.clip(self.temperature, self.min_temp, self.max_temp)
        assert self.temperature.shape == (self.zvals.size, self.rvals.size)
        if np.any(self.zpnts[-1] < 4.0 * self.scaleheight()[0]):
            print("WARNING: Vertical grid does not extent beyond 4 Hp in some regions.")

        # calculate the gas density structure
        self.density_g = self._parse_function("density", self.disk_params["density"])
        self.density_g = np.clip(self.density_g,
                                 self.min_dens * self.mu * sc.m_p * 1e2,
                                 self.max_dens * self.mu * sc.m_p * 1e2,)
        assert self.density_g.shape == self.temperature.shape

        # calculate the molecular abundance structure
        self.abundance = self._parse_function("abundance", self.disk_params["abundance"])
        assert self.abundance.shape == self.temperature.shape

        # calculate the rotational velocities
        self.vphi = self._parse_function("rotation", self.disk_params["rotation"])
        print("Disk model successfully read in.")

    # Gridding functions.

    def _init_grids(self, params):
        """Initialize the grids with values specified in 'disk_grid'."""
        self.nr = int(params.pop("nr", 150))
        self.logr = params.pop("logr", True)
        self.rmin = params.pop("rmin", -1.0)
        self.rmax = params.pop("rmax", 2.0)

        if self.logr:
            self.rvals = np.logspace(self.rmin, self.rmax, self.nr)
            self.rmin = 10 ** self.rmin
            self.rmax = 10 ** self.rmax
        else:
            self.rvals = np.linspace(self.rmin, self.rmax, self.nr)

        self.nz = int(params.pop("nr", 150))
        self.logz = params.pop("logz", True)
        self.zmin = params.pop("zmin", -1.0)
        self.zmax = params.pop("zmax", 1.0)

        if self.logz:
            self.zvals = np.logspace(self.zmin, self.zmax, self.nz)
            self.zmin = 10 ** self.zmin
            self.zmax = 10 ** self.zmax
        else:
            self.zvals = np.linspace(self.zmin, self.zmax, self.nz)

        self.rpnts, self.zpnts = np.meshgrid(self.rvals, self.zvals)
        self.polar_r = np.hypot(self.rpnts, self.zpnts)
        self.polar_t = np.arctan2(self.zpnts, self.rpnts)

        self.rpnts_m = self.rpnts * sc.au
        self.zpnts_m = self.zpnts * sc.au
        self.rvals_m = self.rvals * sc.au
        self.zvals_m = self.zvals * sc.au

    # Wrapper function to parse the arguments.

    def _parse_function(self, func_family, user_input):
        """Initialize the surface density. Values in 'surface_density'."""
        func = user_input["type"]
        try:
            args = user_input["arguments"]
        except KeyError:
            args = {}
        return eval("self._{}_{}(**args)".format(func_family, func))

    # Surface density functions.

    def _surface_density_self_similar(self, **args):
        """Self-similar surface density profile in [g/sqcm]."""
        try:
            r0, sig0, pg1 = args["r0"], args["sig0"], args["pg1"]
            pg2 = args.pop("pg2", 2.0 - pg1)
        except KeyError:
            raise ValueError("Must specify at least `r0`, `sig0`, `pg1`.")
        sigma = self.powerlaw(self.rvals, sig0, -pg1, r0)
        return sigma * np.exp(-(self.rvals / r0) ** pg2)

    def _surface_density_powerlaw(self, **args):
        """Simple powerlaw surface density profile in [g/sqcm]."""
        try:
            r0, sig0, pg1 = args["r0"], args["sig0"], args["pg1"]
        except KeyError:
            raise ValueError("Must specify `r0`, `sig0`, `pg1`.")
        return self.powerlaw(self.rvals, sig0, -pg1, r0)

    # Temperature functions.

    def _temperature_dartois(self, **args):
        """Temperature function from Dartois++ (2003) in [K]."""
        try:
            r0, T0mid = args["r0"], args["T0mid"]
            T0atm = args.pop("T0atm", T0mid)
            Tqmid = args["Tqmid"]
            Tqatm = args.pop("Tqatm", Tqmid)
            delta = args.pop("delta", 2.0)
            ZqHp = args.pop("ZqHp", 4.0)
        except KeyError:
            raise ValueError("Must specify at least `r0`, `T0mid`, `Tqmid`.")
        Tmid = self.powerlaw(self.rpnts, T0mid, Tqmid, r0)
        Tatm = self.powerlaw(self.rpnts, T0atm, Tqatm, r0)
        zatm = ZqHp * self.scaleheight(T=Tmid)
        T = np.cos(np.pi * self.zpnts / 2.0 / zatm) ** (2.0 * delta)
        T = np.where(abs(self.zpnts) < zatm, (Tmid - Tatm) * T, 0.0)
        return T + Tatm

    def scaleheight(self, T=None):
        """Midplane gas pressure scale height in [au]."""
        T = self.temperature if T is None else T
        Hp = sc.k * T * self.rvals_m ** 3
        Hp /= sc.G * self.mstar * disk.mu * sc.m_p
        return np.sqrt(Hp) / sc.au

    def soundspeed(self, T=None):
        """Gas soundspeed in [m/s]."""
        T = self.temperature if T is None else T
        return np.sqrt(sc.k * T / self.mu / sc.m_p)

    # Gas density functions.

    def _density_isothermal(self, **args):
        """Vertically isothermal gas density structure in [g/ccm]."""
        dens = self.sigma_g / self.scaleheight() / np.sqrt(2.0 * np.pi) / sc.au / 1e2
        return self.gaussian(self.zpnts, 0.0, self.Hp, dens)

    def _density_hydrostatic(self, **args):
        """Hydrostatic gas density structure in [g/ccm]."""
        dT = np.diff(np.log(self.temperature), axis=0)
        dT = np.vstack((dT, dT[-1]))
        dz = np.diff(self.zpnts * sc.au, axis=0)
        dz = np.vstack((dz, dz[-1]))
        drho = dz / self.soundspeed() ** 2
        drho *= sc.G * self.mstar * self.zpnts_m
        drho /= np.hypot(self.rpnts_m, self.zpnts_m) ** 3
        rho = np.ones(self.temperature.shape)
        for i in range(self.rvals.size):
            for j in range(1, self.zvals.size):
                rho[j, i] = np.exp(np.log(max(rho[j - 1, i], 1e-100)) - drho[j, i])
        rho *= self.sigma_g[None, :] / np.trapz(rho, self.zvals_m * 1e2, axis=0)
        return rho

    # Molecular abundance functions.

    def _abundance_chemical(self, **args):
        """Chemical parameterization including dissociation and freezeout."""
        try:
            xmol = args["xmol"]
        except KeyError:
            print("Must specify at least `xmol`.")
        not_frzn = self.temperature > args.pop("tfreeze", 21.0)
        not_diss = self.density_g[::-1] / sc.m_p / self.mu / 1e2
        not_diss = cumtrapz(not_diss, self.zvals_m * 1e2, axis=0, initial=0)
        not_diss = not_diss[::-1] > args.pop("ndiss", 1.3e21)
        return np.where(np.logical_and(not_frzn, not_diss), xmol,
                        xmol * args.pop("depletion", 1e-8))

    def _abundance_cut(self, **args):
        """Simple parametric region. TODO: Better name."""
        try:
            xmol = args["xmol"]
        except KeyError:
            print("Must specify at least `xmol`.")
        zrmin = args.pop("zrmin", 0.0)
        zrmax = args.pop("zrmax", 1.0)
        rmin = args.pop("rmin", self.rvals.min())
        rmax = args.pop("rmax", self.rvals.max())
        zr_mask = np.logical_and(self.zpnts / self.rpnts <= zrmax,
                                 self.zpnts / self.rpnts >= zrmin)
        r_mask = np.logical_and(self.rpnts >= rmin, self.rpnts <= rmax)
        return np.where(np.logical_and(zr_mask, r_mask), xmol,
                        xmol * args.pop("depletion", 1e-8))

    def _rotation_keplerian(self, **args):
        """Keplerian rotation."""
        vkep2 = sc.G * self.mstar * self.rpnts_m ** 2
        if args.pop("height", True):
            vkep2 /= np.hypot(self.rpnts_m, self.zpnts_m) ** 3
        else:
            vkep2 /= self.rpnts_m ** 3

        if args.pop("pressure", False):
            n = 1e3 * self.density_g / sc.m_p / self.mu
            P = n * sc.k * self.temperature
            dPdr = np.gradient(P, self.rvals_m, axis=1)
            vprs2 = self.rpnts_m * dPdr / n / sc.m_p / self.mu
            vprs2 = np.where(np.isfinite(vprs2), vprs2, 0.0)
        else:
            vprs2 = 0.0

        if args.pop("selfgrav", False):
            vgrv2 = 0.0
        else:
            vgrv2 = 0.0
        return np.sqrt(vkep2 + vprs2 + vgrv2)

    # Grid output.

    def write_to_grid(self, param, grid):
            
        """Write ``param`` to the grid."""
        towrite = getattr(self, param)
        return towrite

    # Convenience properties.

    @property
    def cs(self):
        """Midplane sound speed in [m/s]."""
        return self.soundspeed(T=self.temperature[0])

    @property
    def Hp(self):
        """Midplane pressure scale height in [au]."""
        return self.scaleheight(T=self.temperature[0])

    @property
    def Nmol(self):
        """Molecular column density in [/sqcm]."""
        nmol = self.density_g * self.abundance / sc.m_p / self.mu
        return np.trapz(nmol, axis=0, x=self.zvals * 1e2)

    @property
    def vmid(self):
        """Midplane rotation velocity in [m/s]."""
        return self.vphi[0]

    # Plotting functions.

    def plot_density(self, fig=None, contourf_kwargs=None, full=True):
        """Plot the density profile."""
        fig, ax = self._grab_axes(fig)
        toplot = np.vstack([self.density_g[::-1], self.density_g])
        toplot = np.log10(toplot / sc.m_p / self.mu)
        yaxis = np.concatenate([-self.zvals[::-1], self.zvals])

        contourf_kwargs = {} if contourf_kwargs is None else contourf_kwargs
        levels = np.linspace(toplot.min(), toplot.max(), 50)
        levels = contourf_kwargs.pop("levels", levels)
        cmap = contourf_kwargs.pop("cmap", "bone_r")
        im = ax.contourf(self.rvals, yaxis, toplot, levels=levels,
                         cmap=cmap, **contourf_kwargs)

        cax = make_axes_locatable(ax)
        cax = cax.append_axes("right", size="4.5%" if full else "3%",
                              pad="2.25%" if full else "1.5%")
        cb = plt.colorbar(im, cax=cax, ticks=np.arange(-30, 30, 2))
        cb.set_label(r"$\log_{10}(n_{\rm H_2}\,\,[{\rm cm^{-2}}])$",
                     rotation=270, labelpad=15)

        self._gentrify_structure_ax(ax, full=full)
        return fig

    def plot_density_contours(self, fig=None, levels=None, contour_kwargs=None, full=True):
        """Overplot the density contours."""
        fig, ax = self._grab_axes(fig)
        toplot = np.vstack([self.density_g[::-1], self.density_g])
        toplot = np.log10(toplot / sc.m_p / self.mu)
        yaxis = np.concatenate([-self.zvals[::-1], self.zvals])
        contour_kwargs = {} if contour_kwargs is None else contour_kwargs
        levels = (np.linspace(toplot.min(), toplot.max(), 20) if levels is None else levels)
        contour_kwargs["linestyles"] = contour_kwargs.pop("linestyles", "-")
        contour_kwargs["linewidths"] = contour_kwargs.pop("linewidths", 1.0)
        contour_kwargs["colors"] = contour_kwargs.pop("colors", "k")
        ax.contour(self.rvals, yaxis, toplot, levels=levels, **contour_kwargs)
        self._gentrify_structure_ax(ax, full=full)
        return fig

    def plot_temperature(self, fig=None, contourf_kwargs=None, full=True):
        """Plot the temperature structure."""
        fig, ax = self._grab_axes(fig)
        toplot = np.vstack([self.temperature[::-1], self.temperature])
        yaxis = np.concatenate([-self.zvals[::-1], self.zvals])
        contourf_kwargs = {} if contourf_kwargs is None else contourf_kwargs
        levels = np.linspace(toplot.min(), toplot.max(), 50)
        levels = contourf_kwargs.pop("levels", levels)
        cmap = contourf_kwargs.pop("cmap", "RdYlBu_r")
        im = ax.contourf(self.rvals, yaxis, toplot, levels=levels, cmap=cmap, **contourf_kwargs)

        cax = make_axes_locatable(ax)
        cax = cax.append_axes("right", size="4.5%" if full else "3%",
                              pad="2.25%" if full else "1.5%")
        cb = plt.colorbar(im, cax=cax, ticks=np.arange(0, toplot.max(), 50))
        cb.set_label(r"$T \,\, [{\rm K}]$", rotation=270, labelpad=15)

        self._gentrify_structure_ax(ax, full=full)
        return fig

    def plot_temperature_contours(
        self, fig=None, levels=None, contour_kwargs=None, full=True):
        """Overplot temperature contours."""
        fig, ax = self._grab_axes(fig)
        toplot = np.vstack([self.temperature[::-1], self.temperature])
        yaxis = np.concatenate([-self.zvals[::-1], self.zvals])
        contour_kwargs = {} if contour_kwargs is None else contour_kwargs
        levels = np.arange(0, 12, 2) if levels is None else levels
        contour_kwargs["linewidths"] = contour_kwargs.pop("linewidths", 1.0)
        contour_kwargs["colors"] = contour_kwargs.pop("colors", "k")
        ax.contour(self.rvals, yaxis, toplot, levels=levels, **contour_kwargs)
        self._gentrify_structure_ax(ax, full=full)
        return fig

    def plot_abundance(self, fig=None, contourf_kwargs=None, full=True):
        """Plot the abundance structure."""
        fig, ax = self._grab_axes(fig)
        toplot = self.density * self.abundance / sc.m_p / self.mu
        toplot = np.log10(np.vstack([toplot[::-1], toplot]))
        yaxis = np.concatenate([-self.zvals[::-1], self.zvals])

        contourf_kwargs = {} if contourf_kwargs is None else contourf_kwargs
        levels = np.linspace(toplot.min(), toplot.max(), 50)
        levels = contourf_kwargs.pop("levels", levels)
        cmap = contourf_kwargs.pop("cmap", "Blues")
        im = ax.contourf(self.rvals, yaxis, toplot, levels=levels, cmap=cmap, **contourf_kwargs)

        cax = make_axes_locatable(ax)
        cax = cax.append_axes("right", size="4.5%" if full else "3%", pad="2.25%" if full else "1.5%")
        cb = plt.colorbar(im, cax=cax, ticks=np.arange(-30, 30, 2))
        cb.set_label(r"$\log_{10}(n_{\rm mol}\,\,[{\rm cm^{-2}}])$", rotation=270, labelpad=15)
        
        self._gentrify_structure_ax(ax, full=full)
        return fig

    def plot_abundance_contour(self, fig=None, full=True):
        """Overplot the abundance structure."""
        fig, ax = self._grab_axes(fig)
        toplot = self.abundance
        toplot = np.log10(np.vstack([toplot[::-1], toplot]))
        yaxis = np.concatenate([-self.zvals[::-1], self.zvals])
        im = ax.contourf(self.rvals, yaxis, toplot, 3, hatches=[" ", "///"], colors=["none"])
        ax.contour(self.rvals, yaxis, toplot, [im.levels[1]], linestyles="-", linewidths=1.0, colors="k",)
        self._gentrify_structure_ax(ax, full=full)
        return fig

    def plot_rotation(self, fig=None, contourf_kwargs=None, full=True):
        """Plot the rotation structure."""
        fig, ax = self._grab_axes(fig)
        toplot = np.vstack([self.vphi[::-1], self.vphi]) / 1e3
        yaxis = np.concatenate([-self.zvals[::-1], self.zvals])

        contourf_kwargs = {} if contourf_kwargs is None else contourf_kwargs
        levels = np.linspace(toplot.min(), toplot.max(), 50)
        levels = contourf_kwargs.pop("levels", levels)
        cmap = contourf_kwargs.pop("cmap", "RdBu_r")
        im = ax.contourf(self.rvals, yaxis, toplot, levels=levels, cmap=cmap, **contourf_kwargs)

        cax = make_axes_locatable(ax)
        cax = cax.append_axes("right", size="4.5%" if full else "3%", pad="2.25%" if full else "1.5%")
        cb = plt.colorbar(im, cax=cax)
        cb.set_label(r"$v_{\phi} \,\, [{\rm km/s}]$", rotation=270, labelpad=15)

        self._gentrify_structure_ax(ax, full=full)
        return fig

    def plot_surface_density(self, fig=None, number_density=True):
        """Plot the gas and dust surface densities."""
        fig, ax = self._grab_axes(fig)
        ax.semilogy(self.rvals, self.sigma_g, ls="-", color="0.2", label="gas")
        ax.semilogy(self.rvals, self.sigma_d, ls="--", color="0.2", label="dust")
        ax.set_xlabel("Radius [au]")
        ax.set_ylabel(r"Surface Density [g cm$^{-2}$]")
        ax.legend(loc=1, markerfirst=False)
        return fig

    @staticmethod
    def _grab_axes(fig):
        """Split off the axis from the figure if provided."""
        if fig is None:
            fig, ax = plt.subplots()
        else:
            ax = fig.axes[0]
        return fig, ax

    @staticmethod
    def _gentrify_structure_ax(ax, full=True):
        """Gentrify the plot."""
        ax.set_xlabel("Radius [au]")
        ax.set_ylabel("Height [au]")
        ax.set_aspect(1)
        if not full:
            ax.set_ylim(0.0, ax.get_ylim()[1])

    # Analytical functions.

    @staticmethod
    def powerlaw(x, y0, q, x0=1.0):
        """Simple powerlaw function."""
        return y0 * (x / x0) ** q

    @staticmethod
    def gaussian(x, x0, dx, A):
        """Simple GaussiTmidan function. `dx` is Doppler width."""
        return A * np.exp(-((x - x0) / dx) ** 2.0)
