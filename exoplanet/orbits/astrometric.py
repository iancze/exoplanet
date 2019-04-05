# -*- coding: utf-8 -*-

r'''
Orbital conventions derived according to

.. math::

    \Delta \delta = X = \rho \cos \theta \\
    \Delta \alpha \cos \delta = Y = \rho \sin \theta

The positive X-axis points North, and the positive Y-axis points East. Rotation angle :math:`\theta` is measured in degrees East of North, i.e.

.. math::

    \theta = \arctan(Y / X).

In our implementation, we use ``np.atan2(Y/X)`` to resolve the quadrant ambiguity.

The *ascending node* is defined as the point where the *secondary* crosses the plane of the sky *receeding* from the observer. It is measured in degrees East of North.

Inclination is defined such that :math:`0 < i < \pi/2` yield prograde orbits (counter-clockwise, such that :math:`\theta` increases), and :math:`\pi/2 < i < \pi` yield retrograde orbits (clockwise, such that :math:`\theta` decreases).

These lead to the following equations of motion

.. math::

    X = r (\cos \Omega \cos(\omega + f) - \sin(\Omega) \sin(\omega + f) \cos(i)) \\
    Y = r (\sin \Omega \cos(\omega + f) + \cos(\Omega) \sin(\omega + f) \cos(i)) \\
    Z = - r \sin(\omega + f) \sin(i)

and

.. math::

    v_{r,1} = K_1 (\cos (\omega_1 + f) + e \cos \omega_1) \\
    v_{r,2} = K_2 (\cos (\omega_2 + f) + e \cos \omega_2).

'''


from __future__ import division, print_function

__all__ = ["AstrometricOrbit"]

# __all__ = ["KeplerianOrbit", "get_true_anomaly"]

import numpy as np
import theano.tensor as tt

from astropy import constants
from astropy import units as u

from ..citations import add_citations_to_model
from ..theano_ops.kepler import (KeplerOp, CircularContactPointsOp, ContactPointsOp)


# Two classes

##################
# Astrometric only
##################
# 7 minimum parameters (according to Pourbaix)
# a (angular), i, ω, Ω, e, P and T

# Use cases
# 1) just fit for these parameters and call it a day
# 2) we have a prior on ϖ, so we want to use it to convert to a real a and M_tot. This means we'll also want to
# add parallax as a real parameter, and a and M_tot as deterministic quantities
# 3) we have a prior on M_tot (perhaps from photometry of the host star, in the case of an exoplanet). This means we'll want to add M_tot as a parameter, and then add a and ϖ as deterministic parameters.

# we can't even do a barycentric orbit with these limited parameters, only a relative orbit.

# Astrometric + RV
# 10 minimum parameters (according to Pourbaix)
# a (angular), i, ω2, Ω2, e, P, T, gamma, ϖ (parallax), and κ

# κ is defined as the ratio of the primary semimajor axis to the relative semi-major axis
# κ = a1 / (a1 + a2)

# from these, we can derive more well-known quantities like V1, V2, and the masses of the stars.

# probably should inherit from KeplerianOrbit, but enough is different for now
class AstrometricOrbit(object):
    """A generalization of a Keplerian orbit with astrometric observations. This is the simplest kind of astrometric orbit.

    This orbit can specify the 3D positions and radial velocities of both stars.

    The minimum parameter set consists of 7 parameters: a (angular), i, ω, Ω, e, P and T

    omega and Omega correspond to the secondary star (omega_2 and Omega_2). They are assumed to be in radians.

    Args:
        positions: A list of separations and position angles

        transit_times: A list (with on entry for each planet) of transit times
            for each transit of each planet in units of days. These times will
            be used to compute the implied (least squares) ``period`` and
            ``t0`` so these parameters cannot also be given.
    """
    __citations__ = ("astropy",)

    def __init__(self, a_ang=None, t0=0.0, period=None,
             incl=None, ecc=None, omega=None, Omega=None,
             model=None, contact_points_kwargs=None,
             **kwargs):
        add_citations_to_model(self.__citations__, model=model)

        # conversion constant
        self.G_grav = constants.G.to(u.R_sun**3 / u.M_sun / u.day**2).value

        self.kepler_op = KeplerOp(**kwargs)

        # Parameters
        self.a_ang = tt.as_tensor_variable(a_ang)
        self.t0 = tt.as_tensor_variable(t0)
        self.period = tt.as_tensor_variable(period)

        self.incl = tt.as_tensor_variable(incl)
        self.cos_incl = tt.cos(self.incl)
        self.sin_incl = tt.sin(self.incl)

        self.ecc = tt.as_tensor_variable(ecc)

        self.omega = tt.as_tensor_variable(omega)
        self.cos_omega = tt.cos(self.omega)
        self.sin_omega = tt.sin(self.omega)

        self.Omega = tt.as_tensor_variable(Omega)
        self.cos_Omega = tt.cos(self.Omega)
        self.sin_Omega = tt.sin(self.Omega)

        self.n = 2 * np.pi / self.period # mean motion

        # Set up the contact points calculation
        if contact_points_kwargs is None:
            contact_points_kwargs = dict()

        # set up some of the parameters for the contact points op
        opsw = 1 + self.sin_omega
        E0 = 2 * tt.arctan2(tt.sqrt(1-self.ecc)*self.cos_omega,
                            tt.sqrt(1+self.ecc)*opsw)
        self.M0 = E0 - self.ecc * tt.sin(E0)
        self.tref = self.t0 - self.M0 / self.n
        self.contact_points_op = ContactPointsOp(**contact_points_kwargs)

    #TODO: not really sure what warp times is here
    # is it just reshaping the time array appropriately?
    def _warp_times(self, t):
        return tt.shape_padright(t)

    def _get_true_anomaly(self, t):
        M = (self._warp_times(t) - self.tref) * self.n
        if self.ecc is None:
            return M
        _, f = self.kepler_op(M, self.ecc + tt.zeros_like(M))
        return f

    def get_relative_position_XY(self, t):
        """Get the position of the secondary star relative to the primary star in rho, theta."""

        f = self._get_true_anomaly(t)

        r = self.a_ang * (1 - self.ecc**2) / (1 + self.ecc * tt.cos(f))

        # X is north (DEC)
        # Y is east (RA)
        X = r * (self.cos_Omega * tt.cos(self.omega + f) - self.sin_Omega * tt.sin(self.omega + f) * self.cos_incl)
        Y = r * (self.sin_Omega * tt.cos(self.omega + f) + self.cos_Omega * tt.sin(self.omega + f) * self.cos_incl)

        return (X,Y)

    def get_relative_position(self, t):

        X,Y = self.get_relative_position_XY(t)

        # calculate rho and theta
        rho = tt.sqrt(X**2 + Y**2) # arcsec
        theta = tt.arctan2(Y,X) # radians

        # do something to make the theta's greater than 0
        # if theta < 0: # ensure that 0 <= theta <= 360
            # theta += 360.

        return (rho, theta)

    def get_physical_a_and_mass(self, parallax):
        """Using a parallax measurement (in arcsec), convert the semi-major axes (in arcseconds) into one measured in AU and compute the total mass of the system."""

        a_phys = self.a_ang / parallax # [AU]

        # TODO use the proper astropy unit conversions
        # a_phys needs to be in cm for this calc
        M_tot = 4 * np.pi**2 * a_phys**3 / G # g

        # constants.G.

        return a_phys, M_tot




# class Binary:
#     '''
#     Binary orbital model that can deliver absolute astrometric position,
#     relative astrometric position (B relative to A), and radial velocities for A and B.
#
#     Args:
#         a (float): semi-major axis [AU]
#         e (float): eccentricity (must be between ``[0.0, 1.0)``)
#         i (float): inclination [deg]
#         omega (float): argument of periastron of the *primary*, i.e. :math:`\omega_1` [degrees]
#         Omega (float): position angle of the ascending node (going into sky) [deg] east of north
#         T0 (float): epoch of periastron passage [JD]
#         M_tot (float): sum of the masses :math:`M_\mathrm{A} + M_\mathrm{B}` [:math:`M_\odot`]
#         M_2 (float): mass of B [:math:`M_\odot`]
#         gamma (float): systemic velocity (km/s)
#         obs_dates (1D np.array): dates of observation (JD)
#     '''
#
#     def __init__(self, a, e, i, omega, Omega, T0, M_tot, M_2, gamma, obs_dates=None, **kwargs):
#         assert (e >= 0.0) and (e < 1.0), "Eccentricity must be between [0, 1)"
#         assert (i >= 0.0) and (i <= 180.), "Inclination must be between [0, 180]"
#         self.a = a # [AU] semi-major axis
#         self.e = e # eccentricity
#         self.i = i # [deg] inclination
#         self.omega = omega # [deg] argument of periastron
#         self.omega_2 = self.omega + 180
#         self.Omega = Omega # [deg] east of north
#         self.T0 = T0 # [JD]
#         self.M_tot = M_tot # [M_sun]
#         self.M_2 = M_2 # [M_sun]
#         self.gamma = gamma # [km/s]
#
#         # Update the derived RV quantities
#         self.q = self.M_2 / (self.M_tot - self.M_2) # [M2/M1]
#         self.P = np.sqrt(4 * np.pi**2 / (C.G * self.M_tot * C.M_sun) * (self.a * C.AU)**3) / (60 * 60 * 24)# [days]
#         self.K = np.sqrt(C.G/(1 - self.e**2)) * self.M_2 * C.M_sun * np.sin(self.i * np.pi/180.) / np.sqrt(self.M_tot * C.M_sun * self.a * C.AU) * 1e-5 # [km/s]
#
#         # If we are going to be repeatedly predicting the orbit at a sequence of dates,
#         # just store them to the object.
#         self.obs_dates = obs_dates
#
#
#     def _theta(self, t):
#         '''Calculate the true anomoly for the A-B orbit.
#         Input is in days.'''
#
#         # t is input in seconds
#
#         # Take a modulus of the period
#         t = (t - self.T0) % self.P
#
#         f = lambda E: E - self.e * np.sin(E) - 2 * np.pi * t/self.P
#         E0 = 2 * np.pi * t / self.P
#
#         E = fsolve(f, E0)[0]
#
#         th = 2 * np.arctan(np.sqrt((1 + self.e)/(1 - self.e)) * np.tan(E/2.))
#
#         if E < np.pi:
#             return th
#         else:
#             return th + 2 * np.pi
#
#     def _v1_f(self, f):
#         '''Calculate the component of A's velocity based on only the inner orbit.
#         f is the true anomoly of this inner orbit.'''
#
#         return self.K * (np.cos(self.omega * np.pi/180 + f) + self.e * np.cos(self.omega * np.pi/180))
#
#     def _v2_f(self, f):
#         '''Calculate the component of B's velocity based on only the inner orbit.
#         f is the true anomoly of this inner orbit.'''
#
#         return self.K/self.q * (np.cos(self.omega_2 * np.pi/180 + f) + self.e * np.cos(self.omega_2 * np.pi/180))
#
#     # Get the position of A in the plane of the orbit, relative to the center of mass
#     def _xy_A(self, f):
#         # find the reduced radius
#         r = self.a * (1 - self.e**2) / (1 + self.e * np.cos(f)) # [AU]
#         r1 = r * self.M_2 / self.M_tot # [AU]
#
#         x = r1 * np.cos(f)
#         y = r1 * np.sin(f)
#
#         return (x,y)
#
#     # Get the position of B in the plane of the orbit, relative to the center of mass
#     def _xy_B(self, f):
#         # find the reduced radius
#         r = self.a * (1 - self.e**2) / (1 + self.e * np.cos(f)) #
#         r2 = r * (self.M_tot - self.M_2) / self.M_tot # [AU]
#
#         x = r2 * np.cos(f)
#         y = r2 * np.sin(f)
#
#         return (x,y)
#
#     # Get the position of B in the plane of the orbit, relative to A
#     def _xy_AB(self, f):
#         r = self.a * (1 - self.e**2) / (1 + self.e * np.cos(f)) # [AU] # Negative sign here because we want B rel to A
#         x = r * np.cos(f)
#         y = r * np.sin(f)
#
#         return (x,y)
#
#     # position of A relative to center of mass
#     def _XYZ_A(self, f):
#
#         # find the reduced radius
#         r = self.a * (1 - self.e**2) / (1 + self.e * np.cos(f)) # [AU]
#         r1 = r * self.M_2 / self.M_tot # [AU]
#
#         Omega = self.Omega * np.pi / 180
#         omega = self.omega * np.pi / 180
#         i = self.i * np.pi / 180
#         X = r1 * (np.cos(Omega) * np.cos(omega + f) - np.sin(Omega) * np.sin(omega + f) * np.cos(i))
#         Y = r1 * (np.sin(Omega) * np.cos(omega + f) + np.cos(Omega) * np.sin(omega + f) * np.cos(i))
#         Z = -r1 * (np.sin(omega + f) * np.sin(i))
#
#         return (X, Y, Z) # [AU]
#
#     # position of B relative to center of mass
#     def _XYZ_B(self, f):
#         # find the reduced radius
#         r = self.a * (1 - self.e**2) / (1 + self.e * np.cos(f)) # [AU]
#         r2 = r * (self.M_tot - self.M_2) / self.M_tot # [AU]
#
#         Omega = self.Omega * np.pi / 180
#         omega = self.omega_2 * np.pi / 180
#         i = self.i * np.pi / 180
#         X = r2 * (np.cos(Omega) * np.cos(omega + f) - np.sin(Omega) * np.sin(omega + f) * np.cos(i))
#         Y = r2 * (np.sin(Omega) * np.cos(omega + f) + np.cos(Omega) * np.sin(omega + f) * np.cos(i))
#         Z = -r2 * (np.sin(omega + f) * np.sin(i))
#
#         return (X, Y, Z) # [AU]
#
#     def _XYZ_AB(self, f):
#         # radius of B to A
#         r = self.a * (1 - self.e**2) / (1 + self.e * np.cos(f))
#         Omega = self.Omega * np.pi / 180
#         omega = self.omega_2 * np.pi / 180
#         i = self.i * np.pi / 180
#         X = r * (np.cos(Omega) * np.cos(omega + f) - np.sin(Omega) * np.sin(omega + f) * np.cos(i))
#         Y = r * (np.sin(Omega) * np.cos(omega + f) + np.cos(Omega) * np.sin(omega + f) * np.cos(i))
#         Z = -r * (np.sin(omega + f) * np.sin(i))
#
#         return (X, Y, Z) # [AU]
#
#     def _get_periastron_A(self):
#         return np.array(self._XYZ_A(0))
#
#     def _get_periastron_B(self):
#         return np.array(self._XYZ_B(0))
#
#     def _get_periastron_BA(self):
#         return np.array(self._XYZ_AB(0))
#
#     def _get_node_A(self):
#         '''
#         The point corresponding to the ascending node
#         '''
#         # set f = 2 * pi - omega
#         return np.array(self._XYZ_A(2 * np.pi - self.omega * np.pi/180))
#
#     def _get_node_B(self):
#         return np.array(self._XYZ_B(2 * np.pi - self.omega_2 * np.pi/180))
#
#     def _get_node_BA(self):
#         return np.array(self._XYZ_AB(2 * np.pi - self.omega_2 * np.pi/180))
#
#     def _get_orbit_t(self, t):
#         '''
#         Given a time, calculate all of the orbital quantaties we might be interseted in.
#         returns (v_A, v_B, (x,y) of A, (x,y) of B, and x,y of B relative to A)
#         '''
#
#         # Get the true anomoly "f" from time
#         f = self._theta(t)
#
#         # Feed this into the orbit equation and add the systemic velocity
#         vA = self._v1_f(f) + self.gamma
#         vB = self._v2_f(f) + self.gamma
#
#         XYZ_A = self._XYZ_A(f)
#         XYZ_B = self._XYZ_B(f)
#         XYZ_AB = self._XYZ_AB(f)
#         xy_A = self._xy_A(f)
#         xy_B = self._xy_B(f)
#         xy_AB = self._xy_AB(f)
#
#         return (vA, vB, XYZ_A, XYZ_B, XYZ_AB, xy_A, xy_B, xy_AB)
#
#
#     def get_orbit(self, dates=None):
#         r'''
#         Deliver only the main quantities useful for performing a joint astrometric + RV fit to real data, namely
#         the radial velocities ``vA``, ``vB``, the relative offsets :math:`\rho`, and relative position angles :math:`\theta`, for all dates provided. Relative offsets are provided in AU, and so must be converted to arcseconds after assuming a distance to the system. Relative position angles are given in degrees east of north.
#
#         Args:
#             dates (optional): if provided, calculate quantities at this new vector of dates, rather than the one provided when the object was initialized.
#
#         Returns:
#             dict: A dictionary with items ``"vAs"``, ``"vBs"``, ``"rhos"``, ``"thetas"``.
#         '''
#
#         if dates is None and self.obs_dates is None:
#             raise RuntimeError("Must provide input dates or specify observation dates upon creation of orbit object.")
#
#         if dates is None and self.obs_dates is not None:
#             dates = self.obs_dates
#
#         dates = np.atleast_1d(dates)
#         N = len(dates)
#
#         vAs = np.empty(N, dtype=np.float64)
#         vBs = np.empty(N, dtype=np.float64)
#         rho_ABs = np.empty(N, dtype=np.float64)
#         theta_ABs = np.empty(N, dtype=np.float64)
#
#         for i,date in enumerate(dates):
#             vA, vB, XYZ_A, XYZ_B, XYZ_AB, xy_A, xy_B, xy_AB = self._get_orbit_t(date)
#             vAs[i] = vA
#             vBs[i] = vB
#
#             # Calculate rho, theta from XY_AB
#             X, Y, Z = XYZ_AB
#
#             rho = np.sqrt(X**2 + Y**2) # [AU]
#             theta = np.arctan2(Y,X) * 180/np.pi # [Deg]
#             if theta < 0: # ensure that 0 <= theta <= 360
#                 theta += 360.
#
#             rho_ABs[i] = rho
#             theta_ABs[i] = theta
#
#         return {"vAs":vAs, "vBs":vBs, "rhos":rho_ABs, "thetas":theta_ABs}
#
#     def get_full_orbit(self, dates=None):
#         '''
#         Deliver the full set of astrometric and radial velocity quantities, namely
#         the radial velocities ``vA``, ``vB``, the position of A and B relative to the center of mass in the plane of the sky (``XY_A`` and ``XY_B``, respectively), the position of B relative to the position of A in the plane of the sky (``XY_AB``), the position of A and B in the plane of the orbit (``xy_A`` and ``xy_B``, respectively), and the position of B relative to the position of A in the plane of the orbit (``xy_AB``), for all dates provided. All positions are given in units of AU, and so must be converted to arcseconds after assuming a distance to the system.
#
#         Args:
#             dates (optional): if provided, calculate quantities at this new vector of dates, rather than the one provided when the object was initialized.
#
#         Returns:
#             dict: A dictionary with items of ``"vAs"``, ``"vBs"``, ``"XYZ_As"``, ``"XYZ_Bs"``, ``"XYZ_ABs"``, ``"xy_As"``, ``"xy_Bs"``, ``"xy_ABs"``
#         '''
#
#
#         if dates is None and self.obs_dates is None:
#             raise RuntimeError("Must provide input dates or specify observation dates upon creation of orbit object.")
#
#         if dates is None and self.obs_dates is not None:
#             dates = self.obs_dates
#
#         dates = np.atleast_1d(dates)
#         N = len(dates)
#
#         vAs = np.empty(N, dtype=np.float64)
#         vBs = np.empty(N, dtype=np.float64)
#         XYZ_As = np.empty((N, 3), dtype=np.float64)
#         XYZ_Bs = np.empty((N, 3), dtype=np.float64)
#         XYZ_ABs = np.empty((N, 3), dtype=np.float64)
#         xy_As = np.empty((N, 2), dtype=np.float64)
#         xy_Bs = np.empty((N, 2), dtype=np.float64)
#         xy_ABs = np.empty((N, 2), dtype=np.float64)
#
#         for i,date in enumerate(dates):
#             vA, vB, XYZ_A, XYZ_B, XYZ_AB, xy_A, xy_B, xy_AB = self._get_orbit_t(date)
#             vAs[i] = vA
#             vBs[i] = vB
#             XYZ_As[i] = np.array(XYZ_A)
#             XYZ_Bs[i] = np.array(XYZ_B)
#             XYZ_ABs[i] = np.array(XYZ_AB)
#             xy_As[i] = np.array(xy_A)
#             xy_Bs[i] = np.array(xy_B)
#             xy_ABs[i] = np.array(xy_AB)
#
#         return {"vAs":vAs, "vBs":vBs, "XYZ_As":XYZ_As, "XYZ_Bs":XYZ_Bs,
#         "XYZ_ABs":XYZ_ABs, "xy_As":xy_As, "xy_Bs":xy_Bs, "xy_ABs":xy_ABs}
