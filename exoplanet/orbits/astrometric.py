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

__all__ = ["AstrometricOrbit", "AstrometricRVOrbit"]

import numpy as np
import theano.tensor as tt

from astropy import constants
from astropy import units as u

from ..citations import add_citations_to_model
from ..theano_ops.kepler import (KeplerOp, CircularContactPointsOp, ContactPointsOp)

# Kepler solver. Need to adjust reference times to use time of periastron, not transit.
# Currently, the Kepler solver uses
# M = (self._warp_times(t) - self.tref) * self.n
# where
# opsw = 1 + self.sin_omega
# E0 = 2 * tt.arctan2(tt.sqrt(1-self.ecc)*self.cos_omega,
#                     tt.sqrt(1+self.ecc)*opsw)
# self.M0 = E0 - self.ecc * tt.sin(E0)
# self.tref = self.t0 - self.M0 / self.n
# self.n = 2 * np.pi / self.period # mean motion

# Here, M0 is probably the mean anomaly at time of transit

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

# we can't even do a center-of-mass orbit with these limited parameters, only a relative orbit.

###################
# Astrometric + RV
###################
# 10 minimum parameters (according to Pourbaix)
# a (angular), i, ω2, Ω2, e, P, T, gamma, ϖ (parallax), and κ

# κ is defined as the ratio of the primary semimajor axis to the relative semi-major axis
# κ = a1 / (a1 + a2)

# from these, we can derive more well-known quantities like V1, V2, and the masses of the stars.

# we probably should inherit from KeplerianOrbit, but enough is different for now that we'll duplicate some


class AstrometricOrbit(object):
    """A single astrometric orbit of a secondary body relative to a primary body.

    This is the simplest kind of astrometric orbit (no radial velocities), where only relative positions are measured on the sky. The minimum parameter set consists of 7 parameters: a (angular), i, ω, Ω, e, P and T, where omega and Omega correspond to the secondary star (omega_2 and Omega_2). They are assumed to be in radians. Following the visual binary field, the ascending node is assumed to be the node where the secondary body is receeding (moving away) from the observer.

    With only astrometric information, there is a 180 degree ambiguity to Ω, and so it may be preferred to fit with the quantities (Ω + ω) and (Ω - ω) instead.

    Args:
        period: The orbital periods of the bodies in days.
        a_ang: The semimajor axes of the orbit in ``arcsec``.
        t0: The time of a reference transit for each orbits in days.
        incl: The inclinations of the orbit in radians.
        ecc: The eccentricities of the orbits. Must be ``0 <= ecc < 1``.
        omega: The argument of periastron of the secondary in radians.
        Omega: The position angle of the ascending node in radians.
    """
    __citations__ = ("astropy",)

    def __init__(self, a_ang=None, t0=0.0, period=None,
             incl=None, ecc=None, omega=None, Omega=None,
             model=None, contact_points_kwargs=None,
             **kwargs):
        add_citations_to_model(self.__citations__, model=model)

        self.kepler_op = KeplerOp(**kwargs)

        # Parameters
        self.a_ang = tt.as_tensor_variable(a_ang)
        self.t0 = tt.as_tensor_variable(t0)
        self.period = tt.as_tensor_variable(period)

        self.incl = tt.as_tensor_variable(incl)
        self.cos_incl = tt.cos(self.incl)
        self.sin_incl = tt.sin(self.incl)

        self.ecc = tt.as_tensor_variable(ecc)

        self.omega = tt.as_tensor_variable(omega) # omega_2
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
        self.M0 = E0 - self.ecc * tt.sin(E0) # calculate mean anomaly at transit?
        # self.tref = self.t0 - self.M0 / self.n # referenced to time of transit?
        self.tref = self.t0 # use t0 as time of periastron?
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
        """The position of the secondary body relative to the primary body.

        Args:
            t: The times where the position should be evaluated.

        Returns:
            The position as specified by X (relative Dec / north) and Y (relative R.A. / east).
        """


        f = self._get_true_anomaly(t)

        r = self.a_ang * (1 - self.ecc**2) / (1 + self.ecc * tt.cos(f))

        # these calculations assume omega = omega_2

        # X is north (DEC)
        # Y is east (RA)
        X = r * (self.cos_Omega * tt.cos(self.omega + f) - self.sin_Omega * tt.sin(self.omega + f) * self.cos_incl)
        Y = r * (self.sin_Omega * tt.cos(self.omega + f) + self.cos_Omega * tt.sin(self.omega + f) * self.cos_incl)

        return (X,Y)

    def get_relative_position(self, t):
        """The position of the secondary body relative to the primary body.

        Args:
            t: The times where the position should be evaluated.

        Returns:
            The position as specified by rho (arcsec) and theta (radians). Theta will be in the range [-π, π]
        """

        X,Y = self.get_relative_position_XY(t)

        # calculate rho and theta
        rho = tt.sqrt(X**2 + Y**2) # arcsec
        theta = tt.arctan2(Y,X) # radians

        return (rho, theta)

    def get_physical_a_and_mass(self, parallax):
        """Using a parallax measurement (in arcsec), convert the orbit into physical units.

        Args:
            parallax: the parallax (in milliarcseconds (mas))

        Returns:
            Semi-major axes (in AU) and total mass (M1 + M2) of the system in M_Sun."""

        a_phys = self.a_ang / (parallax * 1e3) # [AU]

        M_tot = (4 * np.pi**2 * (a_phys * u.au)**3 / constants.G).to(u.Msun)

        return a_phys, M_tot


class AstrometricRVOrbit(object):
    """A generalization of a Keplerian orbit fully described in 3D space. Can be used to model joint astrometric and radial velocity observations.

    This orbit fit requires 10 minimum parameters (following Pourbaix et al. 1998): a (angular), i, ω2, Ω2, e, P, T, gamma, ϖ (parallax), and κ.

    κ (kappa) is defined as the ratio of the primary semimajor axis to the relative semi-major axis
    κ = a1 / (a1 + a2)

    ω (omega) and Ω (Omega) correspond to the secondary star (omega_2 and Omega_2). They are assumed to be in radians.

    Following the visual binary field, the ascending node is assumed to be the node where the secondary body is receeding (moving away) from the observer. Radial velocity information breaks the ambiguity in Ω.

    Args:
        period: The orbital periods of the bodies in days.
        a_ang: The semimajor axes of the orbit in ``arcsec``.
        t0: The time of a reference transit for each orbits in days.
        incl: The inclinations of the orbit in radians.
        ecc: The eccentricities of the orbits. Must be ``0 <= ecc < 1``.
        omega: The argument of periastron for the secondary in radians.
        Omega: The position angle of the ascending node for the secondary in radians.
        gamma: The systemic velocity of the system barycenter (km/s).
        parallax: The parallax of the system (milliarcseconds)
        kappa: κ = a1 / (a1 + a2) ratio of primary semimajor axis to relative semi-major axis.
    """
    __citations__ = ("astropy",)

    def __init__(self, a_ang=None, t0=0.0, period=None,
             incl=None, ecc=None, omega=None, Omega=None, gamma=None, parallax=None, kappa=None,
             model=None, contact_points_kwargs=None,
             **kwargs):
        add_citations_to_model(self.__citations__, model=model)

        self.kepler_op = KeplerOp(**kwargs)

        # Parameters
        self.a_ang = tt.as_tensor_variable(a_ang) # arcsec
        self.t0 = tt.as_tensor_variable(t0)
        self.period = tt.as_tensor_variable(period)

        self.incl = tt.as_tensor_variable(incl)
        self.cos_incl = tt.cos(self.incl)
        self.sin_incl = tt.sin(self.incl)

        self.ecc = tt.as_tensor_variable(ecc)

        self.omega_2 = tt.as_tensor_variable(omega) # radians
        self.cos_omega_2 = tt.cos(self.omega_2)
        self.sin_omega_2 = tt.sin(self.omega_2)

        self.omega_1 = self.omega_2 + np.pi # radians
        self.cos_omega_1 = tt.cos(self.omega_1)
        self.sin_omega_1 = tt.sin(self.omega_1)

        self.Omega = tt.as_tensor_variable(Omega)
        self.cos_Omega = tt.cos(self.Omega)
        self.sin_Omega = tt.sin(self.Omega)

        self.n = 2 * np.pi / self.period # mean motion

        self.gamma = tt.as_tensor_variable(gamma) # km/s
        self.parallax = tt.as_tensor_variable(parallax) # milliarcseconds
        self.kappa = tt.as_tensor_variable(kappa)

        # derived values
        self.a_phys = self.a_ang / (1e3 * self.parallax) # au
        print(self.a_phys)
        self.a1_phys = self.a_phys * self.kappa # au
        self.a2_phys = self.a_phys - self.a1_phys # au

        # semi-amplitudes
        # TODO: check units
        self.M_tot = self.n**2 * self.a_phys**3 / constants.G # solar masses

        self.M_1 = (1 - self.kappa) * self.M_tot # solar masses
        self.M_2 = self.kappa * self.M_tot # solar masses

        self.K_1 = self.a1_phys * self.n * self.sin_incl / tt.sqrt(1 - self.ecc**2) # km/s
        self.K_2 = self.a2_phys * self.n * self.sin_incl / tt.sqrt(1 - self.ecc**2) # km/s

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

    def get_primary_velocity(self, t):
        '''Calculate the radial velocity of the primary body.

        Args:
            t: the time to calculate the radial velocity

        Returns:
            the radial velocity
        '''

        f = self._get_true_anomaly(t)

        return self.K_1 * (tt.cos(self.omega_1 + f) + self.ecc * tt.cos_omega_1) + self.gamma

    def get_secondary_velocity(self, t):
        '''Calculate the radial velocity of the secondary body.

        Args:
            t: the time to calculate the radial velocity

        Returns:
            the radial velocity
        '''

        f = self._get_true_anomaly(t)

        return self.K_2 * (tt.cos(self.omega_2 + f) + self.ecc * tt.cos_omega_2) + self.gamma

    def _XYZ_AB(self, t):
        '''Calculate the 3D relative position of the secondary body relative to the primary (in arcseconds).
        '''

        f = self._get_true_anomaly(t)

        # radius of B to A
        r = self.a_ang * (1 - self.ecc**2) / (1 + self.ecc * tt.cos(f)) # arcsec

        X = r * (self.cos_Omega * tt.cos(self.omega_2 + f) - self.sin_Omega * tt.sin(self.omega_2 + f) * self.cos_incl)
        Y = r * (self.sin_Omega * tt.cos(self.omega_2 + f) + self.cos_Omega * tt.sin(self.omega_2 + f) * self.cos_incl)
        Z = -r * (tt.sin(self.omega_2 + f) * self.sin_incl)

        return (X, Y, Z) # [arcsec]

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
