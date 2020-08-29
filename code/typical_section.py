"""Contains definitions for aeroelastic analysis with typical sections.

This module is developed for the TU Delft master course AE4ASM506
Aeroelasticity. Although it is against software best practice, all
abstractions and analysis are contained within this single module as per
the request of the lecturer.
"""

from __future__ import annotations

import dataclasses
import math
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from functools import cached_property
from typing import Dict, List, NamedTuple, Optional, Tuple, Union

import dufte
import matplotlib
import numpy as np
import rich
import scipy
from gammapy.geometry.airfoil import Airfoil, NACA4Airfoil
from matplotlib import pyplot as plt
from meshpy.triangle import MeshInfo
from rich.table import Table
from scipy import optimize, signal
from sectionproperties.analysis.cross_section import CrossSection
from sectionproperties.post import post
from sectionproperties.pre.sections import Geometry

FigureHandle = Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
"""Defines the type of a plot return."""


@dataclasses.dataclass(frozen=True)
class Wing:
    """Defines Daedelus wing parameters used in aeroelastic anlysis.

    This can be regarded as the single source of truth for the input
    parameters of the entire analysis.

    Note:
        Some of these quantities are expressed on a per-unit length
        basis as is typical for aeroelastic analysis, also
        refered to as local values, see Reader pg. 22.

    Args:
        half_span: Half of the main wing span in SI meter
        chord: Chord-length of the main wing airfoil in SI meter
        airfoil: :py:class:`Airfoil` instance describing the
            outer cross-sectional geometry of the wing.
        lift_gradient: Lift curve slope in SI per radian
        airfoil_mass: Mass of the main wing per unit span in SI
            kilogram per meter
        airfoil_inertia: Mass moment of inertia per unit
            span of the wing airfoil cross-section at the center of
            gravity of the airfoil in SI kilogram meter
        elastic_axis: Normalized location elastic axis measured in
            fraction of the airfoil chord-length
        center_of_gravity: Normalized location of the center of
            gravity of the wing cross-section measured as a fraction
            of the airfoil chord-length.
        bending_rigidity: Product of Modulus of Elasticity, E, and the
            Area Moment of Inertia, I, in SI Newton meter squared
        torsional_rigidity: Product of Modulus of Rigidity, G, and the
            Polar Area Moment of Inertia, J, in SI Newton meter squared
    """

    half_span: float = 16
    chord: float = 1
    lift_gradient: float = 2 * math.pi
    airfoil: Airfoil = NACA4Airfoil("0015")
    airfoil_mass: float = 0.75
    airfoil_inertia: float = 0.1
    elastic_axis: float = 0.5
    center_of_gravity: float = 0.5
    bending_rigidity: float = 2e4
    torsional_rigidity: float = 1e4


@dataclasses.dataclass(frozen=True)
class TypicalSection(Wing):
    """Specialized representation of :py:class:`Wing` as a section.

    Args:
        eta_ts: Normalized location of the typical section along the
            half-span of the wing.
    """

    eta_ts: float = 0.75

    def __post_init__(self):
        """Ensures that :py:attr:`eta_ts` is a float.

        This is useful when a :py:class:`Optimization` inputs a
        :py:class:`numpy.ndarray` instead of a float.
        """
        if isinstance(self.eta_ts, np.ndarray):
            object.__setattr__(self, "eta_ts", self.eta_ts[0])

    @cached_property
    def y_ts(self):
        """Spanwise location of the typical section in SI meter."""
        return self.eta_ts * self.half_span

    @cached_property
    def half_chord(self):
        """Half-chord, b, in SI meter."""
        return self.chord / 2

    @cached_property
    def x_theta(self):
        """Norm. distance from the elastic axis to center of gravity.

        Note:
            This is normalized w.r.t to :py:attr:`half_chord` which is
            why the distance must be multiplied by a factor of 2.
        """
        return (self.center_of_gravity - self.elastic_axis) * 2

    @cached_property
    def i_theta(self):
        """Mass moment of inertia per unit span at the elastic axis.

        Note:
            This uses the Stiener Parallel Axis Theorem to compute the
            mass moment of inertia at the elastic axis using the value
            of the inertia about the center of gravity. This is
            equivalent to Iyy of the inertia tensor of the wing as the
            y-axis is coincident with the elastic axis.
        """
        i_theta_star = self.airfoil_inertia
        x_theta = self.x_theta
        b = self.half_chord
        return i_theta_star + self.airfoil_mass * (x_theta * b) ** 2

    @cached_property
    def lift_moment_arm(self):
        """Distance between the aerodynamic center and the elastic axis.

        Note:
            This distance is equivalent to the distance, ec, and assumes
            that the aerodynamic center is located at quarter chord.
        """
        return (self.elastic_axis - 0.25) * self.chord

    @cached_property
    def elastic_axis_offset(self):
        """Norm. distance from the reference axis to the elastic axis.

        Note:
            This distance is equivalent to the normalized distance, a,
            times the half-chord from the lecture slides.
        """
        return 2 * (self.elastic_axis - 0.5)


class StructuralModel:
    """Defines the linear structural model using the typical section.

    This is left as a separate class since it is useful for the typical
    section location optimization on its own. It also reduces the
    clutter of the Aeroelastic models.

    Args:
        typical_section: A :py:class:`TypicalSection` instance
    """

    def __init__(
        self, typical_section: TypicalSection,
    ):
        self.typical_section = typical_section

    @cached_property
    def structural_mass_matrix(self):
        """Structural mass matrix of the typical section."""
        ts = self.typical_section
        return np.array(
            [
                [
                    ts.airfoil_mass,
                    ts.airfoil_mass * ts.x_theta * ts.half_chord,
                ],
                [ts.airfoil_mass * ts.x_theta * ts.half_chord, ts.i_theta],
            ]
        )

    @cached_property
    def structural_damping_matrix(self) -> np.ndarray:
        """Structural damping is zero due to linear structures."""
        return np.zeros((2, 2), dtype=np.float64)

    @cached_property
    def structural_stiffness_matrix(self):
        """Structural stiffness matrix of the typical section."""
        return np.array(
            [
                [self.unit_bending_stiffness, 0],
                [0, self.unit_torsion_stiffness],
            ]
        )

    @cached_property
    def unit_bending_stiffness(self):
        """Bending stiffness per unit span in SI Pascal."""
        ts = self.typical_section
        EI = ts.bending_rigidity / ts.y_ts
        return (3 * EI) / (ts.y_ts ** 3)

    @cached_property
    def unit_torsion_stiffness(self):
        """Torsion striffness per unit span in SI Newton meter."""
        ts = self.typical_section
        GJ = ts.torsional_rigidity / ts.y_ts
        return GJ / (ts.y_ts)

    @cached_property
    def mass_spring_eigen_solution(self) -> Tuple[np.ndarray, np.ndarray]:
        """Eigen values and vectors of the mass-spring system.

        The typical section can be reduced to only the mass and
        stiffness (spring) terms to find the natural frequencies of the
        system as can be found by solving the eigen value problem below
        where `x` is an eigen-vector::

            (M_s ** -1 @ K_s) @ x = (omega ** 2) @ x

        Returns:
            - [0] Eigen vectors arranged columnwise corresponding to
              heave and torsion
            - [1] Natural frequencies of the heave and torsional
              bending mode in SI radian per second
        """
        eigen_values, eigen_vectors = np.linalg.eig(
            np.linalg.inv(self.structural_mass_matrix)
            @ self.structural_stiffness_matrix
        )
        # Ensuring eigen values are real-valued
        assert np.all(eigen_values.imag == 0)
        return eigen_vectors, np.sqrt(eigen_values.real)

    @cached_property
    def coupled_heave_frequency(self) -> float:
        """Coupled TS heave frequency in SI radian per second."""
        return self.mass_spring_eigen_solution[-1][0]

    @cached_property
    def coupled_torsion_frequency(self) -> float:
        """Coupled TS torsion frequency in SI radian per second."""
        return self.mass_spring_eigen_solution[-1][-1]

    @cached_property
    def uncoupled_heave_frequency(self) -> float:
        """Typical section heave frequency in SI radian per second.

        Note:
            The bending rigidity is converted to a per unit span
            quantity in order to use the aircraft wing mass per unit
            span.
        """
        return math.sqrt(self.unit_bending_stiffness / self.airfoil_mass)

    @cached_property
    def uncoupled_torsion_frequency(self) -> float:
        """Typical section torsion frequency in SI radian per second."""
        return math.sqrt(self.unit_torsion_stiffness / self.i_theta)


class AeroelasticModel(StructuralModel, metaclass=ABCMeta):
    """Specializes the structural model with aerodynamic coupling."""

    def __init__(
        self,
        typical_section: TypicalSection,
        velocity: float = 20,
        density: float = 0.0889,
    ):
        super().__init__(typical_section)
        self.velocity = velocity
        self.density = density

    @property
    @abstractmethod
    def aerodynamic_mass_matrix(self):
        """Relates aerodynamic force to the acceleration of the DoFs."""
        pass

    @property
    @abstractmethod
    def aerodynamic_damping_matrix(self):
        """Relates aerodynamic force to the velocity of the DoFs."""
        pass

    @property
    @abstractmethod
    def aerodynamic_stiffness_matrix(self):
        """Relates aerodynamic force to the displacement of the DoFs."""
        pass

    @property
    @abstractmethod
    def aeroelastic_mass_matrix(self) -> np.ndarray:
        """Coupled aerodynamic and structural mass matrix."""

    @property
    @abstractmethod
    def aeroelastic_damping_matrix(self) -> np.ndarray:
        """Coupled aerodynamic and structural damping matrix."""

    @property
    @abstractmethod
    def aeroelastic_stiffness_matrix(self) -> np.ndarray:
        """Coupled aerodynamic and structural stiffness matrix."""

    @cached_property
    def dynamic_pressure(self) -> float:
        """Dynamic pressure, q, in SI Pascal."""
        return 0.5 * self.density * self.velocity ** 2

    @cached_property
    def state_matrix(self) -> np.ndarray:
        """Returns the state-matrix of the aeroelastic model.

        The state-matrix is comprised of the 4 degrees of freedom of the
        Linear Time Invariant (LTI) system. These are h_theta,
        theta_dot, h, and theta.
        """
        state_matrix = np.zeros((4, 4), dtype=np.float64)
        # Aeroelastic mass matrix
        m_ae_inverse = np.linalg.inv(self.aeroelastic_mass_matrix)
        state_matrix[:2, :2] = -m_ae_inverse @ self.aeroelastic_damping_matrix
        state_matrix[:2, 2:] = (
            -m_ae_inverse @ self.aeroelastic_stiffness_matrix
        )
        # Setting identity matrix for first-order differential terms
        state_matrix[(2, 3), (0, 1)] = 1
        return state_matrix

    def get_eigen_values_at(self, velocity: float) -> np.ndarray:
        """Gets eigen values of the LTI system at ``velocity``."""
        aero_model = self.__class__(
            typical_section=self.typical_section,
            velocity=velocity,
            density=self.density,
        )
        return np.linalg.eig(aero_model.state_matrix)[0]

    @cached_property
    def flutter_speed(self) -> float:
        """Dynamic flutter speed in SI meter per second.

        The flutter boundary is detected by observing the real component
        of the eigen-values of the system that have a non-zero imaginary
        (oscillatory) component. At flutter at least one of
        complex-conjugate eigen-values has a positive real value.
        Therefore, at the flutter-boundary the eigen-values of this
        complex conjugates transition from negative to positive and are
        hence equal to zero.
        """

        def objective_function(velocity: np.ndarray) -> float:
            eigen_values = self.get_eigen_values_at(velocity[0])
            return np.product(eigen_values[eigen_values.imag != 0].real)

        return optimize.fsolve(objective_function, x0=[50])[0]

    @cached_property
    def divergence_speed(self) -> float:
        """Static divergence speed in SI meter per second.

        The divergence speed is detected by observing the real
        component of the eigen-values of the system. At divergence
        """

        def objective_function(velocity: np.ndarray) -> float:
            eigen_values = self.get_eigen_values_at(velocity[0])
            return np.product(eigen_values[eigen_values.imag == 0].real)

        return optimize.fsolve(objective_function, x0=[50])[0]

    @cached_property
    def input_matrix(self) -> np.ndarray:
        """State-space input matrix, B, without dynamic pressure."""
        q = self.dynamic_pressure
        m_ae_inverse = np.linalg.inv(self.aeroelastic_mass_matrix)
        input_matrix = np.array(
            [
                -q
                * self.typical_section.chord ** 2
                * self.typical_section.lift_gradient,
                q
                * self.typical_section.chord ** 2
                * self.typical_section.lift_gradient
                * self.typical_section.lift_moment_arm,
            ]
            # Converting to a column vector to avoid shape mismatch
        ).reshape((2, 1))
        input_matrix = m_ae_inverse @ input_matrix
        return np.vstack((input_matrix, np.zeros((2, 1), dtype=np.float64)))

    @cached_property
    def output_matrix(self) -> np.ndarray:
        """State-space output matrix, C."""
        return np.eye(4, dtype=np.float64)

    @cached_property
    def feedthrough_matrix(self) -> np.ndarray:
        """State-space feedthrough or feedfoward matrix, D."""
        return np.zeros((4, 1), dtype=np.float64)

    def simulate(
        self, alpha_0: float, time: np.ndarray, plot: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Simulates response of the degrees of freedom.

        Args:
            alpha_0: Initial rigid Angle of Attack (AoA) of the wing
            time: Discretized time array to run the simulation at
            plot: Toggles plotting of the linear simulation results

        We need to augment the state-matrix with the constant
        aerodynamic moment term using:
        https://math.stackexchange.com/a/2149853

        THIS IS NO LONGER REQUIRED AS THE MOMENT IS ZERO DUE TO
        THIN AIRFOIL THEORY
        """
        state_matrix = self.state_matrix
        system = signal.StateSpace(
            state_matrix,
            self.input_matrix,
            self.output_matrix,
            self.feedthrough_matrix,
        )
        u = np.ones(time.shape, dtype=np.float64) * math.radians(alpha_0)
        time, y_out, x_out = signal.lsim(system, u, time)
        if plot:
            self.plot_simulation(time, x_out)
        return time, y_out, x_out

    @staticmethod
    def plot_simulation(time: np.ndarray, x_out: np.ndarray) -> None:
        """Plots the degrees of freedom of the aeroelastic model."""
        fig, axes = plt.subplots(nrows=4, sharex=True)
        labels = [
            r"$\dot{h}$ [m/s]",
            r"$\dot{\theta}$ [deg/s]",
            r"$h$ [m]",
            r"$\theta$ [deg]",
        ]
        for i, (ax, label) in enumerate(zip(axes, labels)):
            ax.plot(
                time,
                np.degrees(x_out[:, i]) if "theta" in label else x_out[:, i],
            )
            ax.set_ylabel(label, fontsize=12)
            ax.yaxis.set_tick_params(labelsize=10, pad=1)
        plt.xlabel("Time [s]", fontsize=12)
        fig.align_ylabels(axes)

    def plot_flutter_diagram(self):
        fig, ax = plt.subplots()
        v_range = np.linspace(0.1, 40, 1000)
        eigen_values = [self.get_eigen_values_at(v).real for v in v_range]

        ax.plot(v_range, eigen_values)
        return fig, ax


class SteadyAeroelasticModel(AeroelasticModel):
    """Steady model with zero aerodynamic mass and damping."""

    @cached_property
    def divergence_pressure(self) -> float:
        """Divergence dynamic pressure in SI Pascal."""
        eigen_values, _ = scipy.linalg.eig(
            self.structural_stiffness_matrix,
            self.aerodynamic_stiffness_matrix,
        )
        assert eigen_values[0] == math.inf
        return eigen_values[-1].real

    @cached_property
    def divergence_speed(self) -> float:
        """Divergence speed (velocity) in SI meter per second."""
        return math.sqrt(2 * self.divergence_pressure / self.density)

    @cached_property
    def aerodynamic_mass_matrix(self) -> np.ndarray:
        """Placeholder aerodynamic mass matrix."""
        return np.zeros((2, 2), dtype=np.float64)

    @cached_property
    def aerodynamic_damping_matrix(self) -> np.ndarray:
        """Placeholder aerodynamic damping mass matrix."""
        return np.zeros((2, 2), dtype=np.float64)

    @cached_property
    def aerodynamic_stiffness_matrix(self) -> np.ndarray:
        """Steady model aerodynamic stiffness matrix."""
        c = self.typical_section.chord
        cla = self.typical_section.lift_gradient
        return np.array(
            [
                [0, -c * cla],
                [0, c * cla * self.typical_section.lift_moment_arm],
            ]
        )

    @property
    def aeroelastic_mass_matrix(self) -> np.ndarray:
        """Steady aeroelastic mass matrix."""
        return self.structural_mass_matrix

    @property
    def aeroelastic_damping_matrix(self) -> np.ndarray:
        """Steady aeroelastic damping matrix."""
        return self.structural_damping_matrix

    @cached_property
    def aeroelastic_stiffness_matrix(self) -> np.ndarray:
        """Steady aeroelastic stiffness matrix.

        Note:
            The aerodynamic stiffness matrix reduces the apparent
            structural stiffness of the wing.
        """
        return (
            self.structural_stiffness_matrix
            - self.dynamic_pressure * self.aerodynamic_stiffness_matrix
        )


class TheodorsenQuasiSteadyAeroelasticModel(AeroelasticModel):
    """Quasi-Steady model derived from Theodorsen."""

    @cached_property
    def aerodynamic_mass_matrix(self) -> np.ndarray:
        """Quasi-steady aerodynamic mass matrix."""
        b = self.typical_section.half_chord
        a = self.typical_section.elastic_axis_offset / b
        return (
            np.array([[-1, a * b], [a * b, -(b ** 2) * (0.125 + a ** 2)]])
            * math.pi
            * b ** 2
        )

    @cached_property
    def aerodynamic_damping_matrix(self) -> np.ndarray:
        """Quasi-steady aerodynamic damping matrix."""
        b = self.typical_section.half_chord
        a = self.typical_section.elastic_axis_offset / b
        return (
            np.array(
                [
                    [-2, (2 * a - 2) * b],
                    [(2 * a + 1) * b, (1 - 2 * a) * a * b ** 2],
                ]
            )
            * math.pi
            * b
        )

    @cached_property
    def aerodynamic_stiffness_matrix(self) -> np.ndarray:
        """Quasi-steady aerodynamic stiffness matrix."""
        b = self.typical_section.half_chord
        a = self.typical_section.elastic_axis_offset / b
        return np.array([[0, -2], [0, (2 * a + 1) * b]]) * math.pi * b

    @cached_property
    def aeroelastic_mass_matrix(self) -> np.ndarray:  # noqa: D102
        return (
            self.structural_mass_matrix
            - self.density * self.aerodynamic_mass_matrix
        )

    @cached_property
    def aeroelastic_damping_matrix(self) -> np.ndarray:  # noqa: D102
        return (
            self.structural_damping_matrix
            - self.density * self.velocity * self.aerodynamic_damping_matrix
        )

    @cached_property
    def aeroelastic_stiffness_matrix(self) -> np.ndarray:  # noqa: D102
        return (
            self.structural_stiffness_matrix
            - self.density
            * self.velocity ** 2
            * self.aerodynamic_stiffness_matrix
        )


class LiegeQuasiSteadyAeroelasticModel(TheodorsenQuasiSteadyAeroelasticModel):
    """University of Liege Quasi-Steady Aerodynamic model."""

    @cached_property
    def aerodynamic_damping_matrix(self):
        """Similar aero damping matrix to the QS Theordorsen model.

        Note:
            The addition of the 0.25b^2 term in the pitch moment term is
            monumental in stabilizing the system.
        """
        b = self.typical_section.half_chord
        a = self.typical_section.elastic_axis_offset / b
        return (
            np.array(
                [
                    [-1, -(1 - a) * b],
                    [(a + 0.5) * b, -((a - 0.5) * a + 0.25) * b ** 2],
                ]
            )
            * math.pi
            * 2
            * b
        )


class UnsteadyAeroelasticModel(AeroelasticModel):
    """Unsteady model using Wagner's Indicial Function Approximation.

    Attributes:
        psi_1: First Wagner function curve fitting constant
        psi_2: Second Wagner function curve fitting
        eps_1: First Pole of the Theodorsen function
        eps_2: Second Pole of the Theodorsen function
    """

    psi_1: float = 0.165
    psi_2: float = 0.335
    eps_1: float = 0.0455
    eps_2: float = 0.3

    def reduced_time(self, time: float = 0) -> float:
        """Non-dimensional reduced time, t_star.

        This represents the number of half-chords travelled by the
        typical section during the duration given by ``time``.
        """
        return time * self.velocity / self.typical_section.half_chord

    def wagner_function(self, time: float = 0) -> float:
        """Returns the Wagner function evaluated at ``time``."""
        t_star = self.reduced_time(time)
        return (
            1
            - self.psi_1 * np.exp(-self.eps_1 * t_star)
            - self.psi_2 * np.exp(-self.eps_2 * t_star)
        )

    def wagner_function_derivative(self, time: float = 0) -> float:
        """First-order derivative of the wagner function at ``time``."""
        t_star = self.reduced_time(time)
        u_over_b = self.velocity / self.typical_section.half_chord
        return self.psi_1 * self.eps_1 * u_over_b * np.exp(
            -self.eps_1 * t_star
        ) + self.psi_2 * self.eps_2 * u_over_b * np.exp(-self.eps_2 * t_star)

    @cached_property
    def aerodynamic_mass_matrix(self) -> np.ndarray:
        """Unsteady aerodynamic mass matrix."""
        a = self.typical_section.elastic_axis_offset
        b = self.typical_section.half_chord
        return np.array([[1, -a * b], [-a * b, ((a * b) ** 2 + (b ** 2) / 8)]])

    @cached_property
    def aerodynamic_damping_matrix(self) -> np.ndarray:
        """Unsteady aerodynamic damping matrix."""
        phi_0 = self.wagner_function(time=0)
        c = self.typical_section.chord
        e = self.typical_section.lift_moment_arm / c
        # Collocation point distance
        f = (0.75 - self.typical_section.elastic_axis) * c
        return np.array(
            [
                [phi_0, c / 4 + phi_0 * f],
                [-e * c * phi_0, f * (c / 4 - e * c * phi_0)],
            ]
        )

    @cached_property
    def aerodynamic_stiffness_matrix(self) -> np.ndarray:
        """Unsteady aerodynamic stiffness matrix."""
        phi_0 = self.wagner_function(time=0)
        phi_dot_0 = self.wagner_function_derivative(time=0)
        c = self.typical_section.chord
        e = self.typical_section.lift_moment_arm / c
        # Collocation point distance
        f = (0.75 - self.typical_section.elastic_axis) * c
        return np.array(
            [
                [phi_dot_0, self.velocity * phi_0 + f * phi_dot_0],
                [
                    -e * c * phi_dot_0,
                    -e * c * (self.velocity * phi_0 + f * phi_dot_0),
                ],
            ]
        )

    @cached_property
    def lag_state_matrix(self) -> np.ndarray:
        """Unsteady aerodynamic lag states (wake effect) matrix.

        This is a mathematical trick in order to express the
        wake in the time domain.
        """
        b = self.typical_section.half_chord
        c = self.typical_section.chord
        e = self.typical_section.lift_moment_arm / c

        # Defining individual elements of the lag state matrix
        w_11 = -self.psi_1 * self.eps_1 ** 2 / b
        w_12 = -self.psi_2 * self.eps_2 ** 2 / b
        w_13 = self.psi_1 * self.eps_1 * (1 - self.eps_1 * (1 - 2 * e))
        w_14 = self.psi_2 * self.eps_2 * (1 - self.eps_2 * (1 - 2 * e))
        w_21 = -e * c * w_11
        w_22 = -e * c * w_12
        w_23 = -e * c * w_13
        w_24 = -e * c * w_14
        return (
            2
            * math.pi
            * self.density
            * self.velocity ** 3
            * np.array([[w_11, w_12, w_13, w_14], [w_21, w_22, w_23, w_24]])
        )

    @cached_property
    def initial_lag_state_matrix(self) -> np.ndarray:
        """Matrix of lag state equations from Leibnitz integration."""
        e_1 = self.eps_1 * self.velocity / self.typical_section.half_chord
        e_2 = self.eps_2 * self.velocity / self.typical_section.half_chord
        return np.array(
            [
                [1, 0, -e_1, 0, 0, 0],
                [1, 0, 0, -e_2, 0, 0],
                [0, 1, 0, 0, -e_1, 0],
                [0, 1, 0, 0, 0, -e_2],
            ]
        )

    @cached_property
    def aeroelastic_mass_matrix(self) -> np.ndarray:  # noqa: D102
        return (
            self.structural_mass_matrix
            + math.pi
            * self.density
            * self.typical_section.half_chord ** 2
            * self.aerodynamic_mass_matrix
        )

    @cached_property
    def aeroelastic_damping_matrix(self) -> np.ndarray:  # noqa: D102
        return (
            self.structural_damping_matrix
            + math.pi
            * self.density
            * self.velocity
            * self.typical_section.chord
            * self.aerodynamic_damping_matrix
        )

    @cached_property
    def aeroelastic_stiffness_matrix(self) -> np.ndarray:  # noqa: D102
        return (
            self.structural_stiffness_matrix
            + math.pi
            * self.density
            * self.velocity
            * self.typical_section.chord
            * self.aerodynamic_stiffness_matrix
        )

    @cached_property
    def state_matrix(self) -> np.ndarray:
        """Creates the expanded state-matrix for the unsteady model."""
        state_matrix = np.zeros((8, 8), dtype=np.float64)
        state_matrix[:4, :4] = super().state_matrix
        m_ae_inverse = np.linalg.inv(self.aeroelastic_mass_matrix)

        # Casting the lag state matrix into the upper-right corner
        state_matrix[0:2, 4:] = -m_ae_inverse @ self.lag_state_matrix

        # Casting initial lag matrix into the lower-right corner
        state_matrix[4:, 2:] = self.initial_lag_state_matrix
        return state_matrix

    @cached_property
    def input_matrix(self) -> np.ndarray:
        """Extends the input matrix with lag states."""
        return np.vstack(
            (super().input_matrix, np.zeros((4, 1), dtype=np.float64))
        )

    @cached_property
    def output_matrix(self) -> np.ndarray:
        """Expanded state-space output matrix, C, with lag-states."""
        return np.eye(8, dtype=np.float64)

    @cached_property
    def feedthrough_matrix(self) -> np.ndarray:
        """Expanded state-space feedthrough or feedfoward matrix, D."""
        return np.zeros((8, 1), dtype=np.float64)


class Point(NamedTuple):
    """Defines a 2D point tuple (x, y)."""

    x: float
    y: float


class FEMAnalysisMixin:
    """Defines derived properties from a FEM analysis.

    The main purpose of this class is to make it simpler to run and
    access the results of the a FEM analysis for use in an optimization.
    This FEM analysis is run lazily and will evaluate only when a
    attributes that require them are accessed.

    Args:
        geometry: A :py:class:`Geometry` instance.
        mesh_size: Maximum allowable relative mesh size. Defaults to
            :py:data:`math.inf`.
    """

    mesh_size: float = math.inf

    @cached_property
    def section(self) -> CrossSection:
        """Analysis :py:class:`CrossSection` object."""
        return CrossSection(self, self.mesh)

    @cached_property
    def mesh(self) -> MeshInfo:
        """Meshes :py:attr:`wingbox` using MeshPy."""
        return self.create_mesh(mesh_sizes=[self.mesh_size])

    @property
    def bending_inertia(self) -> float:
        """Area moment of inertia in SI meter to the fourth.

        Note:
            This is the inertia I_xx about the chordline of the
            typical section.
        """
        self.ensure_geometry_has_run()
        return self.section.get_ic()[0]

    @property
    def chordwise_inertia(self) -> float:
        """Area moment of inertia in SI meter to the fourth.

        Note:
            This is the inertia I_zz on about the vertical
            (heave) axis of the typical section.
        """
        self.ensure_geometry_has_run()
        return self.section.get_ic()[1]

    @property
    def polar_inertia(self) -> float:
        """Polar area moment of inertia in SI meter to the fourth."""
        return self.bending_inertia + self.chordwise_inertia

    @property
    def area(self) -> float:
        """Cross-sectional area of the supplied :py:attr:`geometry`."""
        self.ensure_geometry_has_run()
        return self.section.get_area()

    @property
    def torsion_constant(self) -> float:
        """Torsional constant, J, in SI meter to the fourth."""
        self.ensure_warping_has_run()
        return self.section.get_j()

    @property
    def centroid(self) -> Point:
        """Location of the area centroid and the center of gravity."""
        self.ensure_geometry_has_run()
        self.ensure_warping_has_run()
        return Point(*self.section.get_c())

    @property
    def shear_center(self) -> Point:
        """Location of the shear center as computed by FEM."""
        self.ensure_warping_has_run()
        return Point(*self.section.get_sc())

    _has_geometry_run = False
    _has_warping_run = False

    def ensure_geometry_has_run(self) -> None:
        """Ensure that geometric analysis has already run."""
        if not self._has_geometry_run:
            self.section.calculate_geometric_properties()
            self._has_geometry_run = True

    def ensure_warping_has_run(self) -> None:
        """Ensure that warping (shear) analysis has already run."""
        if not self._has_warping_run:
            self.ensure_geometry_has_run()
            self.section.calculate_warping_properties()
            self._has_warping_run = True


class WingBox(Geometry, FEMAnalysisMixin):
    """Creates a hollow airfoil countoured wingbox geometry.

    Args:
        x_start: Normalized start location of the wingbox along the
            airfoil chord length
        x_end: Normalized termination location of the wingbox along the
            airfoil chord length
        t_fs: Normalized thickness of the front spar web
        t_rs: Normalized thickness of the rear spar web
        t_skin: Normalized thickness of the wingbox upper/lower skin
        airfoil: A :py:class:`Airfoil` instance
        n_points: Number of points used to sample the upper and lower
            surfaces of the wingbox
        shift: The translation vector (x, y) applied to the airfoil
        mesh_size: Defines the maximum possible mesh element size
    """

    def __init__(
        self,
        x_start: float = 0.3,
        x_end: float = 0.7,
        t_fs: float = 0.01,
        t_rs: float = 0.01,
        t_skin: float = 0.01,
        airfoil: Airfoil = TypicalSection.airfoil,
        n_points: int = 20,
        shift: Tuple[float, float] = (0, 0),
        mesh_size: float = math.inf,
    ):
        self.x_start = x_start
        self.x_end = x_end
        self.t_fs = t_fs
        self.t_rs = t_rs
        self.t_skin = t_skin
        self.airfoil = airfoil
        self.n_points = n_points
        self.shift = shift
        self.mesh_size = mesh_size

    @cached_property
    def is_solid(self):
        """Determines if the wingbox has a solid geometry (no holes)."""
        return self.x_start + self.t_fs >= self.x_end - self.t_rs

    @cached_property
    def points(self) -> np.ndarray:
        """All points that define the wingbox geometry."""
        if self.is_solid:
            return self.outer_points
        else:
            topology = (
                self.inner_points,
                self.outer_points,
            )
            return np.vstack(topology)

    @cached_property
    def holes(self) -> Union[np.ndarray, List]:
        """Midpoint specifying the interior region of the airfoil."""
        if self.is_solid:
            return []
        else:
            x_mid = np.array([(self.x_end + self.x_start) / 2])
            bot_midpoint = self.airfoil.lower_surface_at(x_mid)
            top_midpoint = self.airfoil.upper_surface_at(x_mid)
            return bot_midpoint + ((top_midpoint - bot_midpoint) / 2)

    @cached_property
    def facets(self) -> List[Tuple[int, int]]:
        """Topology of :py:attr:`points` describing connectivity."""
        if self.is_solid:
            return self.get_facets(self.outer_points.shape[0])
        else:
            n_inner = self.inner_points.shape[0]
            n_outer = self.outer_points.shape[0]
            return [
                *self.get_facets(n_inner),
                *self.get_facets(n_outer, start_idx=n_inner),
            ]

    @cached_property
    def control_points(self) -> np.ndarray:
        """A single point inside the top-left corner of the wingbox."""
        x_left_spar = np.array(
            [self.x_start + self.t_fs / 2], dtype=np.float64
        )
        bot_pt = self.airfoil.lower_surface_at(x_left_spar)
        top_pt = self.airfoil.upper_surface_at(x_left_spar)
        return bot_pt + ((top_pt - bot_pt) / 2)

    @cached_property
    def perimeter(self) -> List[int]:
        """Indices of points making up the outer airfoil."""
        n_points, _ = self.outer_points.shape
        return list(range(n_points))

    @cached_property
    def outer_points(self) -> np.ndarray:
        """Outermost points on the top and bottom of the wingbox."""
        x_start, x_stop = self.x_start, self.x_end
        sample_u = np.linspace(x_start, x_stop, num=self.n_points)
        top_pts = self.airfoil.upper_surface_at(sample_u)
        bot_pts = self.airfoil.lower_surface_at(sample_u[::-1])
        return np.vstack((top_pts, bot_pts))

    @cached_property
    def inner_points(self):
        """Inner points on the top and bottom of the wingbox."""
        t_s = self.t_skin
        x_start, x_stop = self.x_start + self.t_fs, self.x_end - self.t_rs
        sample_u = np.linspace(x_start, x_stop, num=self.n_points)
        top_pts = self.airfoil.upper_surface_at(sample_u) - (0, t_s)
        bot_pts = self.airfoil.lower_surface_at(sample_u[::-1]) + (0, t_s)
        return np.vstack((top_pts, bot_pts))

    @staticmethod
    def get_facets(n_points: int, start_idx: int = 0) -> List[Tuple[int, int]]:
        """List of tuples describing connectivity of curve points.

        Args:
            n_points: Number of points that describe the curve
            start_idx: Starting index of the first point of the curve.
                Defaults to 0.

        Returns:
            Connectivity of curve points comprised of the indices
            describing the location of points within an array. For a
            triangle of 3 points the connectivity would be as follows::

                [(0, 1), (1, 2), (2, 0)]
        """
        return [
            # Generates pairs of numbers [(0, 1), (1, 2)] using zip
            *zip(
                range(start_idx, start_idx + n_points),
                range(start_idx + 1, start_idx + n_points),
            ),
            # Closing the curve (end index to start index)
            (start_idx + n_points - 1, start_idx),
        ]

    def plot_airfoil(self, ax: matplotlib.axes.Axes, n_points: float = 100):
        """Plot :py:attr:`airfoil` on the provided ``ax``.

        Args:
            ax: Plot axes onto which the airfoil should be plotted.
            n_points: Number of points used to sample the upper and
                lower surfaces of the airfoil.
        """
        x = 0.5 * (1 - np.cos(np.linspace(0, np.pi, num=n_points)))

        pts_lower = self.airfoil.lower_surface_at(x[-1:0:-1])
        pts_upper = self.airfoil.upper_surface_at(x)
        pts = np.vstack((pts_lower, pts_upper))

        ax.plot(pts[:, 0], pts[:, 1], label="Airfoil Surface", color="grey")
        ax.set_xlabel("Normalized Chord Location (x/c)")
        ax.set_ylabel("Normalized Thickness (t/c)")
        plt.axis("equal")

    def plot_geometry(self, n_points: float = 100) -> FigureHandle:
        """Plots the geometry of the current :py:class:`WingBox`."""
        plt.style.use(dufte.style)
        fig, ax = plt.subplots()

        self.plot_airfoil(ax, n_points=n_points)
        super().plot_geometry(ax)
        return fig, ax

    def plot_mesh(self, alpha: float = 1.0) -> FigureHandle:
        """Plots the mesh geometry."""
        fig, ax = self.plot_geometry()
        self.section.plot_mesh(ax, alpha=alpha)
        return fig, ax

    def plot_centroids(self) -> FigureHandle:
        """Plots the mesh, centroids, and the principal axes."""
        fig, ax = self.plot_mesh()

        (x_s, y_s) = self.shear_center
        ax.scatter(
            x_s, y_s, c="r", marker="+", s=100, label="Shear Center",
        )

        (x_pc, y_pc) = self.centroid
        ax.scatter(
            x_pc,
            y_pc,
            facecolors="none",
            edgecolors="r",
            marker="o",
            s=100,
            label="Area Centroid",
        )

        post.draw_principal_axis(
            ax,
            self.section.section_props.phi * np.pi / 180,
            self.section.section_props.cx,
            self.section.section_props.cy,
        )
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        return fig, ax


class DesignVariable(NamedTuple):
    """Defines the running variable(s) in an optimization problem."""

    initial: float
    lower: float = None
    upper: float = None


class OptimizationMeta(ABCMeta):
    """Runs post initialization action to gather design variables."""

    def __call__(cls, *args, **kwargs):
        """Runs :py:meth:`__post_init__` after object initialization."""
        obj = super().__call__(*args, **kwargs)
        obj.__post_init__()
        return obj


class Optimization(metaclass=OptimizationMeta):
    """Defines an base optimization problem using scipy.minimize.

    Attributes:
        default_options: Default options passed to the optimizer.
        history: All design vectors used to evaluate
            :py:meth:`objective_function`
    """

    default_options = {}
    history: Dict[str, List[float]] = None
    design_vector: Dict[str, DesignVariable] = None
    normalize: bool = False
    verbose: bool = False

    def __post_init__(self):
        """Traverses the MRO in reverse to get all design variables.

        Iterating in reverse ensures that the specialized class design
        variable or the one optionally passed during initialized is used
        instead of the super class value.
        """
        self.design_vector = {}
        for cls in reversed(self.__class__.mro()):
            if issubclass(cls, Optimization) or cls is Optimization:
                self.design_vector.update(self.get_design_variables(cls))
        self.design_vector.update(self.get_design_variables(self))
        self.history = {name: [] for name in self.variable_names}

    @property
    def bounds(self) -> List[Tuple[float, float]]:
        """Lower and upper bounds of each design variable."""
        return [(v.lower, v.upper) for v in self.design_vector.values()]

    @property
    def normalized_bounds(self) -> List[Tuple[float, float]]:
        """Normalized Lower and upper bounds of each design variable."""
        return [
            (v.lower / v.initial, v.upper / v.initial)
            for v in self.design_vector.values()
        ]

    @property
    def x0(self) -> np.array:
        """Initial design vector of the optimization."""
        return np.array(
            [v.initial for v in self.design_vector.values()], dtype=np.float64,
        )

    @property
    def variable_names(self) -> Tuple[str, ...]:
        """Names of the variables ordered as per the design vector."""
        return tuple(self.design_vector.keys())

    @staticmethod
    def get_design_variables(instance) -> Dict[str, DesignVariable]:
        """Gets all :py:class:`DesignVariable` in ``instance``."""
        return {
            attr: obj
            for attr, obj in vars(instance).items()
            if isinstance(obj, DesignVariable)
        }

    def optimize(self, **optimization_options) -> dict:
        """Runs the optimization and returns the optimization result."""
        self.default_options.update(**optimization_options)
        result = optimize.minimize(
            self.objective_function,
            x0=self.x0 / self.x0 if self.normalize else self.x0,
            bounds=self.normalized_bounds if self.normalize else self.bounds,
            options=self.default_options,
            callback=self.log,
        )
        return result

    def log(self, x) -> None:
        """Logs the current design vector ``x``."""
        x = self.unnormalize(x)
        for name, variable in zip(self.variable_names, x):
            self.history[name].append(variable)
        if self.verbose:
            print(*zip(self.variable_names, x))

    @abstractmethod
    def objective_function(self, x: np.ndarray) -> float:
        """A scalar objective function of the optimization problem.

        Args:
            x: Design vector of the optimization problem
        """

    def unnormalize(self, x) -> float:
        """Turns ``x`` into physical quantities if required."""
        return x * self.x0 if self.normalize else x


class TSOptimization(Optimization):
    """Abstract Base Class (ABC) of a Typical Section optimization."""

    eta_ts = DesignVariable(initial=0.75, lower=1e-6, upper=1)

    def optimize(self) -> dict:
        """Optimizes TS location using :py:meth:`objective_function`."""
        result = super().optimize()
        # Setting :py:attr:`eta_ts_final` to final optimized value.
        # The optimization result "x" is removed and renamed to "eta_ts"
        result["eta_ts"] = result.pop("x")[0]
        result["ts_opt"] = TypicalSection(eta_ts=result["eta_ts"])
        return result

    @staticmethod
    @abstractmethod
    def objective_function(eta_ts: Union[float, np.ndarray]) -> float:
        """Difference between calculated and measured frequencies."""

    @property
    @abstractmethod
    def plot_name(self) -> str:
        """Sets the label of the TS location optimization in plots."""

    def plot_objective(
        self, ax: Optional[matplotlib.axes.Axes] = None
    ) -> matplotlib.axes.Axes:
        """Plots :py:meth:`objective_function` across the half-span."""
        ax = plt.subplots()[-1] if ax is None else ax
        eta_values = np.linspace(0.3, 1.0, num=1000)
        vectorized_obj = np.vectorize(self.objective_function)
        ax.plot(
            eta_values * TypicalSection.half_span,
            vectorized_obj(eta_values),
            label=self.plot_name,
        )
        return ax


class HeaveTSOptimization(TSOptimization):
    """Optimizes TS location to match heave frequency.

    Attributes:
        measured_heave_frequency: Measured structural natural
            frequency of the wing corresponding to the first heave
            mode in SI radian
    """

    plot_name = "Heave"
    target_heave_frequency: float = 2.243

    def objective_function(self, eta_ts: Union[float, np.ndarray]) -> float:
        """Returns the squared residual w.r.t the heave frequency."""
        sm = StructuralModel(typical_section=TypicalSection(eta_ts=eta_ts))
        return (sm.coupled_heave_frequency - self.target_heave_frequency) ** 2


class TorsionTSOptimization(TSOptimization):
    """Optimizes TS location to match torsion frequency.

    Attributes:
        measured_torsion_frequency: Measured structural natural
            frequency of the wing corresponding to the first torsion
            mode in SI radian
    """

    plot_name = "Torsion"
    target_torsion_frequency: float = 31.046

    def objective_function(self, eta_ts: Union[float, np.ndarray]) -> float:
        """Returns the squared residual w.r.t the torsion frequency."""
        sm = StructuralModel(typical_section=TypicalSection(eta_ts=eta_ts))
        return (
            sm.coupled_torsion_frequency - self.target_torsion_frequency
        ) ** 2


class SimultaneousTSOptimization(HeaveTSOptimization, TorsionTSOptimization):
    """Optimizes TS location to match both torsion/heave frequency."""

    plot_name = "Simultaneous"

    def objective_function(self, eta_ts: Union[float, np.ndarray]) -> float:
        """Returns the RSS of both the heave and torsion residual.

        Note:
            RSS stands for Residual Sum of Squares where the
            residual is defined as the difference between the
            frequency calculated at the typical section and the
            measured frequency of the aircraft. The goal is to
            minimize the residual for both the heave and torsional
            frequency simultaneously.
        """
        t_rss = TorsionTSOptimization.objective_function(self, eta_ts=eta_ts)
        h_rss = HeaveTSOptimization.objective_function(self, eta_ts=eta_ts)
        return t_rss + h_rss


class WingBoxOptimization(Optimization):
    """Defines the design variables of a wingbox optimization."""

    x_start = DesignVariable(initial=0.25, lower=0.1, upper=0.45)
    x_end = DesignVariable(initial=0.7, lower=0.55, upper=0.95)
    t_fs = DesignVariable(initial=5e-3, lower=1e-3, upper=0.05)
    t_rs = DesignVariable(initial=0.04, lower=1e-3, upper=0.05)
    t_skin = DesignVariable(initial=0.01, lower=1e-3, upper=0.05)
    normalize = True
    verbose = True
    default_options = {"gtol": 1e-12, "ftol": 1e-12, "disp": 100}

    def __init__(self, typical_section: TypicalSection):
        self.typical_section = typical_section

    def optimize(self) -> dict:
        """Optimizes wingbox geometry to match wing properties."""
        result = super().optimize()
        result["wingbox"] = self.get_wingbox(result["x"])
        return result

    def get_wingbox(self, x) -> WingBox:
        """Instantiates a :py:class:`WingBox` with ``x``."""
        x = self.unnormalize(x)
        kwargs = dict(zip(self.variable_names, x))
        return WingBox(
            **kwargs, airfoil=self.typical_section.airfoil, n_points=5
        )


class GeometricWingBoxOptimization(WingBoxOptimization):
    """Optimizes the wingbox to match geometric properties."""

    def objective_function(self, x: np.ndarray) -> float:
        """Returns the RSS error of the shear center and CG."""
        wbox = self.get_wingbox(x)
        ts = self.typical_section
        shear_error = (wbox.shear_center.x - ts.elastic_axis) ** 2
        centroid_error = (wbox.centroid.y - ts.center_of_gravity) ** 2
        # i_theta = (0.75 / wbox.area) * (
        #     wbox.bending_inertia + wbox.chordwise_inertia
        # )
        # print(shear_error, centroid_error)
        return shear_error + centroid_error


class AeroelasticWingBoxOptimization(WingBoxOptimization):
    """Optimizes the wingbox to match divergence and flutter speed."""

    aeroelastic_model: AeroelasticModel = UnsteadyAeroelasticModel

    def __init__(
        self, typical_section: TypicalSection, initial_wing_box: WingBox,
    ):
        self.typical_section = typical_section
        self.initial_wing_box = initial_wing_box

        # Setting initial design varialbes to initial wing box values
        for k, v in self.get_design_variables(WingBoxOptimization).items():
            initial = getattr(initial_wing_box, k)
            setattr(self, k, DesignVariable(initial, v.lower, v.upper))

    @cached_property
    def material_density(self) -> float:
        """Density of the material in SI kilogram per meter cubed.

        Note:
            That the typical section airfoil mass is expressed
            per unit span and has units of SI kilogram per meter.
        """
        return self.typical_section.airfoil_mass / self.initial_wing_box.area

    @cached_property
    def modulus_of_elasticity(self) -> float:
        """Modulus of elasticity, E, in Newton per meter squared."""
        return (
            self.typical_section.bending_rigidity
            / self.initial_wing_box.bending_inertia
        )

    @cached_property
    def modulus_of_rigidity(self) -> float:
        """Modulus of rigidity, G, in Newton per meter squared."""
        return (
            self.typical_section.bending_rigidity
            / self.initial_wing_box.polar_inertia
        )

    @cached_property
    def inertia_offset(self) -> float:
        """Offset value to match wingbox and typical section inertia."""
        return (
            self.typical_section.airfoil_inertia
            - self.initial_wing_box.polar_inertia * self.material_density
        )

    def get_typical_section(self, wingbox: WingBox):
        """Returns a :py:class:`TypicalSection` using ``wingbox``."""
        c = self.typical_section.chord
        ts_kwargs = dataclasses.asdict(self.typical_section)
        updated_kwargs = dict(
            airfoil_mass=self.material_density * wingbox.area,
            airfoil_inertia=wingbox.polar_inertia * self.material_density
            + self.inertia_offset,
            elastic_axis=wingbox.shear_center.x / c,
            center_of_gravity=wingbox.centroid.x / c,
            bending_rigidity=wingbox.bending_inertia
            * self.modulus_of_elasticity,
            torsional_rigidity=wingbox.polar_inertia
            * self.modulus_of_rigidity,
        )
        ts_kwargs.update(updated_kwargs)
        return TypicalSection(**ts_kwargs)

    def objective_function(self, x):
        wbox = self.get_wingbox(x)
        ts = self.get_typical_section(wbox)
        aero_model = self.aeroelastic_model(typical_section=ts)
        div_speed = aero_model.divergence_speed
        flutter_speed = aero_model.flutter_speed
        error = (div_speed - 37.15) ** 2 + (flutter_speed - 37.15) ** 2
        print(error)
        return error


def run_ts_optimizations(
    print_results: bool = True,
) -> Dict[TSOptimization, dict]:
    """Runs each specialized :py:class:`TSOptimizer`."""
    optimizers = (
        HeaveTSOptimization,
        TorsionTSOptimization,
        SimultaneousTSOptimization,
    )

    results = defaultdict(dict)
    for optimizer in optimizers:
        opt = optimizer()
        results[optimizer]["instance"] = opt
        results[optimizer].update(opt.optimize())

    if print_results:
        table = Table(title="Typical Section Optimization Results")

        table.add_column("Optimizer", justify="left", style="cyan")
        table.add_column("TS Location m", justify="center")
        table.add_column("Span Fraction", justify="center")
        table.add_column("Squared Residual", justify="center", style="red")

        for opt, result in results.items():
            table.add_row(
                opt.__name__,
                f"{result['eta_ts'] * TypicalSection().half_span:.4f}",
                f"{result['eta_ts']:.4f}",
                f"{result['fun']:.4e}",
            )
        rich.print(table)
    return results


def plot_ts_optimization_results(
    results: Dict[TSOptimization, dict]
) -> FigureHandle:
    """Plots error of each specialized :py:class:`TSOptimizer`."""
    plt.style.use(dufte.style)
    fig, ax = plt.subplots()
    ax.set_yscale("log")
    ax.set_ylabel(
        r"Squared Residual $\left[\frac{\mathrm{rad}^2}{\mathrm{s}^2}\right]$"
    )
    ax.set_xlabel("Half-Span [m]")

    for result in results.values():
        result["instance"].plot_objective(ax)
    dufte.legend()
    plt.show()
    return fig, ax


def run_geometric_wbox_optimization():
    pass


def plot_geometric_wbox_optimization(
    optimized_wingbox: WingBox,
) -> FigureHandle:
    fig, (x_ax, t_ax) = plt.subplots(nrows=2, ncols=1)
    for label, data in optimized_wingbox.history.items():
        if "x" in label:
            ax = x_ax
            ylabel = "Normalized Spar Positions"
        else:
            ax = t_ax
            ylabel = "Normalized Thicknesses"
        ax.plot(
            data,
            label=label,
            marker="o",
            markerfacecolor="white",
            markeredgecolor="black",
            markeredgewidth="1",
        )
        ax.set_ylabel(ylabel)
        ax.legend(loc="best")
    plt.xlabel("Design Iteration [-]")
    fig.align_ylabels((x_ax, t_ax))
    return fig, (x_ax, t_ax)


if __name__ == "__main__":
    wing = Wing()
    opt_results = run_ts_optimizations()
    plot_ts_optimization_results(opt_results)

    torsion_ts = opt_results[TorsionTSOptimization]["ts_opt"]
    aero_model = UnsteadyAeroelasticModel(torsion_ts, velocity=30)
    print("Divergence Velocity:\n", aero_model.divergence_speed)
    print("Flutter Velocity:\n", aero_model.flutter_speed)
    _, _, xout = aero_model.simulate(5, np.linspace(0, 10, 1000))
    # wbox.plot_geometry()
    # airfoil = NACA4Airfoil("naca0012")
    # w_opt = GeometricWingBoxOptimization(wing, torsion_ts)
    # result = w_opt.optimize()
    # result["wingbox"].plot_centroids()
    # w_aero_opt = AeroelasticWingBoxOptimization(
    #     wing, torsion_ts, result["wingbox"]
    # )

    # Shear center and centroid at 0.5
    wbox = WingBox(
        x_start=0.234542706789857,
        x_end=0.6839615721216693,
        t_fs=0.005218501897141772,
        t_rs=0.03969576206057929,
        t_skin=0.00967284680430912,
    )
    wbox.plot_centroids()
    opt = AeroelasticWingBoxOptimization(
        typical_section=torsion_ts, initial_wing_box=wbox
    )
    # opt.optimize()

    # Maximum I_theta optimization
    # wbox = WingBox(**dict([('x_start', 0.05), ('x_end', 0.95), ('t_fs', 0.009456086207970144), ('t_rs', 0.033624624987465566), ('t_skin', 0.001)]))
    # wbox.plot_centroids()

    # Optimized Aeroelastic Wingbox:
    wbox_opt = WingBox(
        x_start=0.25940423,
        x_end=0.61488145,
        t_fs=0.00302934,
        t_rs=0.03731402,
        t_skin=0.01064013,
    )
    # opt.objective_function([1.208, 0.92516, 1.1, 0.5, 0.9])
