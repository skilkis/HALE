"""Contains definitions for aeroelastic analysis with typical sections.

This module is developed for the TU Delft master course AE4ASM506
Aeroelasticity. Although it is against software best practice, all
abstractions and analysis are contained within this single module as per
the request of the lecturer.
"""

from __future__ import annotations

import math
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
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


@dataclass(frozen=True)
class Wing:
    """Defines Daedelus wing parameters used in aeroelastic anlysis.

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
        measured_heave_frequency: Structural natural frequency of the
            first bending mode in SI radian
        measured_torsion_frequency: Structural natural frequency of the
            first torsion mode in SI radian
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
    measured_heave_frequency: float = 2.243
    measured_torsion_frequency: float = 31.046
    # TODO add measured divergence and measured flutter velocities


@dataclass(frozen=True)
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


class SteadyAerodynamicModel(AerodynamicModel):

    # TODO incorporate rigid aerodynamic matrices
    # These are currently not needed as

    @cached_property
    def divergence_pressure(self):
        """Divergence dynamic pressure in SI Pascal."""
        eigen_values, _ = scipy.linalg.eig(
            self.typical_section.stiffness_matrix,
            self.aerodynamic_stiffness_matrix,
        )
        assert eigen_values[0] == math.inf
        return eigen_values[-1].real

    @cached_property
    def divergence_speed(self):
        """Divergence speed (velocity) in SI meter per second."""
        return math.sqrt(
            2 * self.divergence_pressure / self.ambient_condition.density
        )

    # TODO redo the coding on this monstrosity
    def plot_frequency_vs_speed(self):

        omega_h_list, omega_theta_list = [], []
        q_list_h = None
        q_list_theta = []
        for q_inf in np.linspace(20, 90, 100):
            k = (
                self.typical_section.stiffness_matrix
                - q_inf * self.aerodynamic_stiffness_matrix
            )
            eigen_values, _ = scipy.linalg.eig(
                k, self.typical_section.mass_matrix
            )
            omega_h, omega_theta = eigen_values.real
            if omega_theta < 0 and omega_h > 0:
                omega_h_list.append(math.sqrt(omega_h))
                if q_list_h is None:
                    q_list_h = q_list_theta[:]
                q_list_h.append(q_inf)
            if omega_h and omega_theta > 0:
                omega_h_list.append(math.sqrt(omega_h))
                omega_theta_list.append(math.sqrt(omega_theta))
                q_list_theta.append(q_inf)

        plt.plot(q_list_h, omega_h_list)
        plt.plot(q_list_theta, omega_theta_list)
        ax = plt.gca()


def run_optimizers(
    aircraft: Wing, print_results: bool = True
) -> Dict[TSOptimizer, dict]:
    """Runs each specialized :py:class:`TSOptimizer`."""
    optimizers = (HeaveTSOptimizer, TorsionTSOptimier, SimultaneousTSOptimizer)

    results = {}
    for optimizer in optimizers:
        obj = optimizer(aircraft)
        results[obj] = obj.optimize()

    if print_results:
        table = Table(title="Typical Section Optimization Results")

        table.add_column("Optimizer", justify="left", style="cyan")
        table.add_column("TS Location m", justify="center")
        table.add_column("Span Fraction", justify="center")
        table.add_column("Squared Residual", justify="center", style="red")

        for opt, result in results.items():
            table.add_row(
                opt.__class__.__name__,
                f"{result['eta_ts'] * aircraft.half_span:.4f}",
                f"{result['eta_ts']:.4f}",
                f"{result['fun']:.4e}",
            )
        rich.print(table)
    return results


def plot_optimizer_results(results: Dict[TSOptimizer, dict]) -> None:
    """Plots error of each specialized :py:class:`TSOptimizer`."""
    plt.style.use(dufte.style)
    fig, ax = plt.subplots()
    ax.set_yscale("log")
    ax.set_ylabel(
        r"Squared Residual $\left[\frac{\mathrm{rad}^2}{\mathrm{s}^2}\right]$"
    )
    ax.set_xlabel("Half-Span [m]")

    for opt, _ in results.items():
        opt.plot_objective(ax)
    dufte.legend()
    plt.show()


class Wingbox(Geometry):
    """Creates a hollow airfoil countoured wingbox section.

    Args:
        airfoil: A :py:class:`Airfoil` instance
        x_start: Normalized start location of the wingbox along the
            airfoil chord length
        x_end: Normalized termination location of the wingbox along the
            airfoil chord length
        t_fs: Normalized thickness of the front spar web
        t_rs: Normalized thickness of the rear spar web
        t_skin: Normalized thickness of the wingbox upper/lower skin
        num: Number of points used to sample the upper and lower
            surfaces of the wingbox
        shift: The translation vector (x, y) applied to the airfoil
    """

    def __init__(
        self,
        airfoil: Airfoil = NACA4Airfoil("0012"),
        x_start: float = 0.3,
        x_stop: float = 0.7,
        t_fs: float = 0.01,
        t_rs: float = 0.01,
        t_skin: float = 0.01,
        num: int = 20,
        shift: Tuple[float, float] = (0, 0),
    ):
        self.airfoil = airfoil
        self.x_start = x_start
        self.x_end = x_stop
        self.t_fs = t_fs
        self.t_rs = t_rs
        self.t_skin = t_skin
        self.num = num
        self.shift = shift

    @cached_property
    def points(self) -> np.ndarray:
        """All points that define the wingbox geometry."""
        topology = (
            self.inner_points,
            self.outer_points,
        )
        return np.vstack(topology)

    @cached_property
    def holes(self) -> np.ndarray:
        """Midpoint specifying the interior region of the airfoil."""
        x_mid = np.array([(self.x_end + self.x_start) / 2])
        bot_midpoint = self.airfoil.lower_surface_at(x_mid)
        top_midpoint = self.airfoil.upper_surface_at(x_mid)
        return bot_midpoint + ((top_midpoint - bot_midpoint) / 2)

    @cached_property
    def facets(self) -> List[Tuple[int, int]]:
        """Topology of :py:attr:`points` describing connectivity."""
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
        sample_u = np.linspace(x_start, x_stop, num=self.num)
        top_pts = self.airfoil.upper_surface_at(sample_u)
        bot_pts = self.airfoil.lower_surface_at(sample_u[::-1])
        return np.vstack((top_pts, bot_pts))

    @cached_property
    def inner_points(self):
        """Inner points on the top and bottom of the wingbox."""
        t_s = self.t_skin
        x_start, x_stop = self.x_start + self.t_fs, self.x_end - self.t_rs
        sample_u = np.linspace(x_start, x_stop, num=self.num)
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


if __name__ == "__main__":
    wing = Wing()
    opt_results = run_optimizers(wing)
    plot_optimizer_results(opt_results)

    torsion_ts = tuple(opt_results.values())[1]["ts_opt"]
    steady_aero = SteadyAerodynamicModel(torsion_ts, AmbientCondition())
    print(steady_aero.divergence_speed)
    airfoil = NACA4Airfoil("naca0012")
    wingbox = Wingbox(
        airfoil=airfoil,
        x_start=0.25,
        x_stop=0.7,
        t_fs=0.005,
        t_rs=0.04,
        t_skin=0.01,
        num=20,
    )
    wingbox.plot()
    mesh = wingbox.create_mesh(mesh_sizes=[1])
    section = CrossSection(wingbox, mesh)
    section.plot_mesh()
    section.calculate_geometric_properties()
    section.calculate_warping_properties()
    section.plot_centroids()
