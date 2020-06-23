"""Contains definitions for aeroelastic analysis with typical sections.

This module is developed for the TU Delft master course AE4ASM506
Aeroelasticity. Although it is against software best practice, the
entire analysis is contained within this single module as per the
request of the lecturer.
"""

import math
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from functools import cached_property
from typing import Dict, Optional, Tuple, Union

import dufte
import matplotlib
import numpy as np
import rich
import scipy
from matplotlib import pyplot as plt
from rich.table import Table
from scipy import optimize


@dataclass(frozen=True)
class Wing:
    """Defines straight wing properties used in aeroelastic anlysis.

    Note:
        Some of these quantities are expressed on a per-unit length
        basis as is typical for aeroelastic analysis, also
        refered to as local values, see Reader pg. 22.

    Args:
        half_span: Half of the main wing span in SI meter
        chord: Chord-length of the main wing airfoil in SI meter
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
        heave_frequency: Structural natural frequency of the first
            bending mode in SI radian
        torsion_frequency; Structural natural frequency of the first
            torsion mode in SI readian
    """

    half_span: float = 16
    chord: float = 1
    airfoil_mass: float = 0.75
    airfoil_inertia: float = 0.1
    elastic_axis: float = 0.5
    center_of_gravity: float = 0.5
    bending_rigidity: float = 2e4
    torsional_rigidity: float = 1e4
    # TODO consider removing, lag rigidity is not important here
    lag_rigidity: float = 4e6
    heave_frequency: float = 2.243
    torsion_frequency: float = 31.046
    lift_gradient: float = 2 * math.pi


@dataclass(frozen=True)
class AmbientCondition:

    altitude: float = 20e3
    density: float = 0.0889


# TODO document arguments
class TypicalSection:
    def __init__(self, wing: Wing, eta_ts: Union[float, np.ndarray] = 0.75):
        self.wing = wing
        self.normalized_location = (
            eta_ts[0] if isinstance(eta_ts, np.ndarray) else eta_ts
        )

    @cached_property
    def y_ts(self):
        """Spanwise location of the typical section in SI meter."""
        return self.normalized_location * self.wing.half_span

    @cached_property
    def half_chord(self):
        """Half-chord, b, in SI meter."""
        return self.wing.chord / 2

    @cached_property
    def x_theta(self):
        """Norm. distance from the elastic axis to center of gravity."""
        return self.wing.center_of_gravity - self.wing.elastic_axis

    @cached_property
    def i_theta(self):
        """Mass moment of inertia per unit span at the elastic axis.

        This uses the Stiener Parallel Axis Theorem to compute the
        mass moment of inertia at the elastic axis using the
        value of the inertia about the center of gravity. This is
        equivalent to Iyy of the inertia tensor of the wing as the
        y-axis is coincident with the elastic axis.
        """
        i_theta_star = self.wing.airfoil_inertia
        x_theta = self.x_theta
        b = self.half_chord
        return i_theta_star + self.wing.airfoil_mass * (x_theta * b) ** 2

    @cached_property
    def mass_matrix(self):
        """Structural mass matrix of the typical section."""
        return np.array(
            [
                [self.wing.airfoil_mass, self.x_theta * self.half_chord],
                [self.x_theta * self.half_chord, self.i_theta],
            ]
        )

    @cached_property
    def unit_bending_stiffness(self):
        """Bending stiffness per unit span in SI Pascal."""
        EI = self.wing.bending_rigidity / self.y_ts
        return (3 * EI) / (self.y_ts ** 3)

    @cached_property
    def unit_torsion_stiffness(self):
        """Torsion striffness per unit span in SI Newton meter."""
        GJ = self.wing.torsional_rigidity / self.y_ts
        return GJ / (self.y_ts)

    @cached_property
    def stiffness_matrix(self):
        """Structural stiffness matrix of the typical section."""
        return np.array(
            [
                [self.unit_bending_stiffness, 0],
                [0, self.unit_torsion_stiffness],
            ]
        )

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
            np.linalg.inv(self.mass_matrix) @ self.stiffness_matrix
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
        return math.sqrt(self.unit_bending_stiffness / self.wing.airfoil_mass)

    @cached_property
    def uncoupled_torsion_frequency(self) -> float:
        """Typical section torsion frequency in SI radian per second."""
        return math.sqrt(self.unit_torsion_stiffness / self.i_theta)


class TSOptimizer(metaclass=ABCMeta):
    """Abstract Base Class (ABC) of a Typical Section optimization."""

    def __init__(self, wing: Wing):
        self.wing = wing

    def optimize(self) -> dict:
        """Optimizes TS location using :py:meth:`objective_function`."""
        result = optimize.minimize(self.objective_function, [0.75])
        # Setting :py:attr:`eta_ts_final` to final optimized value.
        # The optimization result "x" is removed and renamed to "eta_ts"
        result["eta_ts"] = result.pop("x")[0]
        result["ts_opt"] = TypicalSection(
            wing=self.wing, eta_ts=result["eta_ts"]
        )
        return result

    @abstractmethod
    def objective_function(self, eta_ts: float) -> float:
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
        ax.plot(
            eta_values * self.wing.half_span,
            [self.objective_function(e) for e in eta_values],
            label=self.plot_name,
        )
        return ax


class HeaveTSOptimizer(TSOptimizer):
    """Optimizes TS location to match heave frequency."""

    plot_name = "Heave"

    def objective_function(self, eta_ts: float) -> float:
        """Returns the squared residual w.r.t the heave frequency."""
        ts = TypicalSection(wing=self.wing, eta_ts=eta_ts)
        return (ts.coupled_heave_frequency - self.wing.heave_frequency) ** 2


class TorsionTSOptimier(TSOptimizer):
    """Optimizes TS location to match torsion frequency."""

    plot_name = "Torsion"

    def objective_function(self, eta_ts: float) -> float:
        """Returns the squared residual w.r.t the torsion frequency."""
        ts = TypicalSection(wing=self.wing, eta_ts=eta_ts)
        return (
            ts.coupled_torsion_frequency - self.wing.torsion_frequency
        ) ** 2


class SimultaneousTSOptimizer(TSOptimizer):
    """Optimizes TS location to match both torsion/heave frequency."""

    plot_name = "Simultaneous"

    def objective_function(self, eta_ts: float) -> float:
        """Returns the RSS of both the heave and torsion residual.

        Note:
            RSS stands for Residual Sum of Squares where the
            residual is defined as the difference between the
            frequency calculated at the typical section and the
            measured frequency of the aircraft. The goal is to
            minimize the residual for both the heave and torsional
            frequency simultaneously.
        """

        ts = TypicalSection(wing=self.wing, eta_ts=eta_ts)
        return (
            (ts.coupled_heave_frequency - self.wing.heave_frequency) ** 2
            + (ts.coupled_torsion_frequency - self.wing.torsion_frequency) ** 2
        )


class AerodynamicModel(metaclass=ABCMeta):
    def __init__(
        self,
        typical_section: TypicalSection,
        ambient_condition: AmbientCondition,
    ):
        self.typical_section = typical_section
        self.ambient_condition = AmbientCondition

    @cached_property
    def aerodynamic_stiffness_matrix(self):
        c = self.wing.chord
        cla = self.wing.lift_gradient
        return np.array([[0, -c * cla], [0, c * cla * self.lift_moment_arm]])

    @property
    def wing(self):
        """Provides direct access to wing properties."""
        return self.typical_section.wing

    @cached_property
    def lift_moment_arm(self):
        """Distance between the aerodynamic center and the elastic axis.

        Note:
            This assumes that the aerodynamic center is located at
            quarter chord.
        """
        return (self.wing.elastic_axis - 0.25) * self.wing.chord


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


if __name__ == "__main__":
    wing = Wing()
    opt_results = run_optimizers(wing)
    plot_optimizer_results(opt_results)

    torsion_ts = tuple(opt_results.values())[1]["ts_opt"]
    steady_aero = SteadyAerodynamicModel(torsion_ts, AmbientCondition())
    print(steady_aero.divergence_speed)