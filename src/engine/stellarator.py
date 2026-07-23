"""Low-order Fourier geometry for stellarator plasma surfaces."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class FourierMode:
    """One stellarator-symmetric boundary mode."""

    m: int
    n: int
    R_cos: float
    Z_sin: float


@dataclass(frozen=True)
class StellaratorSurface:
    """Structured toroidal-poloidal surface mesh in Cartesian coordinates."""

    X: NDArray[np.float64]
    Y: NDArray[np.float64]
    Z: NDArray[np.float64]
    R: NDArray[np.float64]
    phi: NDArray[np.float64]
    theta: NDArray[np.float64]


def boundary_modes(
    R0: float,
    a: float,
    kappa: float,
    helical_amplitude: float,
) -> tuple[FourierMode, ...]:
    """Build the low-order mode set exposed by the geometry explorer."""
    helical_radius = a * helical_amplitude
    return (
        FourierMode(m=0, n=0, R_cos=R0, Z_sin=0.0),
        FourierMode(m=1, n=0, R_cos=a, Z_sin=kappa * a),
        FourierMode(m=1, n=1, R_cos=helical_radius, Z_sin=helical_radius),
    )


def evaluate_boundary(
    theta: NDArray[np.float64],
    phi: NDArray[np.float64],
    n_field_periods: int,
    modes: tuple[FourierMode, ...],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Evaluate a stellarator-symmetric Fourier boundary in cylindrical coordinates."""
    R = np.zeros(np.broadcast_shapes(theta.shape, phi.shape), dtype=np.float64)
    Z = np.zeros_like(R)
    for mode in modes:
        phase = mode.m * theta - mode.n * n_field_periods * phi
        R += mode.R_cos * np.cos(phase)
        Z += mode.Z_sin * np.sin(phase)
    return R, Z


def calculate_stellarator_surface(
    R0: float,
    a: float,
    kappa: float,
    n_field_periods: int,
    helical_amplitude: float,
    n_phi: int = 256,
    n_theta: int = 256,
) -> StellaratorSurface:
    """Generate a closed structured mesh for a low-order stellarator boundary."""
    modes = boundary_modes(R0, a, kappa, helical_amplitude)
    phi_values = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
    theta_values = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    phi, theta = np.meshgrid(phi_values, theta_values, indexing="ij")
    R, Z = evaluate_boundary(theta, phi, n_field_periods, modes)
    return StellaratorSurface(
        X=R * np.cos(phi),
        Y=R * np.sin(phi),
        Z=Z,
        R=R,
        phi=phi,
        theta=theta,
    )


def calculate_toroidal_slice(
    R0: float,
    a: float,
    kappa: float,
    n_field_periods: int,
    helical_amplitude: float,
    phi: float = 0.0,
    n_theta: int = 256,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return one fixed-toroidal-angle R-Z boundary slice."""
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=True)
    phi_values = np.full_like(theta, phi)
    return evaluate_boundary(
        theta,
        phi_values,
        n_field_periods,
        boundary_modes(R0, a, kappa, helical_amplitude),
    )
