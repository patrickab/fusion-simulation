# Nuclear Fusion Simulation

Work in progress. Parametrizable & differentiable geometry for Tokamak fusion reacotrs. Implemented using JAX.

## Table of Contents
- [Poloidal Plasma Boundary Definition](#poloidal-plasma-boundary-definition)
- [3D Plasma Surface Generation](#3d-plasma-surface-generation)

---

<img width="2161" height="1456" alt="image" src="https://github.com/user-attachments/assets/fada36c9-ee7d-4a51-9e11-5609d53d3b92" />

---

## Poloidal Plasma Boundary Definition

The plasma cross-section in the poloidal (R-Z) plane models tokamak plasma shape with parameters for elongation ($\kappa$) and triangularity ($\delta$).

$$
R(\theta) = R_0 + a \cos \left(\theta + \delta \sin \theta \right)
$$
$$
Z(\theta) = \kappa a \sin \theta
$$

- **$R_0$**: Major radius (center of torus in meters)
- **$a$**: Minor radius (plasma radius)
- **$\kappa$**: Elongation, stretches plasma vertically
- **$\delta$**: Triangularity, shapes plasma cross-section triangularly
- **$\theta$**: Poloidal angle parameter, varies from 0 to $2\pi$

This equation captures the characteristic D-shaped plasma cross-section in modern tokamaks by introducing shaping effects beyond an ideal circular boundary. <image here> illustrating plasma cross-section shapes with varying $\kappa$ and $\delta$.


---

## 3D Plasma Surface Generation

The 2D poloidal plasma contour is revolved around the vertical (Z) axis to form a 3D toroidal plasma surface.

- Use poloidal boundary points $R(\theta), Z(\theta)$.
- Create toroidal angles $\phi$ spanning $0$ to $2\pi$.
- Generate 2D grids $(R, \phi)$ for a parametric surface.
- Convert cylindrical coordinates $(R, \phi, Z)$ to Cartesian:

$$
X = R \cos \phi, \quad Y = R \sin \phi, \quad Z = Z
$$

This produces a symmetric surface representing a tokamak plasma volume warped according to shaping parameters. The procedure efficiently uses numpy broadcasting for mesh generation. <image here> of resulting 3D plasma surface.

---
