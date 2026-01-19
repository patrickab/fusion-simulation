# Nuclear Fusion Simulation

Work in progress. Parametrizable & differentiable estimation of pl. Implemented using JAX.

## Milestones
- ✅ Define parametrizable Reactor Geometry &rarr; (numpy)
- ✅ Migrate to differentiable geometry &rarr; (JAX)
- ✅ Train Physics Informed Neural Network &rarr; (PINN)
- ✅ Predict magnetic flux &rarr; (output of PINN)
- ✅ Predict magnetic field lines &rarr; (spatial derivatives of PINN)
- ✅ Visualize magnetic field lines in 3D
- [ ] Finite Element Methods (FEM) validation
- [ ] Kolmogorov Arnold Netwerk (KAN) with physics loss 

## Table of Contents
- [Poloidal Plasma Boundary Definition](#poloidal-plasma-boundary-definition)
- [3D Plasma Surface Generation](#3d-plasma-surface-generation)

---

<img width="2161" height="1456" alt="image" src="https://github.com/user-attachments/assets/fada36c9-ee7d-4a51-9e11-5609d53d3b92" />

---

## Reactor Geometry

The plasma cross-section in the poloidal (R-Z) plane models tokamak plasma shape with parameters for elongation ($\kappa$) and triangularity ($\delta$).

$$
R(\theta) = R_0 + a \cos \left(\theta + \delta \sin \theta \right)
$$
$$
Z(\theta) = \kappa a \sin \theta
$$

- **$R_0$**: Central radius of torus (m)
- **$a$**: Poloidal section radius (m)
- **$\kappa$**: Elongation, stretches poloidal plasma vertically
- **$\delta$**: Triangularity, shapes poloidal plasma triangularly

This equation yields a coordinate for every input $\theta$
- **$\theta$**: Poloidal angle, ranges from 0 to $2\pi$

---
