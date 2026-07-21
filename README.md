# Nuclear Fusion Simulation

Simulation of magnetic forces in a Tokamak Fusion Reactor using a Physics Informed Neural Network (PINN). Implemented using JAX.

## Milestones
- ✅ Define parametrizable Reactor Geometry &rarr; (numpy)
- ✅ Migrate to differentiable geometry &rarr; (JAX)
- ✅ Train Physics Informed Neural Network
- ✅ Predict magnetic flux
- ✅ Predict magnetic field lines
- ✅ Implement frontend for inference analysis
- ✅ Visualize magnetic field lines in 3D
- ✅ Implement version-controlled benchmarking in frontend & backend
- ✅ Implement hyperparameter optimization pipeline using [Optuna](https://github.com/optuna/optuna)

---

## Table of Contents
- [Reactor Geometry](#reactor-geometry)
- [Physics Problem & Loss Function](#physics-problem--loss-function)
- [Prediction of 3D Magnetic Field Lines](#prediction-of-3d-magnetic-field-lines)
- [Evolution of network performance](#evolution-of-network-performance)
- [Hyperparameter Optimization](#hyperparameter-optimization)
- [Neural Network Architecture](#neural-network-architecture)

---

## Reactor Geometry
Implementation with JAX allows to generate large quantities of diverse geometries as training data for the Neural Network.

The 2D geometry can be expressed through 3 parameters
- **$\kappa$**: Elongation (stretches 2D shape vertically)
- **$\delta$**: Triangularity (shapes poloidal plasma 'triangularly')
- **$a$**: Poloidal section radius (m)

These values define coordinates for any angle $\theta$

$$
R(\theta) = R_0 + a \cos \left(\theta + \delta \sin \theta \right)
$$
$$
Z(\theta) = \kappa a \sin \theta
$$

By defining a central radius for the torus it is possible to project this geoemtry into 3D.

<img width="2377" height="1575" alt="image" src="https://github.com/user-attachments/assets/1fc4224f-f7ff-49bb-a4b6-32f20ac72c44" />

---

## Physics Problem & Loss Function

The **Grad-Shafranov (GS) equation** is a requirement for stable nuclear fusion. It describes the force balance between inward magnetic pressure & the plasma's outward pressure.

For a tokamak, this can be expressed as a 2D partial differential equation (PDE):
- **$\psi(R, Z)$**: poloidal magnetic flux (describes how magnetic field lines wrap around the plasma)
- **$p(\psi)$**: plasma pressure profile (outward force induced by the plasma)
- **$F(\psi)$**: toroidal magnetic field (inward force induced by toroidal coils)

$$
\Delta^* \psi = -\mu_0 R^2 p'(\psi) - F(\psi) F'(\psi)
$$

A solution $\psi$ that satisfies this equation everywhere defines a valid plasma equilibrium.

Unfortunately this problem is not analytically tractable, and is usually simulated via numerical solvers. This is computationally expensive and non-differentiable.

Instead of solving the GS equation numerically, this project trains a neural network to **approximate $\psi(R, Z)$ directly** — for any point, on any reactor geometry. For this purpose the residual of the Grad-Shafranov equation can be treated as differentiable loss signal:

$$
\mathcal{L}_\text{residual} = \left\langle \left( \Delta^* \psi - \text{RHS}(\psi) \right)^2 \right\rangle_\text{interior}
$$

A network that minimises this loss provides a differentiable function for $\psi$ for any reactor shape.

<img width="2366" height="1142" alt="image" src="https://github.com/user-attachments/assets/10e6db77-b49a-44d5-a831-939d507cc6ce" />

---

## Prediction of 3D magnetic field lines

Magnetic field lines in an axisymmetric fusion equilibrium can be reconstructed from the poloidal flux function $\psi(R,Z)$ and the toroidal field function $F(\psi)$. Since the neural network provides a differentiable surrogate $\psi_\theta(R,Z)$, the poloidal magnetic field can be obtained directly through automatic differentiation:

$$
B_R=-\frac{1}{R}\frac{\partial\psi_\theta}{\partial Z},
\qquad
B_Z=\frac{1}{R}\frac{\partial\psi_\theta}{\partial R}.
$$

Together with the toroidal field contribution,

$$
B_\phi=\frac{F(\psi_\theta)}{R}
$$

gives the full axisymmetric 3D magnetic field:


$$
\mathbf B=
\left(
-\frac{1}{R}\frac{\partial\psi_\theta}{\partial Z},
\frac{F(\psi_\theta)}{R},
\frac{1}{R}\frac{\partial\psi_\theta}{\partial R}
\right)_{(R,\phi,Z)}
$$

This is computationally inexpensive and can be performed on consumer hardware in a matter of milliseconds.

The following demo shows real time calculations of 3D magnetic field lines on a 11th Gen Intel-i5.

https://github.com/user-attachments/assets/4b60c624-9668-419b-a0a0-912037ec2a8b

---

## Evolution of network performance

Through testing of different architetures and loss/activation functions the average absolute residual has been reduced by two orders of magnitude from a value range between $[0,10]$ in April to $[0,10^{-2}]$ as of now (July 15th 2026).

---

## Hyperparameter Optimization

In addition a pipeline for bayesian hyperparameter optimization with [Optuna](https://github.com/optuna/optuna) has been implemented. For categorical search spaces the default [TPE optimizer](https://arxiv.org/abs/2304.11127) is used. For continuous search spaces the pipeline automatically switches to Gaussian Process Regression with [Expected Improvement](https://ekamperi.github.io/machine%20learning/2021/06/11/acquisition-functions.html#expected-improvement-ei) (EI). A future integration of [Max-Value Entropy Search](https://arxiv.org/abs/1703.01968) (MES) is planned.

The following video shows 2-dimensional residual plots for different networks. It demonstrates the results from the latest hyperparameter optimization.

https://github.com/user-attachments/assets/53aa94e8-0f4e-43d7-b939-f76465dfef12

---

## Neural Network Architecture

The network is a simple **Multi-Layer Perceptron (MLP)** with Swish activations, mapping spatial coordinates and plasma configuration parameters directly to $\psi$:

$$
\hat{\psi} = \text{MLP}(R, Z,\ R_0, a, \kappa, \delta,\ p_0, F_\text{axis}, \alpha, \gamma)
$$

By conditioning on plasma parameters as network inputs — rather than training a separate model per configuration — the network learns a **universal flux function** across the full geometry and state space.

## **Training setup:**

**Physics-Informed Loss:**
Guided solely by partial differential equations. The network learns by minimizing the violation of force balance (internal plasma physics) and boundary conditions (reactor wall constraints).

**Quasi-Random Sampling:**
Probe points (collocation coordinates) are drawn via Sobol sequences. This prevents point clustering and guarantees an even, space-filling exploration of the physical domain.

**Adaptive Refinement:**
Training engine dynamically places more probe points in highest-loss regions.

**Hybrid Optimization:** 
- *AdamW* searches globally to find the correct physical basin of attraction.
- *L-BFGS* (quasi-Newton) polishes the solution to machine-level precision.

**Simplifying assumptions:**
- Pressure profile $p(\psi) \propto (1 - \psi_\text{norm}^\alpha)$ — power law, zero at the boundary
- Toroidal field profile $F(\psi) = F_\text{axis}(1 - \psi_\text{norm}^\gamma)$ — monotone decrease from axis to edge
- Fixed edge flux $\psi_\text{edge} = 0$ — plasma boundary is a single flux surface
- Axis flux $\psi_\text{axis}$ estimated per-batch as $\min \psi$ over interior samples

---

## JAX-specific design considerations for efficiency

- **2nd order derivatives with (`jax.jvp`)**: Physics loss requires 2nd-order spatial derivatives. Nested JVP computes these without materializing full Hessians — O(1) memory vs O(n²) for reverse-mode Hessians.
- **Double vmap vectorization**: Loss computation is vectorized over both plasma configurations & spatial collocation points, saturating GPU throughput on large tensor batches.
- **Gradient checkpointing (`jax.checkpoint`)**: PDE residuals produce massive computational graphs from nested derivatives. Remat recomputes activations during backprop, allowing to train larger networks with limited VRAM.

