# Nuclear Fusion Simulation

Simulation of magnetic forces in a Tokamak Fusion Reactor using a Physics Informed Neural Network (PINN). Implemented using JAX.

## Milestones
- ✅ Define parametrizable Reactor Geometry &rarr; (numpy)
- ✅ Migrate to differentiable geometry &rarr; (JAX)
- ✅ Train Physics Informed Neural Network &rarr; (PINN)
- ✅ Predict magnetic flux &rarr; (output of PINN)
- ✅ Predict magnetic field lines &rarr;
- ✅ Visualize magnetic field lines in 3D

---

## Table of Contents
- [Reactor Geometry](#reactor-geometry)
- [Physics Problem & Loss Function](#physics-problem--loss-function)
- [Prediction of 3D Magnetic Field Lines](#prediction-of-3d-magnetic-field-lines)
- [Neural Network Architecture](#neural-network-architecture)
- [JAX Design Considerations](#jax-specific-design-considerations-for-efficiency)

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

https://github.com/user-attachments/assets/a83842f5-a1eb-4e32-8735-179937c9f00c

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

<img width="2399" height="1312" alt="image" src="https://github.com/user-attachments/assets/16618ea7-0bbf-41d9-84c7-2b12ae8c20a9" />

---

## Prediction of 3D magnetic field lines

Magnetic field lines in a fusion reactor can be described through the spatial derivatives of poloidal flux $\psi$ and the toroidal magnetic field.

Since the network is a differentiable surrogate for $\psi$, the 2D magnetic field can be recovered analytically via automatic differentiation of the network output wrt. inputs $(R,Z)$.

The full 3D magnetic field can now be reconstructed by incorporating the force of the toroidal magnetic field $F(\psi)$.

$$
\vec{B} = \left[ -\frac{1}{R}\frac{\partial \psi}{\partial Z},\quad \frac{1}{R}\frac{\partial \psi}{\partial R},\quad \frac{F(\psi)}{R} \right]
$$

This is computationally inexpensive and can be performed on consumer hardware in a matter of milliseconds.

The following demo shows real time calculations of 3D magnetic field lines on a 11th Gen Intel-i5.

https://github.com/user-attachments/assets/4b60c624-9668-419b-a0a0-912037ec2a8b

---

## Neural Network Architecture

The network is a simple **Multi-Layer Perceptron (MLP)** with Swish activations, mapping spatial coordinates and plasma configuration parameters directly to $\psi$:

$$
\hat{\psi} = \text{MLP}(R, Z,\ R_0, a, \kappa, \delta,\ p_0, F_\text{axis}, \alpha, \gamma)
$$

By conditioning on plasma parameters as network inputs — rather than training a separate model per configuration — the network learns a **universal flux function** across the full geometry and state space.

**Training setup:**
- **Sampler**:
  -  Points are drawn via Sobol sequences, re-sampled every epoch
  -  Reactor geometries are resampled every 10 epochs to prevent overfitting
- **Mini-Batch**:
  - Each training batch contains a set of diverse plasma configurations
  - Each plasma configuartion contains a large set of $R,Z$ coordinates
- **Optimizer**: Adam with warmup cosine decay schedule
- **Loss**: Physics residual + weighted boundary conditions (Dirichlet & Neumann)

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
