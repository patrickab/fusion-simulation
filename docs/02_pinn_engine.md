# Physics Engine for a Grad–Shafranov PINN

This module encodes the governing laws of axisymmetric MHD equilibrium into a differentiable framework. It transforms the physical problem of plasma confinement into an optimization task, where a Physics-Informed Neural Network (PINN) learns the magnetic topology that balances internal plasma pressure against magnetic tension.

---

## 1. Physical Model & Assumptions

The equilibrium model is based on **ideal, static MHD** under **axisymmetry**. The problem is framed as a **forward problem**, solving for equilibrium geometry given prescribed source profiles.

**Core Assumptions:**
- **Static Equilibrium:** No time dependence ($\partial / \partial t = 0$) and no plasma flow.
- **Toroidal Symmetry:** Derivatives in the toroidal direction vanish ($\partial / \partial \phi = 0$).
- **Force Balance:** Confinement is governed by the competition between outward pressure gradients and inward Lorentz forces:
  $$ -\nabla p + \mathbf{J} \times \mathbf{B} = 0 $$

---

## 2. Reduced Geometry & Coordinate System

### Cylindrical Coordinates
The problem is formulated in cylindrical coordinates $(R, \phi, Z)$. Axisymmetry reduces the domain to the **poloidal plane** $(R, Z)$.

### Scalar Flux Representation
The magnetic field structure is implicitly defined by a single scalar field, the **poloidal magnetic flux** $\psi(R, Z)$. The magnetic field vector $\mathbf{B}$ is decomposed as:

$$
\mathbf{B}
=
\underbrace{\frac{1}{R} \nabla \psi \times \hat{\phi}}_{\text{Poloidal Field}}
+
\underbrace{\frac{F(\psi)}{R} \hat{\phi}}_{\text{Toroidal Field}}
$$

Where $F(\psi) = R B_\phi$ represents the poloidal current function.

---

## 3. The Grad–Shafranov Operator

The geometry of the flux surfaces is encoded by the **Grad–Shafranov operator** ($\Delta^*$). This operator measures the curvature of the flux function within the poloidal plane.

$$
\Delta^* \psi
=
R \frac{\partial}{\partial R}
\left(
\frac{1}{R} \frac{\partial \psi}{\partial R}
\right)
+
\frac{\partial^2 \psi}{\partial Z^2}
$$

### Physical and Computational Nuance
- **Toroidal Curvature:** Unlike the standard Laplacian $\nabla^2$, the $\Delta^*$ operator includes a $-1/R$ term. This accounts for the toroidal geometry (the "donut hole" effect), distinguishing it from planar 2D problems.
- **Implementation Strategy:** The operator is evaluated via **Automatic Differentiation (AD)** (e.g., JAX). This utilizes nested gradient operations or Hessian vector products to compute second-order derivatives with machine precision, avoiding finite-difference errors.

---

## 4. The Grad–Shafranov Equation

Combining force balance with Maxwell’s equations yields the governing elliptic partial differential equation:

$$
\underbrace{\Delta^* \psi}_{\text{Magnetic Tension}}
=
\underbrace{- \mu_0 R^2 \frac{dp}{d\psi} - F(\psi)\frac{dF}{d\psi}}_{\text{Plasma Source Terms}}
$$

### The Shafranov Shift
The $R^2$ scaling factor attached to the pressure gradient term ($\frac{dp}{d\psi}$) has a critical physical implication. It amplifies pressure forces at larger radii (the outer side of the torus). This asymmetry forces the magnetic axis to shift outward relative to the geometric center of the vacuum vessel, a phenomenon known as the **Shafranov Shift**.

---

## 5. Profile Modeling & Dynamic Axis Detection

To evaluate the source terms $p(\psi)$ and $F(\psi)$, the flux must be mapped to a normalized coordinate $\bar{\psi} \in [0, 1]$.

### The Dynamic Axis Problem
The magnetic axis ($\psi_{\text{axis}}$), where pressure is maximal, is not known *a priori*; it is a property of the solution being learned.
- **Detection Strategy:** The network dynamically identifies the axis during the forward pass by locating the minimum flux value in the domain: $\psi_{\text{axis}} \approx \min(\psi_\theta)$.
- **Computational Stability:** To prevent unstable feedback loops during backpropagation, the value of $\psi_{\text{axis}}$ is **detached from the computational graph** (e.g., via a `stop_gradient` operation). This allows the network to "discover" the axis location without treating it as a trainable parameter in the loss landscape.

### Normalization
$$
\bar{\psi}
=
\frac{\psi - \psi_{\text{axis}}}{\psi_{\text{boundary}} - \psi_{\text{axis}}}
$$
This ensures the pressure profile $p(\bar{\psi})$ is correctly mapped from the axis ($\bar{\psi}=0$) to the edge ($\bar{\psi}=1$).

---

## 6. Boundary Conditions & Physical Constraints

To anchor the solution physically, the engine enforces specific boundary conditions (BCs) at the reactor wall, effectively creating a "magnetic cage."

### 1. Dirichlet BC (Gauge Fixing)
$$
\psi(R_{\text{wall}}, Z_{\text{wall}}) = 0
$$
- **Physical Role:** Defines the wall as the Last Closed Flux Surface (LCFS).
- **Mathematical Role:** Fixes the gauge freedom of the potential function $\psi$, preventing trivial solutions (e.g., $\psi=0$ everywhere).

### 2. Neumann BC (Perfect Conductor)
$$
\nabla \psi \cdot \mathbf{n} = 0
$$
- **Physical Role:** Enforces the condition that the magnetic field must be tangential to the wall. This corresponds to a **perfectly conducting boundary** where no magnetic flux penetrates the surface.

---

## 7. Physics Residual Evaluation

At each collocation point in the domain, the physics engine computes the residual $\mathcal{R}_{\text{GS}}$, measuring the local violation of MHD equilibrium:

$$
\mathcal{R}_{\text{GS}}(R, Z)
=
\Delta^* \psi_\theta
+
\mu_0 R^2 p'(\psi)
+
F(\psi)F'(\psi)
$$

The optimization process minimizes the magnitude of this residual, driving the neural network $\psi_\theta$ toward a physically valid state.

---

## 8. Loss Function Structure

The total training loss $\mathcal{L}$ is a weighted sum of the physics residual and boundary constraints:

$$
\mathcal{L}_{\text{total}}
=
\mathcal{L}_{\text{physics}}
+
\lambda_{\text{BC}} (\mathcal{L}_{\text{Dirichlet}} + \mathcal{L}_{\text{Neumann}})
$$

Where:
- $\mathcal{L}_{\text{physics}} = \langle |\mathcal{R}_{\text{GS}}|^2 \rangle$: Enforces internal force balance.
- $\mathcal{L}_{\text{Dirichlet}}$: Enforces the flux value at the wall.
- $\mathcal{L}_{\text{Neumann}}$: Enforces the tangential field condition at the wall.

---

## 9. Conceptual Mental Model

The solving process can be visualized via a membrane analogy:
- **The Membrane:** $\psi$ acts as a flexible sheet stretched across the reactor wall (fixed at $\psi=0$).
- **The Load:** Plasma pressure acts as a distributed weight on the sheet. Due to the $R^2$ term, this weight is heavier at the outer radii.
- **The Equilibrium:** The network adjusts the curvature of the sheet until the tension ($\Delta^* \psi$) exactly counteracts the pressure load at every point, naturally resulting in an asymmetric, shifted topology.

$$
\boxed{
\text{Geometry is learned, physics is enforced, equilibrium emerges.}
}
$$
