# Nuclear Fusion Simulation

## Table of Contents
- [Poloidal Plasma Boundary Definition](#poloidal-plasma-boundary-definition)
- [Toroidal Coil Cross-Section Construction](#toroidal-coil-cross-section-construction)
- [3D Plasma Surface Generation](#3d-plasma-surface-generation)
- [3D Toroidal Coil Generation](#3d-toroidal-coil-generation)

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

## Toroidal Coil Cross-Section Construction

Toroidal coils are represented as 2D cross-sections offset from the plasma boundary by a set distance and have a defined radial thickness.
Essentially, points on the poloidal plasma plane are projected outwards to form the toroidal field coils.

1. Compute the tangent vector at each poloidal boundary point using the gradient:
   - $dR/d\theta$, $dZ/d\theta$
2. Calculate outward poloidal normal vectors by rotating tangents by 90 degrees
3. Offset the plasma boundary by desired distance ($d$) along the normal to find the coil inner boundary.
4. Add radial thickness along normals to define coil outer boundary and center.

- Coil cross-section frames the plasma boundary with a controllable spatial clearance and thickness
- In future the shape of the field coil shall be a matter of optimization, therefore geometric constrains may change

---

## 3D Toroidal Coil Generation

Toroidal coils are extruded into 3D by sweeping the 2D cross-section around the toroidal angle with specified coil count and angular spans.

- The coil circumference is divided into $n$ field coils spaced evenly.
- For each coil, an angular segment centered at $\phi_i$ is generated.
- The 2D coil cross-sections are rotated using $X = r \cos \phi, \quad Y = r \sin \phi, \quad Z = z$
- Surfaces for inner and outer coil boundaries are calculated.
- Coil caps (start and end) are constructed to close the geometry.

