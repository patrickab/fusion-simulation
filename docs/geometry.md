## Toroidal Geometry & Mesh Generation

The simulation represents the fusion plasma as a 3D volume generated via the rotational symmetry of a 2D poloidal cross-section.
This results

### 1. Poloidal Boundary Definition
The plasma boundary is defined in the $(R, Z)$ plane using a parametric representation that accounts for elongation ($\kappa$) and triangularity ($\delta$):

$$ R(\theta) = R_0 + a \cos(\theta + \delta \sin \theta) $$
$$ Z(\theta) = Z_0 + a \kappa \sin \theta $$

Where:
*   $R_0, Z_0$: Magnetic axis major radius and vertical offset.
*   $a$: Minor radius.
*   $\theta$: Poloidal angle $\in [0, 2\pi]$.

## 2. Rotational Revolution
To transition from a 2D slice to a 3D volume, the poloidal contour is revolved around the vertical $Z$-axis. Using a toroidal angle $\phi \in [0, 2\pi]$ a parametric surface mesh is constructed from the 2D plane.

### Coordinate Transformation
The transformation from the local cylindrical coordinates $(R, \phi, Z)$ to global Cartesian coordinates $(X, Y, Z)$ is implemented as follows:

| Coordinate | Transformation | Description |
| :--- | :--- | :--- |
| **X** | $R(\theta) \cdot \cos(\phi)$ | Radial projection onto the X-axis |
| **Y** | $R(\theta) \cdot \sin(\phi)$ | Radial projection onto the Y-axis |
| **Z** | $Z(\theta)$ | Vertical position (invariant under rotation) |

## 3. Numerical Implementation
The generation process is optimized using vectorized operations (via JAX/NumPy) to ensure the mesh is compatible with downstream physics solvers:

1.  **Grid Generation**: A meshgrid of $(\theta, \phi)$ is created, resulting in two-dimensional arrays that represent every discrete point on the toroidal surface.
2.  **Broadcasting**: The 1D poloidal vectors are broadcasted across the toroidal dimension, effectively "sweeping" the poloidal plane around the torus.
3.  **Memory Layout**: The resulting arrays $X, Y, Z$ are shaped as $(N_\phi, N_\theta)$, where each row represents a toroidal "rib" and each column represents a poloidal "ring."
