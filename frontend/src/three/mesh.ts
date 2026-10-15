// Plasma wireframe: the backend ships only the closed poloidal boundary (R, Z);
// revolving it about the Z axis is a pure view transform done here.
import * as THREE from 'three'
import type { Boundary2D } from '../api'

/** `rings` closed poloidal cross-sections plus `circles` closed toroidal circles
 *  (each `circleRes` segments) through evenly spaced boundary points. */
export function revolvedWireframe(
  { R, Z }: Boundary2D,
  rings = 32,
  circles = 32,
  circleRes = 128,
): THREE.BufferGeometry {
  const n = R.length
  const ringStep = Math.max(1, Math.floor(n / circles))
  const nCircles = Math.ceil(n / ringStep)
  const positions = new Float32Array((rings * n + nCircles * circleRes) * 6)
  let o = 0
  const push = (r: number, phi: number, z: number) => {
    positions[o++] = r * Math.cos(phi)
    positions[o++] = r * Math.sin(phi)
    positions[o++] = z
  }
  for (let k = 0; k < rings; k++) {
    const phi = (2 * Math.PI * k) / rings
    for (let j = 0; j < n; j++) {
      const next = (j + 1) % n
      push(R[j], phi, Z[j])
      push(R[next], phi, Z[next])
    }
  }
  for (let j = 0; j < n; j += ringStep) {
    for (let i = 0; i < circleRes; i++) {
      push(R[j], (2 * Math.PI * i) / circleRes, Z[j])
      push(R[j], (2 * Math.PI * (i + 1)) / circleRes, Z[j])
    }
  }
  const geometry = new THREE.BufferGeometry()
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3))
  return geometry
}
