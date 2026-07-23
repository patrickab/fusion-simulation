// Structured-grid wireframe: backend sends (n_phi, n_theta) grids flattened row-major.
import * as THREE from 'three'

/** Line segments along every `stride`-th grid row/column. */
export function sparseWireframe(
  X: number[],
  Y: number[],
  Z: number[],
  nPhi: number,
  nTheta: number,
  stride = 2,
): THREE.BufferGeometry {
  const pts: number[] = []
  const push = (i: number, j: number) => {
    const k = i * nTheta + j
    pts.push(X[k], Y[k], Z[k])
  }
  for (let i = 0; i < nPhi; i += stride) {
    for (let j = 0; j < nTheta - 1; j++) {
      push(i, j)
      push(i, j + 1)
    }
    push(i, nTheta - 1)
    push(i, 0)
  }
  for (let j = 0; j < nTheta; j += stride) {
    for (let i = 0; i < nPhi - 1; i++) {
      push(i, j)
      push(i + 1, j)
    }
    push(nPhi - 1, j)
    push(0, j)
  }
  const geometry = new THREE.BufferGeometry()
  geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(pts), 3))
  return geometry
}
