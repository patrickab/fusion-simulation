// Field lines arrive fully traced from the backend (VTK RK45 through the 48³
// B-grid — the same PyVista path as the legacy Streamlit render). The client's
// only job is assembling one merged LineSegments geometry: thin 1px strands,
// a single draw call, |B| vertex colors over the full range (`b_range`,
// matching VTK's default full-scalar-range normalization).
import * as THREE from 'three'
import type { FieldLinesResponse } from '../api'
import { plasma } from './colormap'

export function buildFieldLineSegments(field: FieldLinesResponse): THREE.BufferGeometry {
  const { points, speeds, line_lengths } = field
  const [lo, hi] = field.b_range
  const span = hi - lo || 1

  let nSegments = 0
  for (const n of line_lengths) nSegments += Math.max(0, n - 1)
  const positions = new Float32Array(nSegments * 6)
  const colors = new Float32Array(nSegments * 6)

  let o = 0
  let start = 0
  for (const n of line_lengths) {
    for (let i = start; i < start + n - 1; i++) {
      for (const j of [i, i + 1]) {
        positions[o] = points[3 * j]
        positions[o + 1] = points[3 * j + 1]
        positions[o + 2] = points[3 * j + 2]
        const [r, g, b] = plasma((speeds[j] - lo) / span)
        colors[o] = r
        colors[o + 1] = g
        colors[o + 2] = b
        o += 3
      }
    }
    start += n
  }

  const geometry = new THREE.BufferGeometry()
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3))
  geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3))
  return geometry
}
