// Field lines arrive fully traced from the backend (RK4 through the axisymmetric
// (R, Z) B-grid) as base64 float32 buffers. The client only decodes them into
// one indexed LineSegments geometry: shared vertices, a single draw call, |B|
// vertex colors over the full range (`b_range`).
import * as THREE from 'three'
import type { FieldLinesResponse } from '../api'
import { plasma } from './colormap'

function decodeFloat32(base64: string): Float32Array {
  const binary = atob(base64)
  const bytes = new Uint8Array(binary.length)
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i)
  return new Float32Array(bytes.buffer)
}

export function buildFieldLineSegments(field: FieldLinesResponse): THREE.BufferGeometry {
  const positions = decodeFloat32(field.points)
  const speeds = decodeFloat32(field.speeds)
  const [lo, hi] = field.b_range
  const span = hi - lo || 1

  const colors = new Float32Array(speeds.length * 3)
  for (let i = 0; i < speeds.length; i++) colors.set(plasma((speeds[i] - lo) / span), 3 * i)

  let nSegments = 0
  for (const n of field.line_lengths) nSegments += Math.max(0, n - 1)
  const index = new Uint32Array(nSegments * 2)
  let o = 0
  let start = 0
  for (const n of field.line_lengths) {
    for (let i = start; i < start + n - 1; i++) {
      index[o++] = i
      index[o++] = i + 1
    }
    start += n
  }

  const geometry = new THREE.BufferGeometry()
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3))
  geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3))
  geometry.setIndex(new THREE.BufferAttribute(index, 1))
  return geometry
}
