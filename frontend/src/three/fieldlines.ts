// Client-side field-line tracing: adaptive RK45 (Fehlberg) streamline
// integration through the backend's 30³ B-vector grid (trilinear
// interpolation), colored by |B|. Matches PyVista's integrator_type=45 —
// a fixed-step integrator has no error control, and over the ~25 toroidal
// transits needed for a dense render, its accumulated drift visibly pulls
// field lines off their true flux surface (collapses them toward the axis
// into an hourglass shape instead of PyVista's evenly nested surfaces).
import * as THREE from 'three'
import type { BFieldResponse } from '../api'
import { plasma } from './colormap'

interface Traced {
  points: THREE.Vector3[]
  speeds: number[]
}

// PyVista traces each streamline up to max_length=1000 (arc length) or
// max_steps=2000, whichever comes first — long enough to wind ~20+ times
// around the torus, which is what makes the reference render look like a
// dense woven surface rather than a handful of loose loops.
const ARC_LENGTH_BUDGET = 500 // per direction, matching PyVista's scale

function traceLines(field: BFieldResponse): Traced[] {
  const { nx, ny, nz, origin, spacing } = field.grid
  const v = field.vectors

  const sample = (p: THREE.Vector3): [number, number, number] | null => {
    const fx = (p.x - origin[0]) / spacing[0]
    const fy = (p.y - origin[1]) / spacing[1]
    const fz = (p.z - origin[2]) / spacing[2]
    if (fx < 0 || fy < 0 || fz < 0 || fx > nx - 1 || fy > ny - 1 || fz > nz - 1) return null
    const ix = Math.min(Math.floor(fx), nx - 2)
    const iy = Math.min(Math.floor(fy), ny - 2)
    const iz = Math.min(Math.floor(fz), nz - 2)
    const tx = fx - ix
    const ty = fy - iy
    const tz = fz - iz
    const idx = (i: number, j: number, k: number) => 3 * ((i * ny + j) * nz + k)
    const out: [number, number, number] = [0, 0, 0]
    for (let c = 0; c < 3; c++) {
      const c000 = v[idx(ix, iy, iz) + c]
      const c100 = v[idx(ix + 1, iy, iz) + c]
      const c010 = v[idx(ix, iy + 1, iz) + c]
      const c110 = v[idx(ix + 1, iy + 1, iz) + c]
      const c001 = v[idx(ix, iy, iz + 1) + c]
      const c101 = v[idx(ix + 1, iy, iz + 1) + c]
      const c011 = v[idx(ix, iy + 1, iz + 1) + c]
      const c111 = v[idx(ix + 1, iy + 1, iz + 1) + c]
      const c00 = c000 + (c100 - c000) * tx
      const c10 = c010 + (c110 - c010) * tx
      const c01 = c001 + (c101 - c001) * tx
      const c11 = c011 + (c111 - c011) * tx
      const c0 = c00 + (c10 - c00) * ty
      const c1 = c01 + (c11 - c01) * ty
      out[c] = c0 + (c1 - c0) * tz
    }
    return out
  }

  // unit direction of B (streamline parametrized by arc length)
  const dir = (p: THREE.Vector3): { d: THREE.Vector3; mag: number } | null => {
    const b = sample(p)
    if (!b) return null
    const mag = Math.hypot(b[0], b[1], b[2])
    if (mag < 1e-9) return null
    return { d: new THREE.Vector3(b[0] / mag, b[1] / mag, b[2] / mag), mag }
  }

  const avgSpacing = (spacing[0] + spacing[1] + spacing[2]) / 3
  const hInit = 0.4 * avgSpacing
  const hMin = 1e-3 * avgSpacing
  const hMax = 2 * avgSpacing
  const tol = 1e-4 * avgSpacing // local error tolerance, in world units

  // Fehlberg RK45: two embedded solutions (4th & 5th order) from the same 6
  // stages — their difference estimates local error, which drives step size
  // up in smooth stretches and down through high-curvature ones.
  const rkf45Step = (p: THREE.Vector3, sign: 1 | -1, h: number) => {
    const sh = sign * h
    const k1 = dir(p)
    if (!k1) return null
    const k2 = dir(p.clone().addScaledVector(k1.d, sh * (1 / 4)))
    if (!k2) return null
    const k3 = dir(p.clone().addScaledVector(k1.d, sh * (3 / 32)).addScaledVector(k2.d, sh * (9 / 32)))
    if (!k3) return null
    const k4 = dir(
      p
        .clone()
        .addScaledVector(k1.d, sh * (1932 / 2197))
        .addScaledVector(k2.d, sh * (-7200 / 2197))
        .addScaledVector(k3.d, sh * (7296 / 2197)),
    )
    if (!k4) return null
    const k5 = dir(
      p
        .clone()
        .addScaledVector(k1.d, sh * (439 / 216))
        .addScaledVector(k2.d, sh * -8)
        .addScaledVector(k3.d, sh * (3680 / 513))
        .addScaledVector(k4.d, sh * (-845 / 4104)),
    )
    if (!k5) return null
    const k6 = dir(
      p
        .clone()
        .addScaledVector(k1.d, sh * (-8 / 27))
        .addScaledVector(k2.d, sh * 2)
        .addScaledVector(k3.d, sh * (-3544 / 2565))
        .addScaledVector(k4.d, sh * (1859 / 4104))
        .addScaledVector(k5.d, sh * (-11 / 40)),
    )
    if (!k6) return null

    const y4 = p
      .clone()
      .addScaledVector(k1.d, sh * (25 / 216))
      .addScaledVector(k3.d, sh * (1408 / 2565))
      .addScaledVector(k4.d, sh * (2197 / 4104))
      .addScaledVector(k5.d, sh * (-1 / 5))
    const y5 = p
      .clone()
      .addScaledVector(k1.d, sh * (16 / 135))
      .addScaledVector(k3.d, sh * (6656 / 12825))
      .addScaledVector(k4.d, sh * (28561 / 56430))
      .addScaledVector(k5.d, sh * (-9 / 50))
      .addScaledVector(k6.d, sh * (2 / 55))

    return { next: y5, err: y5.distanceTo(y4), speed: k1.mag }
  }

  const integrate = (seed: THREE.Vector3, sign: 1 | -1): Traced => {
    const points: THREE.Vector3[] = []
    const speeds: number[] = []
    let p = seed.clone()
    let h = hInit
    let length = 0
    let iterations = 0
    while (length < ARC_LENGTH_BUDGET && iterations++ < 20000) {
      const step = rkf45Step(p, sign, h)
      if (!step) break
      if (step.err > tol && h > hMin) {
        h = Math.max(hMin, h / 2)
        continue
      }
      p = step.next
      length += h
      points.push(p.clone())
      speeds.push(step.speed)
      const factor = step.err > 0 ? 0.9 * (tol / step.err) ** 0.2 : 2
      h = Math.min(hMax, Math.max(hMin, h * Math.min(2, Math.max(0.5, factor))))
    }
    return { points, speeds }
  }

  return field.seed_points
    .map((s) => {
      const seed = new THREE.Vector3(s[0], s[1], s[2])
      const back = integrate(seed, -1)
      const fwd = integrate(seed, 1)
      return {
        points: [...back.points.reverse(), seed, ...fwd.points],
        speeds: [...back.speeds.reverse(), fwd.speeds[0] ?? back.speeds[0] ?? 0, ...fwd.speeds],
      }
    })
    .filter((l) => l.points.length > 8)
}

// All lines merged into one LineSegments geometry: thin 1px strands like the
// PyVista/VTK reference render, and a single draw call. `range` is the full
// |B| span (Tesla), matching VTK's default full-scalar-range normalization.
export function buildFieldLineSegments(field: BFieldResponse): {
  geometry: THREE.BufferGeometry
  range: [number, number]
} {
  const lines = traceLines(field)

  let lo = Infinity
  let hi = -Infinity
  for (const l of lines)
    for (const s of l.speeds) {
      if (s < lo) lo = s
      if (s > hi) hi = s
    }
  if (!isFinite(lo)) [lo, hi] = [0, 1]
  const span = hi - lo || 1

  const nSegments = lines.reduce((n, l) => n + l.points.length - 1, 0)
  const positions = new Float32Array(nSegments * 6)
  const colors = new Float32Array(nSegments * 6)
  let o = 0
  for (const line of lines) {
    for (let i = 0; i < line.points.length - 1; i++) {
      for (const [p, s] of [
        [line.points[i], line.speeds[i]],
        [line.points[i + 1], line.speeds[i + 1]],
      ] as const) {
        positions[o] = p.x
        positions[o + 1] = p.y
        positions[o + 2] = p.z
        const [r, g, b] = plasma((s - lo) / span)
        colors[o] = r
        colors[o + 1] = g
        colors[o + 2] = b
        o += 3
      }
    }
  }

  const geometry = new THREE.BufferGeometry()
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3))
  geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3))
  return { geometry, range: [lo, hi] }
}
