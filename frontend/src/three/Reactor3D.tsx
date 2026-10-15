import { useEffect, useMemo } from 'react'
import * as THREE from 'three'
import type { Boundary2D, FieldLinesResponse } from '../api'
import { buildFieldLineSegments } from './fieldlines'
import { revolvedWireframe } from './mesh'

function useDisposable<T extends THREE.BufferGeometry[]>(geometries: T): T {
  useEffect(() => () => geometries.forEach((g) => g.dispose()), [geometries])
  return geometries
}

// WebGL lineWidth is capped at 1px — "thinner" is done via opacity, not width
export function PlasmaWireframe({
  boundary,
  color = '#2be2cf',
  opacity = 0.45,
}: {
  boundary: Boundary2D
  color?: string
  opacity?: number
}) {
  const [wire] = useDisposable(useMemo(() => [revolvedWireframe(boundary)], [boundary]))
  // ponytail: opacity baked into a dimmed solid color instead of alpha
  // blending. Transparent lines stack brightness wherever they overlap on
  // screen (worse the more strands bunch up, e.g. at grazing view angles —
  // tried capping it with max-blending, but that turned out to brighten
  // everything uniformly instead). An opaque line has nothing to stack: same
  // color from every angle, no matter how many strands overlap.
  const dimmed = useMemo(() => new THREE.Color(color).multiplyScalar(opacity), [color, opacity])
  return (
    <lineSegments geometry={wire}>
      <lineBasicMaterial color={dimmed} />
    </lineSegments>
  )
}

// Thin unlit lines, |B| vertex colors (matches the legacy PyVista render).
// depthWrite stays on (the default) so front strands occlude the ones behind
// them like VTK's tubes do — with it off, strands at a bundled viewing angle
// stack their alpha and flare bright instead of layering.
export function FieldLines({ field }: { field: FieldLinesResponse }) {
  const [geometry] = useDisposable(useMemo(() => [buildFieldLineSegments(field)], [field]))
  return (
    <lineSegments geometry={geometry}>
      <lineBasicMaterial vertexColors transparent opacity={0.6} />
    </lineSegments>
  )
}
