// matplotlib "plasma" colormap, linearly interpolated between anchor stops.
// Perceptually uniform and CVD-safe; brighter and less black than inferno.
const STOPS: [number, [number, number, number]][] = [
  [0.0, [13, 8, 135]],
  [0.125, [84, 2, 163]],
  [0.25, [139, 10, 165]],
  [0.375, [185, 50, 137]],
  [0.5, [219, 92, 104]],
  [0.625, [244, 136, 73]],
  [0.75, [254, 188, 43]],
  [0.875, [247, 233, 33]],
  [1.0, [240, 249, 33]],
]

// ponytail: only 9 anchor points, so a straight RGB line between two
// differently-hued stops (e.g. purple -> orange) dips through a duller,
// grayer color partway — matplotlib's real curve avoids this with hundreds
// of points. Pushing each result away from its own gray average restores
// the saturation without needing a much bigger anchor table.
const SATURATION_BOOST = 1.3

function saturate([r, g, b]: [number, number, number]): [number, number, number] {
  const avg = (r + g + b) / 3
  const clamp = (v: number) => Math.min(1, Math.max(0, v))
  return [clamp(avg + (r - avg) * SATURATION_BOOST), clamp(avg + (g - avg) * SATURATION_BOOST), clamp(avg + (b - avg) * SATURATION_BOOST)]
}

export function plasma(t: number): [number, number, number] {
  const x = Math.min(1, Math.max(0, t))
  for (let i = 1; i < STOPS.length; i++) {
    const [t1, c1] = STOPS[i]
    if (x <= t1) {
      const [t0, c0] = STOPS[i - 1]
      const f = (x - t0) / (t1 - t0)
      return saturate([
        (c0[0] + f * (c1[0] - c0[0])) / 255,
        (c0[1] + f * (c1[1] - c0[1])) / 255,
        (c0[2] + f * (c1[2] - c0[2])) / 255,
      ])
    }
  }
  return saturate([240 / 255, 249 / 255, 33 / 255])
}

// CSS gradient built from the same stops, for the colorbar legend.
export const plasmaGradient = `linear-gradient(90deg, ${STOPS.map(
  ([t, [r, g, b]]) => `rgb(${r},${g},${b}) ${t * 100}%`,
).join(', ')})`

// Same stops as a Plotly `colorscale` array, for heatmaps (e.g. log|R_GS|) that
// want the same dark->bright sequential map as the 3D field-line legend.
export const plasmaPlotlyScale: [number, string][] = STOPS.map(([t, [r, g, b]]) => [
  t,
  `rgb(${r},${g},${b})`,
])
