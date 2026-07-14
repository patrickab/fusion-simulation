import { memo, type ReactNode } from 'react'
import type { Data, Layout } from 'plotly.js'
import { Plot, baseConfig, baseLayout } from './plotly'
import type { GeometryResponse, Grid2D, GridQuantity, Sample } from './api'
import { plasmaPlotlyScale } from './three/colormap'

export type Range2 = [number, number]

/** Matches the backend default (src/engine/model_evaluation.py RESIDUAL_COLOR_RANGE) —
 *  used only until /api/config has loaded. */
export const DEFAULT_RESIDUAL_RANGE: Range2 = [0, 0.01]

/** Viridis, matching GridHeatmap's flux 'Viridis' Plotly colorscale — shared so the
 *  carpet plot and its Colorbar legend never drift apart. */
export const FLUX_COLORBAR =
  'linear-gradient(to right, #440154, #482878, #31688e, #26828e, #35b779, #6cce5a, #fde725)'

/** Backend boundary points are unordered — sort by poloidal angle around the
 *  centroid and close the loop so line traces form a clean closed curve. */
function orderBoundary(R: number[], Z: number[]): { x: number[]; y: number[] } {
  const cR = R.reduce((s, v) => s + v, 0) / R.length
  const cZ = Z.reduce((s, v) => s + v, 0) / Z.length
  // angles precomputed once — atan2 inside the comparator would run O(n log n) times
  const angle = R.map((r, i) => Math.atan2(Z[i] - cZ, r - cR))
  const idx = R.map((_, i) => i).sort((a, b) => angle[a] - angle[b])
  const x = idx.map((i) => R[i])
  const y = idx.map((i) => Z[i])
  x.push(x[0])
  y.push(y[0])
  return { x, y }
}

/** Min/max over a 2D grid in one pass — spreading 45k values into Math.min
 *  allocates the whole argument list and overflows the call stack at high res. */
function minMax2d(rows: number[][]): Range2 {
  let lo = Infinity
  let hi = -Infinity
  for (const row of rows)
    for (const v of row) {
      if (v < lo) lo = v
      if (v > hi) hi = v
    }
  return [lo, hi]
}

function layoutWith(xr?: Range2, yr?: Range2): Partial<Layout> {
  const l = baseLayout()
  if (xr) l.xaxis = { ...l.xaxis, range: xr, autorange: false }
  if (yr) l.yaxis = { ...l.yaxis, range: yr, autorange: false }
  return l
}

/** Aspect-locked container so a plot's true (R, Z) proportions render without
 *  distortion — default 1 (square) for callers with no natural data aspect. */
function PlotSquare({ children, aspect = 1 }: { children: ReactNode; aspect?: number }) {
  return <div className="plot-square" style={{ aspectRatio: aspect }}>{children}</div>
}

/** Min/max of one field over many grids — shared benchmark color scale. */
export function minMaxGridValues(grids: Grid2D[]): Range2 {
  let lo = Infinity
  let hi = -Infinity
  for (const grid of grids) {
    const [a, b] = minMax2d(grid.values)
    if (a < lo) lo = a
    if (b > hi) hi = b
  }
  if (!Number.isFinite(lo)) return [0, 1]
  if (hi - lo < 1e-12) return [lo, lo + 1e-6]
  return [lo, hi]
}

export const GridHeatmap = memo(function GridHeatmap({
  grid,
  quantity,
  zRange,
  residualRange = DEFAULT_RESIDUAL_RANGE,
}: {
  grid: Grid2D
  quantity: GridQuantity
  /** Flux only: shared scale across a montage; defaults to this grid's min/max. */
  zRange?: Range2
  /** Residual only: fixed linear scale for comparability — pass /api/config's
   *  residual_color_range so plot and colorbar agree. */
  residualRange?: Range2
}) {
  const boundary = orderBoundary(grid.boundary_R, grid.boundary_Z)
  // grid.R/Z are 2D — one physical (R, Z) point per (theta, rho) sample, all of
  // them inside this config's own boundary by construction (src/api/network.py).
  // Bounds/aspect come from that true footprint, not a shared box across samples.
  const xr = minMax2d(grid.R)
  const yr = minMax2d(grid.Z)
  const aspect = (xr[1] - xr[0]) / (yr[1] - yr[0])
  // carpet-axis showticklabels is an enum ('none'), not a boolean — false gets
  // coerced back to the default 'start' and litters the plot with labels
  const noAxisLines = { showgrid: false, showline: false, showticklabels: 'none', startline: false, endline: false }
  const { colorscale, lo, hi } =
    quantity === 'residual'
      ? { colorscale: plasmaPlotlyScale, lo: residualRange[0], hi: residualRange[1] }
      : {
          colorscale: 'Viridis',
          lo: zRange?.[0] ?? minMax2d(grid.values)[0],
          hi: zRange?.[1] ?? minMax2d(grid.values)[1],
        }
  const data: Data[] = [
    {
      type: 'carpet',
      carpet: 'g',
      a: grid.theta,
      b: grid.rho,
      x: grid.R,
      y: grid.Z,
      aaxis: noAxisLines,
      baxis: noAxisLines,
    } as unknown as Data,
    {
      type: 'contourcarpet',
      carpet: 'g',
      a: grid.theta,
      b: grid.rho,
      z: grid.values,
      colorscale,
      zmin: lo,
      zmax: hi,
      autocontour: false,
      contours: { coloring: 'fill', showlines: false, start: lo, end: hi, size: (hi - lo) / 256 },
      line: { width: 0 },
      showscale: false,
    } as unknown as Data,
    {
      type: 'scatter',
      x: boundary.x,
      y: boundary.y,
      mode: 'lines',
      line: { color: 'rgba(226,232,240,0.6)', width: 1 },
      hoverinfo: 'skip',
    },
  ]
  // ponytail: carpet traces don't reliably update through Plotly.react — key on a
  // data signature so new data remounts the component (fresh newPlot instead of diff)
  const vs = grid.values
  const mid = vs[vs.length >> 1]
  const sig = `${vs.length}x${vs[0].length}:${vs[0][0]}:${mid[mid.length >> 1]}:${xr}:${yr}:${lo}:${hi}`
  return (
    <PlotSquare aspect={aspect}>
      <Plot key={sig} data={data} layout={layoutWith(xr, yr)} config={baseConfig} style={{ width: '100%', height: '100%' }} useResizeHandler />
    </PlotSquare>
  )
})

export const SampleScatter = memo(function SampleScatter({ sample, xr, yr }: { sample: Sample; xr?: Range2; yr?: Range2 }) {
  const boundary = orderBoundary(sample.boundary_R, sample.boundary_Z)
  const data: Data[] = [
    {
      type: 'scatter',
      x: boundary.x,
      y: boundary.y,
      mode: 'lines',
      line: { color: '#2be2cf', width: 1.5 },
      hoverinfo: 'skip',
    },
    {
      type: 'scatter',
      x: sample.interior_R,
      y: sample.interior_Z,
      mode: 'markers',
      // ponytail: semi-transparent overlapping dots read as a blurry haze; solid + smaller is crisper
      marker: { color: '#d4d4d8', size: 3, opacity: 1 },
    },
  ]
  return (
    <PlotSquare>
      <Plot data={data} layout={layoutWith(xr, yr)} config={baseConfig} style={{ width: '100%', height: '100%' }} useResizeHandler />
    </PlotSquare>
  )
})

export const CrossSection = memo(function CrossSection({ geo, height = 280 }: { geo: GeometryResponse; height?: number }) {
  const data: Data[] = [
    {
      type: 'scatter',
      x: geo.boundary2d.R,
      y: geo.boundary2d.Z,
      mode: 'lines',
      line: { color: '#2be2cf', width: 1.5 },
      hoverinfo: 'skip',
    },
  ]
  return (
    <Plot data={data} layout={baseLayout()} config={baseConfig} style={{ width: '100%', height }} useResizeHandler />
  )
})
