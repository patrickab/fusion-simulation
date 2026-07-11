import type { ReactNode } from 'react'
import type { ColorScale, Data, Layout } from 'plotly.js'
import { Plot, baseConfig, baseLayout } from './plotly'
import type { GeometryResponse, Grid2D, Sample } from './api'

export type Range2 = [number, number]

/** Backend boundary points are unordered — sort by poloidal angle around the
 *  centroid and close the loop so line traces form a clean closed curve. */
function orderBoundary(R: number[], Z: number[]): { x: number[]; y: number[] } {
  const cR = R.reduce((s, v) => s + v, 0) / R.length
  const cZ = Z.reduce((s, v) => s + v, 0) / Z.length
  const idx = R.map((_, i) => i).sort(
    (a, b) => Math.atan2(Z[a] - cZ, R[a] - cR) - Math.atan2(Z[b] - cZ, R[b] - cR),
  )
  const x = idx.map((i) => R[i])
  const y = idx.map((i) => Z[i])
  x.push(x[0])
  y.push(y[0])
  return { x, y }
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

export function GridHeatmap({
  grid,
  colorscale = 'Viridis',
  reversescale = false,
  zmin,
  zmax,
}: {
  grid: Grid2D
  colorscale?: ColorScale
  reversescale?: boolean
  zmin?: number
  zmax?: number
}) {
  const boundary = orderBoundary(grid.boundary_R, grid.boundary_Z)
  // grid.R/Z are 2D — one physical (R, Z) point per (theta, rho) sample, all of
  // them inside this config's own boundary by construction (src/api/network.py).
  // Bounds/aspect come from that true footprint, not a shared box across samples.
  const flatR = grid.R.flat()
  const flatZ = grid.Z.flat()
  const xr: Range2 = [Math.min(...flatR), Math.max(...flatR)]
  const yr: Range2 = [Math.min(...flatZ), Math.max(...flatZ)]
  const aspect = (xr[1] - xr[0]) / (yr[1] - yr[0])
  // carpet-axis showticklabels is an enum ('none'), not a boolean — false gets
  // coerced back to the default 'start' and litters the plot with labels
  const noAxisLines = { showgrid: false, showline: false, showticklabels: 'none', startline: false, endline: false }
  const flatV = grid.values.flat()
  const lo = zmin ?? Math.min(...flatV)
  const hi = zmax ?? Math.max(...flatV)
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
      reversescale,
      zmin: lo,
      zmax: hi,
      // contourcarpet has no 'heatmap' coloring (unlike contour) — 48 fill
      // levels reads as continuous, matching matplotlib contourf
      autocontour: false,
      contours: { coloring: 'fill', showlines: false, start: lo, end: hi, size: (hi - lo) / 48 },
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
  const sig = `${flatV.length}:${flatV[0]}:${flatV[flatV.length >> 1]}:${xr}:${yr}:${lo}:${hi}`
  return (
    <PlotSquare aspect={aspect}>
      <Plot key={sig} data={data} layout={layoutWith(xr, yr)} config={baseConfig} style={{ width: '100%', height: '100%' }} useResizeHandler />
    </PlotSquare>
  )
}

export function SampleScatter({ sample, xr, yr }: { sample: Sample; xr?: Range2; yr?: Range2 }) {
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
}

export function CrossSection({ geo, height = 280 }: { geo: GeometryResponse; height?: number }) {
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
}
