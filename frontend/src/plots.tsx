import type { ReactNode } from 'react'
import type { Data, Layout } from 'plotly.js'
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

/** Square container so a grid column's width and the plot's height always match. */
function PlotSquare({ children }: { children: ReactNode }) {
  return <div className="plot-square">{children}</div>
}

export function GridHeatmap({
  grid,
  colorscale = 'Viridis',
  reversescale = false,
  zmin,
  zmax,
  xr,
  yr,
}: {
  grid: Grid2D
  colorscale?: string
  reversescale?: boolean
  zmin?: number
  zmax?: number
  xr?: Range2
  yr?: Range2
}) {
  const boundary = orderBoundary(grid.boundary_R, grid.boundary_Z)
  const data: Data[] = [
    {
      type: 'heatmap',
      x: grid.R,
      y: grid.Z,
      z: grid.values as unknown as number[][],
      colorscale,
      reversescale,
      zmin,
      zmax,
      showscale: false,
      hoverongaps: false,
    },
    {
      type: 'scatter',
      x: boundary.x,
      y: boundary.y,
      mode: 'lines',
      line: { color: 'rgba(226,232,240,0.6)', width: 1 },
      hoverinfo: 'skip',
    },
  ]
  return (
    <PlotSquare>
      <Plot data={data} layout={layoutWith(xr, yr)} config={baseConfig} style={{ width: '100%', height: '100%' }} useResizeHandler />
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
