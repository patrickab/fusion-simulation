// ponytail: custom bundle with exactly the traces we render — the prebuilt
// cartesian dist lacks carpet/contourcarpet, and the full dist is ~4 MB heavier.
import Plotly from 'plotly.js/lib/core'
import scatter from 'plotly.js/lib/scatter'
import carpet from 'plotly.js/lib/carpet'
import contourcarpet from 'plotly.js/lib/contourcarpet'
import factory from 'react-plotly.js/factory'
import type { Config, Layout } from 'plotly.js'

Plotly.register([scatter, carpet, contourcarpet] as Parameters<typeof Plotly.register>[0])

// CJS default-interop: under Vite dev the module object itself can land in `factory`
const createPlotlyComponent = (factory as unknown as { default?: typeof factory }).default ?? factory

export const Plot = createPlotlyComponent(Plotly)

export const baseLayout = (overrides: Partial<Layout> = {}): Partial<Layout> => ({
  paper_bgcolor: 'rgba(0,0,0,0)',
  plot_bgcolor: 'rgba(0,0,0,0)',
  font: { family: 'ui-monospace, SF Mono, Menlo, monospace', size: 10, color: '#8a8a91' },
  margin: { l: 42, r: 8, t: 8, b: 30 },
  xaxis: { gridcolor: 'rgba(255,255,255,0.05)', zeroline: false },
  yaxis: { gridcolor: 'rgba(255,255,255,0.05)', zeroline: false, scaleanchor: 'x' },
  showlegend: false,
  ...overrides,
})

export const baseConfig: Partial<Config> = { displayModeBar: false, responsive: true }
