import { useEffect, useState } from 'react'
import {
  api,
  invalidate,
  useApi,
  useDebounced,
  type BFieldResponse,
  type GeometryResponse,
  type Grid2D,
  type SampleResponse,
} from '../api'
import { useStore } from '../store'
import { GridHeatmap, SampleScatter, type Range2 } from '../plots'
import { Scene } from '../three/Scene'
import { FieldLines, PlasmaWireframe } from '../three/Reactor3D'
import { plasmaGradient } from '../three/colormap'
import { Colorbar, JsonBlock, Panel, Popover, Section, Segmented, Slider, Spinner, Stat } from '../ui'

const VIEW_MODES = ['New Benchmarks', 'Archive', 'All'] as const
const MODE_LABELS = { 'New Benchmarks': 'New', Archive: 'Archive', All: 'All' } as const
const TABS = ['sampling', 'flux', 'residual', 'topology'] as const
type Tab = (typeof TABS)[number]
const TAB_LABELS: Record<Tab, string> = {
  sampling: 'Sampling',
  flux: 'Flux ψ',
  residual: 'GS residual',
  topology: '3D topology',
}

const fmtExp = (v: number) => v.toExponential(2)
const short = (name: string) => name.replace(/\.flax$/, '')

export function NetworkView() {
  const [viewMode, setViewMode] = useState<(typeof VIEW_MODES)[number]>('New Benchmarks')
  const [bump, setBump] = useState(0)
  const network = useStore((s) => s.network)
  const setNetwork = useStore((s) => s.setNetwork)
  const [tab, setTab] = useState<Tab>('sampling')
  const [seed, setSeed] = useState(0)
  const [sampleSize, setSampleSize] = useState(4)
  const [nLines, setNLines] = useState(50)

  const networks = useApi<string[]>(`networks:${viewMode}:${bump}`, () => api.networks(viewMode))

  useEffect(() => {
    const list = networks.data
    if (list && (!network || !list.includes(network))) setNetwork(list[0] ?? null)
  }, [networks.data, network, setNetwork])

  const dSeed = useDebounced(seed, 400)
  const dLines = useDebounced(nLines, 400)
  const sampleKey = network ? `sample:${network}:${dSeed}:${sampleSize}` : null
  const sample = useApi<SampleResponse>(sampleKey, () => api.sample(network!, dSeed, sampleSize))
  const config = useApi<Record<string, unknown>>(network ? `config:${network}` : null, () => api.config(network!))

  const refresh = () => {
    invalidate()
    setBump((b) => b + 1)
  }
  const doArchive = async () => {
    if (!network) return
    await api.archive(network)
    setNetwork(null)
    refresh()
  }
  const doRename = async () => {
    if (!network) return
    // ponytail: native prompt; inline editor if it ever grates
    const newName = window.prompt('New name (without extension):', short(network))
    if (!newName) return
    const r = await api.rename(network, newName)
    setNetwork(r.name)
    refresh()
  }
  const doDelete = async () => {
    if (!network || !window.confirm(`Delete ${short(network)}?`)) return
    await api.remove(network)
    setNetwork(null)
    refresh()
  }

  return (
    <div className="view">
      <Panel>
        <Section title="Network">
          <div className="ctl">
            <Segmented options={VIEW_MODES} value={viewMode} onChange={setViewMode} labels={MODE_LABELS} small />
          </div>
          <div className="row">
            <select className="select" value={network ?? ''} onChange={(e) => setNetwork(e.target.value)}>
              {(networks.data ?? []).map((n) => (
                <option key={n} value={n}>
                  {short(n)}
                </option>
              ))}
            </select>
            <Popover label="⋯" buttonClassName="pop-btn icon" align="right">
              {(close) => (
                <>
                  <button onClick={() => { close(); void doArchive() }}>Archive</button>
                  <button onClick={() => { close(); void doRename() }}>Rename…</button>
                  <button className="danger" onClick={() => { close(); void doDelete() }}>
                    Delete
                  </button>
                </>
              )}
            </Popover>
          </div>
        </Section>
        {config.data && (
          <Section title="Training config">
            <JsonBlock obj={config.data} />
          </Section>
        )}
        <Section title="Sampling">
          <Slider label="Seed" value={seed} min={0} max={1000} onChange={setSeed} />
          <label className="ctl">
            <span className="ctl-row">
              <span>Sample size</span>
            </span>
            <select className="select" value={sampleSize} onChange={(e) => setSampleSize(Number(e.target.value))}>
              {[1, 2, 4, 6, 8].map((n) => (
                <option key={n} value={n}>
                  {n}
                </option>
              ))}
            </select>
          </label>
          {tab === 'topology' && <Slider label="Field lines" value={nLines} min={1} max={50} onChange={setNLines} />}
        </Section>
      </Panel>
      <div className="view-main">
        <div className="tabs">
          {TABS.map((t) => (
            <button key={t} className={t === tab ? 'on' : ''} onClick={() => setTab(t)}>
              {TAB_LABELS[t]}
            </button>
          ))}
        </div>
        {!network ? (
          <div className="empty">No networks in this view.</div>
        ) : (
          <>
            {tab === 'sampling' && <SamplingTab sample={sample} />}
            {tab === 'flux' && <GridTab kind="flux" network={network} seed={dSeed} sampleSize={sampleSize} sample={sample} />}
            {tab === 'residual' && (
              <GridTab kind="residual" network={network} seed={dSeed} sampleSize={sampleSize} sample={sample} />
            )}
            {tab === 'topology' && (
              <TopologyTab network={network} seed={dSeed} sampleSize={sampleSize} nLines={dLines} sample={sample} />
            )}
          </>
        )}
      </div>
    </div>
  )
}

function SamplingTab({ sample }: { sample: ReturnType<typeof useApi<SampleResponse>> }) {
  let xr: Range2 | undefined
  let yr: Range2 | undefined
  if (sample.data) {
    let rMin = Infinity
    let rMax = -Infinity
    let zMin = Infinity
    let zMax = -Infinity
    for (const s of sample.data.samples) {
      for (const v of s.boundary_R) {
        if (v < rMin) rMin = v
        if (v > rMax) rMax = v
      }
      for (const v of s.boundary_Z) {
        if (v < zMin) zMin = v
        if (v > zMax) zMax = v
      }
    }
    const pad = 0.06 * Math.max(rMax - rMin, zMax - zMin)
    xr = [rMin - pad, rMax + pad]
    yr = [zMin - pad, zMax + pad]
  }
  return (
    <div className="scroll" style={{ position: 'relative' }}>
      {sample.loading && <Spinner />}
      {sample.error && <div className="error">{sample.error}</div>}
      {sample.data && (
        <>
          <div className="grid">
            {sample.data.samples.map((s, i) => (
              <div className="card" key={i}>
                <div className="caption">
                  R₀ {s.R0.toFixed(2)} · a {s.a.toFixed(2)} · κ {s.kappa.toFixed(2)} · δ {s.delta.toFixed(2)}
                </div>
                <SampleScatter sample={s} xr={xr} yr={yr} />
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  )
}

function KpiStats({ sample }: { sample: ReturnType<typeof useApi<SampleResponse>> }) {
  if (!sample.data) return null
  const m = sample.data.metrics
  return (
    <div className="stats">
      <Stat label="avg loss" value={fmtExp(m.avg_loss)} />
      <Stat label="interior" value={fmtExp(m.interior_loss)} />
      <Stat label="boundary" value={fmtExp(m.boundary_loss)} />
      <Stat label="max loss" value={fmtExp(m.max_loss)} />
      <Stat label="max residual" value={fmtExp(m.max_residual)} />
    </div>
  )
}

function GridTab({
  kind,
  network,
  seed,
  sampleSize,
  sample,
}: {
  kind: 'flux' | 'residual'
  network: string
  seed: number
  sampleSize: number
  sample: ReturnType<typeof useApi<SampleResponse>>
}) {
  const resolution = 100
  const grids = useApi<Grid2D[]>(`${kind}:${network}:${seed}:${sampleSize}:${resolution}`, () =>
    api[kind](network, seed, sampleSize, resolution),
  )

  // ponytail: fixed scales matched to the legacy plot_flux_heatmap/plot_gs_residual_heatmap
  // (src/lib/visualization.py) — same absolute zmin/zmax across samples, not per-batch min/max
  const { zmin, zmax } = kind === 'residual' ? { zmin: -0.5, zmax: 0.5 } : { zmin: 0, zmax: 90 }

  let xr: Range2 | undefined
  let yr: Range2 | undefined
  if (grids.data?.length) {
    const { R, Z } = grids.data[0] // all grids share one linspace
    xr = [R[0], R[R.length - 1]]
    yr = [Z[0], Z[Z.length - 1]]
  }

  return (
    <div className="scroll" style={{ position: 'relative' }}>
      {grids.loading && <Spinner />}
      {grids.error && <div className="error">{grids.error}</div>}
      <KpiStats sample={sample} />
      <div className="grid">
        {(grids.data ?? []).map((g, i) => (
          <div className="card" key={i}>
            <div className="caption">sample {i}</div>
            <GridHeatmap
              grid={g}
              colorscale={kind === 'residual' ? 'RdBu' : 'Viridis'}
              reversescale={kind === 'residual'}
              zmin={zmin}
              zmax={zmax}
              xr={xr}
              yr={yr}
            />
          </div>
        ))}
      </div>
    </div>
  )
}

function TopologyTab({
  network,
  seed,
  sampleSize,
  nLines,
  sample,
}: {
  network: string
  seed: number
  sampleSize: number
  nLines: number
  sample: ReturnType<typeof useApi<SampleResponse>>
}) {
  const field = useApi<BFieldResponse>(`bfield:${network}:${seed}:${sampleSize}:${nLines}`, () =>
    api.bfield(network, seed, sampleSize, nLines),
  )
  const geom = sample.data?.geom3d
  const geo = useApi<GeometryResponse>(geom ? `geo3d:${JSON.stringify(geom)}` : null, () =>
    api.geometry({ ...geom!, show_coils: false, mesh_stride: 2 }),
  )

  const [bRange, setBRange] = useState<[number, number]>([0, 1])
  const loading = field.loading || geo.loading || sample.loading
  const error = field.error ?? geo.error ?? sample.error
  return (
    <div style={{ display: 'flex', flexDirection: 'column', flex: 1, minHeight: 0 }}>
      <KpiStats sample={sample} />
      <div className="canvas-wrap">
        {loading && <Spinner />}
        {error && <div className="error">{error}</div>}
        <Scene radius={geom ? (geom.R0 + geom.a) * 2.4 : 22}>
          {geo.data && <PlasmaWireframe surf={geo.data.plasma3d} opacity={0.3} />}
          {field.data && <FieldLines field={field.data} onRange={setBRange} />}
        </Scene>
      </div>
      {field.data && <Colorbar title="|B| (field magnitude)" gradient={plasmaGradient} range={bRange} unit="T" />}
    </div>
  )
}
