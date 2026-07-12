import { useEffect, useMemo, useState } from 'react'
import {
  api,
  invalidate,
  kpiEntries,
  useApi,
  useDebounced,
  type FieldLinesResponse,
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

const short = (name: string) => name.split('/')[1] ?? name

export function NetworkView() {
  const [viewMode, setViewMode] = useState<(typeof VIEW_MODES)[number]>('New Benchmarks')
  const [bump, setBump] = useState(0)
  const network = useStore((s) => s.network)
  const setNetwork = useStore((s) => s.setNetwork)
  const [tab, setTab] = useState<Tab>('sampling')
  const [seed, setSeed] = useState(0)
  const [sampleSize, setSampleSize] = useState(4)
  const [resolution, setResolution] = useState(50)
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
  const config = useApi<Record<string, unknown>>(network ? `config:${network}` : null, () => api.config_file(network!))

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
              {[1, 2, 4, 8].map((n) => (
                <option key={n} value={n}>
                  {n}
                </option>
              ))}
            </select>
          </label>
          {tab === 'topology' && <Slider label="Field lines" value={nLines} min={1} max={50} onChange={setNLines} />}
          {(tab === 'flux' || tab === 'residual') && (
            <Slider label="Resolution" value={resolution} min={50} max={300} step={25} onChange={setResolution} />
          )}
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
            {tab === 'flux' && (
              <GridTab kind="flux" network={network} seed={dSeed} sampleSize={sampleSize} resolution={resolution} />
            )}
            {tab === 'residual' && (
              <GridTab kind="residual" network={network} seed={dSeed} sampleSize={sampleSize} resolution={resolution} />
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
  // memoized so the ranges keep their identity across re-renders — fresh arrays
  // every render would defeat SampleScatter's memo
  const { xr, yr } = useMemo((): { xr?: Range2; yr?: Range2 } => {
    if (!sample.data) return {}
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
    return { xr: [rMin - pad, rMax + pad], yr: [zMin - pad, zMax + pad] }
  }, [sample.data])
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

/** Post-training KPIs from the run's stored kpis.json — nothing recomputed. */
function KpiStats({ network }: { network: string }) {
  const kpis = useApi(`kpis:${network}`, () => api.kpis(network))
  if (!kpis.data) return null
  return (
    <div className="stats">
      {kpiEntries(kpis.data).map(([key, value]) => (
        <Stat key={key} label={key.replaceAll('_', ' ')} value={value.toExponential(2)} />
      ))}
    </div>
  )
}

function GridTab({
  kind,
  network,
  seed,
  sampleSize,
  resolution,
}: {
  kind: 'flux' | 'residual'
  network: string
  seed: number
  sampleSize: number
  resolution: number
}) {
  const dResolution = useDebounced(resolution, 400)
  const grids = useApi<Grid2D[]>(
    `${kind}:${network}:${seed}:${sampleSize}:${dResolution}`,
    () => api.grid(network, kind, seed, sampleSize, dResolution),
  )

  return (
    <div className="scroll" style={{ position: 'relative' }}>
      {grids.error && <div className="error">{grids.error}</div>}
      <KpiStats network={network} />
      <div className="grid">
        {grids.loading
          ? Array.from({ length: sampleSize }, (_, i) => (
              <div className="card" key={i}>
                <div className="caption">sample {i}</div>
                <div className="plot-square">
                  <Spinner center />
                </div>
              </div>
            ))
          : (grids.data ?? []).map((g, i) => (
              <div className="card" key={i}>
                <div className="caption">sample {i}</div>
                <GridHeatmap
                  grid={g}
                  quantity={kind}
                />
              </div>
            ))}
      </div>
      {kind === 'residual' && (
        <Colorbar title="log₁₀|R_GS|" gradient={plasmaGradient} range={[-2, 1]} />
      )}
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
  const field = useApi<FieldLinesResponse>(`fieldlines:${network}:${seed}:${sampleSize}:${nLines}`, () =>
    api.fieldlines(network, seed, sampleSize, nLines),
  )
  const geom = sample.data?.geom3d
  const geo = useApi<GeometryResponse>(geom ? `geo3d:${JSON.stringify(geom)}` : null, () =>
    api.geometry({ ...geom!, show_coils: false, mesh_stride: 2 }),
  )

  const loading = field.loading || geo.loading || sample.loading
  const error = field.error ?? geo.error ?? sample.error
  return (
    <div style={{ display: 'flex', flexDirection: 'column', flex: 1, minHeight: 0 }}>
      <KpiStats network={network} />
      <div className="canvas-wrap">
        {loading && <Spinner />}
        {error && <div className="error">{error}</div>}
        <Scene radius={geom ? (geom.R0 + geom.a) * 2.4 : 22}>
          {geo.data && <PlasmaWireframe surf={geo.data.plasma3d} opacity={0.3} />}
          {field.data && <FieldLines field={field.data} />}
        </Scene>
      </div>
      {field.data && <Colorbar title="|B| (field magnitude)" gradient={plasmaGradient} range={field.data.b_range} unit="T" />}
    </div>
  )
}
