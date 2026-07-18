import { useEffect, useMemo, useState } from 'react'
import {
  api,
  invalidate,
  kpiEntries,
  parseArtifactSlug,
  useApi,
  useDebounced,
  type FieldLinesResponse,
  type GeometryResponse,
  type Grid2D,
  type HpoStudies,
  type NetworkModels,
  type SampleResponse,
} from '../api'
import { useStore } from '../store'
import {
  DEFAULT_RESIDUAL_RANGE,
  FLUX_COLORBAR,
  GridHeatmap,
  SampleScatter,
  minMaxGridValues,
  type Range2,
} from '../plots'
import { Scene } from '../three/Scene'
import { FieldLines, PlasmaWireframe } from '../three/Reactor3D'
import { plasmaGradient } from '../three/colormap'
import { Colorbar, JsonBlock, Panel, Popover, Section, Segmented, Slider, Spinner, Stat } from '../ui'

const VIEW_MODES = ['Single-Configs', 'HPO runs', 'Archive'] as const
const ARCHIVE_MODES = ['Single-Configs', 'HPO runs'] as const
const MODE_LABELS = { 'Single-Configs': 'Single', 'HPO runs': 'HPO', Archive: 'Archive' } as const
const ARCHIVE_LABELS = { 'Single-Configs': 'Single', 'HPO runs': 'HPO' } as const
const TABS = ['sampling', 'flux', 'residual', 'topology'] as const
type Tab = (typeof TABS)[number]
const TAB_LABELS: Record<Tab, string> = {
  sampling: 'Sampling',
  flux: 'Flux ψ',
  residual: 'GS residual',
  topology: '3D topology',
}

export function NetworkView() {
  const [viewMode, setViewMode] = useState<(typeof VIEW_MODES)[number]>('Single-Configs')
  const [archiveMode, setArchiveMode] = useState<(typeof ARCHIVE_MODES)[number]>('Single-Configs')
  const [bump, setBump] = useState(0)
  const network = useStore((s) => s.network)
  const setNetwork = useStore((s) => s.setNetwork)
  const [study, setStudy] = useState('')
  const [tab, setTab] = useState<Tab>('sampling')
  const [seed, setSeed] = useState(0)
  const [sampleSize, setSampleSize] = useState(4)
  const [resolution, setResolution] = useState(50)
  const [nLines, setNLines] = useState(50)

  const singleConfigs = viewMode === 'Single-Configs' || (viewMode === 'Archive' && archiveMode === 'Single-Configs')
  const hpo = viewMode === 'HPO runs' || (viewMode === 'Archive' && archiveMode === 'HPO runs')
  const archived = viewMode === 'Archive'
  const networks = useApi<string[]>(
    singleConfigs ? `networks:${archived ? 'Archive' : 'Single-Configs'}:${bump}` : null,
    () => api.networks(archived ? 'Archive' : 'Single-Configs'),
  )
  const hpoStudies = useApi<HpoStudies>(hpo ? `hpo:${archived}:${bump}` : null, () => api.hpoStudies(archived))
  const studies = Object.keys(hpoStudies.data ?? {})
  const trialRuns = study ? (hpoStudies.data?.[study] ?? []) : []
  const hpoNetwork = study && network?.startsWith(`hpo/${study}/`) ? network : null
  const actionTarget = hpo ? (hpoNetwork ?? (study ? `hpo/${study}` : null)) : network

  useEffect(() => {
    if (!singleConfigs) return
    const list = networks.data
    if (list && (!network || !list.includes(network))) setNetwork(list[0] ?? null)
  }, [singleConfigs, networks.data, network, setNetwork])
  useEffect(() => {
    if (hpo && !studies.includes(study)) setStudy(studies[0] ?? '')
  }, [hpo, studies, study])
  useEffect(() => {
    if (!hpo) return
    const next = study && trialRuns[0] ? `hpo/${study}/${trialRuns[0]}` : null
    const currentRun = hpoNetwork?.split('/')[2]
    if ((!currentRun || !trialRuns.includes(currentRun)) && network !== next) setNetwork(next)
  }, [hpo, study, trialRuns, hpoNetwork, network, setNetwork])

  const dSeed = useDebounced(seed, 400)
  const dLines = useDebounced(nLines, 400)
  const sampleKey = network ? `sample:${network}:${dSeed}:${sampleSize}` : null
  const sample = useApi<SampleResponse>(sampleKey, () => api.sample(network!, dSeed, sampleSize))
  const config = useApi<Record<string, unknown>>(network ? `config:${network}` : null, () => api.config_file(network!))
  const models = useApi<NetworkModels>(network ? `models:${network}` : null, () => api.models(network!))

  const refresh = () => {
    invalidate()
    setBump((b) => b + 1)
  }
  const doArchive = async () => {
    if (!actionTarget) return
    await api.archive(actionTarget)
    setNetwork(null)
    setStudy('')
    refresh()
  }
  const doRename = async () => {
    if (!actionTarget) return
    const slug = hpo ? study : actionTarget
    const current = parseArtifactSlug(slug)?.name
    if (!current) return
    // ponytail: native prompt is the smallest editor with a prefilled current name.
    const newName = window.prompt('New name:', current)
    if (!newName) return
    const r = await api.rename(actionTarget, newName)
    setNetwork(r.name)
    if (hpo) setStudy(r.name.split('/')[1])
    refresh()
  }
  const doDelete = async (target: string, label: string) => {
    if (!window.confirm(`Delete ${label}?`)) return
    await api.remove(target)
    setNetwork(null)
    if (target === `hpo/${study}`) setStudy('')
    refresh()
  }

  return (
    <div className="view">
      <Panel>
        <Section title="Network">
          <div className="ctl">
            <Segmented options={VIEW_MODES} value={viewMode} onChange={setViewMode} labels={MODE_LABELS} small />
          </div>
          {viewMode === 'Archive' && (
            <div className="ctl">
              <Segmented options={ARCHIVE_MODES} value={archiveMode} onChange={setArchiveMode} labels={ARCHIVE_LABELS} small />
            </div>
          )}
          {singleConfigs ? (
            <div className="row">
              <select className="select" value={network ?? ''} onChange={(e) => setNetwork(e.target.value)}>
                {(networks.data ?? []).map((slug) => <option key={slug} value={slug}>{slug}</option>)}
              </select>
              <Actions
                archived={archived}
                hpo={false}
                target={network}
                onArchive={doArchive}
                onRename={doRename}
                onDelete={doDelete}
              />
            </div>
          ) : (
            <>
              <label className="ctl">
                <span className="ctl-row"><span>HPO study</span></span>
                <select className="select" value={study} onChange={(e) => setStudy(e.target.value)}>
                  {studies.map((slug) => <option key={slug} value={slug}>{slug}</option>)}
                </select>
              </label>
              <div className="row">
                <select className="select" value={hpoNetwork?.split('/')[2] ?? ''} disabled={!study || trialRuns.length === 0} onChange={(e) => setNetwork(`hpo/${study}/${e.target.value}`)}>
                  {trialRuns.length === 0 ? <option>No retained trials</option> : trialRuns.map((run) => <option key={run} value={run}>{run}</option>)}
                </select>
                <Actions
                  archived={archived}
                  hpo
                  target={actionTarget}
                  network={hpoNetwork}
                  study={study}
                  onArchive={doArchive}
                  onRename={doRename}
                  onDelete={doDelete}
                />
              </div>
            </>
          )}
        </Section>
        {config.data && (
          <Section title="Training config">
            <JsonBlock obj={config.data} />
          </Section>
        )}
        {models.data && (
          <Section title="Models">
            <div className="model-list">
              <div className="model-row">
                <span className="model-role">Foundation model</span>
                <span className="mono model-name">{models.data.foundation.name}</span>
              </div>
              <div className="model-row">
                <span className="model-role">Corrector</span>
                {models.data.corrector ? (
                  <span className="mono model-name">
                    {models.data.corrector.name} <span className="model-detail">scale {models.data.corrector.scale}</span>
                  </span>
                ) : (
                  <span className="model-detail">Not available</span>
                )}
              </div>
            </div>
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
            <Slider label="Resolution" value={resolution} min={50} max={600} step={50} onChange={setResolution} />
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

function Actions({
  archived,
  hpo,
  target,
  network,
  study,
  onArchive,
  onRename,
  onDelete,
}: {
  archived: boolean
  hpo: boolean
  target: string | null
  network?: string | null
  study?: string
  onArchive: () => Promise<void>
  onRename: () => Promise<void>
  onDelete: (target: string, label: string) => Promise<void>
}) {
  if (!target) return null
  return (
    <Popover label="⋯" buttonClassName="pop-btn icon" align="right">
      {(close) => (
        <>
          {!archived && <button onClick={() => { close(); void onArchive() }}>{hpo ? 'Archive study' : 'Archive'}</button>}
          {!archived && <button onClick={() => { close(); void onRename() }}>{hpo ? 'Rename study…' : 'Rename…'}</button>}
          {hpo && network && <button className="danger" onClick={() => { close(); void onDelete(network, 'network') }}>Delete network</button>}
          <button className="danger" onClick={() => { close(); void onDelete(hpo ? `hpo/${study}` : target, hpo ? 'HPO study' : 'network') }}>
            {hpo ? 'Delete study' : 'Delete'}
          </button>
        </>
      )}
    </Popover>
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

/** Post-training KPIs from the run.json result; nothing recomputed. */
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
  const cfg = useApi('config', api.config)
  const residualRange = cfg.data?.residual_color_range ?? DEFAULT_RESIDUAL_RANGE
  const grids = useApi<Grid2D[]>(
    `${kind}:${network}:${seed}:${sampleSize}:${dResolution}`,
    () => api.grid(network, kind, seed, sampleSize, dResolution),
  )
  const zRange: Range2 | undefined = grids.data?.length
    ? minMaxGridValues(grids.data)
    : undefined

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
                  zRange={zRange}
                  residualRange={residualRange}
                />
              </div>
            ))}
      </div>
      {kind === 'flux' && zRange && (
        <Colorbar title="ψ" gradient={FLUX_COLORBAR} range={zRange} unit="Wb" />
      )}
      {kind === 'residual' && (
        <Colorbar title="|R_GS|" gradient={plasmaGradient} range={residualRange} />
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
