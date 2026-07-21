import { memo, useEffect, useMemo, useRef, useState } from 'react'
import { api, benchmarkFileUrl, benchmarkStream, kpiEntries, parseArtifactSlug, useApi, useDebounced, type BenchmarkEvent, type Grid2D } from '../api'
import {
  DEFAULT_RESIDUAL_RANGE,
  FLUX_COLORBAR,
  GridHeatmap,
  minMaxGridValues,
  type Range2,
} from '../plots'
import { plasmaGradient } from '../three/colormap'
import { Colorbar, Panel, Section, Segmented, Slider, Spinner, Stat, Toggle } from '../ui'

const MODES = ['Flux Prediction', 'GS Residual', 'Both'] as const
const MODE_LABELS = { 'Flux Prediction': 'Flux', 'GS Residual': 'Residual', Both: 'Both' } as const

type Row = Extract<BenchmarkEvent, { type: 'row' | 'row_error' }>

const short = (name: string) => name.startsWith('hpo/') ? name.slice(4) : name
const commitOf = (name: string) => parseArtifactSlug(name.startsWith('hpo/') ? name.split('/')[1] : name)?.commit

export function BenchmarkView() {
  const networks = useApi<string[]>('networks:All', () => api.networks('All'))
  const cfg = useApi('config', api.config)
  const evalConfigCount = cfg.data?.eval_config_count ?? 8
  const residualRange = cfg.data?.residual_color_range ?? DEFAULT_RESIDUAL_RANGE
  const [selected, setSelected] = useState<Set<string>>(new Set())
  const [commit, setCommit] = useState('All')
  const [mode, setMode] = useState<(typeof MODES)[number]>('Both')
  const [seed, setSeed] = useState(0)
  const [resolution, setResolution] = useState(50)
  const [rows, setRows] = useState<Row[]>([])
  const [running, setRunning] = useState(false)
  const [error, setError] = useState<string>()
  const abortRef = useRef<AbortController | null>(null)

  useEffect(() => {
    if (networks.data) setSelected(new Set(networks.data))
  }, [networks.data])
  useEffect(() => () => abortRef.current?.abort(), [])

  const commits = useMemo(
    () => [...new Set((networks.data ?? []).map(commitOf).filter((c): c is string => !!c))],
    [networks.data],
  )

  const toggle = (name: string) =>
    setSelected((s) => {
      const next = new Set(s)
      if (next.has(name)) next.delete(name)
      else next.add(name)
      return next
    })

  const run = async () => {
    setRows([])
    setError(undefined)
    setRunning(true)
    const ac = new AbortController()
    abortRef.current = ac
    try {
      const body = {
        networks: [...selected],
        commit: commit === 'All' ? null : commit,
        mode,
        seed,
        sample_size: evalConfigCount,
        resolution,
      }
      for await (const ev of benchmarkStream(body, ac.signal)) {
        if (ev.type === 'row' || ev.type === 'row_error') setRows((r) => [...r, ev])
        else if (ev.type === 'error') setError(ev.message)
      }
    } catch (e) {
      if (!ac.signal.aborted) setError(String(e))
    }
    setRunning(false)
  }

  return (
    <div className="view">
      <Panel>
        <Section title="Networks">
          <div className="check-list">
            {(networks.data ?? []).map((n) => (
              <Toggle key={n} label={short(n)} checked={selected.has(n)} onChange={() => toggle(n)} />
            ))}
          </div>
        </Section>
        <Section title="Settings">
          <label className="ctl">
            <span className="ctl-row">
              <span>Commit</span>
            </span>
            <select className="select" value={commit} onChange={(e) => setCommit(e.target.value)}>
              {['All', ...commits].map((c) => (
                <option key={c}>{c}</option>
              ))}
            </select>
          </label>
          <div className="ctl">
            <span className="ctl-row">
              <span>Mode</span>
            </span>
            <Segmented options={MODES} value={mode} onChange={setMode} labels={MODE_LABELS} small />
          </div>
          <Slider label="Seed" value={seed} min={0} max={1000} onChange={setSeed} />
          <Slider label="Resolution" value={resolution} min={50} max={600} step={50} onChange={setResolution} />
        </Section>
        <Section title="Run">
          {running ? (
            <button className="btn wide" onClick={() => abortRef.current?.abort()}>
              Cancel
            </button>
          ) : (
            <button className="btn primary wide" disabled={selected.size === 0} onClick={run}>
              Run benchmark
            </button>
          )}
        </Section>
      </Panel>
      <div className="view-main">
        {running && <Spinner />}
        {error && <div className="error">{error}</div>}
        <div className="scroll">
          <SavedBenchmarks />
          {rows.length === 0 && !running && (
            <div className="empty">Select networks and run to compare checkpoints.</div>
          )}
          {rows.map((row) =>
            row.type === 'row_error' ? (
              <div className="bench-row" key={row.network}>
                <div className="bench-head">
                  <span className="name">{short(row.network)}</span>
                  <span className="error" style={{ padding: 0 }}>
                    {row.message}
                  </span>
                </div>
              </div>
            ) : (
              <BenchRow key={row.network} row={row} residualRange={residualRange} />
            ),
          )}
        </div>
      </div>
    </div>
  )
}

// One run at a time, picked via selectbox — a mounted StoredRun fires a residual
// evaluation on the backend, so nothing mounts until it is explicitly selected.
function SavedBenchmarks() {
  const tree = useApi('benchmarks', api.benchmarks)
  const [selected, setSelected] = useState('')
  const commits = Object.entries(tree.data ?? {}).filter(([, runs]) => Object.keys(runs).length > 0)
  if (commits.length === 0) return null
  const [commit, run] = selected.split('/', 2)
  const files = tree.data?.[commit]?.[run]
  return (
    <div className="bench-saved">
      <label className="ctl">
        <span className="ctl-row">
          <span>Saved benchmarks</span>
        </span>
        <select className="select" value={selected} onChange={(e) => setSelected(e.target.value)}>
          <option value="">— select a run —</option>
          {commits.map(([c, runs]) => (
            <optgroup key={c} label={c}>
              {Object.keys(runs).map((r) => (
                  <option key={r} value={`${c}/${r}`}>
                  {r}
                </option>
              ))}
            </optgroup>
          ))}
        </select>
      </label>
      {files && <StoredRun key={selected} run={run} files={files} />}
    </div>
  )
}

function StoredRun({ run, files }: { run: string; files: string[] }) {
  const networkName = run
  const cfg = useApi('config', api.config)
  const evalConfigCount = cfg.data?.eval_config_count ?? 8
  const residualRange = cfg.data?.residual_color_range ?? DEFAULT_RESIDUAL_RANGE
  const [seed, setSeed] = useState(0)
  const [resolution, setResolution] = useState(50)
  const dSeed = useDebounced(seed, 400)
  const dResolution = useDebounced(resolution, 400)
  const hasKpis = files.includes('run.json')
  const kpis = useApi(hasKpis ? `kpis:${networkName}` : null, () =>
    api.kpis(networkName),
  )
  const grids = useApi<Grid2D[]>(
    `stored-run:${networkName}:${dSeed}:${dResolution}`,
    () => api.grid(networkName, 'residual', dSeed, evalConfigCount, dResolution),
  )
  const dataFiles = files.filter((f) => !f.endsWith('.png'))
  return (
    <div className="bench-row">
      <div className="bench-head">
        <span className="name">{run}</span>
        {dataFiles.map((f) => (
          <a className="chip" key={f} href={benchmarkFileUrl(run, f)} target="_blank" rel="noreferrer">
            {f}
          </a>
        ))}
      </div>
      <div className="ctl" style={{ gap: '1rem', flexWrap: 'wrap' }}>
        <Slider label="Seed" value={seed} min={0} max={1000} onChange={setSeed} />
        <Slider label="Resolution" value={resolution} min={50} max={600} step={50} onChange={setResolution} />
      </div>
      {kpis.data && (
        <div className="stats">
          {kpiEntries(kpis.data).map(([key, value]) => (
            <Stat key={key} label={key.replaceAll('_', ' ')} value={value.toExponential(2)} />
          ))}
        </div>
      )}
      {kpis.error && <div className="error">KPIs unavailable for this run</div>}
      {grids.loading && <Spinner />}
      {grids.error && <div className="error">{grids.error}</div>}
      {grids.data && (
        <>
          <div className="bench-grids">
            {grids.data.map((g, i) => (
              <GridHeatmap key={i} grid={g} quantity="residual" residualRange={residualRange} />
            ))}
          </div>
          <Colorbar title="|R_GS|" gradient={plasmaGradient} range={residualRange} />
        </>
      )}
    </div>
  )
}

// memo: each streamed SSE row re-renders the list — without it, appending row N
// re-diffs every already-rendered row's carpet heatmaps (row objects never mutate)
const BenchRow = memo(function BenchRow({
  row,
  residualRange,
}: {
  row: Extract<Row, { type: 'row' }>
  residualRange: Range2
}) {
  const cfg = row.config
  const chips = ['hidden_dims', 'learning_rate_max', 'n_train', 'n_epochs']
    .filter((k) => cfg[k] !== undefined)
    .map((k) => `${k}=${JSON.stringify(cfg[k])}`)
  const fluxRange = row.flux_grids?.length ? minMaxGridValues(row.flux_grids) : undefined
  return (
    <div className="bench-row">
      <div className="bench-head">
        <span className="name">{short(row.network)}</span>
        {chips.map((c) => (
          <span className="chip" key={c}>
            {c}
          </span>
        ))}
      </div>
      <div className="stats">
        {kpiEntries(row.kpis).map(([key, value]) => (
          <Stat key={key} label={key.replaceAll('_', ' ')} value={value.toExponential(2)} />
        ))}
      </div>
      {row.flux_grids && row.flux_grids.length > 0 && (
        <>
          <div className="stats">
            <Stat label="flux ψ" value={`${row.flux_grids.length} samples`} />
          </div>
          <div className="bench-grids">
            {row.flux_grids.map((g, i) => (
              <GridHeatmap key={i} grid={g} quantity="flux" zRange={fluxRange} />
            ))}
          </div>
          {fluxRange && <Colorbar title="ψ" gradient={FLUX_COLORBAR} range={fluxRange} unit="Wb" />}
        </>
      )}
      {row.residual_grids && row.residual_grids.length > 0 && (
        <>
          <div className="stats">
            <Stat label="GS residual" value={`${row.residual_grids.length} samples`} />
          </div>
          <div className="bench-grids">
            {row.residual_grids.map((g, i) => (
              <GridHeatmap key={i} grid={g} quantity="residual" residualRange={residualRange} />
            ))}
          </div>
          <Colorbar title="|R_GS|" gradient={plasmaGradient} range={residualRange} />
        </>
      )}
    </div>
  )
})
