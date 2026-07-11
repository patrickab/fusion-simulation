import { useEffect, useMemo, useRef, useState } from 'react'
import { api, benchmarkStream, useApi, type BenchmarkEvent } from '../api'
import { GridHeatmap } from '../plots'
import { plasmaGradient, plasmaPlotlyScale } from '../three/colormap'
import { Colorbar, Panel, Section, Segmented, Slider, Spinner, Stat, Toggle } from '../ui'

const MODES = ['Flux Prediction', 'GS Residual', 'Both'] as const
const MODE_LABELS = { 'Flux Prediction': 'Flux', 'GS Residual': 'Residual', Both: 'Both' } as const

type Row = Extract<BenchmarkEvent, { type: 'row' | 'row_error' }>

const short = (name: string) => name.replace(/\.flax$/, '')
const commitOf = (name: string) => /_([0-9a-f]{6,})\.flax$/.exec(name)?.[1]

export function BenchmarkView() {
  const networks = useApi<string[]>('networks:All', () => api.networks('All'))
  const [selected, setSelected] = useState<Set<string>>(new Set())
  const [commit, setCommit] = useState('All')
  const [mode, setMode] = useState<(typeof MODES)[number]>('Both')
  const [seed, setSeed] = useState(0)
  const [resolution, setResolution] = useState(100)
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
        sample_size: 4,
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
          <Slider label="Resolution" value={resolution} min={40} max={200} step={20} onChange={setResolution} />
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
              <BenchRow key={row.network} row={row} />
            ),
          )}
        </div>
      </div>
    </div>
  )
}

// ponytail: fixed scales matched to the legacy plot_flux_heatmap/plot_gs_residual_heatmap
// (src/lib/visualization.py) — same absolute zmin/zmax across samples, not per-batch min/max
// (axis ranges are no longer shared across samples — each grid keeps its own true aspect)
// Residual grids are log10|R_GS| (src/api/network.py), matching model_evaluation.py's
// reference montage: -2..1 covers converged-to-diverging orders of magnitude.
function gridScales(residual: boolean) {
  return residual ? { zmin: -2, zmax: 1 } : { zmin: 0, zmax: 90 }
}

function BenchRow({ row }: { row: Extract<Row, { type: 'row' }> }) {
  const cfg = row.config
  const chips = ['hidden_dims', 'learning_rate_max', 'n_train', 'n_epochs']
    .filter((k) => cfg[k] !== undefined)
    .map((k) => `${k}=${JSON.stringify(cfg[k])}`)
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
      {row.flux_grids && row.flux_grids.length > 0 && (
        <>
          <div className="stats">
            <Stat label="flux ψ" value={`${row.flux_grids.length} samples`} />
          </div>
          <div className="bench-grids">
            {row.flux_grids.map((g, i) => (
              <GridHeatmap key={i} grid={g} {...gridScales(false)} />
            ))}
          </div>
        </>
      )}
      {row.residual_grids && row.residual_grids.length > 0 && (
        <>
          <div className="stats">
            <Stat label="GS residual" value={`${row.residual_grids.length} samples`} />
          </div>
          <div className="bench-grids">
            {row.residual_grids.map((g, i) => (
              <GridHeatmap key={i} grid={g} colorscale={plasmaPlotlyScale} {...gridScales(true)} />
            ))}
          </div>
          <Colorbar title="log₁₀|R_GS|" gradient={plasmaGradient} range={[-2, 1]} />
        </>
      )}
    </div>
  )
}
