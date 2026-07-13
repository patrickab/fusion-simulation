// Typed client for the FastAPI backend (src/api/main.py) + a tiny cached-fetch hook.
import { useEffect, useState } from 'react'

/** Stored kpis.json record: KPI numbers plus metadata (date, network, loss label, ...). */
export type Kpis = Record<string, number | string>

/** Numeric KPI entries in display order, metadata keys dropped. */
export const kpiEntries = (kpis: Kpis): [string, number][] =>
  Object.entries(kpis).filter(
    (e): e is [string, number] =>
      typeof e[1] === 'number' &&
      (e[0].startsWith('loss_') ||
        e[0].startsWith('core_') ||
        e[0] === 'edge_loss_p95' ||
        e[0] === 'boundary_leak_max'),
  )

export interface Sample {
  R0: number
  a: number
  kappa: number
  delta: number
  p0: number
  F_axis: number
  pressure_alpha: number
  field_exponent: number
  boundary_R: number[]
  boundary_Z: number[]
  interior_R: number[]
  interior_Z: number[]
}

export interface Geom3D { R0: number; a: number; kappa: number; delta: number }

export interface SampleResponse {
  samples: Sample[]
  geom3d: Geom3D
  state3d: { p0: number; F_axis: number; pressure_alpha: number; field_exponent: number }
}

export type GridQuantity = 'flux' | 'residual'

export interface Grid2D {
  theta: number[]
  rho: number[]
  R: number[][]
  Z: number[][]
  values: number[][]
  boundary_R: number[]
  boundary_Z: number[]
}

export interface SurfaceGrid { n_phi: number; n_theta: number; X: number[]; Y: number[]; Z: number[] }

export interface GeometryResponse {
  boundary2d: { R: number[]; Z: number[] }
  plasma3d: SurfaceGrid
}

export interface GeometryRequest extends Geom3D {
  show_coils: boolean
  mesh_stride?: number
}

/** Server-traced (VTK RK45) field lines: concatenated polylines + |B| per vertex. */
export interface FieldLinesResponse {
  points: number[] // flat xyz
  speeds: number[]
  line_lengths: number[]
  b_range: [number, number]
}

export type BenchmarkEvent =
  | { type: 'row'; network: string; config: Record<string, unknown>; kpis: Kpis; flux_grids?: Grid2D[]; residual_grids?: Grid2D[] }
  | { type: 'row_error'; network: string; message: string }
  | { type: 'error'; message: string }
  | { type: 'done' }

async function toJson<T>(r: Response): Promise<T> {
  if (!r.ok) throw new Error(`${r.status}: ${await r.text()}`)
  return r.json()
}

const post = <T,>(url: string, body: unknown): Promise<T> =>
  fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  }).then((r) => toJson<T>(r))

// network names are 'commit/run' paths — keep slashes as path separators
const enc = (name: string) => encodeURIComponent(name).replaceAll('%2F', '/')

export const api = {
  config: () =>
    fetch('/api/config').then((r) =>
      toJson<{
        eval_config_count: number
        eval_resolution: number
        residual_color_range: [number, number]
      }>(r),
    ),
  networks: (viewMode: string) =>
    fetch(`/api/networks?view_mode=${encodeURIComponent(viewMode)}`).then((r) => toJson<string[]>(r)),
  config_file: (name: string) =>
    fetch(`/api/network/${enc(name)}/config`).then((r) => toJson<Record<string, unknown>>(r)),
  kpis: (name: string) => fetch(`/api/network/${enc(name)}/kpis`).then((r) => toJson<Kpis>(r)),
  archive: (name: string) => fetch(`/api/network/${enc(name)}/archive`, { method: 'POST' }).then((r) => toJson(r)),
  rename: (name: string, newName: string) =>
    post<{ name: string }>(`/api/network/${enc(name)}/rename`, { new_name: newName }),
  remove: (name: string) => fetch(`/api/network/${enc(name)}`, { method: 'DELETE' }).then((r) => toJson(r)),
  sample: (name: string, seed: number, sampleSize: number) =>
    post<SampleResponse>(`/api/network/${enc(name)}/sample`, {
      seed,
      sample_size: sampleSize,
    }),
  grid: (name: string, quantity: GridQuantity, seed: number, sampleSize: number, resolution: number) =>
    post<Grid2D[]>(`/api/network/${enc(name)}/${quantity}`, { seed, sample_size: sampleSize, resolution }),
  fieldlines: (name: string, seed: number, sampleSize: number, nLines: number) =>
    post<FieldLinesResponse>(`/api/network/${enc(name)}/fieldlines`, { seed, sample_size: sampleSize, n_lines: nLines }),
  geometry: (body: GeometryRequest) => post<GeometryResponse>('/api/geometry', body),
  // data/benchmarks tree: {commit: {run: [file, ...]}}
  benchmarks: () => fetch('/api/benchmarks').then((r) => toJson<Record<string, Record<string, string[]>>>(r)),
}

export const benchmarkFileUrl = (commit: string, run: string, file: string) =>
  `/api/benchmarks/files/${commit}/${run}/${file}`

export async function* benchmarkStream(
  body: {
    networks: string[]
    commit: string | null
    mode: string
    seed: number
    sample_size: number
    resolution: number
  },
  signal: AbortSignal,
): AsyncGenerator<BenchmarkEvent> {
  const r = await fetch('/api/benchmark', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    signal,
  })
  if (!r.ok || !r.body) throw new Error(`${r.status}: ${await r.text()}`)
  const reader = r.body.getReader()
  const decoder = new TextDecoder()
  let buf = ''
  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    buf += decoder.decode(value, { stream: true })
    let i
    while ((i = buf.indexOf('\n\n')) >= 0) {
      const chunk = buf.slice(0, i)
      buf = buf.slice(i + 2)
      if (chunk.startsWith('data: ')) yield JSON.parse(chunk.slice(6)) as BenchmarkEvent
    }
  }
}

// ponytail: module-level Map cache keyed by request signature; swap for react-query if it grows.
const cache = new Map<string, unknown>()

export function invalidate(prefix = ''): void {
  for (const k of cache.keys()) if (k.startsWith(prefix)) cache.delete(k)
}

export interface ApiState<T> {
  data?: T
  error?: string
  loading: boolean
}

/** Cached fetch. Pass key=null to stay idle. Previous data is kept while reloading. */
export function useApi<T>(key: string | null, fn: () => Promise<T>): ApiState<T> {
  const [state, setState] = useState<ApiState<T>>({ loading: !!key })
  useEffect(() => {
    if (!key) return
    if (cache.has(key)) {
      setState({ data: cache.get(key) as T, loading: false })
      return
    }
    let live = true
    setState((s) => ({ ...s, error: undefined, loading: true }))
    fn()
      .then((data) => {
        cache.set(key, data)
        if (live) setState({ data, loading: false })
      })
      .catch((e: unknown) => {
        if (live) setState((s) => ({ ...s, error: String(e), loading: false }))
      })
    return () => {
      live = false
    }
    // fn identity changes every render; key encodes all inputs, so it is the only dep
  }, [key])
  return state
}

export function useDebounced<T>(value: T, ms: number): T {
  const [v, setV] = useState(value)
  useEffect(() => {
    const t = setTimeout(() => setV(value), ms)
    return () => clearTimeout(t)
  }, [value, ms])
  return v
}
