import { useEffect, useRef, useState, type ReactNode } from 'react'
import { useStore, type View } from './store'
import { highlightJson } from './shiki'

// Horizontal legend for a sequential-colormap field (e.g. |B| on field lines).
export function Colorbar({
  title,
  gradient,
  range,
  unit,
}: {
  title: string
  gradient: string
  range: [number, number]
  unit?: string
}) {
  const [lo, hi] = range
  const mid = (lo + hi) / 2
  const fmt = (v: number) => `${v.toFixed(2)}${unit ? ` ${unit}` : ''}`
  return (
    <div className="colorbar">
      <div className="colorbar-title">{title}</div>
      <div className="colorbar-track" style={{ background: gradient }} />
      <div className="colorbar-ticks">
        <span>{fmt(lo)}</span>
        <span>{fmt(mid)}</span>
        <span>{fmt(hi)}</span>
      </div>
    </div>
  )
}

export function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="sect">
      <h3>{title}</h3>
      {children}
    </section>
  )
}

export function Slider({
  label,
  value,
  min,
  max,
  step = 1,
  onChange,
  fmt,
}: {
  label: string
  value: number
  min: number
  max: number
  step?: number
  onChange: (v: number) => void
  fmt?: (v: number) => string
}) {
  const pct = ((value - min) / (max - min)) * 100
  return (
    <label className="ctl">
      <span className="ctl-row">
        <span>{label}</span>
        <span className="mono">{fmt ? fmt(value) : value}</span>
      </span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        style={{ background: `linear-gradient(to right, rgba(255,255,255,0.55) ${pct}%, rgba(255,255,255,0.12) ${pct}%)` }}
        onChange={(e) => onChange(Number(e.target.value))}
      />
    </label>
  )
}

export function Segmented<T extends string>({
  options,
  value,
  onChange,
  labels,
  small,
}: {
  options: readonly T[]
  value: T
  onChange: (v: T) => void
  labels?: Record<T, string>
  small?: boolean
}) {
  return (
    <div className={small ? 'seg small' : 'seg'}>
      {options.map((o) => (
        <button key={o} className={o === value ? 'on' : ''} onClick={() => onChange(o)}>
          {labels?.[o] ?? o}
        </button>
      ))}
    </div>
  )
}

export function Toggle({
  label,
  checked,
  onChange,
}: {
  label: string
  checked: boolean
  onChange: (v: boolean) => void
}) {
  return (
    <label className="check">
      <input type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)} />
      {label}
    </label>
  )
}

export function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="stat">
      <span className="label">{label}</span>
      <span className="mono">{value}</span>
    </div>
  )
}

/** Renders an object as syntax-highlighted JSON via Shiki (reuses gigachad-bot's andromeeda theme). */
export function JsonBlock({ obj }: { obj: unknown }) {
  const [html, setHtml] = useState<string | null>(null)
  useEffect(() => {
    let live = true
    highlightJson(obj).then((h) => {
      if (live) setHtml(h)
    })
    return () => {
      live = false
    }
  }, [obj])
  if (!html) return null
  return <div className="json-block" dangerouslySetInnerHTML={{ __html: html }} />
}

// ponytail: skip the indicator for fast local fetches (<200ms) so it doesn't just flash on/off on view switch
export function Spinner({ center = false }: { center?: boolean }) {
  const [show, setShow] = useState(false)
  useEffect(() => {
    const t = setTimeout(() => setShow(true), 200)
    return () => clearTimeout(t)
  }, [])
  return show ? (
    <div className={center ? 'spinner-center' : 'spinner-wrap'}>
      <div className="spinner" />
    </div>
  ) : null
}

/** Click-outside-closing popover. `children` may be a render-prop receiving close(). */
export function Popover({
  label,
  chevron,
  buttonClassName = 'pop-btn',
  align = 'left',
  children,
}: {
  label: ReactNode
  chevron?: boolean
  buttonClassName?: string
  align?: 'left' | 'right'
  children: ReactNode | ((close: () => void) => ReactNode)
}) {
  const [open, setOpen] = useState(false)
  const ref = useRef<HTMLDivElement>(null)
  useEffect(() => {
    if (!open) return
    const onDown = (e: PointerEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false)
    }
    document.addEventListener('pointerdown', onDown)
    return () => document.removeEventListener('pointerdown', onDown)
  }, [open])
  return (
    <div className="pop" ref={ref}>
      <button className={buttonClassName} onClick={() => setOpen((o) => !o)}>
        {label}
        {chevron && <span className={open ? 'chev up' : 'chev'}>▾</span>}
      </button>
      {open && (
        <div className={align === 'right' ? 'pop-menu right' : 'pop-menu'}>
          {typeof children === 'function' ? children(() => setOpen(false)) : children}
        </div>
      )}
    </div>
  )
}

const VIEWS: readonly View[] = ['reactor', 'network', 'benchmark']
const VIEW_LABELS: Record<View, string> = {
  reactor: 'Reactor Visualizer',
  network: 'Network Visualizer',
  benchmark: 'Benchmark Visualizer',
}

/** Left sidebar: view-switcher popover + collapse control, collapsing to a slim rail. */
export function Panel({ children }: { children: ReactNode }) {
  const { view, setView, sidebar, setSidebar } = useStore()
  if (!sidebar) {
    return (
      <aside className="panel rail">
        <button className="icon-btn" title="Show controls" onClick={() => setSidebar(true)}>
          ›
        </button>
      </aside>
    )
  }
  return (
    <aside className="panel">
      <div className="panel-head">
        <Popover label={VIEW_LABELS[view]} chevron>
          {(close) =>
            VIEWS.map((v) => (
              <button
                key={v}
                className={v === view ? 'on' : ''}
                onClick={() => {
                  setView(v)
                  close()
                }}
              >
                {VIEW_LABELS[v]}
              </button>
            ))
          }
        </Popover>
        <button className="icon-btn" title="Hide controls" onClick={() => setSidebar(false)}>
          ‹
        </button>
      </div>
      {children}
    </aside>
  )
}
