import { useState } from 'react'
import { api, useApi, useDebounced, type GeometryResponse } from '../api'
import { CrossSection } from '../plots'
import { Scene } from '../three/Scene'
import { PlasmaWireframe } from '../three/Reactor3D'
import { Panel, Section, Slider, Spinner } from '../ui'

export function ReactorView() {
  const [R0, setR0] = useState(6.2)
  const [a, setA] = useState(3.2)
  const [kappa, setKappa] = useState(1.7)
  const [delta, setDelta] = useState(0.33)

  const body = useDebounced(JSON.stringify({ R0, a, kappa, delta, show_coils: false, mesh_stride: 2 }), 250)
  const { data, error, loading } = useApi<GeometryResponse>(`geometry:${body}`, () => api.geometry(JSON.parse(body)))

  const f1 = (v: number) => v.toFixed(1)
  const f2 = (v: number) => v.toFixed(2)

  return (
    <div className="view">
      <Panel>
        <Section title="Plasma geometry">
          <Slider label="Major radius R₀ (m)" value={R0} min={3} max={10} step={0.1} onChange={setR0} fmt={f1} />
          <Slider label="Minor radius a (m)" value={a} min={1} max={5} step={0.1} onChange={setA} fmt={f1} />
          <Slider label="Elongation κ" value={kappa} min={1} max={3} step={0.1} onChange={setKappa} fmt={f1} />
          <Slider label="Triangularity δ" value={delta} min={0} max={1} step={0.01} onChange={setDelta} fmt={f2} />
        </Section>
        <Section title="Cross-section">{data && <CrossSection geo={data} />}</Section>
      </Panel>
      <div className="view-main">
        {loading && <Spinner />}
        {error && <div className="error">{error}</div>}
        <div className="canvas-wrap">
          <Scene radius={(R0 + a) * 2.4}>{data && <PlasmaWireframe surf={data.plasma3d} />}</Scene>
        </div>
      </div>
    </div>
  )
}
