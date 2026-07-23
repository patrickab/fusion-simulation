import { useState } from 'react'
import {
  api,
  useApi,
  useDebounced,
  type GeometryRequest,
  type GeometryResponse,
  type StellaratorGeometryRequest,
} from '../api'
import { CrossSection } from '../plots'
import { Scene } from '../three/Scene'
import { PlasmaWireframe } from '../three/Reactor3D'
import { Panel, Section, Segmented, Slider, Spinner } from '../ui'

type ReactorRequest =
  | ({ device: 'tokamak' } & GeometryRequest)
  | ({ device: 'stellarator' } & StellaratorGeometryRequest)

export function ReactorView() {
  const [device, setDevice] = useState<'tokamak' | 'stellarator'>('tokamak')
  const [R0, setR0] = useState(6.2)
  const [a, setA] = useState(3.2)
  const [kappa, setKappa] = useState(1.7)
  const [delta, setDelta] = useState(0.33)
  const [stellaratorR0, setStellaratorR0] = useState(5.5)
  const [stellaratorA, setStellaratorA] = useState(1.2)
  const [stellaratorKappa, setStellaratorKappa] = useState(1.4)
  const [nFieldPeriods, setNFieldPeriods] = useState(5)
  const [helicalAmplitude, setHelicalAmplitude] = useState(0.25)

  const body = useDebounced(
    JSON.stringify(
      device === 'tokamak'
        ? { device, R0, a, kappa, delta, show_coils: false, mesh_stride: 2 }
        : {
            device,
            R0: stellaratorR0,
            a: stellaratorA,
            kappa: stellaratorKappa,
            n_field_periods: nFieldPeriods,
            helical_amplitude: helicalAmplitude,
            mesh_stride: 2,
          },
    ),
    250,
  )
  const { data, error, loading } = useApi<GeometryResponse>(`geometry:${body}`, () => {
    const request = JSON.parse(body) as ReactorRequest
    return request.device === 'tokamak'
      ? api.geometry(request)
      : api.stellaratorGeometry(request)
  })

  const f1 = (v: number) => v.toFixed(1)
  const f2 = (v: number) => v.toFixed(2)
  const sceneRadius =
    (device === 'tokamak' ? R0 + a : stellaratorR0 + stellaratorA) * 2.4

  return (
    <div className="view">
      <Panel>
        <Section title="Reactor type">
          <Segmented
            options={['tokamak', 'stellarator']}
            value={device}
            onChange={setDevice}
            labels={{ tokamak: 'Tokamak', stellarator: 'Stellarator' }}
          />
        </Section>
        <Section title="Plasma geometry">
          {device === 'tokamak' ? (
            <>
              <Slider label="Major radius R₀ (m)" value={R0} min={3} max={10} step={0.1} onChange={setR0} fmt={f1} />
              <Slider label="Minor radius a (m)" value={a} min={1} max={5} step={0.1} onChange={setA} fmt={f1} />
              <Slider label="Elongation κ" value={kappa} min={1} max={3} step={0.1} onChange={setKappa} fmt={f1} />
              <Slider label="Triangularity δ" value={delta} min={0} max={1} step={0.01} onChange={setDelta} fmt={f2} />
            </>
          ) : (
            <>
              <Slider label="Major radius R₀ (m)" value={stellaratorR0} min={4} max={12} step={0.1} onChange={setStellaratorR0} fmt={f1} />
              <Slider label="Minor radius a (m)" value={stellaratorA} min={0.4} max={2.5} step={0.1} onChange={setStellaratorA} fmt={f1} />
              <Slider label="Elongation κ" value={stellaratorKappa} min={0.8} max={2.5} step={0.1} onChange={setStellaratorKappa} fmt={f1} />
              <Slider label="Field periods NFP" value={nFieldPeriods} min={2} max={10} onChange={setNFieldPeriods} />
              <Slider label="Helical amplitude h/a" value={helicalAmplitude} min={0} max={0.45} step={0.01} onChange={setHelicalAmplitude} fmt={f2} />
            </>
          )}
        </Section>
        <Section title={device === 'stellarator' ? 'Cross-section (φ = 0)' : 'Cross-section'}>
          {data && <CrossSection geo={data} />}
        </Section>
      </Panel>
      <div className="view-main">
        {loading && <Spinner />}
        {error && <div className="error">{error}</div>}
        <div className="canvas-wrap">
          <Scene radius={sceneRadius}>{data && <PlasmaWireframe surf={data.plasma3d} />}</Scene>
        </div>
      </div>
    </div>
  )
}
