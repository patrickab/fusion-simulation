import { useStore } from './store'
import { ReactorView } from './views/ReactorView'
import { NetworkView } from './views/NetworkView'
import { BenchmarkView } from './views/BenchmarkView'

export default function App() {
  const view = useStore((s) => s.view)
  return (
    <div className="app">
      {view === 'reactor' && <ReactorView />}
      {view === 'network' && <NetworkView />}
      {view === 'benchmark' && <BenchmarkView />}
    </div>
  )
}
