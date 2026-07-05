import { create } from 'zustand'

export type View = 'reactor' | 'network' | 'benchmark'

interface AppState {
  view: View
  setView: (view: View) => void
  network: string | null
  setNetwork: (network: string | null) => void
  sidebar: boolean
  setSidebar: (open: boolean) => void
}

export const useStore = create<AppState>((set) => ({
  view: 'reactor',
  setView: (view) => set({ view }),
  network: null,
  setNetwork: (network) => set({ network }),
  sidebar: true,
  setSidebar: (sidebar) => set({ sidebar }),
}))
