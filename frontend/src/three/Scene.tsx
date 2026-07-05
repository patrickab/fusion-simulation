import type { ReactNode } from 'react'
import { Canvas } from '@react-three/fiber'
import { OrbitControls } from '@react-three/drei'

/** Shared canvas: pure black stage, damped orbit with a slow idle drift. All materials are
 *  unlit, so no lights. Children render inside a group mapping the backend's Z-up to Y-up. */
export function Scene({ children, radius = 17 }: { children: ReactNode; radius?: number }) {
  return (
    <Canvas flat camera={{ position: [radius, radius * 0.5, radius], fov: 40 }} dpr={[1, 2]}>
      <color attach="background" args={['#000000']} />
      <group rotation={[-Math.PI / 2, 0, 0]}>{children}</group>
      <OrbitControls makeDefault enableDamping dampingFactor={0.08} autoRotate autoRotateSpeed={0.35} />
    </Canvas>
  )
}
