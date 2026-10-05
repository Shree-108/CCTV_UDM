import { Suspense } from 'react'
import Feature from '@/frontend/features/analysis/AnalysisWorkspace'

export const metadata = { title: 'Image Analysis' }

export default function Page() {
  return (
    <Suspense>
      <Feature mode="image" />
    </Suspense>
  )
}
