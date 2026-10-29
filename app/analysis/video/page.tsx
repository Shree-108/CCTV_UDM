import { Suspense } from 'react'
import Feature from '@/frontend/features/analysis/VideoAnalysisPage'

export const metadata = { title: 'Video Analysis' }

export default function Page() {
  return (
    <Suspense>
      <Feature />
    </Suspense>
  )
}
