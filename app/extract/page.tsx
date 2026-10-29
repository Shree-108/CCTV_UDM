import { Suspense } from 'react'
import Feature from '@/frontend/features/extraction/ExtractionPage'

export const metadata = { title: 'Frame Extraction' }

export default function Page() {
  return (
    <Suspense>
      <Feature />
    </Suspense>
  )
}
