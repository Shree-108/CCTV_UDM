import { Suspense } from 'react'
import Feature from '@/frontend/features/analytics/HistoryPage'

export const metadata = { title: 'Analysis History' }

export default function Page() {
  return (
    <Suspense>
      <Feature />
    </Suspense>
  )
}
