import { Suspense } from 'react'
import Feature from '@/frontend/features/analytics/AnalyticsPage'

export const metadata = { title: 'Analytics' }

export default function Page() {
  return (
    <Suspense>
      <Feature />
    </Suspense>
  )
}
