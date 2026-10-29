import { Suspense } from 'react'
import Feature from '@/frontend/features/results/ResultsDashboard'

export const metadata = { title: 'Results Dashboard' }

export default function Page() {
  return (
    <Suspense>
      <Feature />
    </Suspense>
  )
}
