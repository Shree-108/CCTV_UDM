import { Suspense } from 'react'
import Feature from '@/frontend/features/alerts/AlertsPage'

export const metadata = { title: 'Alerts' }

export default function Page() {
  return (
    <Suspense>
      <Feature />
    </Suspense>
  )
}
