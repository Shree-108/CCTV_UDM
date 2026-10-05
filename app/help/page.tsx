import { Suspense } from 'react'
import Feature from '@/frontend/features/help/HelpPage'

export const metadata = { title: 'Help Center' }

export default function Page() {
  return (
    <Suspense>
      <Feature />
    </Suspense>
  )
}
