import type { Metadata, Viewport } from 'next'
import './globals.css'
import { AppProvider } from '@/frontend/stores/app'
import { AppShell } from '@/frontend/components/shell/AppShell'

export const metadata: Metadata = {
  title: { default: 'Sightline | Uniform Compliance', template: '%s · Sightline' },
  description: 'AI uniform detection and compliance for uploaded CCTV images and videos.',
  icons: { icon: [{ url: '/icon.svg', type: 'image/svg+xml' }] },
}

export const viewport: Viewport = {
  colorScheme: 'dark light',
  themeColor: '#06151d',
}

// Applies the saved theme before first paint (prevents a light/dark flash).
const THEME_SCRIPT = `(function(){try{var t=JSON.parse(localStorage.getItem('su.theme')||'"dark"');var d=t==='dark'||(t==='system'&&!matchMedia('(prefers-color-scheme: light)').matches);document.documentElement.dataset.theme=d?'dark':'light'}catch(e){document.documentElement.dataset.theme='dark'}})()`

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" data-theme="dark" suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: THEME_SCRIPT }} />
      </head>
      <body>
        <AppProvider>
          <AppShell>{children}</AppShell>
        </AppProvider>
      </body>
    </html>
  )
}
