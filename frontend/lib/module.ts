/**
 * Which Yatrikart AI module this dashboard is. Uniform Compliance and Litter Detection are
 * separate applications with separate models; the module only drives the product name shown
 * under the YATRIKART brand (sidebar, browser tab).
 *
 * Set NEXT_PUBLIC_APP_MODULE in .env.local (uniform | litter). Defaults to uniform - this
 * repository is the Uniform Compliance application.
 */
export type AppModule = 'uniform' | 'litter'

export const MODULES: Record<AppModule, { name: string }> = {
  uniform: { name: 'Uniform Compliance' },
  litter: { name: 'Litter Detection' },
}

const env = process.env.NEXT_PUBLIC_APP_MODULE as AppModule | undefined
export const ACTIVE_MODULE: AppModule = env && env in MODULES ? env : 'uniform'
export const MODULE_NAME = MODULES[ACTIVE_MODULE].name
export const BRAND = 'Yatrikart'
