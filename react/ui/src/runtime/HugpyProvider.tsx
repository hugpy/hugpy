// React provider for the hugpy UI runtime config.
//
// Applies its props to the process-wide config (see ./config) synchronously
// during render, so module-level helpers — `hugpyFetch`, `resolveApiUrl`,
// EventSource URLs — see the configured baseUrl before any child effect fires.
// Also exposes the config via context for components that want to react to it.

import React, { createContext, useContext, useMemo } from 'react'
import {
  configureHugpy,
  getHugpyConfig,
  type HugpyRuntimeConfig,
} from './config'

const HugpyContext = createContext<HugpyRuntimeConfig>(getHugpyConfig())

export interface HugpyProviderProps extends Partial<HugpyRuntimeConfig> {
  children?: React.ReactNode
}

export function HugpyProvider({ children, ...cfg }: HugpyProviderProps) {
  const value = useMemo(() => {
    configureHugpy(cfg)
    return getHugpyConfig()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cfg.baseUrl, cfg.credentials, cfg.fetch, cfg.headers])
  return <HugpyContext.Provider value={value}>{children}</HugpyContext.Provider>
}

/** Read the active runtime config reactively from within the provider. */
export function useHugpyConfig(): HugpyRuntimeConfig {
  return useContext(HugpyContext)
}
