// HugpyConsole — the full hugpy console as a single drop-in component.
//
// This is the "layered" top of the package: it wraps the existing `Console`
// (the tabbed Models/Add/Compute/API surface) in a <HugpyProvider> so an
// integrator can mount the whole UI with one line:
//
//   <HugpyConsole baseUrl="https://api.hugpy.ai" />
//
// `Console` is router-free, so this needs no <BrowserRouter>. The marketing
// <Landing> and the <Auth> login flow are exported separately (they use
// react-router and are opt-in); compose them yourself if you want the full
// app shell rather than just the console.

import React from 'react'
import { Console } from './App/App'
import { HugpyProvider, type HugpyProviderProps } from './runtime/HugpyProvider'

export type HugpyConsoleProps = Omit<HugpyProviderProps, 'children'>

export function HugpyConsole(props: HugpyConsoleProps) {
  return (
    <HugpyProvider {...props}>
      <Console />
    </HugpyProvider>
  )
}
