import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import { HelmetProvider } from 'react-helmet-async'
import './index.css'
import {App} from './App'
import { installDemoModeIfDemoHost } from './showroom/earlyInstall'

// On the dedicated demo host, install the demo-fetch shim BEFORE React renders,
// so the very first boot probes (AuthProvider's /api/auth/config, Landing's
// /api/readiness) are answered from fixtures instead of 404ing against the
// backend-less vhost. No-op on every other host (dev, prod brochure), where the
// existing showroom-active conditions drive install as before.
installDemoModeIfDemoHost()

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <HelmetProvider>
      <BrowserRouter>
        <App />
      </BrowserRouter>
    </HelmetProvider>
  </StrictMode>,
)