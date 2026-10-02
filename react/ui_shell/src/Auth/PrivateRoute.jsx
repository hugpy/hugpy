// src/Auth/PrivateRoute.jsx
import { Navigate, Outlet, useLocation } from 'react-router-dom'
import { useAuth } from './AuthProvider'

export function PrivateRoute() {
  const { state } = useAuth()
  const location = useLocation()

  if (state.status === 'checking') {
    return <div style={{ padding: 24, color: '#888' }}>Checking session…</div>
  }
  if (state.status === 'authed') {
    return <Outlet />
  }
  // guest (external auth, not signed in) → the front door (welcome). "Open the
  // console" there continues into /login, which remembers where they were
  // headed. Open-mode instances are always authed, so they never reach here.
  return <Navigate to="/" replace state={{ from: location }} />
}