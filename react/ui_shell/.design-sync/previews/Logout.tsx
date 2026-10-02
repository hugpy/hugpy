// Logout — a fire-and-navigate control: on mount it calls signOut() from
// useAuth() and redirects home. It renders null (no visible UI of its own), so
// we mount it inside a small labeled frame to show it is wired and inert in a
// preview — there is no form/button to display.
import { Logout, AuthProvider } from '@hugpy/ui'

export function Mounted() {
  return (
    <AuthProvider mode="open">
      <div
        style={{
          fontFamily: 'system-ui, sans-serif',
          fontSize: 13,
          color: '#475569',
          border: '1px dashed #cbd5e1',
          borderRadius: 8,
          padding: '12px 16px',
          background: '#f8fafc',
        }}
      >
        Logout is a side-effect-only control (clears the session, then redirects
        home). It renders no visible UI.
        <Logout />
      </div>
    </AuthProvider>
  )
}
