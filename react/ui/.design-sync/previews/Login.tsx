// Login — the MUI-styled sign-in card (route /login). Reads signIn from
// useAuth(), so it needs an AuthProvider ancestor; mode="open" forces the
// local single-operator posture and skips the GET /api/auth/config round-trip.
import { Login, AuthProvider } from '@hugpy/ui'

export function Default() {
  return (
    <AuthProvider mode="open">
      <Login />
    </AuthProvider>
  )
}
