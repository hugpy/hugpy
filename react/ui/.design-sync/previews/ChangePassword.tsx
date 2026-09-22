// ChangePassword — the MUI-styled credential-rotation card (route
// /change-password): current / new / verify password fields + "Change
// Password". Reads changePassword from useAuth(); mode="open" resolves the
// auth context locally with no config round-trip.
import { ChangePassword, AuthProvider } from '@hugpy/ui'

export function Default() {
  return (
    <AuthProvider mode="open">
      <ChangePassword />
    </AuthProvider>
  )
}
