// Register — the MUI-styled sign-up card (route /register): username, email,
// password fields + "Sign Up". Reads signUp from useAuth(); mode="open" gives
// it a resolved auth context without a config network round-trip.
import { Register, AuthProvider } from '@hugpy/ui'

export function Default() {
  return (
    <AuthProvider mode="open">
      <Register />
    </AuthProvider>
  )
}
