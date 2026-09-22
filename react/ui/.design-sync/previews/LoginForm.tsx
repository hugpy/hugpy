// LoginForm — the dependency-light sign-in card (route /login on the public
// front door): the hugpy lockup, username + password inputs, and a primary
// "Sign in" button, with the app Navbar above it. CSS-styled (no MUI). Reads
// signIn from useAuth(); mode="open" resolves the auth context locally.
import { LoginForm, AuthProvider } from '@hugpy/ui'

export function Default() {
  return (
    <AuthProvider mode="open">
      <LoginForm />
    </AuthProvider>
  )
}
