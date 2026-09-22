// src/Auth/LoginForm.jsx
import { useState } from 'react'
import { useNavigate, useLocation } from 'react-router-dom'
import { useAuth } from './AuthProvider'
import Navbar from '../components/Navbar/Navbar'
import hugpyLockup from '../assets/hugpy.png'
import './LoginForm.css'

export function LoginForm() {
  const { signIn } = useAuth()
  const navigate = useNavigate()
  const location = useLocation()
  const from = location.state?.from?.pathname || '/console'
  // ?next=/video/ — a FULL-NAVIGATION post-login target for destinations that
  // live outside the router (the intelligence arms). Same-origin absolute
  // paths only, so the login page can never be used as an open redirect.
  const nextQ = new URLSearchParams(location.search).get('next')
  const nextFullNav = nextQ && nextQ.startsWith('/') && !nextQ.startsWith('//') ? nextQ : null
  const [err, setErr] = useState(null)
  const [loading, setLoading] = useState(false)

  const submit = async (e) => {
    e.preventDefault()
    setErr(null); setLoading(true)
    const f = new FormData(e.currentTarget)
    try {
      const ok = await signIn(String(f.get('username') || ''), String(f.get('password') || ''))
      if (ok) {
        if (nextFullNav) window.location.assign(nextFullNav)
        else navigate(from, { replace: true })
      }
      else setErr('Credentials incorrect or sign-in not permitted.')
    } catch { setErr('Login request failed.') }
    finally { setLoading(false) }
  }

  return (
    <>
      <Navbar />
      <div className="login-page">
      <form onSubmit={submit} className="login-card">
        <img className="login-lockup" src={hugpyLockup} alt="hugpy — inference you own" />
        {err && <div className="login-error">{err}</div>}
        <input name="username" placeholder="Username" autoComplete="username" autoFocus disabled={loading} />
        <input name="password" type="password" placeholder="Password" autoComplete="current-password" disabled={loading} />
        <button type="submit" className="btn-primary" disabled={loading}>
          {loading ? 'Signing in…' : 'Sign in'}
        </button>
      </form>
      </div>
    </>
  )
}
