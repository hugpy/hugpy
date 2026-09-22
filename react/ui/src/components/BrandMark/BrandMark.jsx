// BrandMark — the canonical hugpy brand mark (symbol + wordmark + tagline).
//
// One static, consistent component used top-left across every page (console,
// landing, docs). Clicking it always returns to the welcome screen ("/" →
// <Landing>). Standardized on the console's mark (the hugpy-mark.png symbol +
// "hugpy" + the "inference you own" tagline) so the brand reads identically
// everywhere. Self-contained CSS keeps it pixel-consistent regardless of the
// host page's stylesheet.

import { Link } from 'react-router-dom'
import hugpyMark from '../../assets/hugpy-mark.png'
import './BrandMark.css'

export default function BrandMark({ word = 'HUGPY', tagline = 'inference you own', className = '' }) {
  return (
    <Link to="/" className={`brandmark ${className}`} title="Back to the welcome page">
      <img className="brandmark-img" src={hugpyMark} alt="" />
      <span className="brandmark-text">
        <span className="brandmark-word">{word}</span>
        {tagline && <span className="brandmark-rule" aria-hidden="true" />}
        {tagline && <span className="brandmark-sub">{tagline}</span>}
      </span>
    </Link>
  )
}
