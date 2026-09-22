/*
 * BrandMark — the canonical hugpy brand mark for the media-intelligence arm.
 *
 * Replicates the main site's BrandMark 1:1 — the hexagon symbol + the two-row
 * "H U G P Y" / hairline rule / "inference you own" lockup (same class names +
 * a copied BrandMark.css) — so the brand reads identically across every page.
 * The arm is a separate SPA served under /media/, so this is a plain <a> (NOT a
 * react-router Link): clicking it leaves the arm and returns to the welcome
 * screen.
 *
 * Destination follows the SAME runtime logic as the Navbar's site links: on a
 * hugpy origin the welcome page is at the CURRENT origin's root — so
 * dev.hugpy.ai/media returns to dev.hugpy.ai and hugpy.ai/media to hugpy.ai,
 * matching wherever the visitor came from. On any other origin (e.g. the
 * hugpy-chat HF Space, which only serves /media + /api) there is no local
 * welcome page, so it falls back to the public platform (hugpyConfig.siteUrl).
 */
import markUrl from "../../../assets/hugpy-mark.png";
import { hugpyConfig } from "../../../config";
import { isEmbedded } from "../../../demo/mode";
import "./BrandMark.css";

// Runtime hostname, not build-time base: one build is served both directly at
// dev.hugpy.ai/media and via reverse proxy at the hugpy-chat HF Space — only the
// live hostname distinguishes them (see Navbar.tsx for the full rationale).
const onHugpyOrigin =
  typeof window !== "undefined" &&
  /(^|\.)hugpy\.ai$/i.test(window.location.hostname);
const welcomeHref = onHugpyOrigin ? "/" : hugpyConfig.siteUrl;

export default function BrandMark(): JSX.Element {
  // Embedded in the showroom <iframe>: escape the frame so the welcome page
  // loads at top level (matches the Navbar's site links).
  const target = isEmbedded() ? "_top" : undefined;
  return (
    <a href={welcomeHref} target={target} title="Back to the welcome page" className="brandmark">
      <img className="brandmark-img" src={markUrl} alt="" />
      <span className="brandmark-text">
        <span className="brandmark-word">HUGPY</span>
        <span className="brandmark-rule" aria-hidden="true" />
        <span className="brandmark-sub">inference you own</span>
      </span>
    </a>
  );
}
