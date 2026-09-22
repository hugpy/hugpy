/*
 * BrandMark — the canonical hugpy brand mark for the video-intelligence arm.
 *
 * Replicates the sitewide BrandMark 1:1 — the hexagon symbol + the two-row
 * "H U G P Y" / hairline rule / "inference you own" lockup (same class names +
 * a copied BrandMark.css) — so the brand reads identically across every page.
 * The arm is a separate SPA served under /video/, so this is a plain <a> (NOT a
 * react-router Link): clicking it leaves the arm and returns to the site root.
 *
 * Destination follows the SAME runtime logic as the Navbar's site links: on a
 * hugpy origin the home page is at the CURRENT origin's root; on any other
 * origin it falls back to the public platform (hugpyConfig.siteUrl).
 *
 * Ported from media_intelligence_ui/src/chat/src/ui/BrandMark.tsx, dropping the
 * demo/mode (`isEmbedded`) import — this arm has no iframe showroom.
 */
import markUrl from "../assets/hugpy-mark.png";
import { hugpyConfig } from "../config";
import "./BrandMark.css";

// Runtime hostname, not build-time base: on a hugpy origin return to that
// origin's root; otherwise the public platform (see Navbar.tsx for the full
// rationale).
const onHugpyOrigin =
  typeof window !== "undefined" &&
  /(^|\.)hugpy\.ai$/i.test(window.location.hostname);
const welcomeHref = onHugpyOrigin ? "/" : hugpyConfig.siteUrl;

export default function BrandMark() {
  return (
    <a href={welcomeHref} title="Back to the welcome page" className="brandmark">
      <img className="brandmark-img" src={markUrl} alt="" />
      <span className="brandmark-text">
        <span className="brandmark-word">HUGPY</span>
        <span className="brandmark-rule" aria-hidden="true" />
        <span className="brandmark-sub">inference you own</span>
      </span>
    </a>
  );
}
