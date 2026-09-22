// DemoBanner — the slim demo-mode strip above the video stations, modeled on
// the media arm's banner MINUS the canned⇄live flip: the video arm has no live
// demo (jobs would drive real worker GPUs — see mode.ts). Announces sample-data
// mode, offers the install command, links to the platform. Suppressed when
// embedded (the host showroom provides this chrome).
import { useState } from "react";
import { getDemoConfig } from "./mode";
import { hugpyConfig } from "../config";
import "./demo.css";

const INSTALL_CMD = "pip install hugpy && hugpy serve --port 7002";

export default function DemoBanner() {
  const [copied, setCopied] = useState(false);
  const cfg = getDemoConfig();
  if (cfg.mode !== "canned" || cfg.embed) return null;

  const copy = () => {
    try {
      void navigator.clipboard.writeText(INSTALL_CMD);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      /* clipboard blocked */
    }
  };

  return (
    <div className="vi-demo-banner" role="note">
      <span className="vi-demo-badge">DEMO</span>
      <span className="vi-demo-text">
        You&apos;re exploring with <strong>sample data</strong> — nothing here
        runs a real model or touches a backend. Run it for real:
      </span>
      <code
        className="vi-demo-cmd"
        title="Click to copy"
        onClick={copy}
        role="button"
        tabIndex={0}
      >
        {copied ? "copied ✓" : INSTALL_CMD}
      </code>
      <a
        className="vi-demo-link"
        href={hugpyConfig.siteUrl}
        target="_blank"
        rel="noreferrer"
      >
        hugpy.ai →
      </a>
    </div>
  );
}
