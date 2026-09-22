// DemoBanner — the slim demo-mode strip above the media chat, modeled on the
// dev UI showroom's banner. Announces the mode, lets a visitor flip the data
// source (canned ⇄ live), offers the install command, and links to the live
// platform. Suppressed when embedded (the host showroom provides this chrome).
import { useState } from "react";
import { getDemoConfig, demoUrl } from "./mode";
import { hugpyConfig } from "../config";
import "./demo.css";

const INSTALL_CMD = "pip install hugpy && hugpy serve --port 7002";

export default function DemoBanner(): JSX.Element | null {
  const [copied, setCopied] = useState(false);
  const cfg = getDemoConfig();
  if (cfg.mode === "off" || cfg.embed) return null;

  const live = cfg.mode === "live";
  const copy = () => {
    try {
      navigator.clipboard.writeText(INSTALL_CMD);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      /* clipboard blocked */
    }
  };

  return (
    <div className="mi-demo-banner" role="note">
      <span className={`mi-demo-badge${live ? " is-live" : ""}`}>
        {live ? "LIVE DEMO" : "DEMO"}
      </span>

      <span className="mi-demo-text">
        {live ? (
          <>
            This is the <strong>live</strong> media-intelligence demo — answers
            come from a real hugpy backend
            {cfg.apiBase ? (
              <>
                {" "}
                at <code>{cfg.apiBase}</code>
              </>
            ) : null}
            .
          </>
        ) : (
          <>
            You're exploring with <strong>sample data</strong> — nothing here
            runs a real model or touches a backend.
          </>
        )}
      </span>

      <span className="mi-demo-toggle" role="group" aria-label="Demo data source">
        <a
          className={!live ? "is-current" : ""}
          href={demoUrl("canned")}
          aria-current={!live ? "true" : undefined}
        >
          Canned
        </a>
        <a
          className={live ? "is-current" : ""}
          href={demoUrl("live")}
          aria-current={live ? "true" : undefined}
        >
          Live
        </a>
      </span>

      <button
        type="button"
        className="mi-demo-cmd"
        onClick={copy}
        title="Copy install command"
      >
        <code>{INSTALL_CMD}</code>
        <span className="mi-demo-cmd-ico">{copied ? "✓ copied" : "⧉"}</span>
      </button>

      <a
        className="mi-demo-cta"
        href={hugpyConfig.siteUrl}
        target="_blank"
        rel="noreferrer"
      >
        Get hugpy →
      </a>
    </div>
  );
}
