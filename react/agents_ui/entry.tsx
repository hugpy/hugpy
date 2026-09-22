// Standalone bootstrap for the agent-fleet arm. The dev UI links to /fleet;
// this mounts the S2 overview page there. A BrowserRouter (basename from the
// Vite base) is provided up front since later slices (S3+) add routed
// sub-pages (/fleet/nodes, /fleet/start — see dev/AGENTS-ARM-PLAN.md §2). S2
// itself is a single route — the overview page — no node board yet.
import React from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import FleetOverview from "./src/FleetOverview";
// Sitewide floating Help button. ONE shared framework-free DOM module mounted
// identically by all four surfaces (main SPA + media/video/fleet arms) — see
// ../ui_shared/help/helpWidget.js for why it isn't a copied React component.
import { mountHelpWidget } from "../ui_shared/help/helpWidget";
import "./src/style/app.css";

const el = document.getElementById("root");
if (!el) throw new Error("agents_ui (fleet): #root not found");

// Router basename must match where the bundle is SERVED, which equals the Vite
// build base: "/fleet/" for the in-hugpy build (served at /fleet), "/" for a
// standalone deploy. A mismatched basename makes React-Router render nothing.
const basename = import.meta.env.BASE_URL.replace(/\/$/, "") || "/";

// Mounted outside the React tree (its own fixed-position node on <body>), so it
// is unaffected by routing and by the arm's Tailwind layout.
mountHelpWidget({ apiBase: "/api", surface: "fleet" });

createRoot(el).render(
  <React.StrictMode>
    <BrowserRouter basename={basename}>
      <Routes>
        <Route path="/" element={<FleetOverview />} />
      </Routes>
    </BrowserRouter>
  </React.StrictMode>,
);
