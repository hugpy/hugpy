// Placeholder body for a not-yet-built station. Shows what the station will do
// and which roadmap phase makes it real, so the shell is honest about being a
// Phase 0 scaffold — no dead controls pretending to work (the media console's
// "shows features that don't function" lesson, learned once, applied here).
import type { StationSpec } from "./types";

export function PlaceholderStation({ spec }: { spec: StationSpec }) {
  return (
    <section className="station-card">
      <header>
        <span className="station-phase">{spec.phase} · builds {spec.buildsSpec}</span>
        <h2>{spec.title}</h2>
        <p className="station-blurb">{spec.blurb}</p>
      </header>
      <ul className="station-planned">
        {spec.planned.map((item) => (
          <li key={item}>{item}</li>
        ))}
      </ul>
      <footer className="station-foot">
        Planned — see <code>ROADMAP.md</code> ({spec.phase}) and{" "}
        <code>hugpy_video_intelligence_map.md</code> for the schema and job wiring.
      </footer>
    </section>
  );
}
