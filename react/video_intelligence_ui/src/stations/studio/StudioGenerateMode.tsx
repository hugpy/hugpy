// The Generate station's "studio" MODE — the in-flow entry point for the fourth tier
// of the image | scene | movie | studio chain. It mounts the SHARED StudioPlane (the
// SAME single plane the standalone Studio station renders): generate controls on top,
// clip viewer + list inline below. A studio job enqueued here is byte-identical to one
// enqueued from the standalone station, and it appears in the plane's inline list +
// viewer without leaving the Generate flow.
//
// This mode does NOT register the studioBridge (registerBridge defaults false): the
// sidebar's "Send to Studio" navigates to the standalone /studio-clips station, which
// owns the registration. The plane is wrapped in the same stacked work-region the other
// Generate modes render into (the classes SectionTabs emits for a no-options work
// column), so switching image → scene → movie → studio reads as one continuous flow.
import { StudioPlane } from "./StudioPlane";

export interface StudioGenerateModeProps {
  /** Set by the top-level Generate tab (Clip → "clip", Cinema → "movie") to lock the
   *  studio surface to one sub-mode and hide its internal Clip|Cinema switcher. */
  lockedSurfaceMode?: "clip" | "movie";
}

export function StudioGenerateMode({ lockedSurfaceMode }: StudioGenerateModeProps = {}) {
  return (
    <div className="vi-section-tabs vi-section-tabs--stacked">
      <section className="vi-section-region vi-section-region--work" aria-label="Studio">
        <StudioPlane lockedSurfaceMode={lockedSurfaceMode} />
      </section>
    </div>
  );
}
