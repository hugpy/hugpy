// Canned IDENTITY PROFILE for the video-intelligence demo (?demo=1): the real
// "luigi" identity (12 refs → 72-view turntable → textured GLB), served from the
// HOSTED demo-media tree (canonical copy /var/www/hugpy-media/identities/luigi/).
//
// CRITICAL: every asset path here is a demoMedia() URL. config.ts's
// mediaBytesUrl() canned-mode bypass passes absolute http(s) URLs straight
// through to the browser, but any "/mnt/"-prefixed store handle would be routed
// back to the API — where the demo shim honestly 501s. Shape mirrors the
// server's _public() projection (zod: src/stations/studio/useIdentityProfiles.ts).
import type { IdentityProfile } from "../stations/studio/useIdentityProfiles";
import { demoMedia } from "./mediaBase";

const pad2 = (n: number) => String(n).padStart(2, "0");

const luigiRefs = Array.from({ length: 12 }, (_, i) =>
  demoMedia(`identities/luigi/refs/ref_${pad2(i)}.jpeg`),
);
const luigiCanonical = Array.from({ length: 4 }, (_, i) =>
  demoMedia(`identities/luigi/canonical/ref_${pad2(i)}.jpg`),
);
const luigiViews = Array.from({ length: 72 }, (_, i) =>
  demoMedia(`identities/luigi/views/view_${pad2(i)}.jpg`),
);

export const demoIdentityLuigi: IdentityProfile = {
  slug: "luigi",
  name: "Luigi",
  created_at: 1784100737.0394623,
  notes: "",
  reference_images: luigiRefs,
  canonical: luigiCanonical,
  canonical_angles: [0.0, 90.0, 180.0, 270.0],
  // The TEXTURED version — NOT the clay base — so the station's activeRecon
  // resolution (active_version → version.recon_id → reconstructions[]) lands on
  // the turntable recon with the done mesh below.
  active_version: "ver_851e3e418dc3869a",
  versions: [
    {
      version_id: "ver_d9f3bf88e5fc7acd",
      name: "base",
      kind: "clay",
      recon_id: "identity_ace73ab07078c529",
      created_at: 1784101503.040569,
      canonical: [],
      notes: "",
    },
    {
      version_id: "ver_851e3e418dc3869a",
      name: "textured-01",
      kind: "textured",
      recon_id: "identity_64853ed017380507",
      created_at: 1784101750.5200672,
      canonical: luigiCanonical,
      notes: "",
    },
  ],
  reconstructions: [
    {
      recon_id: "identity_64853ed017380507",
      mode: "turntable",
      created_at: 1784101750.4968696,
      frame_count: 72,
      degrees_per_frame: 5.0,
      job_id: "b31f8a1eb4e94d4faf8552cb83012290",
      views: luigiViews,
      mesh: {
        status: "done",
        error: null,
        textured: true,
        frame_count: 72,
        auto_promoted: true,
        glb_path: demoMedia("identities/luigi/identity.glb"),
        video_path: demoMedia("identities/luigi/turntable.mp4"),
        version_id: "ver_851e3e418dc3869a",
        version_kind: "textured",
        version_name: "textured-01",
        job_id: "b31f8a1eb4e94d4faf8552cb83012290",
      },
    },
  ],
  gen_settings: {
    texture: true,
    pose: "t-pose",
    frame_count: 72,
    fps: 24,
    width: 768,
    height: 768,
    auto_promote: true,
    front_ref: null,
    remove_background: true,
    vision_model: "Qwen2.5-VL-3B-Instruct-GGUF",
    cleanup_prompt: "",
    negative_prompt: "",
  },
  authorization: {},
};
