// chainSpec.ts


export interface ChainStep {
  pageKey: string;                          // resolved against pagesRegistry
  // Override field defaults for this step. Sourced fields can be re-pointed
  // (e.g. from "text" to a previous step's output — see ChainRuntime below).
  overrides?: Record<string, string | number | boolean>;
}

export interface ChainSpec {
  key: string;                              // e.g. "video.transcribe.summarize"
  title: string;
  category: string;
  description?: string;
  steps: readonly ChainStep[];
}
