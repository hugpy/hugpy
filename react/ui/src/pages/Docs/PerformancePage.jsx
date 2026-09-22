// Performance — the serialization tax (a deliberately shameful page).
// Measured 2026-09-18: the single biggest hole in the hugpy call pipeline is
// that llama-server slots run one sequence at a time (no --parallel / -cb / -fa),
// so concurrent calls to a warm model serialize. This page exists to be
// impossible to ignore. See memory: hugpy-slot-serialization.
export const PERF_TOC = [
  ['perf-shame', 'Read this and wince', []],
  ['perf-evidence', 'The receipts (measured)', []],
  ['perf-why', 'Why it happens', []],
  ['perf-kills', 'It also kills calls', []],
  ['perf-fix', 'The fix', [['perf-flags', 'The flags'], ['perf-size', 'Sizing']]],
  ['perf-verify', 'Verify it yourself', []],
]

const banner = {
  border: '2px solid #d9534f',
  background: 'rgba(217,83,79,0.10)',
  borderRadius: 10,
  padding: '14px 16px',
  margin: '8px 0 20px',
  color: '#d9534f',
  fontWeight: 600,
  lineHeight: 1.5,
}

export default function PerformancePage() {
  return (
    <article className="docs-article">
      <h1>Performance — the serialization tax</h1>

      <section id="perf-shame" className="docs-sec">
        <div style={banner} role="alert">
          ⚠ Every warm model serves <strong>one request at a time</strong>. The
          moment your tools, the station, background heartbeats, and you share a
          model, calls line up single-file and latency stacks ~2.1s per queued
          call. Raw <code>hugpy ask</code> feels fast because it’s the only call
          on an idle slot. This is the molasses. Fix it and forget it — or forget
          it and re-suffer it. This page is here so you can’t.
        </div>
        <p>
          It is <strong>not</strong> the resolver, the Anthropic shim, or the
          network — those are all clean (~0.43&nbsp;s). It is llama.cpp running at
          its single-slot default. One flag change clears the chute.
        </p>
      </section>

      <section id="perf-evidence" className="docs-sec">
        <h2 className="docs-h2">The receipts (measured)</h2>
        <p>Local <code>POST /v1/chat/completions</code>, warm model, 2026-09-18:</p>
        <ul>
          <li><strong>Solo warm call:</strong> <code>~0.43&nbsp;s</code> (a non-canonical name resolves in the same 0.43&nbsp;s — the resolver is not the hole).</li>
          <li><strong>4 concurrent calls, same warm model:</strong> <code>0.6&nbsp;s → 2.7&nbsp;s → 4.8&nbsp;s → 6.9&nbsp;s</code>. Pure serialization: each waits for the one before, ~2.1&nbsp;s added per queued call.</li>
        </ul>
        <p>
          Worse than plain queuing: solo is 0.43&nbsp;s but the marginal cost
          under load is ~2.1&nbsp;s/call — contention <em>also</em> degrades
          per-call throughput, the signature of no continuous batching.
        </p>
      </section>

      <section id="perf-why" className="docs-sec">
        <h2 className="docs-h2">Why it happens</h2>
        <p>
          The slot <code>llama-server</code> is spawned without{' '}
          <code>--parallel</code>/<code>-np</code>,{' '}
          <code>--cont-batching</code>/<code>-cb</code>, or{' '}
          <code>-fa</code> (flash-attention) — confirmed absent in{' '}
          <code>managers/llama/runners/</code>. So it runs the llama.cpp default{' '}
          <code>-np 1</code>: one sequence, no continuous batching. A single
          loaded model therefore cannot overlap requests on the GPU; they queue.
        </p>
      </section>

      <section id="perf-kills" className="docs-sec">
        <h2 className="docs-h2">It also kills calls</h2>
        <p>
          Serialization isn’t just slow — under real load the queue depth × per-request
          time blows past client timeouts and the call is dropped. Stack the
          4-concurrent <strong>cold-load cap</strong> (<code>503 cold_load_capacity</code>)
          and <code>500 “cannot serve”</code> on top, and that’s your kill sources.
          The single-slot default is the root of both the latency and the drops.
        </p>
      </section>

      <section id="perf-fix" className="docs-sec">
        <h2 className="docs-h2">The fix</h2>
        <div className="docs-h3-block" id="perf-flags">
          <h3 className="docs-h3">The flags</h3>
          <p>Add to the slot <code>llama-server</code> spawn:</p>
          <ul>
            <li><code>--parallel N -cb</code> — N concurrent sequences with continuous batching. One loaded model now serves N requests batched on the GPU instead of queued. This is the single biggest win: the linear stack goes near-flat up to N.</li>
            <li><code>-fa</code> — flash-attention: faster prefill/decode <em>and</em> a smaller KV cache, which is what pays for the parallel slots and keeps context.</li>
          </ul>
        </div>
        <div className="docs-h3-block" id="perf-size">
          <h3 className="docs-h3">Sizing — derived per node, not hardcoded</h3>
          <p>
            <code>N</code> is <strong>derived per node</strong> at slot spawn
            (<code>managers/serve/slot_agent.py → _slot_parallel()</code>): from
            this card’s <em>free VRAM headroom after the model</em>, budgeting a
            ctx-scaled KV cache per sequence, capped by CPU cores and a ceiling.
            So a 4×3090 / 128-core box earns far more concurrency than an
            8&nbsp;GB 4060 — no per-node hardcoding. <code>-c</code> is scaled by
            <code>N</code> so each sequence keeps its full context.
          </p>
          <p>Every constant is an <strong>env-overridable fallback</strong>, never a hardcode:</p>
          <ul>
            <li><code>HUGPY_SLOT_PARALLEL=N</code> — force N, bypassing derivation entirely.</li>
            <li><code>HUGPY_SLOT_PARALLEL_MAX</code> — ceiling on the derived N (fallback 8).</li>
            <li><code>HUGPY_SLOT_KV_FRACTION</code> — fraction of headroom spent on KV (fallback 0.7, leaves margin).</li>
            <li><code>HUGPY_SLOT_KV_BYTES_PER_TOKEN</code> — per-seq KV cost estimate (fallback 160&nbsp;KiB/token).</li>
            <li><code>HUGPY_SLOT_FLASH_ATTN=1</code> — enable <code>--flash-attn</code> (smaller KV → more slots), gated on binary support.</li>
          </ul>
          <p>All of it is gated on <code>llama-server</code> flag support and takes effect on the next slot respawn.</p>
        </div>
      </section>

      <section id="perf-verify" className="docs-sec">
        <h2 className="docs-h2">Verify it yourself</h2>
        <p>Fire four at once at a warm model and watch them stack (or, once fixed, stay flat):</p>
        <pre><code>{`B=http://127.0.0.1:7002
body(){ printf '{"model":"%s","max_tokens":8,"messages":[{"role":"user","content":"hi"}]}' "$1"; }
for i in 1 2 3 4; do
  ( curl -s -o /dev/null -w "%{time_total}s\\n" \\
      -H "Authorization: Bearer $HUGPY_KEY" -H "content-type: application/json" \\
      -d "$(body Qwen3-Coder-Next-GGUF)" "$B/v1/chat/completions" ) &
done; wait`}</code></pre>
        <p>
          Before the fix: ~0.6 / 2.7 / 4.8 / 6.9&nbsp;s. After: four times ≈ the
          solo latency. If it still stacks, the slot didn’t respawn with the flags.
        </p>
      </section>
    </article>
  )
}
