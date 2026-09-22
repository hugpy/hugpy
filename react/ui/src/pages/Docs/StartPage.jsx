// Start here — what hugpy does in plain words + the three-command quickstart.
// Keeps the historical `quickstart` slug and its section ids (serve / call /
// fleet / auth / next) so old deep links keep working.
import { CodeTabs, C, S, K } from './docParts'

export const START_TOC = [
  ['what', 'What you can do with hugpy', []],
  ['serve', '1 · Get it running', []],
  ['call', '2 · Call it from code', []],
  ['fleet', '3 · Add more machines', []],
  ['auth', 'Signing in & API keys', []],
  ['next', 'Where to go next', []],
]

export default function StartPage() {
  return (
    <>
      <div className="docs-crumbs">Start here</div>
      <h1>Start here</h1>
      <p className="docs-lede">
        hugpy turns your own computers into a private AI service — your models, your hardware,
        one console.
      </p>

      <section id="what" className="docs-sec">
        <h2>What you can do with hugpy</h2>
        <ul className="docs-routes">
          <li><strong>Chat with local models</strong> — pick any model you&apos;ve downloaded and talk to it in the console, streaming, with images if the model can see. <a className="docs-inline-link" href="#console-chat">Chat with a model</a></li>
          <li><strong>Pull models from Hugging Face</strong> — search the Hub (or Civitai for image checkpoints), pick a variant that fits your disk, and download it with resume and progress. <a className="docs-inline-link" href="#console-models">Find &amp; download models</a></li>
          <li><strong>Spread models across your machines</strong> — join any GPU box to the fleet and place models on it from the console; hugpy routes each request to a machine that has the model. <a className="docs-inline-link" href="#console-workers">Put models on workers</a></li>
          <li><strong>Keep the ones you care about warm</strong> — decide which models stay loaded and ready versus loading on demand. <a className="docs-inline-link" href="#console-serving">Keep models serving</a></li>
          <li><strong>Generate images</strong> — run Stable Diffusion checkpoints or diffusers image models on your GPU workers. <a className="docs-inline-link" href="#console-images">Generate images</a></li>
          <li><strong>Wire it to Discord</strong> — bind models to channels, supervise their replies, and hand agents a scoped token to talk in one channel. <a className="docs-inline-link" href="#console-discord">Discord &amp; comms sessions</a></li>
          <li><strong>Use it like OpenAI</strong> — every model is also behind an OpenAI-compatible API, so existing SDKs and tools just work (see below).</li>
        </ul>
      </section>

      <section id="serve" className="docs-sec">
        <h2>1 · Get it running</h2>
        <p>Two commands. One process serves both the API and this console on port <code>7002</code>.</p>
        <pre className="docs-block"><code>{`pip install hugpy
hugpy serve --port 7002`}</code></pre>
        <p>
          <strong>You should see</strong> the console at <code>http://localhost:7002</code>. The API
          lives under <code>/api/v1</code> on the same port. By default the server listens on every
          network interface (pass <code>--host 127.0.0.1</code> to keep it private to this machine).
        </p>
        <p className="docs-note">
          The released package on PyPI is <code>hugpy</code>; the bleeding-edge dev build is published
          alongside as <code>abstract_hugpy_dev</code>. Both install the same <code>hugpy</code> command.
        </p>
      </section>

      <section id="call" className="docs-sec">
        <h2>2 · Call it from code</h2>
        <p>
          Point any OpenAI SDK at your box. Mint an API key in the console&apos;s <strong>API</strong> tab
          (keys look like <code>hp_…</code>), or run in open mode with no key at all.
        </p>
        <CodeTabs tabs={[
          { label: 'Python', lang: 'python', code: <>{K('from')} openai {K('import')} OpenAI{'\n\n'}client = OpenAI({'\n'}    base_url={S('"http://localhost:7002/api/v1"')},{'\n'}    api_key={S('"hp_…"')},  {C('# or open mode')}{'\n'}){'\n\n'}stream = client.chat.completions.create({'\n'}    model={S('"…"')},{'\n'}    messages=[{'{'}{S('"role"')}: {S('"user"')}, {S('"content"')}: {S('"hi"')}{'}'}],{'\n'}    stream={K('True')},{'\n'}){'\n'}{K('for')} chunk {K('in')} stream:{'\n'}    print(chunk.choices[0].delta.content {K('or')} {S('""')}, end={S('""')})</> },
          { label: 'Shell', lang: 'curl', code: <>curl http://localhost:7002/api/v1/chat/completions \{'\n'}  -H {S('"Authorization: Bearer hp_…"')} \{'\n'}  -H {S('"Content-Type: application/json"')} \{'\n'}  -d {S('\'{"model":"…","messages":[{"role":"user","content":"hi"}],"stream":true}\'')}</> },
        ]} />
        <p>
          <code>/v1/models</code> lists what you can call; <code>/v1/chat/completions</code> streams
          tokens. Long answers are continued automatically — responses are not cut off at a token cap.
        </p>
      </section>

      <section id="fleet" className="docs-sec">
        <h2>3 · Add more machines</h2>
        <p>Join any other box — hugpy registers it, keeps a heartbeat, and routes requests to it.</p>
        <pre className="docs-block"><code>{`# on any GPU box — pip install hugpy first
hugpy worker --central http://your-hugpy:7002`}</code></pre>
        <p>
          <strong>You should see</strong> the new box appear in the console&apos;s <strong>Compute</strong> tab
          marked <em>pending</em>. It does not serve anything until you click
          <strong> ✓ admit</strong> — that approval step is yours. The easier path is the one-line
          install command the Compute tab generates for you; see{' '}
          <a className="docs-inline-link" href="#console-workers">Put models on workers</a>.
        </p>
      </section>

      <section id="auth" className="docs-sec">
        <h2>Signing in &amp; API keys</h2>
        <p>
          <code>hugpy serve</code> starts in <strong>open</strong> mode: you own the box, so there is no
          login wall. For a shared deployment, external login can be switched on (the console then
          delegates sign-in to your auth service). API keys are minted in the <strong>API</strong> tab —
          each key is shown once, stored hashed, and a <em>require key</em> toggle can gate the whole
          API site-wide.
        </p>
      </section>

      <section id="next" className="docs-sec">
        <h2>Where to go next</h2>
        <ul className="docs-routes">
          <li><a href="#console-chat" className="docs-inline-link">Using the console</a> — task-by-task guides for everything above.</li>
          <li><a href="#installation" className="docs-inline-link">Run it yourself</a> — GPU/CUDA setup, building the native engine, storage paths.</li>
          <li><a href="#troubleshooting" className="docs-inline-link">Fix it</a> — symptom-by-symptom fixes (config, routing, memory, workers, Discord) and a worker-box field guide.</li>
          <li><a href="#architecture" className="docs-inline-link">Reference</a> — how it all works inside, for developers.</li>
        </ul>
      </section>

      <div className="docs-pager">
        <a className="docs-pager-card" href="#console-chat"><span className="d">Next</span><span className="t">Chat with a model →</span></a>
      </div>
    </>
  )
}
