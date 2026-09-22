// ChatPanel — the per-model chat surface (header + allocation banner + message
// transcript + composer). Fully prop-driven: App lifts the conversation into a
// per-model, localStorage-backed `messages` array and passes it down with a
// setMessages/onClose pair (see src/App/App.jsx ~line 301). We hand it a literal
// multi-turn transcript so it renders fully without any streaming, mirroring the
// canonical composition. Message shape (from ChatPanel.jsx): { role, content,
// model?, attachment? }. Voice/content adapted from the CHAT_REPLAY fixture.
import { ChatPanel } from '@hugpy/ui'
import { MODELS } from '../../src/showroom/fixtures.js'

const noop = () => {}

const TEXT_MODEL = MODELS.find(m => m.model_key === 'Qwen2.5-3B-Instruct-GGUF')
const VL_MODEL = MODELS.find(m => m.model_key === 'Qwen2.5-VL-3B-Instruct-GGUF')

// A tiny inline SVG so the attached-image thumbnail renders without a backend.
const PHOTO =
  "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='96' height='72'%3E%3Crect width='96' height='72' fill='%23161b22'/%3E%3Crect x='36' y='24' width='24' height='34' rx='5' fill='%2358a6ff'/%3E%3Ccircle cx='48' cy='16' r='8' fill='%23d29922'/%3E%3C/svg%3E"

export function Conversation() {
  // Two-turn text conversation on the installed 3B chat model.
  const messages = [
    { role: 'user', content: 'What is hugpy, in one paragraph?' },
    {
      role: 'assistant',
      model: TEXT_MODEL?.name,
      content:
        'Hugpy unifies mixed hardware — a workstation GPU, an idle laptop, even a phone on llama.cpp — behind a single OpenAI-compatible API. A central node holds the model registry and a live worker pool, and routes each chat request at dispatch time to whichever worker has that model assigned, falling back to local inference when none can serve it. Replies stream token-by-token over SSE and auto-continue past any token cap, so long answers are never cut off.',
    },
    { role: 'user', content: 'Why would I run it instead of a cloud API?' },
    {
      role: 'assistant',
      model: TEXT_MODEL?.name,
      content:
        '• Cost: reuse hardware you already own — no per-token cloud bills, and idle laptops or phones become capacity instead of e-waste.\n• Control: models, prompts, and data never leave your network; the gateway is OpenAI-compatible, so existing SDKs and tooling work unchanged.\n• Resilience: dispatch picks a healthy worker per request and falls back to local inference, so one node dying doesn’t take chat down.',
    },
  ]
  return (
    <ChatPanel
      modelKey={TEXT_MODEL?.model_key}
      model={TEXT_MODEL}
      messages={messages}
      setMessages={noop}
      onClose={noop}
    />
  )
}

export function VisionChat() {
  // Vision-language model (header shows the 🖼 VL tag) with an attached image
  // on the user turn — exercises the image-thumbnail + multimodal path.
  const messages = [
    {
      role: 'user',
      content: 'What protective equipment is the worker on the floor missing?',
      attachment: { name: 'site_floor_2026-06-27.jpg', isImage: true, dataUrl: PHOTO },
    },
    {
      role: 'assistant',
      model: VL_MODEL?.name,
      content:
        'Looking at the floor-cam frame: the worker has a high-visibility vest and gloves, but the head is uncovered — no hard hat — and I don’t see eye protection either. The helmet detector flagged this same frame as no-helmet at 78% confidence, which matches what’s visible. The vest and boots look compliant.',
    },
  ]
  return (
    <ChatPanel
      modelKey={VL_MODEL?.model_key}
      model={VL_MODEL}
      messages={messages}
      setMessages={noop}
      onClose={noop}
    />
  )
}
