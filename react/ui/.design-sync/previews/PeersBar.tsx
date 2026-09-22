// PeersBar — the top-of-console fleet strip: every storage/compute peer the
// central node knows about, each with its role, mount status, and disk headroom.
// Self-fetches /api/llm/peers (DemoProvider answers from fixtures — central +
// two GPU workers). No props; mirrors src/App/App.jsx's bare <PeersBar />.
import { PeersBar } from '@hugpy/ui'

export function Default() {
  return <PeersBar />
}
