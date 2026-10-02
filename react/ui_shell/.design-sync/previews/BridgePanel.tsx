// BridgePanel — supervise model/keeper ↔ Discord-channel bridges from the
// console: a brain (model or keeper) is allocated to a channel with a directive
// and a defer mode (auto / approve-every-reply / brain-decides), and pending
// replies are approved inline. Self-fetches /api/discord/bridges + /channels
// (DemoProvider answers from fixtures); `models` seeds the model picker.
// `embedded` renders the full expanded panel — mirrors src/App/App.jsx.
import { BridgePanel } from '@hugpy/ui'
import { MODELS } from '../../src/showroom/fixtures.js'

export function Default() {
  return <BridgePanel embedded models={MODELS} />
}
