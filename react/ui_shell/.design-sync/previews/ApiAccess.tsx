// ApiAccess — the programmatic-access panel: OpenAI-compatible endpoint, the
// /v1 auth gate, API-key minting + revoke table, the separate media-intelligence
// gate, and a copy-paste curl example. Self-fetches /api/keys + /api/ml/gate
// (DemoProvider answers from fixtures: two live keys, gates open); `models` seeds
// the example model in the curl. `embedded` renders the full expanded panel —
// mirrors src/App/App.jsx's <ApiAccess embedded models={models} />.
import { ApiAccess } from '@hugpy/ui'
import { MODELS } from '../../src/showroom/fixtures.js'

export function Default() {
  return <ApiAccess embedded models={MODELS} />
}
