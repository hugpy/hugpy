// DiscordPanel — bind a model to a Discord channel and/or user so the hugpy bot
// routes @mentions there to that model, plus a compose box to push messages out.
// Self-fetches /api/discord/bindings + /channels + /users (DemoProvider answers
// from fixtures); `models` seeds the model picker. `embedded` renders the full
// expanded panel — mirrors src/App/App.jsx's <DiscordPanel embedded models=… />.
import { DiscordPanel } from '@hugpy/ui'
import { MODELS } from '../../src/showroom/fixtures.js'

export function Default() {
  return <DiscordPanel embedded models={MODELS} />
}
