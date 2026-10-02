// PhoneBrickPanel — the on-prem phone camera pool for distributed video
// analytics. Self-fetches /api/phone-brick/phones (DemoProvider answers from
// fixtures: three online cams with a model loaded + queue depth, one offline),
// then offers an image upload + "Run detection" fan-out. `embedded` renders the
// full expanded panel; mirrors src/App/App.jsx's <PhoneBrickPanel embedded />.
import { PhoneBrickPanel } from '@hugpy/ui'

export function Default() {
  return <PhoneBrickPanel embedded />
}
