// VM Pool — the central VM library + single-VM run host, exposed as a toolserver
// API and a light dashboard at vms.hugpy.ai. Static reference (no live fetch):
// documents the architecture, the pool, the /vmpool HTTP API, and how to run/view
// a VM from any LAN/WireGuard client. Added 2026-08-31.
// 2026-09-01: documented /vmpool/exec (guest-agent exec, no SSH) + friendly-name state/ip/console.
import { CopyBlock } from './docParts'

export const VMPOOL_TOC = [
  ['vmpool-overview', 'Overview', []],
  ['vmpool-architecture', 'Architecture', [
    ['vmpool-hosts', 'Hosts & roles'],
    ['vmpool-model', 'Golden-master model'],
  ]],
  ['vmpool-pool', 'The pool', []],
  ['vmpool-images', 'Image conventions (for agents)', []],
  ['vmpool-dashboard', 'Dashboard', []],
  ['vmpool-api', 'HTTP API', [
    ['vmpool-list', 'GET /vmpool/list'],
    ['vmpool-status', 'GET /vmpool/status'],
    ['vmpool-start', 'POST /vmpool/start'],
    ['vmpool-stop', 'POST /vmpool/stop'],
    ['vmpool-exec', 'POST /vmpool/exec'],
    ['vmpool-inspect', 'state · ip · console'],
  ]],
  ['vmpool-viewing', 'Viewing a console', []],
  ['vmpool-access', 'Access & security', []],
  ['vmpool-adding', 'Adding a VM', []],
]

export default function VmPoolPage() {
  return (
    <article className="docs-article">
      <h1>VM Pool</h1>

      <section id="vmpool-overview" className="docs-sec">
        <p>
          The <strong>VM pool</strong> is a central library of ready-to-run virtual machines
          (Windows, Ubuntu, Fedora, macOS, and a headless LXD box) that any machine on the
          LAN or WireGuard can list, start, stop, and view through a single HTTP API — no SSH,
          libvirt, or local disks required on the client. Images live once, centrally; one VM
          runs at a time on a dedicated host; and a light dashboard at{' '}
          <a href="https://vms.hugpy.ai" target="_blank" rel="noreferrer">vms.hugpy.ai</a> puts
          it all one click away.
        </p>
      </section>

      <section id="vmpool-architecture" className="docs-sec">
        <h2 className="docs-h2">Architecture</h2>
        <p>Three roles, three machines — storage, execution, and authoring are separated:</p>

        <div id="vmpool-hosts" className="docs-h3-block">
          <h3 className="docs-h3">Hosts &amp; roles</h3>
          <ul>
            <li>
              <strong>solcatcher</strong> (<code>192.168.1.100</code>, host <code>ae</code>) —
              the <strong>master library</strong>. The active masters live on a fast Crucial SSD at{' '}
              <code>/var/snap/lxd/common/VMS</code> (disk images, domain XMLs, OSX-KVM scaffolding, LXD
              export); the spinning HDD (<code>/mnt/MEDIA/VMS</code>) keeps the archive. Also the nginx
              hub that fronts <code>vms.hugpy.ai</code>, and — with 124&nbsp;GiB RAM + libguestfs —
              where masters are <strong>edited in place</strong> (<code>virt-customize</code>, no copies).
              Never runs the pool VMs.
            </li>
            <li>
              <strong>computron</strong> (<code>192.168.1.128</code>) — the <strong>run host</strong>.
              On start it pulls that one VM&rsquo;s disk from solcatcher to local storage and boots it.
              Runs <strong>one VM at a time</strong> (a 15&nbsp;GiB RAM ceiling makes that the right
              call anyway). Also hosts the toolserver (<code>:7004</code>) that serves the API.
            </li>
            <li>
              <strong>op</strong> (<code>192.168.1.113</code>) — where masters are <strong>built</strong>
              and refreshed, then pushed to solcatcher.
            </li>
          </ul>
        </div>

        <div id="vmpool-model" className="docs-h3-block">
          <h3 className="docs-h3">Golden-master model</h3>
          <p>
            Masters on solcatcher are <strong>read-only</strong>; every run is <strong>ephemeral</strong> —
            changes to a running disk are discarded on stop (nothing is written back), so there is no
            locking and no divergence. The <em>first</em> start of a VM pulls its disk over the LAN
            (a few minutes); by default the local copy is kept, so the <em>next</em> start boots in
            seconds. Stop with <code>purge</code> to reclaim the space and re-pull next time.
          </p>
          <p>
            Every VM boots straight to its desktop with <strong>no password prompt</strong>
            (Windows autologon, GDM autologin on Ubuntu/Fedora, macOS <code>autoLoginUser</code>;
            the LXD box is headless). Each launch runs as an <strong>independent systemd unit</strong>
            (<code>vmpool-&lt;name&gt;.service</code>), so restarting the toolserver never disturbs a
            running VM.
          </p>
        </div>
      </section>

      <section id="vmpool-pool" className="docs-sec">
        <h2 className="docs-h2">The pool</h2>
        <table className="docs-table">
          <thead>
            <tr><th>name</th><th>OS</th><th>kind</th><th>console / access</th></tr>
          </thead>
          <tbody>
            <tr><td><code>win10</code></td><td>Windows 10 Pro</td><td>libvirt</td><td>VNC · RDP (login <code>windows_test</code>)</td></tr>
            <tr><td><code>ubuntu</code></td><td>Ubuntu Desktop (GNOME)</td><td>libvirt</td><td>SPICE · user <code>hugpy</code></td></tr>
            <tr><td><code>fedora</code></td><td>Fedora 44 Workstation</td><td>libvirt</td><td>SPICE · user <code>hugpy</code></td></tr>
            <tr><td><code>macos</code></td><td>macOS Monterey</td><td>libvirt</td><td>VNC · user <code>hugpy</code></td></tr>
            <tr><td><code>hugpy</code></td><td>Ubuntu 24.04 (hugpy agent)</td><td>lxd</td><td>headless · <code>lxc exec ubuntu-hugpy -- bash</code></td></tr>
          </tbody>
        </table>
      </section>

      <section id="vmpool-images" className="docs-sec">
        <h2 className="docs-h2">Image conventions (for agents)</h2>
        <p>The images are built so an agent can drive them end-to-end — including unattended installs:</p>
        <ul>
          <li>
            <strong>Bare by design.</strong> Images ship minimal (e.g. no <code>curl</code>) so running
            an install <em>surfaces its own missing dependencies</em>. The pool is a clean test bed:
            what an installer needs shows up on a fresh, unmodified OS.
          </li>
          <li>
            <strong>Passwordless sudo.</strong> The <code>hugpy</code> user has <code>NOPASSWD:ALL</code>
            on Ubuntu, Fedora, and macOS (via <code>/etc/sudoers.d/</code>), so an agent runs{' '}
            <code>sudo</code> installs with no prompt. Windows keeps its password (no sudo; auth is part
            of its UX); the LXD box runs as root.
          </li>
          <li>
            <strong>Auto-login.</strong> Every VM lands on its desktop with no login prompt (Windows
            autologon, GDM autologin on Ubuntu/Fedora, macOS <code>autoLoginUser</code>).
          </li>
          <li>
            <strong>Headless-drivable <em>and</em> console-drivable.</strong> The libvirt guests run{' '}
            <code>qemu-guest-agent</code>, so commands run <em>inside</em> a guest with no SSH via{' '}
            <a href="#vmpool-exec"><code>POST /vmpool/exec</code></a> (the LXD box uses{' '}
            <code>lxc exec</code>). For GUI work a guest is still fully drivable through the libvirt
            console — <code>virsh screenshot</code> + <code>virsh send-key</code> (+ QMP mouse) — the
            same path the dashboard exposes.
          </li>
        </ul>
        <p>Credentials (kept intact; passwordless sudo is layered on top):</p>
        <table className="docs-table">
          <thead><tr><th>VM</th><th>login</th></tr></thead>
          <tbody>
            <tr><td><code>win10</code></td><td><code>windows_test</code> / <code>changeme</code></td></tr>
            <tr><td><code>ubuntu</code> · <code>fedora</code> · <code>macos</code></td><td><code>hugpy</code> / <code>changeme</code></td></tr>
            <tr><td><code>hugpy</code> (LXD)</td><td>root, via <code>lxc exec</code></td></tr>
          </tbody>
        </table>
      </section>

      <section id="vmpool-dashboard" className="docs-sec">
        <h2 className="docs-h2">Dashboard</h2>
        <p>
          <a href="https://vms.hugpy.ai" target="_blank" rel="noreferrer">https://vms.hugpy.ai</a>{' '}
          (bare domain redirects to <code>/vmpool/</code>) — OS cards with live running status,
          Start / Stop buttons (Start disabled while another VM runs), the running VM&rsquo;s console
          URL to copy, and a settings panel (purge-on-stop, refresh interval). It polls the same API
          documented below. Reachable from the LAN or WireGuard only (see Access &amp; security).
        </p>
      </section>

      <section id="vmpool-api" className="docs-sec">
        <h2 className="docs-h2">HTTP API</h2>
        <p>
          Base URL <code>https://vms.hugpy.ai</code> (via the nginx gate) or, directly on the LAN,
          <code>http://192.168.1.128:7004</code>. Bodies accept JSON, form-encoded, or query-string
          params. All examples use <code>curl</code>.
        </p>

        <div id="vmpool-list" className="docs-h3-block">
          <h3 className="docs-h3">GET /vmpool/list</h3>
          <p>Every pool member and whether it&rsquo;s running.</p>
          <CopyBlock
            title="list the pool"
            code={`curl -s https://vms.hugpy.ai/vmpool/list

# {"vms":[
#   {"name":"win10","domain":"win10-hugpy","kind":"libvirt","running":false},
#   {"name":"ubuntu","domain":"ubuntu-desktop","kind":"libvirt","running":true},
#   {"name":"fedora","domain":"fedora-hugpy","kind":"libvirt","running":false},
#   {"name":"macos","domain":"macos-ventura","kind":"libvirt","running":false},
#   {"name":"hugpy","domain":"ubuntu-hugpy","kind":"lxd","running":false}
# ]}`}
          />
        </div>

        <div id="vmpool-status" className="docs-h3-block">
          <h3 className="docs-h3">GET /vmpool/status</h3>
          <p>
            What&rsquo;s running and its console URL. <code>console</code> is <code>null</code> until
            the VM has booted (SPICE or VNC, resolved via <code>virsh domdisplay</code>).
          </p>
          <CopyBlock
            title="current status"
            code={`curl -s https://vms.hugpy.ai/vmpool/status

# {"running":"ubuntu","console":"spice://192.168.1.128:5900"}
# {"running":null,"console":null}   # when idle`}
          />
        </div>

        <div id="vmpool-start" className="docs-h3-block">
          <h3 className="docs-h3">POST /vmpool/start</h3>
          <p>
            Pull (if needed) and boot a VM. Returns <code>202</code> immediately and works
            asynchronously — the first start pulls the disk, so <strong>poll{' '}
            <code>/vmpool/status</code></strong> until <code>console</code> appears. Refuses with{' '}
            <code>409</code> if another VM is already running, <code>400</code> for an unknown name.
          </p>
          <CopyBlock
            title="start a VM, then poll for the console"
            code={`curl -s https://vms.hugpy.ai/vmpool/start -d '{"name":"fedora"}'
# {"name":"fedora","state":"starting","unit":"vmpool-fedora","hint":"poll GET /vmpool/status until 'console' is set"}

# wait for it, then open the console URL:
while true; do
  c=$(curl -s https://vms.hugpy.ai/vmpool/status | grep -o '"console":"[^"]*"' | cut -d'"' -f4)
  [ -n "$c" ] && break; sleep 5
done
remote-viewer "$c"`}
          />
        </div>

        <div id="vmpool-stop" className="docs-h3-block">
          <h3 className="docs-h3">POST /vmpool/stop</h3>
          <p>
            Stop the VM. Pass <code>{'{"purge":true}'}</code> to also delete the local disk copy and
            reclaim the space (the next start re-pulls it). Without <code>purge</code> the disk is
            cached, so restarting that VM is near-instant.
          </p>
          <CopyBlock
            title="stop (optionally purge)"
            code={`curl -s https://vms.hugpy.ai/vmpool/stop -d '{"name":"fedora"}'
curl -s https://vms.hugpy.ai/vmpool/stop -d '{"name":"fedora","purge":true}'
# {"name":"fedora","stopped":true,"purged":true,"output":"Stowed fedora (purged local scratch)."}`}
          />
        </div>

        <div id="vmpool-exec" className="docs-h3-block">
          <h3 className="docs-h3">POST /vmpool/exec</h3>
          <p>
            Run a shell command <strong>inside</strong> a pool VM and get its exit code + stdout/stderr
            back — no SSH, no console. Libvirt guests execute via <code>qemu-guest-agent</code>{' '}
            (<code>guest-exec</code>); the headless LXD box uses <code>lxc exec</code>. Accepts the
            <strong> pool name</strong> (not the libvirt domain). Commands run as <strong>root</strong>{' '}
            in the guest by default; pass <code>"user"</code> to drop to a guest account. Optional{' '}
            <code>"timeout"</code> in seconds (default 120, max 900).
          </p>
          <CopyBlock
            title="install into a fresh venv in fedora (the clean-room)"
            code={`curl -s https://vms.hugpy.ai/vmpool/exec \\
  -d '{"name":"fedora","cmd":"python3 -m venv /tmp/v && /tmp/v/bin/pip install requests","timeout":300}'

# {"name":"fedora","domain":"fedora-hugpy","via":"guest-agent","exit":0,
#  "stdout":"...","stderr":"","truncated":false}

# run as the pool user instead of root:
curl -s https://vms.hugpy.ai/vmpool/exec -d '{"name":"fedora","cmd":"whoami","user":"hugpy"}'`}
          />
          <p>
            The pool is <em>bare by design</em>: a command that needs a missing tool surfaces that
            dependency on a fresh, unmodified OS — and every change is discarded on stop.
          </p>
        </div>

        <div id="vmpool-inspect" className="docs-h3-block">
          <h3 className="docs-h3">state · ip · console (friendly-name)</h3>
          <p>
            Convenience wrappers over <code>virsh</code> that accept the <strong>pool name</strong>. The
            generic <code>vm/*</code> tools require the raw libvirt domain (e.g. <code>fedora-hugpy</code>)
            and return 500 on a pool name like <code>fedora</code>; these resolve it for you.
          </p>
          <CopyBlock
            title="inspect a VM by pool name"
            code={`curl -s https://vms.hugpy.ai/vmpool/state   -d '{"name":"fedora"}'   # {"state":"running", ...}
curl -s https://vms.hugpy.ai/vmpool/ip      -d '{"name":"fedora"}'   # domifaddr table
curl -s https://vms.hugpy.ai/vmpool/console -d '{"name":"fedora"}'   # {"console":"vnc://192.168.1.128:0"}`}
          />
        </div>
      </section>

      <section id="vmpool-viewing" className="docs-sec">
        <h2 className="docs-h2">Viewing a console</h2>
        <p>
          <code>/vmpool/status</code> hands back a <code>spice://</code> or <code>vnc://</code> URL
          pointed at the run host. Open it with <code>remote-viewer</code> (handles both), or paste it
          into any SPICE/VNC client. Windows also answers RDP on its NAT address; the LXD box is
          headless — shell in with <code>lxc exec ubuntu-hugpy -- bash</code> on computron.
        </p>
        <CopyBlock
          title="open the running VM"
          code={`remote-viewer "$(curl -s https://vms.hugpy.ai/vmpool/status | grep -o '"console":"[^"]*"' | cut -d'"' -f4)"`}
        />
      </section>

      <section id="vmpool-access" className="docs-sec">
        <h2 className="docs-h2">Access &amp; security</h2>
        <p>
          <code>vms.hugpy.ai</code> is served by solcatcher&rsquo;s nginx (Let&rsquo;s Encrypt TLS)
          and proxied to the toolserver on computron. It is gated{' '}
          <strong>LAN (<code>192.168.1.0/24</code>) + WireGuard (<code>192.168.2.0/24</code>) only</strong>{' '}
          — the same wall as <code>toolserver.hugpy.ai</code>. That gate <em>is</em> the auth: the API
          can start and stop VMs, so it is deliberately kept off the public internet. To reach it from
          elsewhere, connect the WireGuard VPN — no config change needed.
        </p>
      </section>

      <section id="vmpool-adding" className="docs-sec">
        <h2 className="docs-h2">Adding a VM</h2>
        <p>The pattern (how Fedora 44 was added):</p>
        <ol>
          <li>Build the VM on <strong>op</strong> (e.g. <code>virt-install</code> + a kickstart for an
            unattended install); set the pool conventions — user <code>hugpy</code>, auto-login,
            passwordless sudo.</li>
          <li>Push the disk master + its domain XML to solcatcher&rsquo;s SSD library at{' '}
            <code>/var/snap/lxd/common/VMS</code>. Small tweaks to an existing master can be made in
            place there with <code>virt-customize</code> (no copy needed).</li>
          <li>Add a case to <code>vm-run</code> / <code>vm-stow</code> and the <code>POOL</code> map in
            <code>vmpool.py</code>, then redeploy the blueprint.</li>
        </ol>
        <p>
          It then appears in <code>/vmpool/list</code> and on the dashboard automatically, startable
          like any other member.
        </p>
      </section>
    </article>
  )
}
