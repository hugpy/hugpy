import http from 'node:http'
import { readFile } from 'node:fs/promises'
import { existsSync, statSync } from 'node:fs'
import path from 'node:path'
import { chromium } from 'playwright'

const ROOT = path.resolve(process.argv[2] || './dist')
const MIME = { '.html':'text/html', '.js':'text/javascript', '.css':'text/css', '.png':'image/png', '.svg':'image/svg+xml', '.json':'application/json', '.woff2':'font/woff2', '.map':'application/json' }

const server = http.createServer(async (req, res) => {
  try {
    let p = decodeURIComponent(req.url.split('?')[0])
    let fp = path.join(ROOT, p)
    if (!existsSync(fp) || statSync(fp).isDirectory()) {
      // SPA fallback unless it's an asset request
      fp = path.extname(p) ? fp : path.join(ROOT, 'index.html')
      if (!existsSync(fp)) fp = path.join(ROOT, 'index.html')
    }
    const buf = await readFile(fp)
    res.writeHead(200, { 'content-type': MIME[path.extname(fp)] || 'application/octet-stream' })
    res.end(buf)
  } catch (e) { res.writeHead(404); res.end('nf') }
})

await new Promise(r => server.listen(0, '127.0.0.1', r))
const port = server.address().port
const base = `http://127.0.0.1:${port}`
console.log('serving', ROOT, 'at', base)

const browser = await chromium.launch({ executablePath: '/usr/bin/google-chrome-stable' })
const page = await browser.newPage({ viewport: { width: 1280, height: 860 }, deviceScaleFactor: 2 })
page.on('pageerror', e => console.log('PAGEERROR:', e.message))

await page.goto(base + '/console', { waitUntil: 'networkidle' })
await page.waitForTimeout(1500)

const out = '.design-sync/.cache'
await page.screenshot({ path: out + '/brand-console-full.png' })

// crop the top-left navbar/brand region
const nav = await page.$('.navbar') || await page.$('nav')
if (nav) {
  await nav.screenshot({ path: out + '/brand-navbar.png' })
  console.log('captured .navbar')
} else {
  console.log('no .navbar found; full page only')
}

// also a tight crop top-left
await page.screenshot({ path: out + '/brand-topleft.png', clip: { x: 0, y: 0, width: 520, height: 120 } })

await browser.close()
server.close()
console.log('done')
