import http from 'node:http'
import { readFile } from 'node:fs/promises'
import { existsSync, statSync } from 'node:fs'
import path from 'node:path'
import { chromium } from 'playwright'

const ROOT = path.resolve(process.argv[2] || './dist')
const MIME = { '.html':'text/html','.js':'text/javascript','.css':'text/css','.png':'image/png','.svg':'image/svg+xml','.json':'application/json','.woff2':'font/woff2' }
const server = http.createServer(async (req,res)=>{
  try{ let p=decodeURIComponent(req.url.split('?')[0]); let fp=path.join(ROOT,p)
    if(!existsSync(fp)||statSync(fp).isDirectory()) fp=path.extname(p)?fp:path.join(ROOT,'index.html')
    const buf=await readFile(fp); res.writeHead(200,{'content-type':MIME[path.extname(fp)]||'application/octet-stream'}); res.end(buf)
  }catch(e){res.writeHead(404);res.end('nf')}
})
await new Promise(r=>server.listen(0,'127.0.0.1',r))
const port=server.address().port
const browser=await chromium.launch({executablePath:'/usr/bin/google-chrome-stable'})
const page=await browser.newPage({viewport:{width:760,height:470},deviceScaleFactor:2})
await page.goto(`http://127.0.0.1:${port}/__brandmock.html`,{waitUntil:'networkidle'})
await page.waitForTimeout(600)
await page.screenshot({path:'.design-sync/.cache/brand-variants.png'})
await browser.close(); server.close(); console.log('done')
