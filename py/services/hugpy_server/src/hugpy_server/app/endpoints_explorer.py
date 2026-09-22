"""endpoints_explorer — a drop-in interactive API explorer for any Flask /
``abstract_flask`` app. Framework-only (Flask + stdlib), no app-specific imports.

This is the reusable generalization of hugpy's ``/endpoints`` page — the intended
*upgrade to* ``abstract_flask.generator``. Where the generator turns Python functions
into routes (and wires ``offer_help`` so each supports ``?help``), this turns an app's
``url_map`` into a browsable, searchable, **try-it** console:

    from hugpy_server.app.endpoints_explorer import install_endpoints_explorer
    install_endpoints_explorer(app)                      # one call, zero config

One call gives you, at ``/endpoints``:
  * curl / ``Accept: application/json`` / ``?format=json`` -> the faithful
    ``[{endpoint,url,methods}]`` JSON (same shape ``abstract_flask``'s inspector
    already serves — programmatic clients are unaffected).
  * a browser -> a rendered page: search, group-by-prefix, and an inline **try-it**
    form per endpoint (path params from the rule, query, JSON body, method, Send ->
    live same-origin response, and a ``params (?help)`` button that surfaces the
    ``offer_help`` schema for generator-built routes).

Two optional hooks make it curate + gate itself for apps that have a notion of
"internal / operator-only" routes (permissive defaults, so a bare app needs neither):

    install_endpoints_explorer(app,
        classify_internal=lambda url, methods: ...,   # True -> hidden by default
        can_view_internal=lambda: is_operator(),      # gate for ?all=1
        brand="my-api", accent="#8ab4ff")

It overrides ``abstract_flask``'s existing ``global_endpoint_inspector`` /
``prefix_inspector`` views IN PLACE when present (no duplicate routes), else registers
``/endpoints`` (+ ``/prefixes`` if that inspector exists). Curation is a docs nicety,
NOT a security boundary — real routes must still enforce their own auth.
"""

from __future__ import annotations

import html as _html
import json as _json
from typing import Callable, Dict, List, Optional

from flask import Response, jsonify, request

Classifier = Callable[[str, List[str]], bool]
Gate = Callable[[], bool]

_MUTATING = {"POST", "PUT", "PATCH", "DELETE"}
_METHOD_ORDER = {"GET": 0, "POST": 1, "PUT": 2, "PATCH": 3, "DELETE": 4}


# ── collection ────────────────────────────────────────────────────────────────

def collect_endpoints(app, classify: Optional[Classifier] = None) -> List[Dict]:
    """Rich per-route records: ``{endpoint,url,methods,args,internal}``, sorted by
    url. ``classify(url, methods) -> bool`` flags internal routes (default: none)."""
    out: List[Dict] = []
    for rule in app.url_map.iter_rules():
        if rule.endpoint == "static":
            continue
        methods = sorted((rule.methods or set()) - {"HEAD", "OPTIONS"})
        url = str(rule)
        internal = False
        if classify is not None:
            try:
                internal = bool(classify(url, methods))
            except Exception:
                internal = False
        out.append({
            "endpoint": rule.endpoint, "url": url, "methods": methods,
            "args": sorted(rule.arguments or ()), "internal": internal,
        })
    return sorted(out, key=lambda x: x["url"])


def _visible(entries: List[Dict], include_internal: bool) -> List[Dict]:
    return entries if include_internal else [e for e in entries if not e["internal"]]


def _public_json(entries: List[Dict], include_internal: bool) -> List[Dict]:
    return [{"endpoint": e["endpoint"], "url": e["url"], "methods": e["methods"]}
            for e in _visible(entries, include_internal)]


def _top_segment(url: str) -> str:
    stripped = url.lstrip("/")
    if not stripped:
        return "/"
    return "/" + stripped.split("/")[0].split("<")[0].rstrip("/")


def collect_prefixes(entries: List[Dict], include_internal: bool) -> List[str]:
    return sorted({_top_segment(e["url"]) for e in _visible(entries, include_internal)})


# ── request policy (negotiation + gating) ─────────────────────────────────────

def _wants_html(req) -> bool:
    fmt = (req.args.get("format") or "").strip().lower()
    if fmt in ("json", "raw"):
        return False
    if fmt in ("html", "view", "page"):
        return True
    accept = req.accept_mimetypes
    best = accept.best_match(["application/json", "text/html"])
    return best == "text/html" and accept["text/html"] >= accept["application/json"]


def _all_requested(req) -> bool:
    return (req.args.get("all") or "").strip().lower() in ("1", "true", "yes", "on")


# ── HTML rendering (self-contained: inline CSS/JS, Google-fonts editorial look) ─

def render_html(entries: List[Dict], *, host: str, show_all: bool, all_allowed: bool,
                title: str, accent: str, call_base: str = "") -> str:
    include = show_all and all_allowed
    visible = _visible(entries, include)
    hidden_internal = 0 if include else sum(1 for e in entries if e["internal"])

    groups: Dict[str, List[Dict]] = {}
    for e in visible:
        groups.setdefault(_top_segment(e["url"]), []).append(e)

    rows: List[str] = []
    for seg in sorted(groups):
        items = groups[seg]
        rows.append(
            f'<tr class="grp" data-seg="{_html.escape(seg)}">'
            f'<td colspan="3"><span class="seg">{_html.escape(seg)}</span>'
            f'<span class="segn">{len(items)}</span></td></tr>'
        )
        for e in items:
            methods = sorted(e["methods"], key=lambda m: _METHOD_ORDER.get(m, 9))
            badges = "".join(
                f'<span class="m m-{_html.escape(m.lower())}">{_html.escape(m)}</span>'
                for m in methods)
            mutating = bool(set(methods) & _MUTATING)
            flags = ""
            if e["internal"]:
                flags += '<span class="flag flag-int" title="operator-gated / internal">internal</span>'
            if mutating:
                flags += '<span class="flag flag-mut" title="mutating — changes state">mutates</span>'
            data = _html.escape(_json.dumps({
                "url": e["url"], "methods": methods, "args": e["args"],
                "internal": e["internal"], "mutating": mutating, "endpoint": e["endpoint"],
            }), quote=True)
            hay = _html.escape(f'{e["url"]} {e["endpoint"]} {" ".join(methods)}'.lower())
            rows.append(
                f'<tr class="ep" data-h="{hay}" data-ep="{data}">'
                f'<td class="c-m">{badges}</td>'
                f'<td class="c-u"><code>{_html.escape(e["url"])}</code>{flags}</td>'
                f'<td class="c-e">{_html.escape(e["endpoint"])}</td></tr>'
            )

    host_line = f" · <span class='host'>{_html.escape(host)}</span>" if host else ""
    if include:
        toggle = '<a href="?">public only</a>'
    elif show_all and not all_allowed:
        toggle = '<span class="muted">internal view needs operator auth</span> · <a href="?">public</a>'
    else:
        extra = f" ({hidden_internal} internal hidden)" if hidden_internal else ""
        toggle = f'<a href="?all=1">show all{extra}</a>'

    return (_PAGE
            .replace("__TITLE__", _html.escape(title))
            .replace("__ACCENT__", accent)
            .replace("__CALLBASE__", _json.dumps(call_base))
            .replace("__TOTAL__", str(len(visible)))
            .replace("__HOSTLINE__", host_line)
            .replace("__TOGGLE__", toggle)
            .replace("__ROWS__", "\n".join(rows)))


_PAGE = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="theme-color" content="#050505">
<title>__TITLE__</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=Newsreader:ital,wght@0,400;0,500;1,400&display=swap" rel="stylesheet">
<style>
:root{
  --bg:#050505;--panel:#111113;--fg:#ededed;--dim:#ffffffb3;--faint:#ffffff66;
  --line:#ffffff26;--line-lt:#ffffff14;--accent:__ACCENT__;--warn:#e3b341;--bad:#f87171;
  --serif:'Newsreader',Georgia,'Times New Roman',serif;
  --mono:'IBM Plex Mono',ui-monospace,SFMono-Regular,Menlo,monospace;
}
*{box-sizing:border-box}
html,body{margin:0;background:var(--bg);color:var(--fg);overflow-x:hidden;}
body{font:14px/1.6 var(--mono);-webkit-font-smoothing:antialiased;}
header{position:sticky;top:0;background:rgba(5,5,5,.92);backdrop-filter:blur(8px);border-bottom:1px solid var(--line);padding:20px 22px 14px;z-index:5;}
h1{margin:0;font-family:var(--serif);font-size:26px;font-weight:500;letter-spacing:-.01em;}
h1 .n{color:var(--accent);font-style:italic;}
.sub{color:var(--dim);font-size:12px;margin-top:4px;letter-spacing:.02em;}
.sub a{color:var(--accent);text-decoration:none;border-bottom:1px solid transparent;}
.sub a:hover{border-bottom-color:var(--accent);}
.muted{color:var(--faint);}
.tools{display:flex;gap:10px;align-items:center;margin-top:12px;flex-wrap:wrap;}
#q{flex:1;min-width:200px;background:var(--panel);border:1px solid var(--line);color:var(--fg);border-radius:8px;padding:9px 12px;font:13px var(--mono);}
#q:focus{outline:none;border-color:var(--accent);}
#q::placeholder{color:var(--faint);}
#count{color:var(--dim);font-size:12px;white-space:nowrap;}
.wrap{max-width:1080px;margin:0 auto;padding:0 22px 80px;}
table{width:100%;border-collapse:collapse;}
td{padding:8px 8px;border-bottom:1px solid var(--line-lt);vertical-align:top;}
tr.grp td{border-bottom:1px solid var(--line);padding:26px 8px 8px;}
.seg{font-family:var(--serif);font-style:italic;font-size:16px;color:var(--fg);}
.segn{color:var(--faint);font-size:11px;margin-left:9px;}
tr.ep{cursor:pointer;}
tr.ep:hover td,tr.ep.open td{background:var(--panel);}
.c-m{width:118px;white-space:nowrap;}
.c-u code{font:12.5px/1.6 var(--mono);color:var(--fg);word-break:break-all;}
.c-e{color:var(--faint);font:11.5px/1.6 var(--mono);word-break:break-all;}
.m{display:inline-block;font:500 10px/1 var(--mono);padding:3px 6px;border-radius:5px;margin-right:4px;letter-spacing:.04em;}
.m-get{background:#2ea04326;color:#7ee787;}.m-post{background:#8ab4ff26;color:#8ab4ff;}
.m-put{background:#e3b34126;color:#e3b341;}.m-patch{background:#a371f726;color:#c9a2ff;}.m-delete{background:#f8717126;color:#f87171;}
.flag{display:inline-block;font:500 9px/1 var(--mono);padding:2px 5px;border-radius:4px;margin-left:7px;vertical-align:middle;letter-spacing:.04em;text-transform:uppercase;}
.flag-int{background:#a371f71f;color:#c9a2ff;border:1px solid #a371f73d;}
.flag-mut{background:#e3b3411f;color:var(--warn);border:1px solid #e3b3413d;}
.panel td{padding:0;background:var(--panel);}
.tryit{padding:16px 18px;border-left:2px solid var(--accent);margin:0 0 6px;}
.tryit .row{display:flex;gap:10px;align-items:center;margin:8px 0;flex-wrap:wrap;}
.tryit label{font:500 10.5px/1 var(--mono);color:var(--faint);min-width:98px;text-transform:uppercase;letter-spacing:.06em;}
.tryit input,.tryit textarea,.tryit select{background:var(--bg);border:1px solid var(--line);color:var(--fg);border-radius:7px;padding:7px 10px;font:12.5px var(--mono);}
.tryit input,.tryit textarea{flex:1;min-width:170px;}
.tryit textarea{min-height:66px;resize:vertical;width:100%;}
.tryit input:focus,.tryit textarea:focus,.tryit select:focus{outline:none;border-color:var(--accent);}
.tryit .u{font:12.5px var(--mono);color:var(--dim);word-break:break-all;}
.btn{background:var(--accent);color:#08131f;border:none;border-radius:7px;padding:8px 16px;font:500 12px var(--mono);cursor:pointer;letter-spacing:.02em;}
.btn:hover{filter:brightness(1.08);}
.btn.sec{background:transparent;color:var(--accent);border:1px solid var(--line);}
.btn:disabled{opacity:.5;cursor:default;}
.warn{color:var(--warn);font:11.5px var(--mono);margin:6px 0;}
.resp{margin-top:10px;}
.resp .st{font:500 11.5px var(--mono);margin-bottom:5px;letter-spacing:.02em;}
.resp pre{background:var(--bg);border:1px solid var(--line);border-radius:8px;padding:12px;overflow:auto;max-height:380px;font:12px/1.5 var(--mono);white-space:pre-wrap;word-break:break-word;margin:0;color:var(--fg);}
.st.ok{color:#7ee787;}.st.err{color:var(--bad);}
.empty{color:var(--faint);padding:40px 8px;text-align:center;font-family:var(--serif);font-style:italic;font-size:16px;}
::-webkit-scrollbar{width:10px;height:10px;}::-webkit-scrollbar-thumb{background:var(--line);border-radius:6px;}
</style></head><body>
<header>
  <h1>__TITLE__</h1>
  <div class="sub"><span id="total">__TOTAL__</span> routes__HOSTLINE__ ·
    <a href="?format=json">raw JSON</a> · <a href="/prefixes">/prefixes</a> · __TOGGLE__</div>
  <div class="tools">
    <input id="q" type="search" placeholder="filter by path, method, or endpoint name…" autocomplete="off" autofocus>
    <span id="count"></span>
  </div>
</header>
<div class="wrap"><table><tbody id="t">
__ROWS__
</tbody></table><div class="empty" id="none" hidden>no endpoints match your filter</div></div>
<script>
(function(){
  var CALL_BASE=__CALLBASE__;
  var q=document.getElementById('q'),t=document.getElementById('t'),
      cnt=document.getElementById('count'),none=document.getElementById('none'),
      eps=[].slice.call(t.querySelectorAll('tr.ep')),
      grps=[].slice.call(t.querySelectorAll('tr.grp'));
  var ESC={'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'};
  function esc(s){return String(s==null?'':s).replace(/[&<>"']/g,function(c){return ESC[c];});}
  function buildURL(tpl,argVals){
    var out=tpl.replace(/<[^>]+>/g,function(tok){
      var isPath=/<path:/.test(tok);
      var name=tok.replace(/[<>]/g,'').split(':').pop();
      var v=argVals[name]!=null?argVals[name]:'';
      return isPath?v.split('/').map(encodeURIComponent).join('/'):encodeURIComponent(v);
    });
    return out.replace(/^\/{2,}/,'/');
  }
  // Route the actual try-it call through the host's API base (e.g. "/api") when
  // set: the dev front proxies only /api to Flask and SPA-swallows every other
  // bare path into index.html, so a bare fetch would return HTML, not the
  // endpoint. Strip an existing /api first so both bare- and /api-listed routes
  // resolve to the one proxied path. CALL_BASE="" (default) leaves URLs as-is.
  function callPath(u){
    if(!CALL_BASE)return u;
    u=u.replace(/^\/api(?=\/|$)/,'');
    return CALL_BASE+(u||'/');
  }
  function sameOrigin(path){try{return new URL(path,location.href).origin===location.origin;}catch(e){return false;}}
  function panelFor(row){
    var ep=JSON.parse(row.getAttribute('data-ep'));
    var methods=ep.methods.slice(),defM=methods.indexOf('GET')>-1?'GET':methods[0];
    var wrap=document.createElement('tr');wrap.className='panel';
    var td=document.createElement('td');td.colSpan=3;wrap.appendChild(td);
    var h='<div class="tryit">';
    if(methods.length>1){
      h+='<div class="row"><label>method</label><select class="mm">'+
        methods.map(function(m){return '<option'+(m===defM?' selected':'')+'>'+esc(m)+'</option>';}).join('')+'</select></div>';
    }else{h+='<div class="row"><label>method</label><span class="u">'+esc(defM)+'</span></div>';}
    (ep.args||[]).forEach(function(a){
      h+='<div class="row"><label>'+esc(a)+'</label><input class="pa" data-a="'+esc(a)+'" placeholder="'+esc(a)+'"></div>';
    });
    h+='<div class="row"><label>query</label><input class="qq" placeholder="key=val&amp;k2=v2"></div>';
    h+='<div class="row bodyrow"><label>body (JSON)</label><textarea class="bb" placeholder="{ }"></textarea></div>';
    if(ep.mutating){h+='<div class="warn">⚠ this method changes state'+(ep.internal?' and is operator-gated':'')+' — it runs for real when you Send.</div>';}
    h+='<div class="row"><span class="u" data-role="preview"></span></div>';
    h+='<div class="row"><button class="btn go">Send</button>';
    h+='<button class="btn sec help">params (?help)</button></div>';
    h+='<div class="resp" hidden><div class="st"></div><pre></pre></div>';
    h+='</div>';
    td.innerHTML=h;
    var pas=[].slice.call(td.querySelectorAll('.pa')),
        mm=td.querySelector('.mm'),qq=td.querySelector('.qq'),bb=td.querySelector('.bb'),
        bodyrow=td.querySelector('.bodyrow'),preview=td.querySelector('[data-role=preview]'),
        resp=td.querySelector('.resp'),st=td.querySelector('.st'),pre=td.querySelector('pre');
    function curMethod(){return mm?mm.value:defM;}
    function argVals(){var o={};pas.forEach(function(i){o[i.getAttribute('data-a')]=i.value;});return o;}
    function fullPath(){var u=callPath(buildURL(ep.url,argVals())),query=(qq.value||'').trim();return u+(query?((u.indexOf('?')>-1?'&':'?')+query):'');}
    function refresh(){var m=curMethod();bodyrow.style.display=(m==='GET'||m==='DELETE')?'none':'';preview.textContent=m+' '+fullPath();}
    pas.concat([qq]).forEach(function(i){i.addEventListener('input',refresh);});
    if(mm)mm.addEventListener('change',refresh);
    refresh();
    function send(path,method,useBody){
      if(!sameOrigin(path)){resp.hidden=false;st.className='st err';st.textContent='refused: not same-origin';pre.textContent='This tool only calls '+location.origin;return;}
      var opts={method:method,headers:{}};
      if(useBody&&bb.value.trim()){opts.headers['Content-Type']='application/json';opts.body=bb.value;}
      resp.hidden=false;st.className='st';st.textContent='…';pre.textContent='';
      var t0=Date.now();
      fetch(path,opts).then(function(r){return r.text().then(function(txt){
        st.className='st '+(r.ok?'ok':'err');
        st.textContent=r.status+' '+r.statusText+'  ·  '+(Date.now()-t0)+'ms  ·  '+(r.headers.get('content-type')||'');
        try{pre.textContent=JSON.stringify(JSON.parse(txt),null,2);}catch(e){pre.textContent=txt.slice(0,20000);}
      });}).catch(function(e){st.className='st err';st.textContent='network error';pre.textContent=String(e);});
    }
    td.querySelector('.go').addEventListener('click',function(){
      var m=curMethod();
      if(m!=='GET'&&!confirm(m+' '+fullPath()+'\n\nThis calls the API for real'+(ep.mutating?' and may change state':'')+'. Continue?'))return;
      send(fullPath(),m,m!=='GET'&&m!=='DELETE');
    });
    td.querySelector('.help').addEventListener('click',function(){
      var u=callPath(buildURL(ep.url,argVals()));send(u+(u.indexOf('?')>-1?'&':'?')+'help','GET',false);
    });
    return wrap;
  }
  eps.forEach(function(row){
    row.addEventListener('click',function(e){
      if(e.target.closest('.panel'))return;
      var nx=row.nextElementSibling;
      if(nx&&nx.classList.contains('panel')){nx.remove();row.classList.remove('open');return;}
      row.classList.add('open');
      row.parentNode.insertBefore(panelFor(row),row.nextElementSibling);
    });
  });
  function apply(){
    var s=q.value.trim().toLowerCase(),shown=0;
    eps.forEach(function(r){
      var m=!s||r.dataset.h.indexOf(s)>-1;r.hidden=!m;if(m)shown++;
      var nx=r.nextElementSibling;if(nx&&nx.classList.contains('panel'))nx.hidden=!m;
    });
    grps.forEach(function(g){
      var n=g.nextElementSibling,any=false;
      while(n&&!n.classList.contains('grp')){if(n.classList.contains('ep')&&!n.hidden)any=true;n=n.nextElementSibling;}
      g.hidden=!any;
    });
    cnt.textContent=s?(shown+' shown'):'';none.hidden=shown>0;
  }
  q.addEventListener('input',apply);apply();
})();
</script>
</body></html>"""


# ── install ──────────────────────────────────────────────────────────────────

def install_endpoints_explorer(app, *, classify_internal: Optional[Classifier] = None,
                               can_view_internal: Optional[Gate] = None,
                               brand: str = "API endpoints",
                               accent: str = "#8ab4ff",
                               call_base: str = "") -> None:
    """Install the explorer on ``app``. Overrides ``abstract_flask``'s inspector
    views in place when present; otherwise registers ``/endpoints``.

    classify_internal(url, methods) -> True to hide a route by default (docs
    curation). can_view_internal() -> True to allow ?all=1 to reveal internal
    routes (called in request context). Both optional; defaults keep everything
    public and permissive, so a bare app works with a bare call.

    call_base: prefix the try-it FETCH with this (e.g. "/api") when the page is
    served behind a front that only proxies that base to the app and SPA-swallows
    other bare paths (the dev webpack front does exactly this — a bare fetch would
    return index.html, not the endpoint). An existing "/api" on the route is
    stripped first, so both bare- and /api-listed routes resolve to the one
    proxied path. Default "" leaves try-it URLs exactly as listed."""
    def _gate() -> bool:
        if can_view_internal is None:
            return True
        try:
            return bool(can_view_internal())
        except Exception:
            return False

    def _include() -> bool:
        return _all_requested(request) and _gate()

    def endpoints_view(*_a, **_k):
        entries = collect_endpoints(app, classify_internal)
        if _wants_html(request):
            return Response(render_html(
                entries, host=request.host or "", show_all=_all_requested(request),
                all_allowed=_gate(), title=brand, accent=accent,
                call_base=call_base), mimetype="text/html",
                # Never let a browser serve a cached copy of this page: it carries
                # the try-it JS, and a stale copy silently calls old (bare) URLs.
                headers={"Cache-Control": "no-store"})
        return jsonify(_public_json(entries, include_internal=_include())), 200

    def prefixes_view(*_a, **_k):
        entries = collect_endpoints(app, classify_internal)
        return jsonify(collect_prefixes(entries, include_internal=_include())), 200

    if "global_endpoint_inspector" in app.view_functions:
        app.view_functions["global_endpoint_inspector"] = endpoints_view
    else:
        app.add_url_rule("/endpoints", endpoint="global_endpoint_inspector",
                         view_func=endpoints_view, methods=["GET"])

    if "prefix_inspector" in app.view_functions:
        app.view_functions["prefix_inspector"] = prefixes_view
