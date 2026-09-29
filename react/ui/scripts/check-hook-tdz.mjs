// Hook-order TDZ scanner. Finds a `const/let` declared in a function body that
// is referenced EARLIER in the same body from code that runs during render:
// a hook's dependency array, a useMemo/useState-initializer factory, or a bare
// statement — i.e. exactly "can't access lexical declaration 'load' before
// initialization" (CallsPanel, 2026-09-29). References from inside callbacks
// (useCallback/useEffect bodies, handlers) are fine — they run later.
//
//   node scripts/check-hook-tdz.mjs [dir-or-file ...]   (default: src)
// Exit 1 with one line per finding. Exported for the node:test wrapper.
import { readFileSync, readdirSync, statSync } from 'node:fs'
import { join, extname } from 'node:path'
import { parse } from '@babel/parser'
import traverseMod from '@babel/traverse'
const traverse = traverseMod.default || traverseMod

const EXT = new Set(['.js', '.jsx', '.ts', '.tsx', '.mjs'])
const HOOK_RE = /^use[A-Z]/
// Hook args whose code runs synchronously during render: (hook, argIndex).
const RENDER_TIME_ARGS = { useMemo: [0], useState: [0], useReducer: [1], useSyncExternalStore: [], useId: [] }

export function scanSource(code, file = '<src>') {
  let ast
  try {
    ast = parse(code, { sourceType: 'module', plugins: ['jsx', 'typescript'], errorRecovery: true })
  } catch (e) { return [{ file, line: 0, msg: `parse error: ${e.message}` }] }
  const out = []
  const checkFn = (fnPath) => {
    const body = fnPath.get('body')
    if (!body.isBlockStatement()) return
    // top-level lexical declarations of this function body → declaration start
    const decl = new Map()
    for (const st of body.get('body')) {
      if (!st.isVariableDeclaration() || st.node.kind === 'var') continue
      for (const d of st.node.declarations) {
        if (d.id.type === 'Identifier') decl.set(d.id.name, { start: d.start, line: d.loc.start.line })
      }
    }
    if (!decl.size) return
    // walk render-time code positions: for each statement, the parts that execute now
    const renderTime = []   // [start, end] ranges evaluated during render
    const factories = new Set()   // function nodes whose BODY runs during render (useMemo factory …)
    const ctx = { renderTime, factories }
    for (const st of body.get('body')) {
      const n = st.node
      if (st.isVariableDeclaration()) {
        for (const d of n.declarations) if (d.init) collectRenderTime(d.init, ctx)
      } else if (st.isExpressionStatement()) {
        collectRenderTime(n.expression, ctx)
      } else if (st.isReturnStatement() && n.argument) {
        collectRenderTime(n.argument, ctx)
      } else if (st.isIfStatement()) {
        collectRenderTime(n.test, ctx)
      }
    }
    const scope = fnPath.scope
    fnPath.traverse({
      Identifier(p) {
        const name = p.node.name
        const d = decl.get(name)
        if (!d || !p.isReferencedIdentifier()) return
        if (p.node.start >= d.start) return                       // after the declaration: fine
        const b = p.scope.getBinding(name)
        if (!b || b.scope !== scope) return                       // shadowed / not this declaration
        if (!renderTime.some(([s, e]) => p.node.start >= s && p.node.end <= e)) return
        // Inside a nested function? Only a render-time factory (useMemo …)
        // executes now; a handler / effect / callback body runs later.
        const fp = p.getFunctionParent()
        if (fp !== fnPath && !factories.has(fp.node)) return
        out.push({ file, line: p.node.loc.start.line, name, declLine: d.line,
          msg: `${file}:${p.node.loc.start.line} '${name}' used during render before its declaration on line ${d.line} (TDZ)` })
      },
    })
  }
  traverse(ast, {
    'FunctionDeclaration|FunctionExpression|ArrowFunctionExpression'(p) { checkFn(p) },
  })
  return out
}

// Push the code ranges of `node` that execute during render. Function bodies
// do NOT (they run later) unless they are render-time hook args (useMemo etc.),
// which are recorded in `factories` so the identifier check admits them.
function collectRenderTime(node, ctx) {
  if (!node) return
  const { renderTime: acc, factories } = ctx
  if (node.type === 'ArrowFunctionExpression' || node.type === 'FunctionExpression') return
  if (node.type === 'CallExpression') {
    const callee = node.callee
    const hook = callee.type === 'Identifier' ? callee.name
      : callee.type === 'MemberExpression' && callee.property.type === 'Identifier' ? callee.property.name : ''
    if (HOOK_RE.test(hook)) {
      const runNow = RENDER_TIME_ARGS[hook]
      node.arguments.forEach((a, i) => {
        if (a.type === 'ArrowFunctionExpression' || a.type === 'FunctionExpression') {
          if (runNow && runNow.includes(i)) { acc.push([a.start, a.end]); factories.add(a) }
          return
        }
        acc.push([a.start, a.end])                                       // deps array / plain args: evaluated now
      })
      return
    }
    acc.push([callee.start, callee.end])
    node.arguments.forEach(a => collectRenderTime(a, ctx))
    return
  }
  // Any other expression executes now; nested functions inside it are
  // excluded by the getFunctionParent check in the identifier visitor.
  acc.push([node.start, node.end])
}

function walk(dir, files = []) {
  for (const ent of readdirSync(dir)) {
    if (ent === 'node_modules' || ent === 'dist' || ent.startsWith('.')) continue
    const p = join(dir, ent)
    const st = statSync(p)
    if (st.isDirectory()) walk(p, files)
    else if (EXT.has(extname(ent)) && !ent.endsWith('.test.mjs')) files.push(p)
  }
  return files
}

export function scanPaths(paths) {
  const files = []
  for (const p of paths) (statSync(p).isDirectory() ? walk(p, files) : files.push(p))
  const findings = []
  for (const f of files) findings.push(...scanSource(readFileSync(f, 'utf8'), f))
  return findings
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const targets = process.argv.slice(2).length ? process.argv.slice(2) : ['src']
  const findings = scanPaths(targets)
  for (const f of findings) console.log(f.msg)
  console.log(`[check-hook-tdz] ${findings.length} finding(s)`)
  process.exit(findings.length ? 1 : 0)
}
