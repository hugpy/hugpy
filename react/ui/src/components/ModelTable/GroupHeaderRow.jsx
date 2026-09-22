/**
 * MODEL GROUPS — the group header row and the per-member verdict chips.
 *
 * A header row spans the table and carries the group's three ticks
 * (quality / speed / priority — keeper owns nomenclature, these names are
 * fixed). Members render indented beneath it as ordinary model rows, each with
 * a chip saying what it does on each worker, or why it lost there.
 *
 * Styling reuses ModelTable.css conventions: flat `mt-`-prefixed classes,
 * theme tokens from index.css, and the same `.mt-worker-fit` chip family the
 * expanded per-worker fit block already uses.
 */
import React from 'react'

import { TICKS, TICK_HELP, memberVerdicts } from './useModelGroups'

/** One tick switch. Reports the server's authority, never guesses at it. */
function TickSwitch ({ groupKey, tick, on, disabled, hint, onFlip }) {
  const title = [TICK_HELP[tick], hint].filter(Boolean).join('\n\n')
  return (
    <label className={`mt-tick${on ? ' mt-tick-on' : ''}`
      + (disabled ? ' mt-tick-disabled' : '')} title={title}>
      <input
        type="checkbox"
        checked={!!on}
        disabled={!!disabled}
        aria-label={`${tick} tick for ${groupKey}`}
        onChange={e => onFlip(tick, e.target.checked)}
      />
      <span>{tick}</span>
    </label>
  )
}

export function GroupHeaderRow ({ group, colSpan, enabled, offHint, onFlip,
                                 collapsed, onToggleCollapse }) {
  const ticks = group.ticks || {}
  const memberCount = (group.members || []).length
  const anyTick = TICKS.some(t => ticks[t])
  return (
    <tr className="mt-group-row">
      <td colSpan={colSpan}>
        <div className="mt-group-head">
          <button
            type="button"
            className="mt-group-toggle"
            aria-expanded={!collapsed}
            title={collapsed ? 'Show members' : 'Hide members'}
            onClick={onToggleCollapse}
          >
            {collapsed ? '▸' : '▾'}
          </button>

          <span className="mt-group-key" title={
            group.derived
              ? 'Auto-derived from the members’ base name'
              : 'Membership set by an operator override'}>
            {group.group_key}
          </span>
          <span className="mt-group-count">
            {memberCount} iteration{memberCount === 1 ? '' : 's'}
          </span>
          {!group.derived && (
            <span className="badge badge-blue" title="Operator override">
              override
            </span>
          )}

          <span className="mt-group-ticks">
            {TICKS.map(t => (
              <TickSwitch
                key={t}
                groupKey={group.group_key}
                tick={t}
                on={ticks[t]}
                disabled={!enabled}
                hint={enabled ? null : offHint}
                onFlip={onFlip}
              />
            ))}
          </span>

          {!enabled && anyTick && (
            // An operator who ticked something while the feature is off should
            // see that it is stored but inert, rather than wonder why nothing
            // changed. Honest beats quiet.
            <span className="mt-group-inert" title={offHint}>
              ticks stored · inert while groups are off
            </span>
          )}
        </div>
      </td>
    </tr>
  )
}

/** The per-worker verdict chips shown under a member's name. */
export function MemberVerdicts ({ group, modelKey }) {
  const chips = memberVerdicts(group, modelKey)
  if (!chips.length) return null
  return (
    <span className="mt-verdicts">
      {chips.map(c => (
        <span
          key={`${c.worker}:${c.tone}`}
          className={`mt-worker-fit ${c.tone === 'ok' ? 'mt-fit-ok' : 'mt-fit-partial'}`}
          title={c.title}
        >
          {c.text}
        </span>
      ))}
    </span>
  )
}

export default GroupHeaderRow
