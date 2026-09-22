# Compartmentalized WorkersPanel

This package splits the original 5,706-line React module by responsibility while
preserving its component bodies, API calls, props, state behavior, and default
export.

## Integration

Copy the contents of `src/components/WorkersPanel/` over the existing
`src/components/WorkersPanel/` directory. Keep the project's existing
`WorkersPanel.css`: the pasted source referenced that stylesheet, but its bytes
were not included in the attachment, so this archive intentionally does not
replace it.

Existing imports that target `WorkersPanel.jsx` remain valid. The added
`index.js` also permits directory imports where the build setup supports them.

## Module map

- `WorkersPanel.jsx` — fleet data, API mutations, registration, and orchestration.
- `WorkerRow.jsx` — one worker's serving-table state and row interactions.
- `WorkerLoadTable.jsx` — load/allocation table behavior.
- `GroupAssignPanel.jsx` — multi-worker assignment UI.
- `AllocationControls.jsx` / `allocation.js` — allocation UI and pure policy transforms.
- `ResourceStrip.jsx`, `WorkerStorageBar.jsx`, `BudgetBars.jsx` — resource and storage views.
- `ResidencyMenu.jsx`, `SpillBadge.jsx` — focused controls and indicators.
- `workerMetrics.js`, `formatters.js`, `storageBadge.js` — pure helpers.
- `constants.js`, `useNarrowContainer.js` — shared configuration and responsive hook.

## Scope

The two stateful orchestrators remain intact deliberately. Extracting their
internal callbacks into custom hooks would be a second-stage refactor best done
inside the full repository with its test suite and backend contracts available.

See `VALIDATION.md` for the mechanical parity and bundle checks performed on
this package.
