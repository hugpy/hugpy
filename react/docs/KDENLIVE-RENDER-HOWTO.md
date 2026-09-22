# Render your Kdenlive edits on hugpy (headless melt)

Edit on your Windows machine, render on the server. No GUI needed server-side.

## One-time setup (Windows)
1. Map the shares: clips live on `\\192.168.1.250\studio` (read-only);
   save your projects to `\\192.168.1.250\studio-edits` (read-write, user `ubuntu`).
2. Any custom fonts you use in title clips must ALSO be installed on the hugpy VM,
   or melt silently substitutes them. DejaVu + Liberation are pre-installed —
   ask the keeper to install others.
3. Match versions when convenient: server renders with **MLT 7.22 / frei0r 1.8**.

## Each edit
1. In Kdenlive, reference clips from the `studio` share, build your timeline,
   and **save the `.kdenlive` project into `\\192.168.1.250\studio-edits\`**.
2. Kick the render (operator token required):

```bash
curl -X POST https://dev.hugpy.ai/api/video/mlt/render \
  -H "X-Operator-Token: $TOK" -H 'Content-Type: application/json' \
  -d '{"project": "edits/myfilm.kdenlive"}'
# -> {"job_id": "..."}
```

Optional knobs: `"output"` (rel path, default `edits/renders/<project>.mp4`),
`"width"/"height"/"fps"` or `"profile"` (default: the project's own profile —
your authored resolution is respected), `"drive_letter":"Z:"` if your project
references clips through a mapped drive instead of UNC paths.

3. Watch it: `GET /video/jobs/<job_id>` (progress %); cancel via the normal
   job cancel. The finished mp4 appears **atomically** in
   `\\192.168.1.250\studio-edits\renders\` — if you can see it, it's complete.

## What the server does for you
- Rewrites the Windows paths (`\\192.168.1.250\studio\...`, any casing, either
  slash direction, or your declared drive letter) to the server's storage.
- Refuses honestly with a list of missing clips instead of rendering blanks.
- Renders ride the normal job queue (CPU-only, no GPU reservation), so they
  show up in Active Processes like everything else.

## Troubleshooting
- `melt_missing` error → melt was removed from the VM; keeper reinstalls.
- `unresolved_resources` → the listed clips aren't under the shares; move them
  into `studio`/`studio-edits` and re-save the project.
- Titles look wrong → font mismatch; install the font on the VM.
