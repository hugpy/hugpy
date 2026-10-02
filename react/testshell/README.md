# @hugpy/testshell

A deliberately simple, read-only console: **the UI is the database.** One table
per model, holding everything the `hugpy` Postgres DB knows about it — the
`models` row (attributes, serving settings), its `model_quants`, and one row per
worker from `model_workers` with the current allocation, the per-worker knobs
(`user_settings`) and activity. Worker names/liveness come from
`hugpy_worker_registry` and the `liveness` row of `hugpy_feed`. Nothing writes.

```
api.py          stdlib HTTP server + psycopg: GET /api/models, /api/models/<id>,
                /api/workers, /api/health; serves dist/ when built
model_full.sql  the view: one row per model (spec/serving/quants/workers=intent, live=mechanics)
src/App.jsx     the React view (polls /api/models every 5 s)
dev.sh          starts api.py (:7013) + vite dev server (:7014) together
```

## Run

```sh
cd react/testshell
./dev.sh                       # http://<host>:7014/
```

`dev.sh` symlinks `node_modules` to `../ui_shell/node_modules` if missing (same
react/vite versions, no install needed). To install standalone instead:
`npm install --workspaces=false`.

Env: `HUGPY_TESTSHELL_DSN` (default `dbname=hugpy`, local socket / peer auth),
`HUGPY_TESTSHELL_PORT` (api, 7013), `HUGPY_TESTSHELL_UI_PORT` (vite, 7014),
`HUGPY_TESTSHELL_BIND` (vite host, 0.0.0.0).

Production-style: `npm run build` then `python3 api.py` serves the SPA and the
API from one port.
