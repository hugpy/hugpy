# Console development shell

This is an independent copy of `react/ui`. Make console development changes here.
The console talks to the real central API on port 7002, and its worker roster is
scoped to this host's `ae-worker`. Model discovery, allocation and eviction
actions therefore exercise the actual Python backend and local worker.
The copied app uses local open-mode auth for this isolated dev shell; it does
not change authentication on the original console.

Start from this directory:

```sh
HUGPY_DEV_BIND=192.168.1.100 \
HUGPY_DEV_PORT=4173 \
HUGPY_DEV_HOST=192.168.1.100 \
HUGPY_DEV_WS_URL=ws://192.168.1.100:4173/ws \
npm run dev
```

Open <http://192.168.1.100:4173/>. The root redirects to `/console`.
