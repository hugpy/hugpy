import path from 'node:path'
import { fileURLToPath } from 'node:url'
import HtmlWebpackPlugin from 'html-webpack-plugin'

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)

export default {
  entry: './src/main.jsx',

  output: {
    path: path.resolve(__dirname, 'dist'),
    filename: 'assets/[name].[contenthash].js',
    clean: true,
    publicPath: '/',
  },

  resolve: {
    extensions: ['.js', '.jsx', '.ts', '.tsx'],
    fallback: {
      util: false,        // stub out Node's util; the auth path doesn't need it in-browser
    },
  },
  devServer: {
  // All host wiring is env-overridable so other deployments aren't pinned to
  // this box's topology; the defaults keep the dev.hugpy.ai setup working.
  host: process.env.HUGPY_DEV_BIND || '0.0.0.0',   // reachable from the main server's nginx
  port: Number(process.env.HUGPY_DEV_PORT) || 7001,
  // demo.hugpy.ai is the host-side public demo vhost proxying to this watcher
  // (2026-07-21) — the browser's hostname drives demo mode, so serving it from
  // here is safe; without it webpack answers "Invalid Host header".
  allowedHosts: process.env.HUGPY_DEV_HOST ? [process.env.HUGPY_DEV_HOST] : ['dev.hugpy.ai', 'demo.hugpy.ai'],
  // HMR over nginx's TLS — tell the client to reach the socket via wss:443
  client: {
    webSocketURL: process.env.HUGPY_DEV_WS_URL
      || `wss://${process.env.HUGPY_DEV_HOST || 'dev.hugpy.ai'}:443/ws`,
    overlay: {
      errors: true,
      warnings: false,
      // "ResizeObserver loop…" is a benign browser notification (xterm's FitAddon
      // triggers it when the Console tab's terminal fits) — don't pop the overlay.
      runtimeErrors: (error) => !/ResizeObserver loop/.test(error && error.message || ''),
    },
  },
  // proxy /api to the local dev API; strip the /api prefix like prod nginx does
  proxy: [
    {
      // hugpy Station showroom (2026-08-13): the canned Station demo runs as
      // console-showroom.service on THIS VM's loopback in prefix mode, so
      // demo.hugpy.ai/station (and dev) serve it same-origin. The demo vhost
      // hard-404s only ^/api/, so /station rides straight through nginx here.
      context: ['/station'],
      target: process.env.HUGPY_STATION_DEMO_URL || 'http://127.0.0.1:8795',
      changeOrigin: true,
      ws: false,
    },
    {
      context: ['/api'],
      target: process.env.HUGPY_API_URL || 'http://127.0.0.1:7002',
      changeOrigin: true,
      pathRewrite: { '^/api': '' },
    },
    {
      // Expose the abstract_flask endpoint inspector at the bare public paths
      // (dev.hugpy.ai/endpoints, /prefixes) — Flask serves these at :7002 WITHOUT
      // the /api prefix, so DON'T rewrite the path (unlike /api above). Without
      // this, historyApiFallback swallows /endpoints and returns the SPA shell.
      //
      // EXACT-match only. A bare string context is prefix-matched (indexOf===0 in
      // http-proxy-middleware), so ['/endpoints'] would ALSO forward /endpointsFOO
      // to Flask, which has no such route and silently serves its stale bundled
      // catch-all instead of the live dev SPA. Anchor to the two real routes (plus
      // any real subpath) via a function matcher.
      context: (pathname) =>
        pathname === '/endpoints' || pathname === '/prefixes'
        || pathname.startsWith('/endpoints/') || pathname.startsWith('/prefixes/'),
      target: process.env.HUGPY_API_URL || 'http://127.0.0.1:7002',
      changeOrigin: true,
    },
    {
      // The OpenAI-compatible API lives at the BARE /v1/* (the external
      // contract every OpenAI SDK expects — baseURL https://dev.hugpy.ai/v1).
      // Flask serves it at :7002 without a prefix, so no rewrite. Same
      // exact-match discipline as /endpoints above (keeper 2026-07-14: public
      // /v1 was 404ing off the SPA devServer while local :7002 worked).
      context: (pathname) => pathname === '/v1' || pathname.startsWith('/v1/'),
      target: process.env.HUGPY_API_URL || 'http://127.0.0.1:7002',
      changeOrigin: true,
    },
  ],
  // Serve the standalone ARMs (built separately with their own Vite + Tailwind
  // v4 toolchains) on this same origin — loosely coupled, NOT bundled into the
  // SPA. Built output lives in each arm's own dist.
  static: [{
    directory: path.resolve(__dirname, '../media_intelligence_ui/dist'),
    publicPath: '/media',
  }, {
    directory: path.resolve(__dirname, '../video_intelligence_ui/dist'),
    publicPath: '/video',
  }, {
    directory: path.resolve(__dirname, '../agents_ui/dist'),
    publicPath: '/fleet',
  }],
  // SPA routing — but each arm's deep links fall back to the arm's own index,
  // not the SPA's.
  historyApiFallback: {
    rewrites: [
      { from: /^\/media(\/.*)?$/, to: '/media/index.html' },
      { from: /^\/video(\/.*)?$/, to: '/video/index.html' },
      { from: /^\/fleet(\/.*)?$/, to: '/fleet/index.html' },
    ],
  },
  // /studio — the public name for the Studio experience, whose stations live
  // INSIDE the video arm (/video/). A REDIRECT, not a historyApiFallback
  // rewrite: the video arm's BrowserRouter takes its basename from the Vite
  // `base` ("/video"), so serving its index.html under a /studio URL would
  // mismatch the basename and render a blank screen (the trap already
  // documented in video_intelligence_ui/entry.tsx). Moving the address bar to
  // /video/ first is the only version that actually works, and it costs one
  // extra hop. 302, not 301 — the mapping is a routing decision we may revisit,
  // and a permanent redirect would be pinned in browser caches forever.
  // Prod has the matching hop in flask_app/wsgi_app.py `_mount_ui`.
  // Registered as a bare `use` middleware matching on req.path rather than an
  // express route pattern: this runs across express 4 and 5 (dev-server 6 is on
  // express 5, where RegExp/param route syntax changed) and lands BEFORE the
  // internal middlewares — including historyApiFallback, which would otherwise
  // swallow /studio into the SPA shell.
  setupMiddlewares: (middlewares, devServer) => {
    devServer.app.use((req, res, next) => {
      const m = /^\/studio(?:\/(.*))?$/.exec(req.path || '')
      if (!m) return next()
      res.redirect(302, '/video/' + (m[1] || ''))
    })
    return middlewares
  },
},
  module: {
    rules: [
      {
        test: /\.(js|jsx|ts|tsx)$/,
        exclude: /node_modules/,
        use: {
          loader: 'babel-loader',
          options: {
            presets: [
              ['@babel/preset-env', { targets: 'defaults' }],
              ['@babel/preset-react', { runtime: 'automatic' }],
              '@babel/preset-typescript',
            ],
          },
        },
      },
      {
        test: /\.css$/,
        use: ['style-loader', 'css-loader'],
      },
      {
        test: /\.(png|jpe?g|svg|gif|webp)$/,
        type: 'asset/resource',
        generator: { filename: 'assets/[name].[contenthash][ext]' },
      },
    ],
  },

  plugins: [
    new HtmlWebpackPlugin({
      template: './index.html',
      favicon: './src/assets/hugpy-favicon-hex.svg',
    }),
  ],

}
