# @hugpy/ui-shared

The shared pieces every hugpy surface mounts: the ONE floating Help widget
(`help/helpWidget.js` + `.css`) and the cross-app navbar link registry
(`navbar/links.js` + `navbar.css`).

Plain DOM + CSS on purpose — no React, no framework, no dependencies — so the
same widget mounts identically in the webpack console, the vite arms, and any
future surface. Import the JS, include the CSS, done.

Licensed under the hugpy Source-Available License (see LICENSE).
