import e, { Component as t, Fragment as n, createContext as r, useCallback as i, useContext as a, useEffect as o, useMemo as s, useRef as c, useState as l } from "react";
import { Fragment as u, jsx as d, jsxs as f } from "react/jsx-runtime";
import { Link as p, Navigate as m, Outlet as h, useLocation as g, useNavigate as _ } from "react-router-dom";
import v from "@mui/material/Button";
import y from "@mui/material/TextField";
import b from "@mui/material/Stack";
import x from "@mui/material/Box";
import S from "@mui/material/Typography";
import { Alert as C, Card as w, CardContent as T, Collapse as E } from "@mui/material";
import D from "@mui/material/Grid";
//#region \0rolldown/runtime.js
var O = Object.create, k = Object.defineProperty, A = Object.getOwnPropertyDescriptor, j = Object.getOwnPropertyNames, M = Object.getPrototypeOf, N = Object.prototype.hasOwnProperty, P = (e, t) => () => (t || (e((t = { exports: {} }).exports, t), e = null), t.exports), F = (e, t, n, r) => {
	if (t && typeof t == "object" || typeof t == "function") for (var i = j(t), a = 0, o = i.length, s; a < o; a++) s = i[a], !N.call(e, s) && s !== n && k(e, s, {
		get: ((e) => t[e]).bind(null, s),
		enumerable: !(r = A(t, s)) || r.enumerable
	});
	return e;
}, I = (e, t, n) => (n = e == null ? {} : O(M(e)), F(t || !e || !e.__esModule ? k(n, "default", {
	value: e,
	enumerable: !0
}) : n, e)), L = {
	baseUrl: "",
	fetch: (...e) => globalThis.fetch(...e)
}, R = { ...L };
function z(e) {
	R = {
		...R,
		...e
	};
}
function B() {
	return R;
}
function V() {
	R = { ...L };
}
function ee(e) {
	return e.replace(/\/+$/, "");
}
var te = /^[a-z][a-z0-9+.-]*:\/\//i;
function H(e) {
	if (te.test(e)) return e;
	let t = ee(R.baseUrl || "");
	return t ? t + (e.startsWith("/") ? e : "/" + e) : e;
}
function ne() {
	return ee(R.baseUrl || "") || (typeof window < "u" && window.location ? window.location.origin : "");
}
async function re(e) {
	let t = R, n = new Headers(e?.headers || {});
	if (t.headers) {
		let e = typeof t.headers == "function" ? await t.headers() : t.headers;
		new Headers(e).forEach((e, t) => {
			n.has(t) || n.set(t, e);
		});
	}
	let r = {
		...e,
		headers: n
	};
	return t.credentials && r.credentials == null && (r.credentials = t.credentials), r;
}
async function U(e, t) {
	return R.fetch(H(e), await re(t));
}
//#endregion
//#region src/runtime/HugpyProvider.tsx
var W = r(B());
function G({ children: e, ...t }) {
	let n = s(() => (z(t), B()), [
		t.baseUrl,
		t.credentials,
		t.fetch,
		t.headers
	]);
	return /* @__PURE__ */ d(W.Provider, {
		value: n,
		children: e
	});
}
function ie() {
	return a(W);
}
//#endregion
//#region src/api.ts
function ae(e, t) {
	if (e && typeof e == "object") {
		let t = e;
		if (typeof t.error == "string" && t.error.trim()) return t.error;
		if (typeof t.detail == "string" && t.detail.trim()) return t.detail;
		if (typeof t.message == "string" && t.message.trim()) return t.message;
	}
	return t;
}
function oe(e, t) {
	try {
		return JSON.parse(e);
	} catch {
		throw Error(e.trim() || `HTTP ${t} (empty response)`);
	}
}
async function K(e, t) {
	let n = await U(e, t), r = oe(await n.text(), n.status);
	if (!n.ok) throw Error(ae(r, `HTTP ${n.status}`));
	return r;
}
async function se(e) {
	let t = new FormData();
	return t.append("file", e), K("/api/uploads", {
		method: "POST",
		body: t
	});
}
//#endregion
//#region src/assets/hugpy-mark.png
var ce = "data:image/png;base64,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";
//#endregion
//#region src/components/BrandMark/BrandMark.jsx
function le({ word: e = "HUGPY", tagline: t = "inference you own", className: n = "" }) {
	return /* @__PURE__ */ f(p, {
		to: "/",
		className: `brandmark ${n}`,
		title: "Back to the welcome page",
		children: [/* @__PURE__ */ d("img", {
			className: "brandmark-img",
			src: ce,
			alt: ""
		}), /* @__PURE__ */ f("span", {
			className: "brandmark-text",
			children: [
				/* @__PURE__ */ d("span", {
					className: "brandmark-word",
					children: e
				}),
				t && /* @__PURE__ */ d("span", {
					className: "brandmark-rule",
					"aria-hidden": "true"
				}),
				t && /* @__PURE__ */ d("span", {
					className: "brandmark-sub",
					children: t
				})
			]
		})]
	});
}
//#endregion
//#region src/Auth/authConfig.ts
var ue = {
	mode: "open",
	base: "/api/auth-svc",
	reachable: !1
}, de = null;
async function fe() {
	try {
		let e = new AbortController(), t = setTimeout(() => e.abort(), 4e3), n;
		try {
			n = await U("/api/auth/config", {
				headers: { Accept: "application/json" },
				signal: e.signal
			});
		} finally {
			clearTimeout(t);
		}
		if (n.status >= 500) return "retry";
		if (!n.ok) return { ...ue };
		let r = await n.text();
		try {
			let e = JSON.parse(r);
			return e && (e.mode === "open" || e.mode === "external") ? {
				mode: e.mode,
				base: e.base ?? ue.base,
				reachable: !0
			} : { ...ue };
		} catch {
			return { ...ue };
		}
	} catch {
		return "retry";
	}
}
async function pe() {
	for (let e = 0; e < 3; e++) {
		let t = await fe();
		if (t !== "retry") return t;
		e < 2 && await new Promise((t) => setTimeout(t, 600 * (e + 1)));
	}
	return { ...ue };
}
function me() {
	return de ||= pe(), de;
}
async function he() {
	return (await me()).base ?? ue.base;
}
//#endregion
//#region ../ui_shared/navbar/links.js
var ge = [
	{
		key: "docs",
		label: "Docs",
		path: "/docs"
	},
	{
		key: "console",
		label: "Console",
		path: "/console"
	},
	{
		key: "media",
		label: "Media",
		path: "/media/"
	},
	{
		key: "video",
		label: "Video",
		path: "/video/"
	},
	{
		key: "fleet",
		label: "Fleet",
		path: "/fleet/"
	}
];
function _e(e = {}) {
	let { currentKey: t, siteBase: n = "", hrefByKey: r = {} } = e, i = n ? n.replace(/\/$/, "") : "";
	return ge.map((e) => ({
		key: e.key,
		label: e.label,
		href: r[e.key] == null ? `${i}${e.path}` : r[e.key],
		current: t === e.key
	}));
}
//#endregion
//#region src/runtime/localCentral.js
function ve() {
	try {
		return /^demo\.hugpy\.ai$/i.test(window.location.hostname);
	} catch {
		return !1;
	}
}
//#endregion
//#region src/components/Navbar/Navbar.jsx
var ye = /* @__PURE__ */ new Set(["docs", "console"]);
function be({ brand: e = !0, left: t, children: n, banner: r, className: i = "" }) {
	let [a, s] = l("/console");
	return o(() => {
		let e = !0;
		return me().then((t) => {
			e && s(t.mode === "external" ? "/login" : "/console");
		}), () => {
			e = !1;
		};
	}, []), /* @__PURE__ */ f("header", {
		className: "hugpy-navbar-stack",
		children: [/* @__PURE__ */ f("nav", {
			className: `hugpy-navbar ${i}`,
			children: [
				/* @__PURE__ */ f("span", {
					className: "hugpy-navbar-brand",
					children: [t, e && /* @__PURE__ */ d(le, {})]
				}),
				/* @__PURE__ */ f("span", {
					className: "hugpy-navbar-links",
					children: [_e({ hrefByKey: { console: a } }).map((e) => ye.has(e.key) ? /* @__PURE__ */ d(p, {
						to: e.href,
						children: e.label
					}, e.key) : /* @__PURE__ */ d("a", {
						href: e.href,
						children: e.label
					}, e.key)), ve() && /* @__PURE__ */ f(u, { children: [/* @__PURE__ */ d("a", {
						href: "/station/",
						target: "_blank",
						rel: "noreferrer",
						children: "Station"
					}), /* @__PURE__ */ d("a", {
						href: "/station/?open=tern:keeper",
						target: "_blank",
						rel: "noreferrer",
						children: "Agent"
					})] })]
				}),
				/* @__PURE__ */ d("span", {
					className: "hugpy-navbar-side",
					children: n
				})
			]
		}), r]
	});
}
//#endregion
//#region src/components/ApiAccess/InstallLinks.jsx
function xe(e) {
	return e ? (/* @__PURE__ */ new Date(e * 1e3)).toLocaleString() : "–";
}
function Se({ text: e, label: t = "copy" }) {
	let [n, r] = l(!1);
	return /* @__PURE__ */ d("button", {
		className: "aa-copy",
		onClick: () => {
			navigator.clipboard?.writeText(e).then(() => {
				r(!0), setTimeout(() => r(!1), 1500);
			});
		},
		children: n ? "✓ copied" : t
	});
}
var Ce = [{
	id: "agent",
	label: "hugpy-agent",
	hint: "the hugpy-agent installer (all platforms)"
}, {
	id: "console",
	label: "hugpy Station",
	hint: "the desktop Station .deb + its hugpy key (Linux)"
}], we = [
	{
		id: "v1",
		hint: "chat / models (/v1)"
	},
	{
		id: "ml",
		hint: "media intelligence (/ml)"
	},
	{
		id: "agent-register",
		hint: "fleet enroll (/agent/register)"
	},
	{
		id: "full",
		hint: "everything"
	}
], q = [
	{
		label: "1 hour",
		s: 3600
	},
	{
		label: "24 hours",
		s: 86400
	},
	{
		label: "7 days",
		s: 604800
	}
], Te = [
	1,
	3,
	10
];
function Ee() {
	let [e, t] = l([]), [n, r] = l("agent"), [a, s] = l(""), [c, p] = l(["v1"]), [m, h] = l(86400), [g, _] = l(1), [v, y] = l(""), [b, x] = l(null), [S, C] = l(!1), [w, T] = l(null), E = i(() => {
		K("/api/agent/install-links").then((e) => {
			t(e.links ?? []), T(null);
		}).catch((e) => T(e.message));
	}, []);
	o(() => {
		E();
	}, [E]);
	let D = i((e) => {
		p((t) => t.includes(e) ? t.filter((t) => t !== e) : [...t, e]);
	}, []), O = i(() => {
		if (!a.trim()) {
			alert("An install link needs a label.");
			return;
		}
		if (c.length === 0) {
			alert("Pick at least one scope.");
			return;
		}
		C(!0);
		let e = {
			label: a.trim(),
			scopes: c,
			link_ttl_s: m,
			max_uses: g
		};
		if (v) {
			let t = new Date(v).getTime();
			Number.isFinite(t) && (e.key_expires_at = t / 1e3);
		}
		K(n === "console" ? "/api/agent/console/install-links" : "/api/agent/install-links", {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify(e)
		}).then((e) => {
			x(e), s(""), E();
		}).catch((e) => alert(`Install-link mint failed: ${e.message}`)).finally(() => C(!1));
	}, [
		n,
		a,
		c,
		m,
		g,
		v,
		E
	]), k = i((e) => {
		confirm(`Revoke install link "${e.label}"? Its key stops working too.`) && K(`/api/agent/install-links/${e.link_id}`, { method: "DELETE" }).then(E).catch((e) => alert(`Revoke failed: ${e.message}`));
	}, [E]), A = i((e) => {
		let t = e.status === "active" ? `Remove ACTIVE install link "${e.label}"? Its unused key dies with it.` : `Remove install link "${e.label}" from the list? (Already-installed machines keep working.)`;
		confirm(t) && K(`/api/agent/install-links/${e.link_id}?purge=1`, { method: "DELETE" }).then(E).catch((e) => alert(`Remove failed: ${e.message}`));
	}, [E]), j = i(() => {
		confirm("Remove ALL used-up / expired / revoked links from the list? Active links are untouched.") && K("/api/agent/install-links/prune", { method: "POST" }).then((e) => {
			E(), e?.pruned != null && alert(`${e.pruned} dead link${e.pruned === 1 ? "" : "s"} removed.`);
		}).catch((e) => alert(`Prune failed: ${e.message}`));
	}, [E]), M = (e, t) => e?.commands?.[t] ? e.commands[t] : t === "windows" ? `irm ${e.url}.ps1 | iex` : `curl -fsSL ${e.url}.sh | bash`, N = (e, t) => e?.downloads?.[t] ? e.downloads[t] : `${e.url}.zip`, P = (b?.kind ?? "agent") === "console", F = e.filter((e) => e.status !== "active").length;
	return /* @__PURE__ */ f("div", {
		className: "aa-install-links",
		children: [
			/* @__PURE__ */ f("div", {
				className: "aa-row",
				children: [
					/* @__PURE__ */ d("span", {
						className: "aa-label",
						children: "Install link"
					}),
					/* @__PURE__ */ d("select", {
						className: "aa-select",
						value: n,
						onChange: (e) => r(e.target.value),
						title: "What the link installs.",
						children: Ce.map((e) => /* @__PURE__ */ d("option", {
							value: e.id,
							title: e.hint,
							children: e.label
						}, e.id))
					}),
					/* @__PURE__ */ d("input", {
						className: "aa-name",
						placeholder: "label (who / what box this is for)…",
						value: a,
						onChange: (e) => s(e.target.value),
						onKeyDown: (e) => {
							e.key === "Enter" && O();
						}
					}),
					/* @__PURE__ */ d("select", {
						className: "aa-select",
						value: m,
						onChange: (e) => h(Number(e.target.value)),
						title: "How long the download link itself stays fetchable.",
						children: q.map((e) => /* @__PURE__ */ f("option", {
							value: e.s,
							children: ["link: ", e.label]
						}, e.s))
					}),
					/* @__PURE__ */ d("select", {
						className: "aa-select",
						value: g,
						onChange: (e) => _(Number(e.target.value)),
						title: "How many times the installer may be downloaded.",
						children: Te.map((e) => /* @__PURE__ */ f("option", {
							value: e,
							children: [
								e,
								" use",
								e > 1 ? "s" : ""
							]
						}, e))
					}),
					/* @__PURE__ */ d("button", {
						className: "btn-primary",
						onClick: O,
						disabled: S,
						children: S ? "…" : "+ Mint link"
					})
				]
			}),
			/* @__PURE__ */ f("div", {
				className: "aa-row",
				children: [
					/* @__PURE__ */ d("span", {
						className: "aa-label",
						children: "Scopes"
					}),
					/* @__PURE__ */ d("div", {
						className: "aa-scope-picks",
						children: we.map((e) => /* @__PURE__ */ f("label", {
							className: "aa-toggle",
							title: e.hint,
							children: [/* @__PURE__ */ d("input", {
								type: "checkbox",
								checked: c.includes(e.id),
								onChange: () => D(e.id)
							}), e.id]
						}, e.id))
					}),
					/* @__PURE__ */ f("label", {
						className: "aa-toggle",
						title: "Optional expiry for the KEY the link mints (the link's own TTL is separate).",
						children: ["key expires", /* @__PURE__ */ d("input", {
							className: "aa-name aa-key-expiry",
							type: "datetime-local",
							value: v,
							onChange: (e) => y(e.target.value)
						})]
					}),
					/* @__PURE__ */ d("span", {
						className: "aa-note",
						children: n === "console" ? "Installs the hugpy Station .deb and writes a freshly minted key (with these scopes) to ~/.fleet/console-hugpy.env on the target box — the credential its hugpy agents use. The raw key is never shown here." : "The download bakes a freshly minted key (with these scopes) into the installer. The raw key is never shown — it exists only inside the one-time download."
					})
				]
			}),
			w && /* @__PURE__ */ d("div", {
				className: "aa-error",
				children: w
			}),
			b && /* @__PURE__ */ f("div", {
				className: "aa-fresh",
				children: [
					/* @__PURE__ */ f("div", {
						className: "aa-fresh-head",
						children: [/* @__PURE__ */ d("span", { children: P ? "hugpy Station install link minted — run this on the target Linux box:" : "Install link minted — hand this one-liner to the target box:" }), /* @__PURE__ */ d("button", {
							className: "aa-fresh-dismiss",
							onClick: () => x(null),
							children: "✕"
						})]
					}),
					/* @__PURE__ */ f("div", {
						className: "aa-fresh-key",
						children: [
							/* @__PURE__ */ d("span", {
								className: "aa-label",
								children: "Linux"
							}),
							/* @__PURE__ */ d("code", { children: M(b, "linux") }),
							/* @__PURE__ */ d(Se, {
								text: M(b, "linux"),
								label: "copy linux"
							})
						]
					}),
					!P && /* @__PURE__ */ f("div", {
						className: "aa-fresh-key",
						children: [
							/* @__PURE__ */ d("span", {
								className: "aa-label",
								children: "macOS"
							}),
							/* @__PURE__ */ d("code", { children: M(b, "macos") }),
							/* @__PURE__ */ d(Se, {
								text: M(b, "macos"),
								label: "copy macos"
							}),
							/* @__PURE__ */ d("a", {
								className: "aa-copy",
								href: N(b, "macos_zip"),
								download: !0,
								children: "download installer (.zip)"
							}),
							b.downloads?.macos_pkg && /* @__PURE__ */ d("a", {
								className: "aa-copy",
								href: b.downloads.macos_pkg,
								download: !0,
								children: "download installer (.pkg)"
							})
						]
					}),
					!P && /* @__PURE__ */ f("div", {
						className: "aa-fresh-key",
						children: [
							/* @__PURE__ */ d("span", {
								className: "aa-label",
								children: "Windows"
							}),
							/* @__PURE__ */ d("code", { children: M(b, "windows") }),
							/* @__PURE__ */ d(Se, {
								text: M(b, "windows"),
								label: "copy windows"
							})
						]
					}),
					P ? /* @__PURE__ */ d("span", {
						className: "aa-note",
						children: "Fetching the command's URL consumes the link's use and delivers the key, so paste it straight into a terminal on the target box (sudo is used only for the apt install step). The script verifies the .deb's sha256 before installing."
					}) : /* @__PURE__ */ f("span", {
						className: "aa-note",
						children: ["Paste the command into a terminal on the target box — don't download the script and double-click/run it (downloaded files aren't executable, and macOS quarantines them). On macOS you can instead use the zip download above: after extracting it, the first run needs right-click → Open (not double-click) since the installer is unsigned and Gatekeeper blocks a plain double-click once — after that first approval it runs normally.", b.downloads?.macos_pkg && /* @__PURE__ */ d(u, { children: " The .pkg gives an installer window instead of a terminal. Same unsigned first run (right-click → Open, or System Settings → Privacy & Security → “Open Anyway” on macOS 15+), and because an installer shows no output it writes everything to ~/hugpy-agent/install.log — read that if anything goes wrong." })]
					}),
					/* @__PURE__ */ f("span", {
						className: "aa-note",
						children: [
							b.max_uses,
							" download",
							b.max_uses > 1 ? "s" : "",
							" ·",
							" ",
							"expires ",
							xe(b.expires_at),
							" · scopes: ",
							b.scopes.join(", ")
						]
					})
				]
			}),
			e.length > 0 && /* @__PURE__ */ f(u, { children: [/* @__PURE__ */ f("table", {
				className: "aa-keys",
				children: [/* @__PURE__ */ d("thead", { children: /* @__PURE__ */ f("tr", { children: [
					/* @__PURE__ */ d("th", { children: "Label" }),
					/* @__PURE__ */ d("th", { children: "Kind" }),
					/* @__PURE__ */ d("th", { children: "Scopes" }),
					/* @__PURE__ */ d("th", { children: "Status" }),
					/* @__PURE__ */ d("th", { children: "Uses" }),
					/* @__PURE__ */ d("th", { children: "Created" }),
					/* @__PURE__ */ d("th", { children: "Expires" }),
					/* @__PURE__ */ d("th", {})
				] }) }), /* @__PURE__ */ d("tbody", { children: e.map((e) => /* @__PURE__ */ f("tr", { children: [
					/* @__PURE__ */ d("td", {
						className: "aa-key-name",
						children: e.label
					}),
					/* @__PURE__ */ d("td", { children: /* @__PURE__ */ d("span", {
						className: "aa-chip",
						children: (e.kind ?? "agent") === "console" ? "fleet-console" : "hugpy-agent"
					}) }),
					/* @__PURE__ */ d("td", { children: (e.scopes ?? []).map((e) => /* @__PURE__ */ d("span", {
						className: "aa-chip",
						children: e
					}, e)) }),
					/* @__PURE__ */ d("td", { children: /* @__PURE__ */ d("span", {
						className: `aa-chip aa-status-${e.status}`,
						children: e.status
					}) }),
					/* @__PURE__ */ f("td", {
						className: "aa-dim",
						children: [
							(e.max_uses ?? 0) - (e.uses_left ?? 0),
							"/",
							e.max_uses
						]
					}),
					/* @__PURE__ */ d("td", {
						className: "aa-dim",
						children: xe(e.created_at)
					}),
					/* @__PURE__ */ d("td", {
						className: "aa-dim",
						children: xe(e.expires_at)
					}),
					/* @__PURE__ */ f("td", {
						className: "aa-key-actions",
						children: [e.status === "active" && /* @__PURE__ */ d("button", {
							className: "aa-revoke",
							onClick: () => k(e),
							children: "revoke"
						}), /* @__PURE__ */ d("button", {
							className: "aa-revoke",
							title: "Remove this row from the ledger",
							onClick: () => A(e),
							children: "remove"
						})]
					})
				] }, e.link_id)) })]
			}), F > 0 && /* @__PURE__ */ f("div", {
				className: "aa-row",
				children: [/* @__PURE__ */ f("button", {
					className: "aa-revoke",
					onClick: j,
					children: [
						"clear ",
						F,
						" dead link",
						F === 1 ? "" : "s"
					]
				}), /* @__PURE__ */ d("span", {
					className: "aa-note",
					children: "Removes every used-up / expired / revoked row above. Machines installed from a used-up link keep their key — this only cleans the list."
				})]
			})] })
		]
	});
}
//#endregion
//#region src/components/ApiAccess/ConsoleDownload.jsx
function De(e) {
	return !e && e !== 0 ? "–" : e > 1 << 20 ? (e / (1 << 20)).toFixed(1) + " MB" : Math.round(e / 1024) + " KB";
}
function Oe({ label: e, info: t, hint: n }) {
	return t ? /* @__PURE__ */ f("div", {
		className: "aa-row",
		children: [
			/* @__PURE__ */ d("span", {
				className: "aa-label",
				children: e
			}),
			/* @__PURE__ */ f("a", {
				className: "aa-copy",
				href: t.url,
				download: !0,
				children: [
					"download ",
					t.filename,
					" (",
					De(t.size_bytes),
					")"
				]
			}),
			t.sha256 && /* @__PURE__ */ f("span", {
				className: "aa-note",
				title: "sha256 " + t.sha256,
				children: [
					"sha256 ",
					t.sha256.slice(0, 12),
					"…"
				]
			}),
			n && /* @__PURE__ */ d("span", {
				className: "aa-note",
				children: n
			})
		]
	}) : /* @__PURE__ */ f("div", {
		className: "aa-row",
		children: [/* @__PURE__ */ d("span", {
			className: "aa-label",
			children: e
		}), /* @__PURE__ */ d("span", {
			className: "aa-note",
			children: "none staged on this deployment"
		})]
	});
}
function ke() {
	let [e, t] = l(null), [n, r] = l(null);
	return o(() => {
		K("/api/agent/console/info").then(t).catch((e) => r(e.message || String(e)));
	}, []), n ? null : /* @__PURE__ */ f("div", {
		className: "aa-install-links",
		children: [
			/* @__PURE__ */ f("div", {
				className: "aa-row",
				children: [/* @__PURE__ */ d("span", {
					className: "aa-label",
					children: "hugpy Station"
				}), /* @__PURE__ */ d("span", {
					className: "aa-note",
					children: "The operator's desktop console (Linux .deb) and the hugpy-agent wheel it calls into — install the wheel first, then the deb, then launch from the desktop entry (not the bare binary)."
				})]
			}),
			/* @__PURE__ */ d(Oe, {
				label: "console (.deb)",
				info: e?.deb,
				hint: "sudo apt install ./<file>"
			}),
			/* @__PURE__ */ d(Oe, {
				label: "agent (.whl)",
				info: e?.agent_whl,
				hint: "pip install --upgrade ./<file>"
			}),
			e?.install?.example && /* @__PURE__ */ f("div", {
				className: "aa-row",
				children: [
					/* @__PURE__ */ d("span", {
						className: "aa-label",
						children: "one-liner"
					}),
					/* @__PURE__ */ d("code", { children: e.install.example }),
					/* @__PURE__ */ d("button", {
						className: "aa-copy",
						onClick: () => navigator.clipboard?.writeText(e.install.example),
						children: "copy"
					}),
					/* @__PURE__ */ d("span", {
						className: "aa-note",
						children: "Downloads + sha256-verifies + installs the newest staged artifact (deb/rpm/pacman/AppImage auto-detected). HUGPY_TOKEN takes an API key minted above — the installer provisions it into ~/.config/hugpy-station/station.env on the target — or an operator token (authorizes the download only, never persisted)."
					})
				]
			})
		]
	});
}
//#endregion
//#region src/components/ApiAccess/ApiAccess.jsx
function Ae(e) {
	return e ? (/* @__PURE__ */ new Date(e * 1e3)).toLocaleString() : "–";
}
function je({ text: e, label: t = "copy" }) {
	let [n, r] = l(!1);
	return /* @__PURE__ */ d("button", {
		className: "aa-copy",
		onClick: () => {
			navigator.clipboard?.writeText(e).then(() => {
				r(!0), setTimeout(() => r(!1), 1500);
			});
		},
		children: n ? "✓ copied" : t
	});
}
function Me({ models: e = [], embedded: t = !1 }) {
	let [n, r] = l(!1), [a, s] = l([]), [c, u] = l(!1), [p, m] = l(!1), [h, g] = l(""), [_, v] = l(""), [y, b] = l(null), [x, S] = l(!1), [C, w] = l(null), [T, E] = l(null), [D, O] = l(""), [k, A] = l(!1), [j, M] = l(null), N = `${ne()}/api/v1`, P = e.find((e) => e.status === "installed" && ["text-generation", "image-text-to-text"].includes(e.primary_task ?? e.task))?.model_key ?? e.find((e) => e.status === "installed")?.model_key ?? "<model_key>", F = i(() => {
		K("/api/keys").then((e) => {
			s(e.keys ?? []), u(!!e.require_key), w(null);
		}).catch((e) => w(e.message)), K("/api/ml/gate").then((e) => m(!!e.require_key)).catch(() => {}), K("/api/llm/hf/auth").then((e) => {
			E(e), M(null);
		}).catch((e) => M(e.message));
	}, []);
	o(() => {
		(t || n) && F();
	}, [
		t,
		n,
		F
	]);
	let I = i(() => {
		S(!0), K("/api/keys", {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({
				name: h,
				pool: _.trim()
			})
		}).then((e) => {
			b(e), g(""), v(""), F();
		}).catch((e) => alert(`Key creation failed: ${e.message}`)).finally(() => S(!1));
	}, [
		h,
		_,
		F
	]), L = i((e) => {
		confirm(`Revoke key "${e.name}" (${e.prefix}…)? Calls using it will stop working.`) && K(`/api/keys/${e.id}`, { method: "DELETE" }).then(F).catch((e) => alert(`Revoke failed: ${e.message}`));
	}, [F]), R = i(() => {
		let e = a.filter((e) => e.expired).length, t = { expired: !0 }, n = e ? `Remove ${e} expired key${e > 1 ? "s" : ""}?` : "No expired keys.", r = prompt(`${n}\n\nAlso remove keys older than how many days? (blank = expired only; only never-used keys are swept by age)`, "");
		if (r === null) return;
		let i = parseFloat(r);
		if (!Number.isNaN(i) && i > 0) t = {
			...t,
			older_than_days: i,
			unused_only: !0
		};
		else if (!e) return;
		K("/api/keys/prune", {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify(t)
		}).then((e) => {
			alert(`Removed ${e.count} key${e.count === 1 ? "" : "s"}.`), F();
		}).catch((e) => alert(`Prune failed: ${e.message}`));
	}, [a, F]), z = i(() => {
		K("/api/keys/require", {
			method: "PUT",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({ require: !c })
		}).then((e) => u(!!e.require_key)).catch((e) => alert(`Toggle failed: ${e.message}`));
	}, [c]), B = i(() => {
		K("/api/ml/gate", {
			method: "PUT",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({ require: !p })
		}).then((e) => m(!!e.require_key)).catch((e) => alert(`Media gate toggle failed: ${e.message}`));
	}, [p]), V = i(() => {
		let e = D.trim();
		e && (A(!0), M(null), K("/api/llm/hf/auth", {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({ token: e })
		}).then((e) => {
			E(e), O("");
		}).catch((e) => M(e.message)).finally(() => A(!1)));
	}, [D]), ee = i(() => {
		confirm("Remove the saved Hugging Face token? HF calls go back to anonymous (rate-limited).") && (A(!0), M(null), K("/api/llm/hf/auth", { method: "DELETE" }).then((e) => E(e)).catch((e) => M(e.message)).finally(() => A(!1)));
	}, []), te = [
		`curl ${N}/chat/completions \\`,
		"  -H \"Content-Type: application/json\" \\",
		...c ? ["  -H \"Authorization: Bearer <your-key>\" \\"] : [],
		`  -d '{"model": "${P}", "stream": true,`,
		"       \"messages\": [{\"role\": \"user\", \"content\": \"Hello!\"}]}'"
	].join("\n");
	return /* @__PURE__ */ f("section", {
		className: "api-access",
		children: [!t && /* @__PURE__ */ f("div", {
			className: "section-strip clickable",
			onClick: () => r((e) => !e),
			title: "Programmatic access to the local models (OpenAI-compatible)",
			children: [
				/* @__PURE__ */ d("span", {
					className: "section-caret",
					children: n ? "▾" : "▸"
				}),
				/* @__PURE__ */ d("span", {
					className: "section-title",
					children: "API access"
				}),
				!n && /* @__PURE__ */ f("span", {
					className: "aa-strip-hint",
					children: [N, " · OpenAI-compatible"]
				}),
				/* @__PURE__ */ d("span", {
					className: `section-count ${c ? "aa-locked" : "aa-open"}`,
					children: c ? "🔒 key required" : "open · no key needed"
				})
			]
		}), (t || n) && /* @__PURE__ */ f("div", {
			className: "aa-body",
			children: [
				C && /* @__PURE__ */ d("div", {
					className: "aa-error",
					children: C
				}),
				/* @__PURE__ */ f("div", {
					className: "aa-row aa-endpoint",
					children: [
						/* @__PURE__ */ d("span", {
							className: "aa-label",
							children: "Endpoint"
						}),
						/* @__PURE__ */ d("code", {
							className: "aa-url",
							children: N
						}),
						/* @__PURE__ */ d(je, { text: N }),
						/* @__PURE__ */ f("span", {
							className: "aa-note",
							children: [
								"OpenAI-compatible: ",
								/* @__PURE__ */ d("code", { children: "/chat/completions" }),
								" & ",
								/* @__PURE__ */ d("code", { children: "/models" }),
								" — works with any OpenAI SDK via ",
								/* @__PURE__ */ d("code", { children: "base_url" }),
								"."
							]
						})
					]
				}),
				/* @__PURE__ */ f("div", {
					className: "aa-row",
					children: [
						/* @__PURE__ */ d("span", {
							className: "aa-label",
							children: "Auth"
						}),
						/* @__PURE__ */ f("label", {
							className: "aa-toggle",
							children: [/* @__PURE__ */ d("input", {
								type: "checkbox",
								checked: c,
								onChange: z
							}), "require an API key for /v1 calls"]
						}),
						/* @__PURE__ */ d("span", {
							className: "aa-note",
							children: c ? "Calls must send Authorization: Bearer <key>." : "The API is open; keys are optional until you turn this on."
						})
					]
				}),
				/* @__PURE__ */ f("div", {
					className: "aa-row",
					children: [
						/* @__PURE__ */ d("span", {
							className: "aa-label",
							children: "New key"
						}),
						/* @__PURE__ */ d("input", {
							className: "aa-name",
							placeholder: "key name (e.g. laptop, notebook, cron)…",
							value: h,
							onChange: (e) => g(e.target.value),
							onKeyDown: (e) => {
								e.key === "Enter" && I();
							}
						}),
						/* @__PURE__ */ d("input", {
							className: "aa-name",
							placeholder: "pool (optional — dedicated worker)",
							title: "Bind this key to a dedicated worker pool: its requests route to that pool's reserved workers.",
							value: _,
							onChange: (e) => v(e.target.value),
							onKeyDown: (e) => {
								e.key === "Enter" && I();
							}
						}),
						/* @__PURE__ */ d("button", {
							className: "btn-primary",
							onClick: I,
							disabled: x,
							children: x ? "…" : "+ Create key"
						})
					]
				}),
				/* @__PURE__ */ f("div", {
					className: "aa-row aa-media-gate",
					children: [
						/* @__PURE__ */ d("span", {
							className: "aa-label",
							children: "Media gate"
						}),
						/* @__PURE__ */ f("label", {
							className: "aa-toggle",
							children: [/* @__PURE__ */ d("input", {
								type: "checkbox",
								checked: p,
								onChange: B
							}), "require an API key for media-intelligence (/ml) access"]
						}),
						/* @__PURE__ */ d("span", {
							className: `section-count ${p ? "aa-locked" : "aa-open"}`,
							children: p ? "🔒 key required" : "open · no key needed"
						}),
						/* @__PURE__ */ f("span", {
							className: "aa-note",
							children: [
								p ? "Media-intelligence /ml calls must send Authorization: Bearer <key>." : "Media-intelligence is open within the console; turn on to gate it.",
								" ",
								"Separate from the console login and the /v1 gate above.",
								" ",
								"Mint a key below to get a shareable ",
								/* @__PURE__ */ d("strong", { children: "live-demo link" }),
								" that opens the media UI against this backend."
							]
						})
					]
				}),
				/* @__PURE__ */ f("div", {
					className: "aa-row aa-hf-gate",
					children: [/* @__PURE__ */ d("span", {
						className: "aa-label",
						children: "HF login"
					}), /* @__PURE__ */ f("div", {
						className: "aa-hf-body",
						children: [
							/* @__PURE__ */ d("div", {
								className: "aa-hf-status",
								children: T == null ? /* @__PURE__ */ d("span", {
									className: "aa-note",
									children: "Checking Hugging Face status…"
								}) : T.authenticated === !0 ? /* @__PURE__ */ f("span", {
									className: "section-count aa-open",
									children: [
										"✓ Authenticated",
										T.username ? ` as ${T.username}` : "",
										T.source === "env" ? " (env)" : "",
										T.token_last4 ? ` · …${T.token_last4}` : ""
									]
								}) : T.authenticated === null ? /* @__PURE__ */ d("span", {
									className: "section-count aa-locked",
									children: "Hugging Face unreachable — status unknown"
								}) : T.token_last4 ? /* @__PURE__ */ f("span", {
									className: "section-count aa-locked",
									children: ["⚠ Saved token rejected by Hugging Face", T.note ? ` (${T.note})` : ""]
								}) : /* @__PURE__ */ d("span", {
									className: "section-count aa-locked",
									children: "Anonymous — HF calls are rate-limited"
								})
							}),
							/* @__PURE__ */ f("div", {
								className: "aa-hf-input",
								children: [
									/* @__PURE__ */ d("input", {
										className: "aa-name",
										type: "password",
										autoComplete: "off",
										placeholder: "hf_… token (write-only, never shown again)",
										value: D,
										onChange: (e) => O(e.target.value),
										onKeyDown: (e) => {
											e.key === "Enter" && V();
										}
									}),
									/* @__PURE__ */ d("button", {
										className: "btn-primary",
										onClick: V,
										disabled: k || !D.trim(),
										children: k ? "…" : "Save"
									}),
									T && (T.source || T.token_last4) && /* @__PURE__ */ d("button", {
										className: "aa-revoke",
										onClick: ee,
										disabled: k,
										children: "clear"
									})
								]
							}),
							j && /* @__PURE__ */ d("div", {
								className: "aa-error",
								children: j
							}),
							/* @__PURE__ */ d("span", {
								className: "aa-note",
								children: "Save a Hugging Face token so model search, metadata, and downloads go out authenticated instead of anonymously rate-limited. Stored server-side (0600, never returned). A token is validated against HF before it is saved."
							})
						]
					})]
				}),
				y && /* @__PURE__ */ f("div", {
					className: "aa-fresh",
					children: [
						/* @__PURE__ */ f("div", {
							className: "aa-fresh-head",
							children: [/* @__PURE__ */ d("span", { children: "Key created — copy it now, it won't be shown again:" }), /* @__PURE__ */ d("button", {
								className: "aa-fresh-dismiss",
								onClick: () => b(null),
								children: "✕"
							})]
						}),
						/* @__PURE__ */ f("div", {
							className: "aa-fresh-key",
							children: [/* @__PURE__ */ d("code", { children: y.key }), /* @__PURE__ */ d(je, {
								text: y.key,
								label: "copy key"
							})]
						}),
						/* @__PURE__ */ f("div", {
							className: "aa-fresh-demo",
							children: [/* @__PURE__ */ d("span", {
								className: "aa-note",
								children: "Live-demo link — opens the media UI against this backend, keyed:"
							}), /* @__PURE__ */ f("div", {
								className: "aa-fresh-key",
								children: [/* @__PURE__ */ d("code", { children: `${ne()}/media/?demo=live&key=${y.key}` }), /* @__PURE__ */ d(je, {
									text: `${ne()}/media/?demo=live&key=${y.key}`,
									label: "copy demo link"
								})]
							})]
						}),
						/* @__PURE__ */ f("div", {
							className: "aa-fresh-demo",
							children: [/* @__PURE__ */ d("span", {
								className: "aa-note",
								children: "Station install one-liner — installs hugpy Station on any Linux box and provisions this key into it:"
							}), /* @__PURE__ */ f("div", {
								className: "aa-fresh-key",
								children: [/* @__PURE__ */ d("code", { children: `curl -fsSL ${ne()}/api/agent/console/install.sh | HUGPY_TOKEN=${y.key} bash` }), /* @__PURE__ */ d(je, {
									text: `curl -fsSL ${ne()}/api/agent/console/install.sh | HUGPY_TOKEN=${y.key} bash`,
									label: "copy install cmd"
								})]
							})]
						})
					]
				}),
				a.length > 0 && /* @__PURE__ */ d("div", {
					className: "aa-keys-toolbar",
					children: /* @__PURE__ */ f("button", {
						className: "aa-prune",
						onClick: R,
						title: "Revoke expired keys (and optionally keys older than a cutoff).",
						children: ["Remove dated keys", a.some((e) => e.expired) && /* @__PURE__ */ f("span", {
							className: "aa-chip aa-chip-expired",
							children: [a.filter((e) => e.expired).length, " expired"]
						})]
					})
				}),
				a.length > 0 && /* @__PURE__ */ f("table", {
					className: "aa-keys",
					children: [/* @__PURE__ */ d("thead", { children: /* @__PURE__ */ f("tr", { children: [
						/* @__PURE__ */ d("th", { children: "Name" }),
						/* @__PURE__ */ d("th", { children: "Label" }),
						/* @__PURE__ */ d("th", { children: "Scopes" }),
						/* @__PURE__ */ d("th", { children: "Key" }),
						/* @__PURE__ */ d("th", { children: "Created" }),
						/* @__PURE__ */ d("th", { children: "Last used" }),
						/* @__PURE__ */ d("th", {})
					] }) }), /* @__PURE__ */ d("tbody", { children: a.map((e) => /* @__PURE__ */ f("tr", { children: [
						/* @__PURE__ */ f("td", {
							className: "aa-key-name",
							children: [e.name, e.created_by === "install-link" && /* @__PURE__ */ d("span", {
								className: "aa-chip aa-chip-install",
								title: "Minted by a one-time install link — never a standing operator key.",
								children: "install-link"
							})]
						}),
						/* @__PURE__ */ d("td", {
							className: "aa-dim",
							children: e.label || "–"
						}),
						/* @__PURE__ */ d("td", { children: (e.scopes ?? ["full"]).map((e) => /* @__PURE__ */ d("span", {
							className: "aa-chip",
							children: e
						}, e)) }),
						/* @__PURE__ */ d("td", { children: /* @__PURE__ */ f("code", { children: [e.prefix, "…"] }) }),
						/* @__PURE__ */ f("td", {
							className: "aa-dim",
							children: [Ae(e.created_at), e.expired && /* @__PURE__ */ d("span", {
								className: "aa-chip aa-chip-expired",
								children: "expired"
							})]
						}),
						/* @__PURE__ */ d("td", {
							className: "aa-dim",
							children: Ae(e.last_used)
						}),
						/* @__PURE__ */ d("td", {
							className: "aa-key-actions",
							children: /* @__PURE__ */ d("button", {
								className: "aa-revoke",
								onClick: () => L(e),
								children: "revoke"
							})
						})
					] }, e.id)) })]
				}),
				/* @__PURE__ */ f("div", {
					className: "aa-row aa-install-head",
					children: [/* @__PURE__ */ d("span", {
						className: "aa-label",
						children: "Install links"
					}), /* @__PURE__ */ d("span", {
						className: "aa-note",
						children: "Secure one-time download links for the hugpy-agent installer — each link mints its own labeled, scoped, revocable key."
					})]
				}),
				/* @__PURE__ */ d(Ee, {}),
				/* @__PURE__ */ d(ke, {}),
				/* @__PURE__ */ f("details", {
					className: "aa-example",
					children: [
						/* @__PURE__ */ d("summary", { children: "curl example" }),
						/* @__PURE__ */ d("pre", { children: /* @__PURE__ */ d("code", { children: te }) }),
						/* @__PURE__ */ d(je, {
							text: te,
							label: "copy example"
						})
					]
				})
			]
		})]
	});
}
//#endregion
//#region node_modules/react-fast-compare/index.js
var Ne = /* @__PURE__ */ P(((e, t) => {
	var n = typeof Element < "u", r = typeof Map == "function", i = typeof Set == "function", a = typeof ArrayBuffer == "function" && !!ArrayBuffer.isView;
	function o(e, t) {
		if (e === t) return !0;
		if (e && t && typeof e == "object" && typeof t == "object") {
			if (e.constructor !== t.constructor) return !1;
			var s, c, l;
			if (Array.isArray(e)) {
				if (s = e.length, s != t.length) return !1;
				for (c = s; c-- !== 0;) if (!o(e[c], t[c])) return !1;
				return !0;
			}
			var u;
			if (r && e instanceof Map && t instanceof Map) {
				if (e.size !== t.size) return !1;
				for (u = e.entries(); !(c = u.next()).done;) if (!t.has(c.value[0])) return !1;
				for (u = e.entries(); !(c = u.next()).done;) if (!o(c.value[1], t.get(c.value[0]))) return !1;
				return !0;
			}
			if (i && e instanceof Set && t instanceof Set) {
				if (e.size !== t.size) return !1;
				for (u = e.entries(); !(c = u.next()).done;) if (!t.has(c.value[0])) return !1;
				return !0;
			}
			if (a && ArrayBuffer.isView(e) && ArrayBuffer.isView(t)) {
				if (s = e.length, s != t.length) return !1;
				for (c = s; c-- !== 0;) if (e[c] !== t[c]) return !1;
				return !0;
			}
			if (e.constructor === RegExp) return e.source === t.source && e.flags === t.flags;
			if (e.valueOf !== Object.prototype.valueOf && typeof e.valueOf == "function" && typeof t.valueOf == "function") return e.valueOf() === t.valueOf();
			if (e.toString !== Object.prototype.toString && typeof e.toString == "function" && typeof t.toString == "function") return e.toString() === t.toString();
			if (l = Object.keys(e), s = l.length, s !== Object.keys(t).length) return !1;
			for (c = s; c-- !== 0;) if (!Object.prototype.hasOwnProperty.call(t, l[c])) return !1;
			if (n && e instanceof Element) return !1;
			for (c = s; c-- !== 0;) if (!((l[c] === "_owner" || l[c] === "__v" || l[c] === "__o") && e.$$typeof) && !o(e[l[c]], t[l[c]])) return !1;
			return !0;
		}
		return e !== e && t !== t;
	}
	t.exports = function(e, t) {
		try {
			return o(e, t);
		} catch (e) {
			if ((e.message || "").match(/stack|recursion/i)) return console.warn("react-fast-compare cannot handle circular refs"), !1;
			throw e;
		}
	};
})), Pe = /* @__PURE__ */ P(((e, t) => {
	t.exports = function(e, t, n, r, i, a, o, s) {
		if (process.env.NODE_ENV !== "production" && t === void 0) throw Error("invariant requires an error message argument");
		if (!e) {
			var c;
			if (t === void 0) c = /* @__PURE__ */ Error("Minified exception occurred; use the non-minified dev environment for the full error message and additional helpful warnings.");
			else {
				var l = [
					n,
					r,
					i,
					a,
					o,
					s
				], u = 0;
				c = Error(t.replace(/%s/g, function() {
					return l[u++];
				})), c.name = "Invariant Violation";
			}
			throw c.framesToPop = 1, c;
		}
	};
})), Fe = /* @__PURE__ */ P(((e, t) => {
	t.exports = function(e, t, n, r) {
		var i = n ? n.call(r, e, t) : void 0;
		if (i !== void 0) return !!i;
		if (e === t) return !0;
		if (typeof e != "object" || !e || typeof t != "object" || !t) return !1;
		var a = Object.keys(e), o = Object.keys(t);
		if (a.length !== o.length) return !1;
		for (var s = Object.prototype.hasOwnProperty.bind(t), c = 0; c < a.length; c++) {
			var l = a[c];
			if (!s(l)) return !1;
			var u = e[l], d = t[l];
			if (i = n ? n.call(r, u, d, l) : void 0, i === !1 || i === void 0 && u !== d) return !1;
		}
		return !0;
	};
})), Ie = /* @__PURE__ */ I(Ne()), Le = /* @__PURE__ */ I(Pe()), Re = /* @__PURE__ */ I(Fe()), ze = /* @__PURE__ */ ((e) => (e.BASE = "base", e.BODY = "body", e.HEAD = "head", e.HTML = "html", e.LINK = "link", e.META = "meta", e.NOSCRIPT = "noscript", e.SCRIPT = "script", e.STYLE = "style", e.TITLE = "title", e.FRAGMENT = "Symbol(react.fragment)", e))(ze || {}), Be = {
	link: { rel: [
		"amphtml",
		"canonical",
		"alternate"
	] },
	script: { type: ["application/ld+json"] },
	meta: {
		charset: "",
		name: [
			"generator",
			"robots",
			"description"
		],
		property: [
			"og:type",
			"og:title",
			"og:url",
			"og:image",
			"og:image:alt",
			"og:description",
			"twitter:url",
			"twitter:title",
			"twitter:description",
			"twitter:image",
			"twitter:image:alt",
			"twitter:card",
			"twitter:site"
		]
	}
}, Ve = Object.values(ze), He = {
	accesskey: "accessKey",
	charset: "charSet",
	class: "className",
	contenteditable: "contentEditable",
	contextmenu: "contextMenu",
	"http-equiv": "httpEquiv",
	itemprop: "itemProp",
	tabindex: "tabIndex"
}, Ue = Object.entries(He).reduce((e, [t, n]) => (e[n] = t, e), {}), J = "data-rh", We = {
	DEFAULT_TITLE: "defaultTitle",
	DEFER: "defer",
	ENCODE_SPECIAL_CHARACTERS: "encodeSpecialCharacters",
	ON_CHANGE_CLIENT_STATE: "onChangeClientState",
	TITLE_TEMPLATE: "titleTemplate",
	PRIORITIZE_SEO_TAGS: "prioritizeSeoTags"
}, Ge = (e, t) => {
	for (let n = e.length - 1; n >= 0; --n) {
		let r = e[n];
		if (Object.prototype.hasOwnProperty.call(r, t)) return r[t];
	}
	return null;
}, Ke = (e) => {
	let t = Ge(e, "title"), n = Ge(e, We.TITLE_TEMPLATE);
	if (Array.isArray(t) && (t = t.join("")), n && t) return n.replace(/%s/g, () => t);
	let r = Ge(e, We.DEFAULT_TITLE);
	return t || r || void 0;
}, qe = (e) => Ge(e, We.ON_CHANGE_CLIENT_STATE) || (() => {}), Je = (e, t) => t.filter((t) => t[e] !== void 0).map((t) => t[e]).reduce((e, t) => ({
	...e,
	...t
}), {}), Ye = (e, t) => t.filter((e) => e.base !== void 0).map((e) => e.base).reverse().reduce((t, n) => {
	if (!t.length) {
		let r = Object.keys(n);
		for (let i = 0; i < r.length; i += 1) {
			let a = r[i].toLowerCase();
			if (e.indexOf(a) !== -1 && n[a]) return t.concat(n);
		}
	}
	return t;
}, []), Xe = (e) => console && typeof console.warn == "function" && console.warn(e), Ze = (e, t, n) => {
	let r = {};
	return n.filter((t) => Array.isArray(t[e]) ? !0 : (t[e] !== void 0 && Xe(`Helmet: ${e} should be of type "Array". Instead found type "${typeof t[e]}"`), !1)).map((t) => t[e]).reverse().reduce((e, n) => {
		let i = {};
		n.filter((e) => {
			let n, a = Object.keys(e);
			for (let r = 0; r < a.length; r += 1) {
				let i = a[r], o = i.toLowerCase();
				t.indexOf(o) !== -1 && !(n === "rel" && e[n].toLowerCase() === "canonical") && !(o === "rel" && e[o].toLowerCase() === "stylesheet") && (n = o), t.indexOf(i) !== -1 && (i === "innerHTML" || i === "cssText" || i === "itemprop") && (n = i);
			}
			if (!n || !e[n]) return !1;
			let o = e[n].toLowerCase();
			return r[n] || (r[n] = {}), i[n] || (i[n] = {}), r[n][o] ? !1 : (i[n][o] = !0, !0);
		}).reverse().forEach((t) => e.push(t));
		let a = Object.keys(i);
		for (let e = 0; e < a.length; e += 1) {
			let t = a[e], n = {
				...r[t],
				...i[t]
			};
			r[t] = n;
		}
		return e;
	}, []).reverse();
}, Qe = (e, t) => {
	if (Array.isArray(e) && e.length) {
		for (let n = 0; n < e.length; n += 1) if (e[n][t]) return !0;
	}
	return !1;
}, $e = (e) => ({
	baseTag: Ye(["href"], e),
	bodyAttributes: Je("bodyAttributes", e),
	defer: Ge(e, We.DEFER),
	encode: Ge(e, We.ENCODE_SPECIAL_CHARACTERS),
	htmlAttributes: Je("htmlAttributes", e),
	linkTags: Ze("link", ["rel", "href"], e),
	metaTags: Ze("meta", [
		"name",
		"charset",
		"http-equiv",
		"property",
		"itemprop"
	], e),
	noscriptTags: Ze("noscript", ["innerHTML"], e),
	onChangeClientState: qe(e),
	scriptTags: Ze("script", ["src", "innerHTML"], e),
	styleTags: Ze("style", ["cssText"], e),
	title: Ke(e),
	titleAttributes: Je("titleAttributes", e),
	prioritizeSeoTags: Qe(e, We.PRIORITIZE_SEO_TAGS)
}), et = (e) => Array.isArray(e) ? e.join("") : e, tt = (e, t) => {
	let n = Object.keys(e);
	for (let r = 0; r < n.length; r += 1) if (t[n[r]] && t[n[r]].includes(e[n[r]])) return !0;
	return !1;
}, nt = (e, t) => Array.isArray(e) ? e.reduce((e, n) => (tt(n, t) ? e.priority.push(n) : e.default.push(n), e), {
	priority: [],
	default: []
}) : {
	default: e,
	priority: []
}, rt = (e, t) => ({
	...e,
	[t]: void 0
}), it = [
	"noscript",
	"script",
	"style"
], at = (e, t = !0) => t === !1 ? String(e) : String(e).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#x27;"), ot = (e) => Object.keys(e).reduce((t, n) => {
	let r = e[n] === void 0 ? `${n}` : `${n}="${e[n]}"`;
	return t ? `${t} ${r}` : r;
}, ""), st = (e, t, n, r) => {
	let i = ot(n), a = et(t);
	return i ? `<${e} ${J}="true" ${i}>${at(a, r)}</${e}>` : `<${e} ${J}="true">${at(a, r)}</${e}>`;
}, ct = (e, t, n = !0) => t.reduce((t, r) => {
	let i = r, a = Object.keys(i).filter((e) => !(e === "innerHTML" || e === "cssText")).reduce((e, t) => {
		let r = i[t] === void 0 ? t : `${t}="${at(i[t], n)}"`;
		return e ? `${e} ${r}` : r;
	}, ""), o = i.innerHTML || i.cssText || "";
	return `${t}<${e} ${J}="true" ${a}${it.indexOf(e) === -1 ? "/>" : `>${o}</${e}>`}`;
}, ""), lt = (e, t = {}) => Object.keys(e).reduce((t, n) => {
	let r = He[n];
	return t[r || n] = e[n], t;
}, t), ut = (t, n, r) => {
	let i = lt(r, {
		key: n,
		[J]: !0
	});
	return [e.createElement("title", i, n)];
}, dt = (t, n) => n.map((n, r) => {
	let i = {
		key: r,
		[J]: !0
	};
	return Object.keys(n).forEach((e) => {
		let t = He[e] || e;
		if (t === "innerHTML" || t === "cssText") {
			let e = n.innerHTML || n.cssText;
			i.dangerouslySetInnerHTML = { __html: e };
		} else i[t] = n[e];
	}), e.createElement(t, i);
}), ft = (e, t, n = !0) => {
	switch (e) {
		case "title": return {
			toComponent: () => ut(e, t.title, t.titleAttributes),
			toString: () => st(e, t.title, t.titleAttributes, n)
		};
		case "bodyAttributes":
		case "htmlAttributes": return {
			toComponent: () => lt(t),
			toString: () => ot(t)
		};
		default: return {
			toComponent: () => dt(e, t),
			toString: () => ct(e, t, n)
		};
	}
}, pt = ({ metaTags: e, linkTags: t, scriptTags: n, encode: r }) => {
	let i = nt(e, Be.meta), a = nt(t, Be.link), o = nt(n, Be.script);
	return {
		priorityMethods: {
			toComponent: () => [
				...dt("meta", i.priority),
				...dt("link", a.priority),
				...dt("script", o.priority)
			],
			toString: () => `${ft("meta", i.priority, r)} ${ft("link", a.priority, r)} ${ft("script", o.priority, r)}`
		},
		metaTags: i.default,
		linkTags: a.default,
		scriptTags: o.default
	};
}, mt = (e) => {
	let { baseTag: t, bodyAttributes: n, encode: r = !0, htmlAttributes: i, noscriptTags: a, styleTags: o, title: s = "", titleAttributes: c, prioritizeSeoTags: l } = e, { linkTags: u, metaTags: d, scriptTags: f } = e, p = {
		toComponent: () => [],
		toString: () => ""
	};
	return l && ({priorityMethods: p, linkTags: u, metaTags: d, scriptTags: f} = pt(e)), {
		priority: p,
		base: ft("base", t, r),
		bodyAttributes: ft("bodyAttributes", n, r),
		htmlAttributes: ft("htmlAttributes", i, r),
		link: ft("link", u, r),
		meta: ft("meta", d, r),
		noscript: ft("noscript", a, r),
		script: ft("script", f, r),
		style: ft("style", o, r),
		title: ft("title", {
			title: s,
			titleAttributes: c
		}, r)
	};
}, ht = [], gt = !!(typeof window < "u" && window.document && window.document.createElement), _t = class {
	instances = [];
	canUseDOM = gt;
	context;
	value = {
		setHelmet: (e) => {
			this.context.helmet = e;
		},
		helmetInstances: {
			get: () => this.canUseDOM ? ht : this.instances,
			add: (e) => {
				(this.canUseDOM ? ht : this.instances).push(e);
			},
			remove: (e) => {
				let t = (this.canUseDOM ? ht : this.instances).indexOf(e);
				(this.canUseDOM ? ht : this.instances).splice(t, 1);
			}
		}
	};
	constructor(e, t) {
		this.context = e, this.canUseDOM = t || !1, t || (e.helmet = mt({
			baseTag: [],
			bodyAttributes: {},
			encodeSpecialCharacters: !0,
			htmlAttributes: {},
			linkTags: [],
			metaTags: [],
			noscriptTags: [],
			scriptTags: [],
			styleTags: [],
			title: "",
			titleAttributes: {}
		}));
	}
}, vt = parseInt(e.version.split(".")[0], 10) >= 19, yt = e.createContext({}), bt = class n extends t {
	static canUseDOM = gt;
	helmetData;
	constructor(e) {
		super(e), vt ? this.helmetData = null : this.helmetData = new _t(this.props.context || {}, n.canUseDOM);
	}
	render() {
		return vt ? /* @__PURE__ */ e.createElement(e.Fragment, null, this.props.children) : /* @__PURE__ */ e.createElement(yt.Provider, { value: this.helmetData.value }, this.props.children);
	}
}, xt = (e, t) => {
	let n = document.head || document.querySelector("head"), r = n.querySelectorAll(`${e}[${J}]`), i = [].slice.call(r), a = [], o;
	return t && t.length && t.forEach((t) => {
		let n = document.createElement(e);
		for (let e in t) if (Object.prototype.hasOwnProperty.call(t, e)) if (e === "innerHTML") n.innerHTML = t.innerHTML;
		else if (e === "cssText") {
			let e = t.cssText;
			n.appendChild(document.createTextNode(e));
		} else {
			let r = e, i = t[r] === void 0 ? "" : t[r];
			n.setAttribute(e, i);
		}
		n.setAttribute(J, "true"), i.some((e, t) => (o = t, n.isEqualNode(e))) ? i.splice(o, 1) : a.push(n);
	}), i.forEach((e) => e.parentNode?.removeChild(e)), a.forEach((e) => n.appendChild(e)), {
		oldTags: i,
		newTags: a
	};
}, St = (e, t) => {
	let n = document.getElementsByTagName(e)[0];
	if (!n) return;
	let r = n.getAttribute(J), i = r ? r.split(",") : [], a = [...i], o = Object.keys(t);
	for (let e of o) {
		let r = t[e] || "";
		n.getAttribute(e) !== r && n.setAttribute(e, r), i.indexOf(e) === -1 && i.push(e);
		let o = a.indexOf(e);
		o !== -1 && a.splice(o, 1);
	}
	for (let e = a.length - 1; e >= 0; --e) n.removeAttribute(a[e]);
	i.length === a.length ? n.removeAttribute(J) : n.getAttribute(J) !== o.join(",") && n.setAttribute(J, o.join(","));
}, Ct = (e, t) => {
	e !== void 0 && document.title !== e && (document.title = et(e)), St("title", t);
}, wt = (e, t) => {
	let { baseTag: n, bodyAttributes: r, htmlAttributes: i, linkTags: a, metaTags: o, noscriptTags: s, onChangeClientState: c, scriptTags: l, styleTags: u, title: d, titleAttributes: f } = e;
	St("body", r), St("html", i), Ct(d, f);
	let p = {
		baseTag: xt("base", n),
		linkTags: xt("link", a),
		metaTags: xt("meta", o),
		noscriptTags: xt("noscript", s),
		scriptTags: xt("script", l),
		styleTags: xt("style", u)
	}, m = {}, h = {};
	Object.keys(p).forEach((e) => {
		let { newTags: t, oldTags: n } = p[e];
		t.length && (m[e] = t), n.length && (h[e] = p[e].oldTags);
	}), t && t(), c(e, m, h);
}, Tt = null, Et = (e) => {
	Tt && cancelAnimationFrame(Tt), e.defer ? Tt = requestAnimationFrame(() => {
		wt(e, () => {
			Tt = null;
		});
	}) : (wt(e), Tt = null);
}, Dt = class extends t {
	rendered = !1;
	shouldComponentUpdate(e) {
		return !(0, Re.default)(e, this.props);
	}
	componentDidUpdate() {
		this.emitChange();
	}
	componentWillUnmount() {
		let { helmetInstances: e } = this.props.context;
		e.remove(this), this.emitChange();
	}
	emitChange() {
		let { helmetInstances: e, setHelmet: t } = this.props.context, n = null, r = $e(e.get().map((e) => {
			let { context: t, ...n } = e.props;
			return n;
		}));
		bt.canUseDOM ? Et(r) : mt && (n = mt(r)), t(n);
	}
	init() {
		if (this.rendered) return;
		this.rendered = !0;
		let { helmetInstances: e } = this.props.context;
		e.add(this), this.emitChange();
	}
	render() {
		return this.init(), null;
	}
}, Ot = [], kt = (e) => {
	let t = {};
	for (let n of Object.keys(e)) t[Ue[n] || n] = e[n];
	return t;
}, At = (e) => {
	let t = {};
	for (let n of Object.keys(e)) {
		let r = He[n];
		t[r || n] = e[n];
	}
	return t;
}, jt = (e, t) => {
	if (!gt) return;
	let n = document.getElementsByTagName(e)[0];
	if (!n) return;
	let r = "data-rh-managed", i = n.getAttribute(r), a = i ? i.split(",") : [], o = Object.keys(t);
	for (let e of a) o.includes(e) || n.removeAttribute(e);
	for (let e of o) {
		let r = t[e];
		r == null || r === !1 ? n.removeAttribute(e) : r === !0 ? n.setAttribute(e, "") : n.setAttribute(e, String(r));
	}
	o.length > 0 ? n.setAttribute(r, o.join(",")) : n.removeAttribute(r);
}, Mt = () => {
	let e = {}, t = {};
	for (let n of Ot) {
		let { htmlAttributes: r, bodyAttributes: i } = n.props;
		r && Object.assign(e, kt(r)), i && Object.assign(t, kt(i));
	}
	jt("html", e), jt("body", t);
}, Nt = class extends t {
	componentDidMount() {
		Ot.push(this), Mt();
	}
	componentDidUpdate() {
		Mt();
	}
	componentWillUnmount() {
		let e = Ot.indexOf(this);
		e !== -1 && Ot.splice(e, 1), Mt();
	}
	resolveTitle() {
		let { title: e, titleTemplate: t, defaultTitle: n } = this.props;
		return e && t ? t.replace(/%s/g, () => Array.isArray(e) ? e.join("") : e) : e || n || void 0;
	}
	renderTitle() {
		let t = this.resolveTitle();
		if (t === void 0) return null;
		let n = this.props.titleAttributes || {};
		return e.createElement("title", At(n), t);
	}
	renderBase() {
		let { base: t } = this.props;
		return t ? e.createElement("base", At(t)) : null;
	}
	renderMeta() {
		let { meta: t } = this.props;
		return !t || !Array.isArray(t) ? null : t.map((t, n) => e.createElement("meta", {
			key: n,
			...At(t)
		}));
	}
	renderLink() {
		let { link: t } = this.props;
		return !t || !Array.isArray(t) ? null : t.map((t, n) => e.createElement("link", {
			key: n,
			...At(t)
		}));
	}
	renderScript() {
		let { script: t } = this.props;
		return !t || !Array.isArray(t) ? null : t.map((t, n) => {
			let { innerHTML: r, ...i } = t, a = At(i);
			return r && (a.dangerouslySetInnerHTML = { __html: r }), e.createElement("script", {
				key: n,
				...a
			});
		});
	}
	renderStyle() {
		let { style: t } = this.props;
		return !t || !Array.isArray(t) ? null : t.map((t, n) => {
			let { cssText: r, ...i } = t, a = At(i);
			return r && (a.dangerouslySetInnerHTML = { __html: r }), e.createElement("style", {
				key: n,
				...a
			});
		});
	}
	renderNoscript() {
		let { noscript: t } = this.props;
		return !t || !Array.isArray(t) ? null : t.map((t, n) => {
			let { innerHTML: r, ...i } = t, a = At(i);
			return r && (a.dangerouslySetInnerHTML = { __html: r }), e.createElement("noscript", {
				key: n,
				...a
			});
		});
	}
	render() {
		return e.createElement(e.Fragment, null, this.renderTitle(), this.renderBase(), this.renderMeta(), this.renderLink(), this.renderScript(), this.renderStyle(), this.renderNoscript());
	}
}, Pt = class extends t {
	static defaultProps = {
		defer: !0,
		encodeSpecialCharacters: !0,
		prioritizeSeoTags: !1
	};
	shouldComponentUpdate(e) {
		return !(0, Ie.default)(rt(this.props, "helmetData"), rt(e, "helmetData"));
	}
	mapNestedChildrenToProps(e, t) {
		if (!t) return null;
		switch (e.type) {
			case "script":
			case "noscript": return { innerHTML: t };
			case "style": return { cssText: t };
			default: throw Error(`<${e.type} /> elements are self-closing and can not contain children. Refer to our API for more information.`);
		}
	}
	flattenArrayTypeChildren(e, t, n, r) {
		return {
			...t,
			[e.type]: [...t[e.type] || [], {
				...n,
				...this.mapNestedChildrenToProps(e, r)
			}]
		};
	}
	mapObjectTypeChildren(e, t, n, r) {
		switch (e.type) {
			case "title": return {
				...t,
				[e.type]: r,
				titleAttributes: { ...n }
			};
			case "body": return {
				...t,
				bodyAttributes: { ...n }
			};
			case "html": return {
				...t,
				htmlAttributes: { ...n }
			};
			default: return {
				...t,
				[e.type]: { ...n }
			};
		}
	}
	mapArrayTypeChildrenToProps(e, t) {
		let n = { ...t };
		return Object.keys(e).forEach((t) => {
			n = {
				...n,
				[t]: e[t]
			};
		}), n;
	}
	warnOnInvalidChildren(e, t) {
		return (0, Le.default)(Ve.some((t) => e.type === t), typeof e.type == "function" ? "You may be attempting to nest <Helmet> components within each other, which is not allowed. Refer to our API for more information." : `Only elements types ${Ve.join(", ")} are allowed. Helmet does not support rendering <${e.type}> elements. Refer to our API for more information.`), (0, Le.default)(!t || typeof t == "string" || Array.isArray(t) && !t.some((e) => typeof e != "string"), `Helmet expects a string as a child of <${e.type}>. Did you forget to wrap your children in braces? ( <${e.type}>{\`\`}</${e.type}> ) Refer to our API for more information.`), !0;
	}
	mapChildrenToProps(t, n) {
		let r = {};
		return e.Children.forEach(t, (e) => {
			if (!e || !e.props) return;
			let { children: t, ...i } = e.props, a = Object.keys(i).reduce((e, t) => (e[Ue[t] || t] = i[t], e), {}), { type: o } = e;
			switch (typeof o == "symbol" ? o = o.toString() : this.warnOnInvalidChildren(e, t), o) {
				case "Symbol(react.fragment)":
					n = this.mapChildrenToProps(t, n);
					break;
				case "link":
				case "meta":
				case "noscript":
				case "script":
				case "style":
					r = this.flattenArrayTypeChildren(e, r, a, t);
					break;
				default:
					n = this.mapObjectTypeChildren(e, n, a, t);
					break;
			}
		}), this.mapArrayTypeChildrenToProps(r, n);
	}
	render() {
		let { children: t, ...n } = this.props, r = { ...n }, { helmetData: i } = n;
		return t && (r = this.mapChildrenToProps(t, r)), i && !(i instanceof _t) && (i = new _t(i.context, !0), delete r.helmetData), vt ? /* @__PURE__ */ e.createElement(Nt, { ...r }) : i ? /* @__PURE__ */ e.createElement(Dt, {
			...r,
			context: i.value
		}) : /* @__PURE__ */ e.createElement(yt.Consumer, null, (t) => /* @__PURE__ */ e.createElement(Dt, {
			...r,
			context: t
		}));
	}
}, Ft = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAASUAAAE6CAYAAABQ50Q5AAAAiXpUWHRSYXcgcHJvZmlsZSB0eXBlIGV4aWYAAHjaVY7LDcMwDEPvmiIj6GfKGqcIEqAbdPzIsAu37yBRhECQrs/7pmMgrOQtOhLgwtNTXyU6T4xZlGXsmpO1TUrptsl0CmQP9v3oy//SDB13eAQaTpxa6XqZmdasGjRSedTIHaJtKfv365TfcHoARSQsT1V5DXcAAAoGaVRYdFhNTDpjb20uYWRvYmUueG1wAAAAAAA8P3hwYWNrZXQgYmVnaW49Iu+7vyIgaWQ9Ilc1TTBNcENlaGlIenJlU3pOVGN6a2M5ZCI/Pgo8eDp4bXBtZXRhIHhtbG5zOng9ImFkb2JlOm5zOm1ldGEvIiB4OnhtcHRrPSJYTVAgQ29yZSA0LjQuMC1FeGl2MiI+CiA8cmRmOlJERiB4bWxuczpyZGY9Imh0dHA6Ly93d3cudzMub3JnLzE5OTkvMDIvMjItcmRmLXN5bnRheC1ucyMiPgogIDxyZGY6RGVzY3JpcHRpb24gcmRmOmFib3V0PSIiCiAgICB4bWxuczpleGlmPSJodHRwOi8vbnMuYWRvYmUuY29tL2V4aWYvMS4wLyIKICAgIHhtbG5zOnRpZmY9Imh0dHA6Ly9ucy5hZG9iZS5jb20vdGlmZi8xLjAvIgogICBleGlmOlBpeGVsWERpbWVuc2lvbj0iMjkzIgogICBleGlmOlBpeGVsWURpbWVuc2lvbj0iMzE0IgogICB0aWZmOkltYWdlV2lkdGg9IjI5MyIKICAgdGlmZjpJbWFnZUxlbmd0aD0iMzE0IgogICB0aWZmOk9yaWVudGF0aW9uPSIxIi8+CiA8L3JkZjpSREY+CjwveDp4bXBtZXRhPgogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgCjw/eHBhY2tldCBlbmQ9InciPz76ppCoAAAABHNCSVQICAgIfAhkiAAAIABJREFUeNrsnXecHNWV77+VOvd0T47KEYkgJBQJElFkCxtY2xgwtvdhvN61d9/uvmcbh931ene9u37vrcPaOIDBkSzAJIEEkpCEEgghoayRNDl3DlVd9f7oUUujGYE0mpFGcL586jNoerqr+ta9vzrn3HPPVQAHQRCEEYIqTSAIgoiSIAiCiJIgCCJKgiAIIkqCIIgoCYIgiCgJgiCiJAiCIKIkCIKIkiAIgoiSIAiCiJIgCCJKgiAIIkqCIIgoCYIgiCgJgiCiJAiCIKIkCIKIkiAIgoiSIAgiSoIgCCJKgiAIIkqCIIgoCYIgiCgJgiCiJAiCIKIkCIKIkiAIgoiSIAgiSoIgCCJKgiCIKAmCIIgoCYIgiCgJgiCiJAiCIKIkCIKIkiAIgoiSIAgiSoIgCCJKgiCIKAmCIIgoCYIgoiQIgiCiJAiCIKIkCIKIkiAIgoiSIAgiSoIgCCJKgiCIKAlCL16vl/LycrxerzSGcMro0gTCYNF0nQkTJnDFFVcydepUduzYwfLlr7J3715yliUNJAwKBXCkGYSTJRgMct31N3Drbbdx+eWX4/X6SKWSrFixgscfe4wXnv8TsVhMGkoQURKGn7nz5vHpT9/BNYsXU1c3qt/rDQ2HePmll/jd737Lm+vWSYMJIkrC8FBeXsFn7ryTj3/8E5x73nnoet77z2azJBIJfD4fbrcbAMuyeHfrVp588gl+88gjtLe3SQMKIkrC0KDrOpdedhlf/vJfMXvOHEKhEIqiANDU1MgjDz/C6lUruXDmTL54333U1NQC4DgOkUiEDevX86Mf/RerVq7EkliTcAI4cshxvKOsvNy5/1vfdvbsrXe6emJOJJZ0IrGk0xNNOE8tfdaZPv1cp6KiwikpKXVcLpczbdp056mlzzo90UThb7t6Ys6evfXO/d/6tlNWXi7tKscHHdIIcvQ/PB6Ps3DhIuf1VaudSCxROLp6os6uPfucOz5zp6MoiqNqmlNcUuIUhUKF9yqK4tzxmTudXXv2OV090T7vf23lamfhwkWOx+ORdpZDREmODz7cbrczbvx459/+/T+dlrZOpyeacHqiCaezO+rs3rvf+dGP/9upra0r/L2uG05JaZnjDwT6fVZtbZ3zox//t7N7736nszta+KyWtk7n377/H864ceMdl8sl7S6HiJIc/Q9VVZ1Ro0c7n7nzTmfzW1uc7mjc6Y7Gna5IzKk/1Ogsfe5PzpJbbnG8Xm+f97lcLqesrNzxeLwDfq7X63WW3HKLs/S5Pzn1hxqdrkis8Nmb3trifOrTdzijRo12VFWV+yCHAzga8B0Jq320CYfDXHLJpXzpL77Ml/7iy1RUVABgmha7du7gt795hP/7gx+wbt1aTNPsGwQ3DDweL6lkAtu2+322ZVns2rWLDW+uJ5lMUFpSSjhcjKaqFBcXc/kVV1BTU0sqmaSzs5NMJi035COOiNJHfFbtnHOmcfc99/DF+77E/AULcLlcOEAkEuHpp5/iJz/5MU88/hitra04Tv+JWpfhwuPxEI/HB3z98CxcV1cXmzdvYveePTi2zajRo3G53eiGweQpU7jwwpkUF4fp6Oigq6vruJ8lfPiRlICPKIFAgI8tuYXbbv8zLrxwJoFAoPDaho0beOhXv2TVqpU0NjS87+f4/QH8/gBtbS0nfO66ujouufQy7vnc57nootmF38fjcd7avIlHH/0jzyx9mng8LjdKREn4KDBlylT+/n/9b+bOm0dFRRWqms856unp5oGf/Yxnlj7N3r17+rlq/TqPohAsKsLQDTo7O07qGgzDYMKEidz8sSX8j3vvJRwuBsC2HdraWnhz3Tq+/2//ys6dO+SGfQSR4NpH5PD6fM5fffWvnS3vbnea2zqc9q6I094VcVrau5zHn3jaueTSyxy/P3BSwfHi4hInFAoP+pr8/oBzyaWXOk88udRp7eguXFNzW4ez5d3tzl9+5auSPiCzb3J8GI/Zc+Y6L7z4stPc2uG0d/YUju07djt/9ZWvOuFw2FEU5eRmSTTNKS0rPykhG+hQFMUJh8POX33lq872Hbv7XF9za4fzwosvO7Nnz5H7KKIkx9l+6Lru1NbWOV//xjednbv3Om2d3YVjb/1B57d/+KMzc+asU/r8ysoqxz2ElszMWbOc3/3hj87e+oN9rnfn7r3O17/xTae2ts7RdV3ur6QECGdVoFBRKC0t5fIrruSb3/oOt3ziE/h8fgAymQy7du7kxz/6Id/77j9x8ODBQZ9H0zT8gSDxWGzAdIDB0NzczAvPP0+kJ0J1dTXBohCapuPxepk9ezYzLryQSCRCZ0cn6bSkD3wYEVH6kKHrBhdddBGf/dzn+Ou/+VsmTZqEqijYts2BA/U8/9xzfPub9/PqK8s+MJB9IsFqvz9AJNIzpN/BNE02b97E6lWr8Hg8hEIhikIhNFVl1KhRXLpwEeFQiFgsSltr25AJoiCiJAwxY8eN49bbbufLf/UVrr3uetweDw4QjUVZsXw5D/zsp/zi5z+jrW1oyoi4PR5chjFsU/ddXV2sWvk6TU1NGIaLisoKXC43Ho+HmbNmMX36ebg9bro6O+jp6ZEOIKIkjBR8fj/X33Aj937xPu74zJ2FwmtWzuK9bdt46MEH+cUvHmDjhvWnbB0djbfXJUylksP23SzLYs+e3bz99lt0dXVRWlpGSUkJqqJSWVnF3LnzGDN2LI7j0NDQMKTfTzhD4QckT+nsvXmKwtSp5/DpOz7D1dfkq0CqWn4viFg0ytKlT/P4o39k2/ZtJIbYmlEUhZKSUkzLJBqJnJbvGwgEmDZtOrfd/kluXrKEYDAIgJ2zOdRwkGUvv8Tvfvsbdu7YIRnhIkrC6SYQCHDjTR/j7s/ew6TJk/F6fYXXdu/eyb/9y7+w/s11dHd3DUvMRVEUKiqriEUjJJPJ0/a9VVWluLiEOfPm8b/+99eYNGlK7ysOqWSKXbt38dCvfsmfnnuGRCIhHeUsRaYhz6JDURRn7Lhxzs9+/itn974DTkNzm9PY0u40trQ7+w80Ol+//5tOaWmpo2nasF9Hbd1oxzCMMzNtrGlOaWmpc/+3vu3sP9hYaIOG5jZn974Dzk8f+IUzduy4k869kkPylOQ4CRFwu93OXZ/9nLNtxy6nobnNaWhucw41tTr1h5qcZ59/0Tn/ghmnbRAqiuKMHTv+jA96RVGcCy+c6Tz7/ItO/aEm51BTa6Ft3n1vp3Pn3Z913G63iJPkKQlD6SYFg0EuuGAGP/i//8Vdd9+D2+0B8gX7Dx48yM8f+Bn/+O1vcaB+/2m7LpfLhc/nH/J0gMHQ0tLMSy88TyKRoKa2Fp/Pj6ZpeDxerrzqaubOm8++vXuIxmISCD8LEFEawbjdbiZOnMQdd97F1+//JpMmTUZRwLZtOjraefWVZfzHv/8rzz6zlGTy9MZPPB4vmq4PeQB9sKRSKTZt2si7W9/B6/NSWlaG1+tBURRqa2u58uprcLs9dHV2EovFyOVy0sFElIQTt45UKqsquWbxdXz5L7/CTTd/jEDvTJNpZlm//k0efughfv6z/2bPnj1nJHkwEAhg2zapVGrEtJtt2zQ2NrD+zXV0dnbi9weorKxE1XR8Pj8zZ85i6jnnYOUsOjo6TmuAXhBROmvRdYP5CxZwz+e+wGfuvIspU6aiaRqQ3+Txt795mF/+/AFefWXZGRWEUDhMJp0mm82MuDZMpVJsfWcLO3ZsJxKJUFdXRzAYRNM06upGMWPGhVRX15BIJGhubpKM8JH2UEZSAkYMpWVlfPaez3PlVVczefKUwmaPqVSKV5a9zNNPPcHaNW+MiCd83agxdLS3jvj1Zz6fjwUXX8KSWz7BlVddjdfrBXrL9O7cwbJlL/Pwrx+ks6NDOqCIknDEOtK55tprueuzn+Pc6efiP6oK5M4dO3j4oQdZ+foKGhubcJyR8VQfP2ES9fv3nhVWhqKq1NbUsnDRIu767OeYPGVK4bV4LMa2be/y0IO/YtlLL0qsSdw3AeCL9/0Ff/mVv2bK5Km4XPltrxOJBE898QT/9q//zJo3VtPd3T1inh+aphEOhenq6jw7GthxiEaj7Nq5kzffXIdhGIwdOx6X4cLlclNbW8esi2ajqSqbNm6QDjkSbpkcZ+4Ih8PO408udfYdaCwcq9ducBYvvm7Ebjvk8XidulGjz+rtpK674QbnjXUb+rT7o0887QSCQemXZ/r+iCafWVxuN6quF+7I66+9xu2fWMKyZS+NWNfI5XKRzWZHrrumKKiqiqppaJqG4XLhDwQoKSmlurqGUaPHsH3be/z55+/htddWFNpeNwx8Pp90yjMdzpAmONNBPQXFcaB3Aen69euwbZtwOEwikcA0zREnTobLRTYzMmbdVFVFVVUURe0VIhVd1zEMV+9PA1VRyWQyZLMZeiIRstkMOcsC4I3VK1m4cFH+XjgOCop0ShGlj7oqKUcW/ADZbIa2tlaCwSJCoWIymTSpVArTzI6Yle8ul4tIKnmam6nX+lHz1o+qqWiahnb436qGpqmAgmmZmGaWVDKJaWaxegXo/eIXhZ+KiJKIkkDeUHIKI8O2bSKRHgzDRSAQIBgMYpomiUR8RMwOuVzu4bWUFAVN09A1HV3X0HQdTdVQFDWvGYqC0iselmmRyWQwLRPLNE++fY5qe6l2IqIkFJ7PzpGBcdQrppmlp6cbj8eDx+OlpKSUVCpFPB47s+6Sonyg9XFSnVDX0Q0DQzfQDSOfn+Xk28Tu/enYNpaVJZezsKwcuZw1JALtcNQDoY/NKogofaQtJeeYgdH3tVQqRTabxe324PF6qQoG6enuOiOJi7quD1qQFEUpxHkMw43LlRch23awe0Uml8uRMrPYOZtcLodt53p/Dldc7agHgphKIkrCAA/o44yLXC5HMpkgk8ngdrspCoXxBywiPT1DarV8EIbhImue2Mybbhi43W5crvxhGDqO7WCaJqZlkkqlsHp3QnEcu/enUxAgXdepqR1F1szS1tI8PMJ0Am0viCh95ASp14E7oXGRy1kkkxaZbIZAIEBFRRXRaIREIn5anvSGYWAdU/7jsAXkdnvy1pzHjdvjwbJyZDNpMpkM8ViEbDZbEJ4TsU6uue5m6kaPwTSzbN3yFhvfXDNczvORthdhElESDrtvJ/eenJW3klLJJKVl5QQCATo62snlcsMiTocDyy63m1zOIlhUVBAhl8uFY9ukewWoq7ubTDpVuA7l2BktRUFVlLwl5XJjuPKZ1S63G1fv/7vdbi6YeRGhUIhEPE5LUyOKogzLdyvMMYj7JqIkHPvMPnmy2SzNTY34/QGqa+qIx6LE4jFylvW+g0xRlLyguF1YpkW6V0SU3pktRTmS8+PuFQy3201NTR3RSA/xRJxsJksqlSQa7cFxnPxsmWEQCoUxysoxDAOtEEM6kjeUD2YbWJaFaWYxs1my2Symmf95+Fq6OjsoKiqivb2Nnp7uYRQNESMRJeG4ltJgB14iESeZTBAKF1NWWk4iESOdTh+30qLH42XBpYsYO2EiXR0dbN2ymXgshstl9LFaFEUtBKAdx8bozUAvChXn84R6D0XN/51lmoWYkWXmRSsW7f2dafa+nj3udem6jtfro6S0jN073sMyTd7atJ5dO7aP2LYXRJQ+tMI0FJ/R092Fy+UiGCzC5XKTyWRIpVLkcn2D4W6Ph6nTz6OkpJTa2lEoQFPjIWzbLhxZM4uZTWFZVj6Y7jgYuk5bawvZTKbwe8syT9ltVBQFl9uNz+fHMAwymTSaofH68pfYs3vXiG97QUTpQ+a4OX2SJ0+VbDZLV1cnXq8Pj9eL2+0mlU6RSiYL57FMk66ONkpKSmltaaJ+/1727t5JOp0XoYFygDweL5l0io72od0qW1VVfH4/brcby7SwezO0PW4PTU2Nw+41H8kRE3ESURJ6n9RHzwA5Q/SZTiGFwOv14vXkj1gsSjabJZPNsH3bVpoaG6nft4dYLNobxM6ROU62tm7oQ55+oKoa4XAYB4d4LI7P58M0TSoqq2lvbyM9zNU1nd7/Dt8HQURJOJwoM0wDIpeziMdjZDJp3B4PJaVlZNIZUukkOSvH25s3EOnpzrtPLjeBQACvz0tHe3s/t8bQDaycNWTujtvjobS0jFg0SiqVxO12o6gqZjpFaXk5B09XETnnmHshiCiJLg3/+ivTNLEsi3QqRSBYxISJkwkEAsSikYJllcmksSwTr9fHqNFjaG9r7VMHXNcNUumhWYgbDhfj8/tpb2vFsixcLhduj5dkMkFJaRlmJktPd/fpsVKH0HUWhsB6liYYCe6bc1qe0Y7jYFkW0Ug+CzyTzVJZWY3L5TrKssoRj8dobDhEOFxCcXFJP/ftVCwlVVWpqq5B0zSamxoxTTMfU/IFsEyTdCpFVXUtyVTitK3xK1QXE/9NREno98g+LY9rVVUpLi5h5/atRGMRSssqCIXDhY0KDotTa2sLqqZRXV2L15svfuYM0p1SFAWv10dVdQ3xWIyurs5CXpTX60PXNaLRCKFwMY5jE4v0nKY6Us5RbS+I+ybgALZzdFbx8J+zuKSUdCZd2BUlk04TCoUpKS0jHouR6S2CZts5Ojva8fp8lJSU4OqN+Zwsmqbj751d62hv77Mtk9vtwef309nRDopCWVk5ZjZLNBo9vc+Cw/dBuqSIkpCXptM5HV03eiyNhw4U/m1ZFp2dHXg8HkLhYjy9blQmk86v2k8miRsuKv352k44zglVKMhnjbvx+QKAQ3d3V5/ZO8MwCBYVEY30kMvlcHs8eP1+cqZJMpk4be6UIytyRZSE/o/r45UuGWo8Hi/BYJDOzv77nKXTadItzQQCwXzekMdNOpUinU5j2zadnR3Yto3fH8DldhOPxbHt49c0CgSCuFwuMtkMiXjfBcOqqlJcUkoykSgIXLi4pBBXso6T8T0sknQ6zVRBROlsceFOl1VQVV1DW2tLoUb1QMTjMVLpFD6fD58/gMfjRTcMstkssWgEw3Dh8/spKS0llU5jWRaKqhTqIqkohEJhrJxJPB4nk+lvVRUXl2Jms4VgtqbrBAJBHMcmHo+d1rrkAxXYE0SUPuKGknPaavrUjR7DWxvXf+Df5SyLeCyGYaTxeL2Ul1QRj0WJx6JksxkUXSdcWkZ5aRmWZZFMJLAdG4/Lg8tlEI/0EItGB0y2DBYVgQI9PUem/P0+P5qm9Za3TZ8+UTpqbkFm30SUhAHiGsMZWwoXF5PJZEgk4icsltlsNp8mEIuiqioVldVkbRtHUTBtGzIZ/D4/Xl9+V9+cnSMRj5NTNQxfgFw82mfGzuv14nF7+q36DxQVoWk60UgEy7ROazxJlpeMLCQlYAT4bodXqg/3OKyqrqW5qeGk36coCtlslu7ubrJ2joqaOsorK1EVBTObJZFM4PV5KQoVkU6l8paUquHy+fAGiwr1lHRdJ1gUIhaL9qkS4Ha78Xp9ZNL5XVuOXTw8/JbqUUmUok9iKQmHXYjhzSrWNI3SsnL27dk9qPfiOOguF46i0tbWgt/nZ/SYcVSNGcO8y6/CGwjQ1dHOjo0beHvTBro7OwAF3e3F8GTJppKEi0tIJZN9Zu4URcnHrbxeOtpasXvLpHyY2l4QUTorXbfhdt/KKirp7OwgZ5/8gFc1DRQFw+PBUfMCZVkWE2bM5Mo7PktM0clkLbxV45gzcRp6IMg7a1bT0d6GhYXL68PjcpGzLJLJxJHOpxvMmHUREyZOpaOtha7eypmnU5ROR9sLIkpnnygNs+umKAqlpWV0drRjD2LAa6qGquvkOFIS94I5c7niU3cRQcPMWpi5HKZpEbN1pi++mdb6/di2TSIew3IZ2Ok00Z6uPoJTVVPLgksWEQgECQQDtHe00dPz3hlp/6N/CmcWiSmNHP9t2J7VPp8fTdeJxwe3ucDhHWhVVcXv9+MPBJl92SLimgvTzBUEybRymGaOnpzGjEsW0dbaguFyUV5RhappZI/JPXIcOz87p4Cu6dg5+zTmJx3tsTlIhQCxlIRjntS2PXxxjeKSUhLxWJ/lHSdhZqFqKi6XC9WVL4Ubi0QoKi4lc5SFdFiQTMvCyuWorqomWFSEgkIqmQJFIRgMkognCoHs1uYmNm1YR3lFJY5tk7NzpzU/6XB7H257sZTEUhKOeV4PtSq5PR7GT5zMlGnT85s7DqJAm6Zp+Hx+3B4PpmURjUTImlk621qxbbufIJlWDhyH9qYm/D4/gWCQZCpJIhFHUVSKi4txuz1AfnnL5o1vsmbVa7yzZTNul4viktIPRdsLYimd9ZbSUGcVa7rOuPGTWHDpIoLBIHYuR2tLM6lU8qQEqbi4BK/HS6S7i6wDiqYBsPLFP3HTlPOI2VofQTKtHKWaxZpVyzEtk+6ebnRVJeDzE+3pxsxmCYXDpNNpopGefFZ3LIaqqhyM11NVXYOZzRKJ9Jw+x9kRS0ksJaHPsDi8OeNQ5skYuk5ZeTkVFZWES0qorRuNz+8/8Y6hqpSVVaAoCtFohHg0Ss46Eu/Zte1dXv/9QwSsNDg2Vi6HgkOparF56aNEujrp7OwkmUgQ6e4iFu0hFA7j8/uJ9HSjaxo1NXX53VD0/I4oPd1dHDywnzHjJuD1ek+bodSn/cVaEktJJIm+O8YO0aDIZLO0NDfR1dVJSWkZ7W1t+WqPXh+RaH4JyPGC3l6vj+qaw4mWCn6/H9MyyaUcNMMARcM0s2xcsYyeliZmX3ENZeESujvaeG35K8Qj3bS1tub3nrNtMskkZjpJMpGgqChEZVU1XV2dpFJJRo0aQyaTIZPNLy2Jx2LU79vDtPMu4K2N64c9xnT0pg2Hi+0JIkoiTI4z5CVZHdtm397dxKIRyiuqUBRobmrEtnMEi0KUjC0jmYgT662f5Ng2iqIQDhfj9njYtzefZOnz+0FRsHv3dNMMF4bXh6IomFmT7VveZvuWtwsF20KhEFbOJptJ5ytdZtJYmVThe0YiPSQSccrKylFUlfaOdkaNGk0sFkXT8iVMotEI9Xv3csHM2Wx9e9OQb1YwkKUEsvZNREno58INpaUEFOJIrS3NTJg0mbKKShoO1tPe1kqn2o4/ECAUDqOqKmbWxOv1kkgmaGlu6uPGKVDIL0rF8sXXdJenEF9yHAd6K0iCQiIew7FzmJk02QHSECzLoqWlGa/PR0V5BQoKtm1TXlFJNNJDKpWms7Mdw2Uwaco57N2zi2wmM1wtf9rKxggSUzqLLCWGPKZ0LIcO1FNcXEJRUSgvWLZNLBqlo72tt/JkCBQFBQgEi/B6vRiGgaqqfbLAHdsmFY2QivVgplPkzCx2Loeh6xiGTrSnm2wyQToWJRuP4TjHd79SySQdnR1Yufw+c7ZtEwoVUxQKYRgGLc1NRCMR6kaNwe12n4aYkvRFsZSEwqAo5MoM02my2SyHDuxn7PiJvPvO2+Ry+R1EfP4Atm1z4EA9uVwOr9eD1+vD7XKDAj6vj3Q6haqqhfiO4ziY6TRWJoOq67jcHnS3i0hHG4lYFPskNhdwuz20tbaQTCYJBAIoLjehUBiX4SKRjNPc1EBN3SiqamppbmwcXK7VB8T0juQpyYJcsZSEPjElZ5jXm3R1dRKPxxg7fgJ+f4BgsAjLNIlGI5hmFtvOkUgk6OhoJxLpwTRNdMPA7fFQXFJKUVHegjn6urFtXJpGKh4j3tNNzjRPWJDymwZ4SaWSmGY+DSAajRCLRvD5fJSUlBEuLqGzvQ3HdqisqkbXjSE3U/vOvgkiSmIo9RkUwz0uDtbvo6a2jorKKuLxGLFYbMAFsKaZJZmIE4/HiER6yKTTaLpOcUkpZeUV+Hx+VFXNV6XUdRKJ+EnPlOm6ng+Y9y4tsW2bVCpJNBqhvaMNM5uhpKSUktJSopEedF2noqoKVVWHUpP6tr90SXHfhGM2RBzGYeHxeCgrq2DH9m1MmDSZ+vp973s+RVFRUEgmk/mZN01D03RcLhf+QICKikrcHg+HDtYPKlvc7w+QTqX6WSi5XI5kIkE2kyEej1NaVk5ZeQXJRIJQcQmmadLR1jpElo0jyZMiSsIAw+Ko8hnDQ1FRCJ/fT2trC7mchcfrZfyESezZteP4ZrSaX/d2WHAOlxUxzSypVBKPx0M00kNJaRnFJQ6Rnh6SycQJW0yBQIDOzs7jvm5ZFpYVJ5PJEAqFKK+sxDQzjB03gUxvRviQWKpSukTcN+H0xZQ0TaO8vAK320N7W2tv7MimtaUJt8f9vmvNVE3Dtu1+FonjOBQXlxCNRmlubqKh4RCdnR0EgkFGjx5DWXkFhmEUKk4eD6/Pf0LLXizLpLOzg317dmPnbOycxUWz5zJ2/ETOnzGLYLBIYkpiKQlDGlXqk9E9NOT3XPNQVBQik0kTjUb6DLpMOk1LUxN1o8YQjUb6uV+KoqBrOpbVv5RIMFiEqulEOtoLAzudStGSSuV3JfEHqKquwbIs4vE46VQS27b7CJw/EDjpbbmz2Sz79u6muLuUyxZdxd2f/yKRnm46L5zF04//gVgsOpjWP2aLJREmESXRpGMyuk99UGiahtfnw+vxEotFj2uNxKIRYkUhqqtraWo81MftUhQFXdexTKtfXCoUCtHQcGjAz8xZFpFID5FIDx6Ph0CgiGAwSCadJpPJYJpZLMsicBKipOsGhmGgG/mfHo+nkPRYUlpONBqhKBQelCj1eSBISoCIktDXfRsKS+nwNtiaqtLT092nQP9AlkdXZwdV1TUEi0JEjtryCPKVBjJHZVK7XC6KisK0tbWe0LWk02nS6XReJL0+PF4vXq+XXC5HcXEp0UgURVH6bVJpGC7cHg9utxu3x4OhHxEkTdcxs1kaGg4SLi7B7XLT1tJCd3fX4GNKssxEREk4dmA4R6yUQQ4MRVEIBIO4DBdZM0t39MSshlg0QlFRiJLSUpLJBGY2W/g8TdMxe903XdcJBItIJOLvK3QDWk+5HPEwFs/mAAAgAElEQVR4jEQijuFyEQ4VoyoqPp8Pf8CPoqioqobL7cLt9vSb8s+k0/R0d5NKJUglU+RyFqqm0d3RgdvjobOjneQJbhs1kKVUSAoVM0lESTiiQ6diKRmGQVEohG07xBNxMun0SVlpbW0tjB03gXC4mI72tt5rOey+mXnBCwTJWRapVHLQFoXjODi2g8fnQ9VUakePpagoiGVa2I6DZWaJxWK9699SvS5fesDz2bkc+/ftGQrvWVICRJSE93PfTlaWvF4f/kCATDpNMpkY1E4g2UyG1pZmRo8ZSzQaIZNOo2oq9FpwgUAQVdOIRnpOMkFSwePxECwqIlgUoqgoVNieu62lie7ubpoaDpLL5XAcB1VR0HWj99z5Wbfhd6kccd9ElIRjXTfHOfmBoaoqwaIi3G5PQUhOZVBFIz3EYjFqakdRv28Phm5gmSZutwe3x008Hv/AEiKKouD3BwgXFxMKF1MUKgYF4tEIsWg0n2SZyxEOh2lpbiKbzfYLrmuahq7r+Hx+qqpr88tgYhFSyeSwiMaxbS8unIiS0M9S+mA0TcvnFznQ0d6ObeeG5BoO7N/L3AWX0tbajKEb2LaN3+8nk8kO6BKqqkooFKa4tIyS0lLC4RIymTTdXZ10dXWyf++e3o0njwhvMFhELBrtJ0iHryGfMGmRyWTo7u7C5/MRCoWpqKgiFo30rtMzh7z9xVISURL6BTY+2Hs7bEmUV1SSiMeJRiNDPji3v7uFSy69nHQmTU93Fy3NTUSiEQzDhWEYBItChMNhQsUlBAJBEvEY3d2dHNi/jy3dmzDN7Puew9O7APeDBODw64lEgkQiUdjyu6q6FssyiUXzqQ627bxveZShantBROkjaykdz33QdR2vz084XExba3OfqfqhRFVVpp8/o5AisG7NKkpKy/D7A+i6QSIZJ9LTTet724hGIycVY8qvndMwsycfK7Isi+6uTnq6u/B6vQSCRYRCIVKpFKlUCssyCzWZBuM+i6UkoiQcz31z+ltHLrcbv9+Pgsqhg/XDei1FofxOI8WlZRguN8GiEPX79nCwfn+vZTJ4q8TldmOZ5im5m47jkEwmSSaT6LqO3+8nFAqTs3NksxnMrFlI0DwhkZFyuCJKwkDug4M9wLR0fs81H263h3Q6TWKwuTgnQWd7O82NDaRSSbo6O3jn7U20HlUe91Rwu9xYljWoGcLjWU+RSIRoNIrb7c7PRPr92HY+QTObzZLJpN/3fI5Doe2RjG4RJeEoF6JggeRHhWEYBAJBUCAWyweGT8eTPJGIs3rlCkLhEIl4gvbWliFzC3XdIJ1ODfkOJY7j9Mked3s8uFwuvD4fPr8fy7RIpZNkM5kB2vBI28vMm4iSQP8tlgC8Ph/BQJBUKkUymSxsc326aG9rob2tZWg7mm4AzvDuTMKRWkypZLKwNMVluCgqCqEoCqlUklQyWbgO5xj3TWRJREk4pmyGx+slGAgS6d0e2xnmfc9OF4bLyGdtW+ZpalYHM5vFzGZJq/nqBYZu5Nu3sgjTNInHYqAcE1OSuJKIktBXlMaMGVeIhXyYMAwXjm0Pu6U0ELZtYx8WqN5NEDxeH3WjRnHueecfE+gWURJR+oiTyfadKZo1ey5jx0/gkQd/zvo31w77DrGnpZPpOqqikM1mz/i1HG7Pc889jzvv+XNKS8sKbW+ZFplMVjrlGUYDviPNcOYws1mCRSFqamp7N3PMr2ebceEsKquq6GhvI5lInNXi5HLlV/8nU8khm3kbnLVmMGr0GP7sU3ey5OO3Eyw6UrGys7ODla8v592tW6RTiigJO3dsp6OjndKSMrw+P7quo+s6Y8eNZ8rUaViW9b7F2kY6bo8Hw3CRGGC33NOBoiiUl1cwZ97FfPKOuzl/xkx0Pe8kpFIp9u/dw+N//B0rXn1ZOuMIQBEneuRQFAqx+NobuXDWbGrrRhUGTiaTYfPG9byx6jW2b3/3pEqTnGnyC4fzM189gyzEdqqCOG3auVx82SJmzppT2GnXsiwaGw7x1qYNvPTic0QjEemAIkrCgKarpjHlnOksuPhSLrhwFuFw8REXo6OdN1a/zoY313Ggft9ZkYGs6wahcJhUMkEyefosPUVRGDN2PLPnzuPiSxZSWlZeeK2nu5stb29izRur2PnetjPqUgoiSmfHTVEUQuEw551/IZctvJwJk6YUrKZcLsfuXTt4c+1q1q9bO7i61KfTUnHnd0xpb2s9bYM/GCxizrz5zJ1/CZMmT0XTtIJ1tGf3Tla9voKt77xFpKdHlpaIKAkng2EYVFbVMHf+xSy64iqKisJAfuo6Go2w/d13WPHqS+zZvWtEPu0P11fy+ny0n2Bd71O1MidOmszlVy5m2rnnFxImAaLRHl5b/gpvrn2D1pamIS9/IogofaTwen3UjhrFbX/2GaZMnVb4vWVZdHa0s2rlCl5b/nI+GXAEoaoqxcUlZE2TWHR4YzaBYJDLr7iGSy67nNKy8oJlCfmJhMf++BsaDx06aycLRJSEEWl16LrOpZddwcdv/xR+f6DwWi6X40D9Ph759S+o37d3xFyzpmlUVlbT0dE2rDlKY8dP4M67v8CYseMLrhrk1/E98ejvWb1y+YlXDRBElISTp6Kyik/c/mnOv+BCXC53wUWxLJNlLz7Pi88/QyIRP+O5TS6Xm4rKShoOHRwWK8zvD3Dt9TdzzbU3oPVaRo7jkM1meGfLZp549Pe0tbZIhxFREk4HHo+HWbPnccVVi6muqcPtdvXeTmhqauSpx3/P7p07iMdjZ8xCCBcXo6oaXZ0dQ2oxBoJBJk2eyi23foqamtreVxwymSzNTQ0sf+UlNm1Y11uKVzjbkOTJsxTLsmg4dIC9e3bhkJ9x8ni9KIpCMBhk+nkzCIVCJBJxkonEGVlzVlJaRjQSGbJzezwexo6fwDWLb+CGj32c4pISIL90pKO9jfVvvsHSJx9l+7atZ+T7CiJKAhCNRti9awedHR35DQWKS9A0DUPXqRs9hrFjx2MYLqLRyCnt2XbSHUvTKA6X0NV16laSqqqUV1Qy/5LLuPaGm5l27gW4DAN66yht3fIWr7z8Aq+veIXuri7pFOK+CSOF8opKLpgxi7kLLmb0mHGF36dSKXbt2Mb6dWvY8vbmwi64w0kwGMTnD9Da0nxKn2O4XFwwYyZz5i1g8tTpeL3ewmsHD+znzTWr2fL25tOSciCIKAmDtFDGjp/ARbPnMWf+xYVZOsdx6GhvY8f2d3l12Qu0NDcP63VUVdWQSMSInUKaQlV1NVdefT3nTD+X0rLyQkA/kYizfu0bbFy/jvr9eyUjW0RJGPE3tTdpcdKUqVxx1WLGT5yMqh7Zdba1pYXVK1ewdvXrw7YryoSJk9m/b8+gZgDdbjcLLl3IxZdeTmVVVW/VynzsaO+eXax45SV279xBIhGXaX4RJeFss5qKQmEuXXQFly68ok9uUzqVYs/unTz/7NPU7x/a3CaPx0tZefmgUgHGjp/A9TfdwsSJk/Ec5aolEnFWvb6cVa8tJxLpwRbrSERJOLsZP3ESH7/tU4wZOw5VPZJgGItFefXlF1i9cgWpIVowW1paTs7OnVRVAJ/fzyWXXc4VV19LMHikzpFt56jfv48nH/09+/ftkRspoiR8mDBcLi6/4mouu/wqgkWhQvazbdvs2vEeL7/wLAfq959yKd66UaNpbW05oYC62+1hzLjxLL7uRiZPPQdFybuZuVyOWDTCa8tf4fUVy05LcF4QURLOEFXVtVx/0xLGT5hEUejIotVkIsFrK5bx1sYNtLW1DMpFcrlcVFRU0tjY8L7xHk3TqKisYsas2Sy6/Gp8fj/Qu9g40sPe3bt44U/P0NLcKDdMREn4KODxeJgxczYXzZnP6LHj8Hg8hdf279vL6pXL2bNzB90nWZitKBTC0F10dXUcV5SKS0qZNPkcLll4OWPHjS/8Pp1Oc7B+PxvXr+GtzRvPqmJ2goiSMASoqkpVdQ0zL5rL+TNmUV5egdIbb0qnkryzZTNvbVzP3j27yGZPbJausqqaWDRKMpkYwIpyM2HiZGZeNIfzLpiJp7cmuW3n6Ghv5Z23N7N545u0NDd9KDZMEAaHZHR/hHEch1gsyqFDB2htbkJVVUrLylA1FV3Xqa6ppW70aHw+P5FIhNQAQtPXJdMJBIPEY9E+oqIoChWVVSy4bBELL7+KiZPPwTDym1NmsxneeWsTy195kbc2rae7q1Om+cVSEktJyAtHcXEJEydP5arFN1BWXlF4LZVK0nDwAGtWv872be8cN+js8/nx+f10d3UWEhoNl4vp557PgksWUTtqTJ+M7I72Npa99Cf27NpBT3eXiJEgoiT0R9d1wsUlXLrwSuYuuAzD0HtdLJtEIs72d7fy8gtL6enu7vfekpJScrkc0WgEx3EoLill8fU3MXXaeQQCgcLMmmmavLlmFatef5We7i5ZPCuIKAkfjGEYjBs/kZtuuZ3qQnmQvDjF4zFeePYpNq5fW7BuFEWhoqKqsBXU7LkLuO7GW/AHAoVscoCGQwd47unHqa/fhyUlaQURJeFkCRaFuGThFcyeuwC//4jAOI7Drh3beW7pE7S3t+YtrHAxbo+XxdffzKTJUwupBrZtk0zEeXPtalavXD7iyvYKIkrCWYaqqkycNIVLF13Vmz6Qr9vkANGeblavXEFTw0EmT53GzIvmEigK5TuW45BOpThwYB+rXnuFvbt3yayaIKIkDB2BYJCL5izg/Bkzqa6pLSxXMU0T08zicrkLBfvtXI6mpka2btnExvVrxToSRJSEYeowisLoseOYddE8Jk+dRqi4pN/f9HR3sWvHNjZtWMfB+v3SaIKIkjD8eDwepk4/jwtmzGLCpKkYhoFpZtm9cwfvvL2JHe+9KxnZgoiScHpRVZWy8gqmTJ1OVXUtzU0N7Nyxjc6OdokdCSJKwpnD5XLh9njJpFPDur+bIKIkCIJw+i1waQJBEESUBEEQRJQEQRBREgRBEFESBEFESRAEQURJEAQRJUEQBBElQRBElARBEESUBEEQRJQEQRBREgRBEFESBEFESRAEQURJEAQRJUEQBBElQRBElARBEESUBEEQURIEQRBREgRBEFESBEFESRAEQURJEAQRJUEQBBElQRBElARBEESUBEEQURIEQRBREgRBREkQBEFESRAEQURJEAQRJUEQBBElQRBElARBEESUBEEQURIEQRBREgRBREkQBEFESRAEESVBEAQRJUEQBBElQRBElI5D4M/+SGssQaTP0cPKv5uKdrw3FX2S37cf+54EHSv/jinaKVyM+xZ+PcDndr75Tc7TB36LPuu7bOo+9j1xGn5+A+4hayU31TOXcO+3/ouHnnqF1W/tZE9DG61dUbojUTo62ji4bzdbNqxh2Z+e4pFf/ph//db/5AufvJbzyrVTPrt+4Xd4s7t/uxx79ERjdLS3sHfbep5/+Pvcd80EfKdwXiV8HT96L0ok2sJz903k5L+Jlzn/uImOaJxDT3+e0SP9EaxUcNnPd/KrnQ08uP11bjtfH97zGQv57BsHeXD7yuE/18n0N9HlkY0x6lq+9uP/x18sqsOjHOfJ4vYTKvcTKq9h7NSjX8nw4hfH88nf9uAMxcU4Djnbfp9HnIrhCVI2ejplo6dz8ZLPcvdDX+DjX32GJnsQp+t5ie/90wtc/983cvHf/wOfeOozPNpy4t9En3Iv3713Cnp6M//v/oc5aI9wTaq9kQVzfSgA2ljmfmwOT7+zBlMsJWHE3JzKm/l/z/+Bv7n8+IJ0OrG7f8MnK4ooCR/nKApSXj2eWVd8iq//eiOdto9z7v4x37+9ksFdvk3To/fzg3VxlLKb+ObXr6ToRD9IreNT//S3zPZa7Pnl1/jJuyN9aGtU33grE10O8X27ieY0ShbfyjTfR7Dfy9AfqY/NMIu/8598cqyBctZctEM23sqeDc/w4y/fwKd/tANTKeaae24bvOtk7eYXX/tvtmVVRt3xz3zlQs+JNB7F136Tr19TjNP0GN/+9zUkR3rT6dOZf/M0NLuRdf/0v1l1wEIpvYYFC8Nn0f0XUfpw35jqj/M/bqkeMI7iWFEatq5l5auv8Oorh49XeX31OjZt2c7uA410xDLknDP5DeJs+PXv2WopuM6fx6xTeOKn3/oB3/rNQWzXdO773heY+EHBJd88/uYf/4xqpZtX/vkfeKHLGfH325h5K3PH6uQOPMOa9RtZ++xOcmqI85csPnHr8EOCxJRGpplEaOG1zPMp/SwR68CT/PVtX+KR9+IfGCfSPEX4iHGmhmSuYS/1WYeZ7jpGV2mwJzdIAyzK8u99m2dv+hVL5v8937n9Ce76fTMDh4gMzrn3e3xhsk5q/X/yrd83YI/4++3nnCU3UqaZHHr2SQ5YOZw/Pcn++6YxYd6tXFT9GK822R+Z3i+W0gh9Vpwz41zc/TQpyYrv/90JCRJALh0llj6DVkIuQiTugBIiHDq1x73T+iT/8P3VxJVSrr//G1x5nM9TR32af/ybi/BaO3jg6z9lh3UWPIKKLufiK8sgu4W1f9qNDTiHnmHNxjS4L2LBDeM+UgNVRGlE4qZ2dGX/m2O+y+ur2nHOlq/hWFiWA4oLl+tUfZAc+x78Gj96J4M66g7+4a9n0S+6pJRw/f33c2XYofH33+D/bEidFVZx8KpbOT+kYG58gjcP9VpEdgublq4ijcGYj93CKO2j0/tFlEbkXSmlvFTtF+B0sgepbzmLzHjFwDAUcLJks0MgpZm3+eE3HmR/zmDaF/+FL0zqG33wzf87vnNbNUrn8/zTd5fRczaot1rL7CWX4CbOtqUv0GMfcdVjy59ga8RGG38L8y8wRJTOLAYXfGsTXcdL1Gv8Jdd7Psx3JURxqP+tceJRYrmz6HvoJRQHFHCi9ESGRiHiK/+V7zzViuOfx9/80+3UHG4mYxpf/Oc/Z6KeYO2/389jzWeHeKtjP8bFM90QWcGa5Z19rGAn+hpvvNqBo9Uxe8kCXCJKwpl031wD9EAnkyJzVK/Vpv0vVnW9X5Z1D8v/ehJnyvLXJ01jskvBMZtoaB0iNXXaefY73+O1qELptd/ia1eFUcinC3x1pgdz20+4/5d7ODu0W2fUTR9ntOEQffVxtkaPFe4E7y39E105jeKrP8G5QREl4Qy6PS5jgBhMzuKsMZSUMq754qeYrDuYW9eyOTF0H20ffJj7/+9mUmodn/ru/2TOqJv45tevIuQc4Dff+AFvpc+SNnLNZMFNE9FyTWxc+gaZAf7E3PQkbx60UIqvZsGiko9EzpKIkjCEvcnAW1zH9Ms+xdd+s4xf3DEaze7k+Qf+yP4h9aZMtv34a/xqTw596pf49bL/4tYqaHvmO/zrithZMxHgnnsrc2o1cgefYc2mzMB/ZL3Duud2kVMCnLvkOsIfgREreUp9fQNs24EheB7lP2ewl2GSNQd4v6ZzJidh1JI7eazzzhP/GnYPm3/4ef72sdahF4rkGv7jm49yy28/TU2tByf2Gt//9pO0ni2KpBRx3pLrCKkWDc89Rf1xUxdyND77JPvv/QYT5tzKnLrf89LBD3fO0ggVJZN3vreQxf9n58CJb8Hb+PV7P+Va91Cf18Ia8ITK8WVKAUXpL25WzjqFgZghmx3gVG5Pn9wlu2Ep3/nCvvzTUxvPbf98P9dVDt+j1HFyWFnruMmIjmOTyyboaannvQ0rWPrwA/xxTROZYXqAdL3wXX64/lb+Zb7G/oe+yyP7z55ZAKXkGhYsCoP5Fmuf3fW+CZ72oWdZu/lvmTB3BvNvnMSyn+zEFlE6AzaLlSGTTg8cQ3GZ2M6wnBRzoHWbunZcC0XRB7ZeTPMUFoDaEboj/budEghTdNTJnOgOXn1yR/4fxhwu+No3uK5yGO9J9+/49KQv8nJ2hHQSu5P2ThtQ6G7vOItW06sUL76V6X4FRZnJ7S8f4PYTfGfdTR9n7AP/wj6LDy0SU+oz6hLE4wOIgceH7zimkuLz4x3AUkrGEoO3lOxOOrrsfu9X3KMZV6PJfTrb0cYy92NzMXBwcjnsEzwcx0EbczMLZro/1M0jMaU+WhIlGu8vJWppHdUe6P8oVgjU1NJ/xYNNNBI9BRM7Q8OBVmzG9LXC9OksuqySH+5twpa7dfZaApNuYcF5BiSX8curP8cbnSfw+FJCXPSD1Xzp+louWnIJj61/dZjcYrGURha5Zhqa+juMivcyPnn7+P7Ja+4J3Hb7/AHWqOVoaWo9BeGw2LFlW5+cpF6zjMv+5z9z61iP3KuzFoPxH7uFGs0m8drjvHWiFQycCFufepGIrVJ01a2c/yEuHSCWUt+gCdu2HiJ35TGlV9UwV/1gPe/e+w47DnYSt8DwlzL6nPOZUuntHwTPHWDb9lOZmnboWfkyG9LXsbCPb6igj7mdB9Yt4N7XX+ftfa1EUr0BdbWG+SWK3MORjnc+F18/CtXuYPPTK0ieRCfJrHuM9Y23c03d5Vx8VRkbnzyL1kGKKA0Wk3defJmGv5zImGNCN4rqpfKcuVSecwIGV8MyXtl2apFIu/EJfvHs17n09op+5qzir+Oi6+/gIrlhZ58mXXwrsyo07KbnWLPuJEvPZTez5tk9XPmlyZyz5AZKnn6Izg+hHy/u27FPo7U/5j9f7Rq862V389r/+SnrTnWGyuniuW//PU8dsj6UT8OPJEoJM265moBq0fLc4+w56T5icWjp4xwwwZj5CeaO+XBOeogo9ROVeh7+80/y3WWH+sd0PkhHMo0s/96n+PMH9w9JINpueIz7rr+LH65qOulrEUagJlVez8UXB8Daztpntw1qyZB9cCmrN6bBOI/5N0/9UA5gcd8GNFLe4D8/fiF/mHcjn7jxKuZeOI2JY2qoLA0R8LrRVbCtLKl4D11tTRzYs5231y7n+SefZc2hoa0Gnalfyjevf4n/nn09S268grkzpjNpXB1VpWGCfg+GCrlsimQ8Sk9nB21trbS0NNN86AD1+3axeUUDObmlI+L5X379rUzxgLXxCd7cN8i7Yjez8amV3DZvMTU3fIIJP9nG7g/ZdicKiHcgCIK4b4IgCCJKgiCIKAmCIIgoCYIgoiQIgiCiJAiCiJIgCIKIkiAIIkqCIAgiSoIgCCJKgiCIKAmCIIgoCYIgoiQIgiCiJAiCiJIgCIKIkiAIIkqCIAgiSoIgiCgJgiCIKAmCIIgoCYIgoiQIgvA+yGaUR6NN4mNfnMq7P32WvUftFajWXs2X/izMaw88zrtxB1Cpvep/MK/pFzyx3er7/r/5BBMzSfL7A9q0rf0dj244sg24Wns1X7rzArRktnfDPZumlQ/zxNvR3n8rlCz4HPdeEiSRtkFR0ZU425//Ay/tih+1Sd8xfwdgd7Hh0d+xts3u8zdfmLSVXzyyni4bUIqZf9cNmE/+lo2xo7b8U0s49/olLJoQQMnZ2FYDK3+zlK1x5yTON8BTr3Q+d97i4eUHV9Ccyz8HKy69m+tyz/LImo4j7VJyHjcsuYxxARXHAYUMzRv+xNJ1jRzZ3Vpj1DX3Mq/p5zz2bu8OjPo5fPzesWz+6QvUf9D+jkYVs2++ibm1XhQHFCVH4uAbLH32bTpyR57TVYu+wNWJP/CbDVEcFIrn3cN9U9/lx79eT8QBpXg+d9/k8PQj6+hxTrKPCCJKQ4Z3MpdfPpG9z+0m9X7bdya38fQxonYsuX0v8cCjWzn+VvI5mlY+xMPrenAAY8xi/vySKfh2byLhHP/vjqu1dZdw9Xk7eHRL9Lh/p42ex6KSbfzuR2/SlTux6zoR7M5NrGq6h8vO28ijb8fAP43Lprax6uGOo7Y216i96FJKtv+OH63rxgYU71SWfGE+k956nG2ZobmFxvj5zNfW8csfbs23o1rEzE/exZzRW3l+/+EvbdN+sJHgjFqMDVGyipfRY0OY4bGM9mxga8rBXVuH0fgGUWeQfUQQ9+3UcbB2r2NH9ZVcOsp12s9u9XQTc3nwKIN5d46GDe/gueQKpviO9wEK3tJiMgfq6RnyPb6z7H9jI9qcBYw2dGrmzcW1eTX7M327YVGRQktTtCBUTrqFpkiAkF8ZoutQ8IYCxJuajwiGHae5xSRYZPRtsaaDtFeMolID9FGM9W1jzZ4wY+t0QKNqVAmtB9uwR1AfEUvpo4jVzJplLu6+Zj7vPLj61LzE8ddw75cX5q2N9Hs8++CrHMgdfzB5KiopSh0c4OmrUX3JXfzFRfnhkat/hV89t4NjDQu7ayOvvHsrtywcy/6Xega2Ilw62Uz2AyygEztfP0mPbGHlvnu4YtFCrNEHWPnryDHn0dC0HJZ19G8tcjkNtzZ0O8trmkbOsvp8Ws7Koet63/3rM4c4lJrFqJBCU2AMpS17WLbPzx1jK9H2pqmrinDw9dyw9hERJeGErKVM/ess7/wci2dvZ/nx/sw3jZu/NAbTAexGVj78NO/EnWPct5f55fu6bxo1Cz/HV+bZoHsJKs289oeXBxClHM2rHz4Bd8qiZe0y9n5+MRfXPE1q0G1woufrf/7GdetI33ctzgs/ofEUQiyO4wx4b4bUW3KiHGxwcUmdlz3hanrqXyd1KERs/jhKvDFqnUO8knYG30cEEaWh66xJdr66mhl3X8P5jQo0DRRT2s4zHxBTOpHB3/T6r/KD3zWRmz5/Ph3tmVMbeNmDrHy1hc8vnsNBp/+lm1kLV8DF0NklxzRdooFD7W04jckBPj9HLqdh6Ep/6ynn9Bn0uVwOVT0q6qCoqLkcuRO46Fwuh+42+nxHTe9vPYFN28EWSqZMZnxRhvrNWZxkPQdy1zNhUpRA0xZ67FPoI4LElIb2IfoOyzYYnHdu8elpvGw9W+srOX+yn1OLrjikdq9gZWwqF1Sr/V/r6sYzegyhM9IjbKJRqKwpKrSp4qmmJhQnkugrSrFIktLaatyH3c6KKkqSUWLOCXz/SJxATTXew//BzksAAAFzSURBVA2pBqiuMohGzf4C1niQntELmGkcpD7hgBNh/yEvMxfU0HWohdxg+4jqJhjyiTUgltLQDqDOjS+x7oK7KBt0TGkx9/3V5cdJCejv+hx6dy/XXjaV4NZNx8z4aNRc9ln+cs4JTtE7Md59ZSUzxkztPwgPrOP1827hji/PwbYcFKeNtX94nM3dzuDPdxKWYeOmVXQvuYMvz1ZwAEUxad34LLszfYUlsX012865nvu+amDboOa62frSE/1nwgbA3LeOtefdxOf+cmHeVFIgfWg1Tx/sLzFO+hCHssWUNtT3TvvbtO9vwr2gjEMN5qD7iOI/j//frh3UAACDQBDERH3hAFUoqrFa4NMXMyI2XELWiduDF4alfl3rAOYbIEoAogSIEoAoAaIEIEoAogSIEoAoAaIEIEqAKAGIEiBKAKIEiBKAKAGIEiBKAKIEiBKAKAGiBCBKgCgBiBIgSgCiBCBKgCgBiBIgSgCiBIgSgCgBCzx/8uhGABWTwwAAAABJRU5ErkJggg==", It = {
	ok: "rp-ok",
	warn: "rp-warn",
	bad: "rp-bad",
	muted: "rp-muted"
};
function Lt({ state: e, title: t, desc: n, fix: r, link: i }) {
	return /* @__PURE__ */ f("div", {
		className: "rp-row",
		children: [/* @__PURE__ */ d("span", { className: `rp-dot ${It[e] || "rp-muted"}` }), /* @__PURE__ */ f("div", {
			className: "rp-bodyc",
			children: [
				/* @__PURE__ */ d("div", {
					className: "rp-title",
					children: t
				}),
				n && /* @__PURE__ */ d("div", {
					className: "rp-desc",
					children: n
				}),
				r && /* @__PURE__ */ d("pre", {
					className: "rp-fix",
					children: r
				}),
				i && /* @__PURE__ */ d("a", {
					className: "rp-link",
					href: i.href,
					children: i.label
				})
			]
		})]
	});
}
function Rt() {
	let [e, t] = l(null);
	if (o(() => {
		let e = !0;
		return K("/api/readiness").then((n) => {
			e && t(n);
		}).catch(() => {}), () => {
			e = !1;
		};
	}, []), !e || e.central && e.central.auth_mode && e.central.auth_mode !== "open") return null;
	let n = typeof window < "u" && window.location && window.location.origin || e.connect && e.connect.base_url || "", r = e.storage || {}, i = e.serving || {}, a = e.console || {}, s = (r.model_count || 0) > 0;
	return /* @__PURE__ */ f("section", {
		className: "rp",
		id: "your-instance",
		children: [/* @__PURE__ */ f("h2", {
			className: "landing-section-title",
			children: ["Your instance", e.central && e.central.version ? ` · v${e.central.version}` : ""]
		}), /* @__PURE__ */ f("div", {
			className: "rp-cards",
			children: [
				/* @__PURE__ */ d(Lt, {
					state: r.error ? "bad" : s ? "ok" : r.exists ? "warn" : "bad",
					title: `Model storage${r.free_gb == null ? "" : ` — ${r.free_gb} GB free`}`,
					desc: r.error ? `Could not read storage: ${r.error}` : r.exists ? `${r.root} — ${r.model_count || 0} model(s)${r.writable ? "" : " (not writable)"}` : `Storage root ${r.root || "unset"} does not exist yet.`,
					fix: !r.error && !s ? "export DEFAULT_ROOT=/path/with/space\nhugpy models pull <model-key>   # or drop a .gguf there" : null
				}),
				/* @__PURE__ */ d(Lt, {
					state: i.any_serving ? "ok" : "warn",
					title: "A model is serving",
					desc: i.any_serving ? "At least one slot is live." : i.enabled === !1 ? "Model slots are not enabled on this install." : "No slot is serving a model yet.",
					fix: i.any_serving ? null : `curl -s ${n}/api/llm/slots   # check slot health`
				}),
				/* @__PURE__ */ d(Lt, {
					state: "ok",
					title: "Connect a bot or worker",
					desc: "Point any hugpy arm at this central:",
					fix: `export HUGPY_BASE_URL=${n}\nhugpy bot       # or: hugpy worker --central ${n}`
				}),
				a.configured ? /* @__PURE__ */ d(Lt, {
					state: "ok",
					title: "Console",
					desc: "A delegated console endpoint is configured.",
					link: {
						href: a.url,
						label: "Open console →"
					}
				}) : /* @__PURE__ */ d(Lt, {
					state: "muted",
					title: "Console (optional)",
					desc: "Run the separate @hugpy/console broker on its own port and point hugpy at it — hugpy never spawns the shell itself, it only links to the delegated endpoint.",
					fix: "python3 pty_broker.py --resolver local --port 8801\nexport HUGPY_CONSOLE_URL=http://<host>:8801"
				})
			]
		})]
	});
}
//#endregion
//#region src/components/Landing/Landing.jsx
var zt = {
	width: 22,
	height: 22,
	viewBox: "0 0 24 24",
	fill: "none",
	stroke: "currentColor",
	strokeWidth: 1.6,
	strokeLinecap: "round",
	strokeLinejoin: "round"
}, Bt = () => /* @__PURE__ */ f("svg", {
	...zt,
	children: [
		/* @__PURE__ */ d("rect", {
			x: "3",
			y: "6",
			width: "18",
			height: "12",
			rx: "2"
		}),
		/* @__PURE__ */ d("rect", {
			x: "6",
			y: "9",
			width: "6",
			height: "6",
			rx: "1"
		}),
		/* @__PURE__ */ d("path", { d: "M15 9v6M18 9v6" })
	]
}), Vt = () => /* @__PURE__ */ f("svg", {
	...zt,
	children: [/* @__PURE__ */ d("rect", {
		x: "7",
		y: "3",
		width: "10",
		height: "18",
		rx: "2"
	}), /* @__PURE__ */ d("path", { d: "M11 18h2" })]
}), Ht = () => /* @__PURE__ */ f("svg", {
	...zt,
	children: [/* @__PURE__ */ d("circle", {
		cx: "8",
		cy: "14",
		r: "4"
	}), /* @__PURE__ */ d("path", { d: "M11 11l8-8M16 6l2 2M19 3l2 2" })]
}), Ut = () => /* @__PURE__ */ d("svg", {
	...zt,
	children: /* @__PURE__ */ d("path", { d: "M21 12a8 8 0 0 1-11.5 7.2L4 21l1.8-5.5A8 8 0 1 1 21 12z" })
});
function Wt({ icon: e, label: t, sub: n }) {
	return /* @__PURE__ */ f("div", {
		className: "lf-node",
		children: [
			/* @__PURE__ */ d("div", {
				className: "lf-ic",
				children: e
			}),
			/* @__PURE__ */ d("div", {
				className: "lf-lb",
				children: t
			}),
			/* @__PURE__ */ d("div", {
				className: "lf-sb",
				children: n
			})
		]
	});
}
var Gt = [
	["Model registry & downloads", "Search the Hugging Face Hub, pick a quantization, and pull with resumable, cancellable jobs. Everything lands in one manifest-backed registry."],
	["Streaming chat, never truncated", "Token-streamed chat across transformers and llama.cpp backends, with unbounded auto-continuation past per-pass token caps."],
	["OpenAI-compatible API", "/v1/chat/completions and /v1/models on your own box. Point any OpenAI SDK at it. Mint and revoke API keys from the console — or run it open."],
	["GPU worker fleet", "Join any box with `hugpy worker`. Central registers, heartbeats, assigns models, probes VRAM fit, and routes requests with local fallback."],
	["Cross-machine sharding", "Models too big for any single GPU split across the fleet via llama.cpp RPC — a deterministic allocator picks the placement, hugpy runs the lead."],
	["One process, one port", "Console and API ship in a single pip package. `hugpy serve` and you’re at http://localhost:7002 — no nginx, no node."],
	["Phones as a video-analytics pool", "Add phones as workers, not just GPUs. The phone-brick pool runs an ONNX-YOLO vision worker on each handset; fan one frame across the fleet and the console resolves a live consensus verdict with per-phone class, confidence, and detections."],
	["Discord bot & cross-machine comms", "Bind any model — or a Claude keeper — to a Discord channel or user, and the hugpy bot relays both ways. Console↔Discord bridges plus any↔any comms mailboxes let models, keepers, and people talk across the whole fleet."],
	["Portable agent fleet", "Enroll any box as an autonomous agent that uses your hugpy fleet as its brain — a workspace-jailed toolset, crash-safe memory, and policy gates with operator approval over Discord. One command to install, no local GPU required."]
];
function Kt() {
	let [e, t] = l(!1), [n, r] = l("/console");
	return o(() => {
		let e = !0;
		return me().then((t) => {
			e && r(t.mode === "external" ? "/login" : "/console");
		}), () => {
			e = !1;
		};
	}, []), /* @__PURE__ */ f("div", {
		className: "landing",
		children: [
			/* @__PURE__ */ f(Pt, { children: [
				/* @__PURE__ */ d("title", { children: "Hugpy — Inference You Own · Self-Hosted LLM Server" }),
				/* @__PURE__ */ d("meta", {
					name: "description",
					content: "A self-hosted LLM console: pull Hugging Face models, expose an OpenAI-compatible API with keys you mint yourself, and pool GPUs across machines — all on your hardware."
				}),
				/* @__PURE__ */ d("link", {
					rel: "canonical",
					href: "https://hugpy.ai/"
				}),
				/* @__PURE__ */ d("meta", {
					property: "og:title",
					content: "Hugpy — Inference You Own"
				}),
				/* @__PURE__ */ d("meta", {
					property: "og:url",
					content: "https://hugpy.ai/"
				})
			] }),
			/* @__PURE__ */ d(be, { brand: !1 }),
			/* @__PURE__ */ f("div", {
				className: "landing-hero",
				children: [
					/* @__PURE__ */ d("img", {
						className: "landing-lockup",
						src: Ft,
						alt: "HUGPY AI"
					}),
					/* @__PURE__ */ f("h1", { children: ["Inference you own", /* @__PURE__ */ d("span", {
						className: "landing-dim",
						children: "."
					})] }),
					/* @__PURE__ */ d("p", {
						className: "landing-sub",
						children: "A self-hosted LLM console. Pull models from the Hugging Face Hub, chat with streaming auto-continuation, expose an OpenAI-compatible API with keys you mint yourself, and pool GPUs across machines — all on your hardware."
					}),
					/* @__PURE__ */ f("div", {
						className: "landing-cta-row",
						children: [/* @__PURE__ */ d(p, {
							className: "landing-btn primary",
							to: n,
							children: "Open the console →"
						}), /* @__PURE__ */ d("a", {
							className: "landing-btn",
							href: "#quickstart",
							children: "Run your own"
						})]
					}),
					/* @__PURE__ */ f("div", {
						className: "landing-pip",
						onClick: () => {
							navigator.clipboard?.writeText("pip install hugpy").then(() => {
								t(!0), setTimeout(() => t(!1), 1500);
							});
						},
						title: "copy",
						children: [
							/* @__PURE__ */ d("span", {
								className: "landing-dollar",
								children: "$"
							}),
							" pip install hugpy",
							e && /* @__PURE__ */ d("span", {
								className: "landing-copied",
								children: "✓ copied"
							})
						]
					})
				]
			}),
			/* @__PURE__ */ d(Rt, {}),
			/* @__PURE__ */ f("div", {
				className: "landing-fleet",
				children: [
					/* @__PURE__ */ d(Wt, {
						icon: /* @__PURE__ */ d(Bt, {}),
						label: "GPU box",
						sub: "--role rpc"
					}),
					/* @__PURE__ */ d("div", { className: "lf-wire" }),
					/* @__PURE__ */ d(Wt, {
						icon: /* @__PURE__ */ d(Vt, {}),
						label: "Phone",
						sub: "ONNX-YOLO"
					}),
					/* @__PURE__ */ d("div", { className: "lf-wire" }),
					/* @__PURE__ */ f("div", {
						className: "lf-core",
						children: [/* @__PURE__ */ d("div", {
							className: "lf-ring",
							children: /* @__PURE__ */ d("img", {
								src: ce,
								width: 48,
								height: 48,
								alt: "hugpy"
							})
						}), /* @__PURE__ */ d("div", {
							className: "lf-corelb",
							children: "hugpy central"
						})]
					}),
					/* @__PURE__ */ d("div", { className: "lf-wire" }),
					/* @__PURE__ */ d(Wt, {
						icon: /* @__PURE__ */ d(Ht, {}),
						label: "OpenAI API",
						sub: "/v1"
					}),
					/* @__PURE__ */ d("div", { className: "lf-wire" }),
					/* @__PURE__ */ d(Wt, {
						icon: /* @__PURE__ */ d(Ut, {}),
						label: "Discord",
						sub: "relay"
					})
				]
			}),
			/* @__PURE__ */ f("section", {
				id: "features",
				children: [/* @__PURE__ */ d("h2", {
					className: "landing-section-title",
					children: "What's in the box"
				}), /* @__PURE__ */ d("div", {
					className: "landing-grid",
					children: Gt.map(([e, t]) => /* @__PURE__ */ f("div", {
						className: "landing-card",
						children: [/* @__PURE__ */ f("h3", { children: [/* @__PURE__ */ d("span", {
							className: "landing-glyph",
							children: "⬢"
						}), e] }), /* @__PURE__ */ d("p", { children: t })]
					}, e))
				})]
			}),
			/* @__PURE__ */ f("section", {
				id: "quickstart",
				children: [/* @__PURE__ */ d("h2", {
					className: "landing-section-title",
					children: "Quickstart"
				}), /* @__PURE__ */ f("div", {
					className: "landing-steps",
					children: [
						/* @__PURE__ */ f("div", {
							className: "landing-step",
							children: [/* @__PURE__ */ d("div", {
								className: "landing-step-head",
								children: "1 · serve"
							}), /* @__PURE__ */ d("pre", { children: "# console + API in one process\npip install hugpy\nhugpy serve --port 7002" })]
						}),
						/* @__PURE__ */ f("div", {
							className: "landing-step",
							children: [/* @__PURE__ */ d("div", {
								className: "landing-step-head",
								children: "2 · call it like OpenAI"
							}), /* @__PURE__ */ d("pre", { children: "client = OpenAI(\n  base_url=\"http://localhost:7002/api/v1\",\n  api_key=\"hp_…\")  # or open mode" })]
						}),
						/* @__PURE__ */ f("div", {
							className: "landing-step",
							children: [/* @__PURE__ */ d("div", {
								className: "landing-step-head",
								children: "3 · grow the fleet"
							}), /* @__PURE__ */ d("pre", { children: "# on any GPU box\nhugpy worker --central http://your-hugpy:7002\n# or lend the GPU to the shard pool\nhugpy worker --role rpc" })]
						})
					]
				})]
			}),
			/* @__PURE__ */ f("footer", {
				className: "landing-footer",
				children: [
					/* @__PURE__ */ f("div", {
						className: "landing-footer-links",
						children: [
							/* @__PURE__ */ d("a", {
								href: "https://hugpy.ai",
								children: "hugpy.ai"
							}),
							/* @__PURE__ */ d("a", {
								href: "https://x.com/hugpyai",
								target: "_blank",
								rel: "noreferrer",
								children: "X"
							}),
							/* @__PURE__ */ d("a", {
								href: "https://www.instagram.com/hugpy.ai/",
								target: "_blank",
								rel: "noreferrer",
								children: "Instagram"
							}),
							/* @__PURE__ */ d("a", {
								href: "https://www.threads.com/@hugpy.ai",
								target: "_blank",
								rel: "noreferrer",
								children: "Threads"
							}),
							/* @__PURE__ */ d("a", {
								href: "https://www.reddit.com/user/hugpy/",
								target: "_blank",
								rel: "noreferrer",
								children: "Reddit"
							}),
							/* @__PURE__ */ d("a", {
								href: "https://www.minds.com/hugpy/",
								target: "_blank",
								rel: "noreferrer",
								children: "Minds"
							}),
							/* @__PURE__ */ d("a", {
								href: "https://huggingface.co/hugpy-ai",
								target: "_blank",
								rel: "noreferrer",
								children: "Hugging Face"
							}),
							/* @__PURE__ */ d("a", {
								href: "https://github.com/hugpy",
								target: "_blank",
								rel: "noreferrer",
								children: "GitHub"
							}),
							/* @__PURE__ */ d("a", {
								href: "https://pypi.org/project/hugpy/",
								target: "_blank",
								rel: "noreferrer",
								children: "PyPI"
							}),
							/* @__PURE__ */ d("a", {
								href: "https://www.npmjs.com/org/hugpy",
								target: "_blank",
								rel: "noreferrer",
								children: "npm"
							})
						]
					}),
					/* @__PURE__ */ f("div", {
						className: "landing-footer-emails",
						children: [
							/* @__PURE__ */ d("a", {
								href: "mailto:hello@hugpy.ai",
								children: "hello@hugpy.ai"
							}),
							/* @__PURE__ */ d("span", {
								className: "em-note",
								children: "social"
							}),
							/* @__PURE__ */ d("a", {
								href: "mailto:support@hugpy.ai",
								children: "support@hugpy.ai"
							}),
							/* @__PURE__ */ d("span", {
								className: "em-note",
								children: "support"
							}),
							/* @__PURE__ */ d("a", {
								href: "mailto:abuse@hugpy.ai",
								children: "abuse@hugpy.ai"
							}),
							/* @__PURE__ */ d("span", {
								className: "em-note",
								children: "abuse reports"
							})
						]
					}),
					"© 2026 hugpy · source-available · inference you own."
				]
			})
		]
	});
}
//#endregion
//#region src/components/ChatPanel/ChatPanel.jsx
var qt = "/system <text> · /clear · /tokens <N>";
function Jt(e) {
	return e ? Array.isArray(e.tasks) && e.tasks.includes("image-text-to-text") ? !0 : (e.primary_task || e.task) === "image-text-to-text" : !1;
}
function Yt(e) {
	return new Promise((t, n) => {
		let r = new FileReader();
		r.onload = () => t(r.result), r.onerror = () => n(r.error), r.readAsDataURL(e);
	});
}
function Xt({ modelKey: e, model: t, onClose: n, messages: r = [], setMessages: a, chats: u = {}, models: p = [], onSwitchChat: m }) {
	let [h, g] = l(""), [_, v] = l(""), [y, b] = l(null), [x, S] = l(!1), [C, w] = l(null), [T, E] = l(null), D = c(null), O = c(null), k = c(null), A = c(null), j = c(null), M = Jt(t), N = s(() => {
		let t = u || {}, n = Object.keys(t).filter((e) => (t[e]?.length ?? 0) > 0);
		return e && !n.includes(e) && n.push(e), n.map((n) => {
			let r = t[n] || [];
			return {
				modelKey: n,
				name: p.find((e) => (e.model_key ?? e.key) === n)?.name ?? n,
				count: r.length,
				active: n === e && x || r.some((e) => e?.status)
			};
		});
	}, [
		u,
		p,
		e,
		x
	]);
	o(() => {
		O.current?.focus(), E(null);
	}, [e]), o(() => {
		D.current?.scrollIntoView({ behavior: "smooth" });
	}, [r]);
	let P = i(async (e) => {
		let t = e.target.files?.[0];
		if (e.target.value = "", !t) return;
		let n = t.type.startsWith("image/");
		w({
			name: t.name,
			isImage: n,
			uploading: !0
		});
		try {
			if (n) {
				let e = await Yt(t);
				w({
					name: t.name,
					isImage: n,
					dataUrl: e
				});
			} else {
				let e = await se(t);
				w({
					name: t.name,
					isImage: n,
					dataUrl: null,
					path: e.path
				});
			}
		} catch (e) {
			alert(`Could not attach file: ${e?.message ?? e}`), w(null);
		}
	}, []), F = i(async () => {
		let n = h.trim(), i = C;
		if (!n && !i || x || i?.uploading) return;
		if (g(""), n === "/clear") {
			a([]);
			return;
		}
		if (n.startsWith("/system ")) {
			v(n.slice(8).trim());
			return;
		}
		if (n.startsWith("/tokens ")) {
			let e = parseInt(n.slice(8).trim(), 10);
			isNaN(e) || b(e);
			return;
		}
		let o = {
			role: "user",
			content: n
		};
		i && (o.attachment = i);
		let s = [...r, o];
		a(s), w(null), S(!0), E(null);
		let c = s.filter((e) => !e.error).map(({ role: e, content: t, attachment: n }) => {
			let r = {
				role: e,
				content: t
			};
			return n?.path && (r.file = n.path), r;
		});
		_ && c.unshift({
			role: "system",
			content: _
		});
		let l = crypto?.randomUUID?.() || `${Date.now()}-${Math.random().toString(16).slice(2)}`;
		j.current = l;
		let u = {
			model_key: e,
			messages: c,
			prompt: n,
			request_id: l
		};
		y && (u.max_new_tokens = y), i?.path && (u.file = i.path), i?.isImage && i?.dataUrl && (u.images = [i.dataUrl]);
		let d = t?.name ?? e;
		a((e) => [...e, {
			role: "assistant",
			content: "",
			model: d
		}]);
		let f = new AbortController();
		A.current = f;
		try {
			let e = await U("/api/chat/stream", {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify(u),
				signal: f.signal
			});
			if (!e.ok) throw Error(`${e.status}: ${await e.text()}`);
			let t = e.body.getReader(), n = new TextDecoder(), r = "";
			for (;;) {
				let { done: e, value: i } = await t.read();
				if (e) break;
				r += n.decode(i, { stream: !0 });
				let o = r.split("\n");
				r = o.pop() ?? "";
				for (let e of o) {
					if (!e.startsWith("data: ")) continue;
					let t = e.slice(6).trim();
					if (!(!t || t === "[DONE]")) try {
						let e = JSON.parse(t);
						if (e.type === "request") e.request_id && (j.current = e.request_id);
						else if (e.type === "status" && e.served_by != null) E({
							servedBy: e.served_by,
							workerId: e.worker_id || "",
							workerName: e.worker_name || (e.served_by === "local" ? "local" : "")
						});
						else if (e.type === "status") {
							let t = e.progress == null ? "" : ` ${Math.round(e.progress * 100)}%`;
							a((n) => {
								let r = [...n], i = r[r.length - 1];
								return i?.role === "assistant" && (r[r.length - 1] = {
									...i,
									status: `${e.message || e.stage || "working"}${t}`
								}), r;
							});
						} else e.type === "token" ? a((t) => {
							let n = [...t], r = n[n.length - 1];
							return r?.role === "assistant" && (n[n.length - 1] = {
								...r,
								content: r.content + e.text,
								status: null
							}), n;
						}) : e.type === "error" && a((t) => {
							let n = [...t], r = n[n.length - 1];
							return r?.role === "assistant" && (n[n.length - 1] = {
								...r,
								content: `[Error: ${e.message}]`,
								error: !0
							}), n;
						});
					} catch {}
				}
			}
		} catch (e) {
			e.name === "AbortError" ? a((e) => {
				let t = [...e], n = t[t.length - 1];
				return n?.role === "assistant" && (t[t.length - 1] = {
					...n,
					status: null,
					content: n.content + " ⏹",
					stopped: !0
				}), t;
			}) : a((t) => {
				let n = [...t], r = n[n.length - 1];
				return r?.role === "assistant" && r.content === "" ? n[n.length - 1] = {
					role: "assistant",
					content: `[Error: ${e.message}]`,
					error: !0
				} : n.push({
					role: "assistant",
					content: `[Error: ${e.message}]`,
					error: !0
				}), n;
			});
		} finally {
			S(!1), A.current = null, j.current = null;
		}
	}, [
		h,
		C,
		r,
		e,
		_,
		y,
		x
	]), I = i(() => {
		let e = j.current;
		e && U(`/api/llm/chat/cancel/${encodeURIComponent(e)}`, { method: "POST" }).catch(() => {}), A.current?.abort();
	}, []), L = i((e) => {
		e.key === "Enter" && !e.shiftKey && (e.preventDefault(), F());
	}, [F]);
	return /* @__PURE__ */ f("aside", {
		className: "chat-panel",
		children: [
			/* @__PURE__ */ f("div", {
				className: "chat-header",
				children: [/* @__PURE__ */ f("div", {
					className: "chat-title",
					children: [/* @__PURE__ */ d("span", {
						className: "chat-model",
						children: t?.name ?? e
					}), /* @__PURE__ */ f("span", {
						className: "chat-meta",
						children: [
							t?.framework,
							" · ",
							/* @__PURE__ */ d("span", {
								title: Array.isArray(t?.tasks) && t.tasks.length > 1 ? `all tasks: ${t.tasks.join(", ")}` : void 0,
								children: t?.primary_task ?? t?.task
							}),
							_ && /* @__PURE__ */ d("span", {
								className: "system-set",
								title: _,
								children: " · sys"
							}),
							" · ",
							y ? `max ${y} tok` : "unbounded (auto-continue)",
							M && /* @__PURE__ */ d("span", {
								className: "vl-tag",
								title: "vision-language model",
								children: " · 🖼 VL"
							})
						]
					})]
				}), /* @__PURE__ */ f("div", {
					className: "chat-header-right",
					children: [
						N.length > 1 && /* @__PURE__ */ d("select", {
							className: "chat-switcher",
							value: e,
							onChange: (t) => {
								let n = t.target.value;
								n && n !== e && m?.(n);
							},
							title: "Switch to another model's saved conversation",
							children: N.map((e) => /* @__PURE__ */ f("option", {
								value: e.modelKey,
								children: [
									e.active ? "● " : "",
									e.name,
									" · ",
									e.count,
									" msg",
									e.count === 1 ? "" : "s"
								]
							}, e.modelKey))
						}),
						/* @__PURE__ */ d("span", {
							className: `chat-status ${x ? "is-streaming" : "is-idle"}`,
							title: x ? "Generating a response…" : "Conversation saved locally — survives tab changes and reloads",
							children: x ? "● generating…" : `● saved · ${r.length} msg${r.length === 1 ? "" : "s"}`
						}),
						/* @__PURE__ */ d("button", {
							className: "btn-clear-chat",
							onClick: () => a([]),
							title: "Clear conversation",
							disabled: x,
							children: "✕ clear"
						}),
						/* @__PURE__ */ d("button", {
							className: "btn-close",
							onClick: n,
							title: "Close chat",
							children: "✕"
						})
					]
				})]
			}),
			/* @__PURE__ */ f("div", {
				className: `chat-alloc ${T ? T.servedBy === "local" ? "is-local" : "is-worker" : "is-pending"}`,
				title: "The allocation that served the most recent request",
				children: [/* @__PURE__ */ d("span", {
					className: "alloc-label",
					children: "allocation"
				}), /* @__PURE__ */ d("span", {
					className: "alloc-value",
					children: T ? T.servedBy === "local" ? "local (this node)" : `${T.workerName || T.workerId}${T.workerId && T.workerId !== T.workerName ? ` (${T.workerId})` : ""}` : "—"
				})]
			}),
			/* @__PURE__ */ f("div", {
				className: "chat-messages",
				children: [
					r.length === 0 && /* @__PURE__ */ f("div", {
						className: "chat-empty",
						children: [
							/* @__PURE__ */ f("p", { children: ["Chat with ", /* @__PURE__ */ d("strong", { children: t?.name ?? e })] }),
							/* @__PURE__ */ f("p", {
								className: "chat-hint",
								children: ["Commands: ", qt]
							}),
							/* @__PURE__ */ f("p", {
								className: "chat-hint",
								children: [
									"Attach a file with 📎",
									M ? " (images supported)" : "",
									"."
								]
							})
						]
					}),
					r.map((e, t) => /* @__PURE__ */ f("div", {
						className: `msg msg-${e.role} ${e.error ? "msg-error" : ""}`,
						children: [
							/* @__PURE__ */ d("span", {
								className: "msg-role",
								children: e.role === "assistant" ? e.model ?? "assistant" : e.role
							}),
							e.attachment?.isImage && e.attachment.dataUrl && /* @__PURE__ */ d("img", {
								className: "msg-thumb",
								src: e.attachment.dataUrl,
								alt: e.attachment.name,
								title: e.attachment.name
							}),
							e.attachment && !e.attachment.isImage && /* @__PURE__ */ f("span", {
								className: "msg-file",
								title: e.attachment.name,
								children: ["📎 ", e.attachment.name]
							}),
							e.status && !e.content && /* @__PURE__ */ f("div", {
								className: "msg-status",
								children: ["⏳ ", e.status]
							}),
							/* @__PURE__ */ f("pre", {
								className: "msg-content",
								children: [e.content, e.role === "assistant" && x && t === r.length - 1 && /* @__PURE__ */ d("span", {
									className: "cursor",
									children: "▌"
								})]
							})
						]
					}, t)),
					/* @__PURE__ */ d("div", { ref: D })
				]
			}),
			C && /* @__PURE__ */ f("div", {
				className: "attach-strip",
				children: [
					C.isImage ? /* @__PURE__ */ d("img", {
						src: C.dataUrl,
						alt: C.name,
						className: "attach-thumb"
					}) : /* @__PURE__ */ d("span", {
						className: "attach-file",
						children: "📎"
					}),
					/* @__PURE__ */ f("span", {
						className: "attach-name",
						title: C.name,
						children: [C.name, C.uploading ? " · uploading…" : ""]
					}),
					/* @__PURE__ */ d("button", {
						className: "attach-remove",
						onClick: () => w(null),
						disabled: x,
						title: "Remove attachment",
						children: "×"
					})
				]
			}),
			/* @__PURE__ */ f("div", {
				className: "chat-input-row",
				children: [
					/* @__PURE__ */ d("input", {
						ref: k,
						type: "file",
						style: { display: "none" },
						onChange: P
					}),
					/* @__PURE__ */ d("button", {
						className: "btn-attach",
						onClick: () => k.current?.click(),
						disabled: x,
						title: "Attach file",
						children: "📎"
					}),
					/* @__PURE__ */ d("textarea", {
						ref: O,
						className: "chat-input",
						rows: 3,
						value: h,
						onChange: (e) => g(e.target.value),
						onKeyDown: L,
						placeholder: "Message… (Enter to send, Shift+Enter for newline)",
						disabled: x
					}),
					x ? /* @__PURE__ */ d("button", {
						className: "btn-stop",
						onClick: I,
						title: "Stop generating",
						children: "⏹ Stop"
					}) : /* @__PURE__ */ d("button", {
						className: "btn-send",
						onClick: F,
						disabled: !h.trim() && !C || C?.uploading,
						children: "↑ Send"
					})
				]
			})
		]
	});
}
//#endregion
//#region src/hooks/useSessionState.js
function Y(e, t) {
	let [n, r] = l(() => {
		try {
			let t = sessionStorage.getItem(e);
			if (t != null) return JSON.parse(t);
		} catch {}
		return typeof t == "function" ? t() : t;
	});
	return [n, i((t) => {
		r((n) => {
			let r = typeof t == "function" ? t(n) : t;
			try {
				sessionStorage.setItem(e, JSON.stringify(r));
			} catch {}
			return r;
		});
	}, [e])];
}
//#endregion
//#region src/components/DownloadsQueue/DownloadsQueue.jsx
function Zt(e) {
	if (e == null) return "—";
	let t = [
		"B",
		"KB",
		"MB",
		"GB",
		"TB"
	], n = Number(e), r = 0;
	for (; n >= 1024 && r < t.length - 1;) n /= 1024, r++;
	return `${n.toFixed(+(n < 10 && r > 0))} ${t[r]}`;
}
var Qt = /* @__PURE__ */ new Set(["queued", "running"]), $t = /* @__PURE__ */ new Set([
	"completed",
	"failed",
	"cancelled",
	"expired"
]), en = /* @__PURE__ */ new Set(["failed", "expired"]);
function tn({ variant: e = "tile" }) {
	let [t, n] = l([]), [r, a] = l(!1), s = c(!0), p = c(/* @__PURE__ */ new Set()), m = i(async () => {
		try {
			let e = await K("/api/jobs");
			if (!s.current) return;
			n(Array.isArray(e) ? e : []);
		} catch {}
	}, []);
	o(() => {
		s.current = !0, m();
		let e = setInterval(m, 2500);
		return () => {
			s.current = !1, clearInterval(e);
		};
	}, [m]), o(() => {
		let e = t.find((e) => en.has(e.status) && !e.diagnosis && !p.current.has(e.id));
		e && (p.current.add(e.id), K(`/api/jobs/${e.id}/diagnose`, { method: "POST" }).then(m).catch(() => {}));
	}, [t, m]);
	let h = i((e) => {
		K(`/api/jobs/${e}/cancel`, { method: "POST" }).then(m).catch(() => {});
	}, [m]), g = i((e) => {
		K(`/api/jobs/${e}/retry`, { method: "POST" }).then(m).catch(() => {});
	}, [m]), _ = i((e) => {
		K(`/api/jobs/${e}/discard`, { method: "POST" }).then(m).catch((e) => alert(`Discard refused: ${e.message || e}`));
	}, [m]), v = i((e) => {
		p.current.add(e), K(`/api/jobs/${e}/diagnose`, { method: "POST" }).then(m).catch((e) => alert(`Diagnose failed: ${e.message || e}`));
	}, [m]), y = t.filter((e) => Qt.has(e.status)), b = t.filter((e) => $t.has(e.status)), x = y.length > 0, S = b.filter((e) => en.has(e.status)).length, C = y.length > 0 || b.length > 0, w = /* @__PURE__ */ f(u, { children: [y.length > 0 && /* @__PURE__ */ f(u, { children: [/* @__PURE__ */ f("div", {
		className: "dlq-head",
		children: ["Downloading · ", y.length]
	}), y.map((e) => /* @__PURE__ */ d(nn, {
		job: e,
		onCancel: h
	}, e.id))] }), b.length > 0 && /* @__PURE__ */ f(u, { children: [/* @__PURE__ */ f("div", {
		className: "dlq-head dlq-head-muted",
		children: ["Recent", S > 0 ? ` · ${S} failed (kept until discarded)` : ""]
	}), b.map((e) => /* @__PURE__ */ d(rn, {
		job: e,
		onRetry: g,
		onDiscard: _,
		onDiagnose: v
	}, e.id))] })] });
	return e === "panel" ? /* @__PURE__ */ f("div", {
		className: "dlq-panel",
		children: [!C && /* @__PURE__ */ d("div", {
			className: "dlq-panel-empty",
			children: "No model downloads right now. Every download shows here — including ones started from other sessions or before a reload. Failed or expired downloads stay here, with the reason and a keeper diagnosis, until an admin discards them."
		}), w]
	}) : /* @__PURE__ */ f("div", {
		className: `sb-seg dlq ${x ? "dlq-busy" : ""}`,
		onMouseEnter: () => a(!0),
		onMouseLeave: () => a(!1),
		children: [
			/* @__PURE__ */ d("span", {
				className: "sb-k",
				children: "Downloads"
			}),
			/* @__PURE__ */ d("button", {
				type: "button",
				className: "dlq-toggle",
				onClick: () => a((e) => !e),
				title: x ? `${y.length} model download${y.length > 1 ? "s" : ""} in progress — separate from the inference queue` : "Model downloads — separate from the inference queue",
				children: x ? /* @__PURE__ */ f("span", {
					className: "sb-v dlq-count",
					children: [y.length, " downloading"]
				}) : S > 0 ? /* @__PURE__ */ f("span", {
					className: "sb-v dlq-count",
					children: [S, " failed"]
				}) : /* @__PURE__ */ d("span", {
					className: "sb-v sb-zero",
					children: "idle"
				})
			}),
			r && C && /* @__PURE__ */ d("div", {
				className: "dlq-pop",
				role: "status",
				children: w
			})
		]
	});
}
function nn({ job: e, onCancel: t }) {
	let n = Math.round((e.progress ?? 0) * 100), r = e.status === "running" && !e.total_bytes, i = e.bytes_per_second;
	return /* @__PURE__ */ f("div", {
		className: "dlq-row dlq-row-active",
		children: [
			/* @__PURE__ */ d("span", {
				className: "dlq-name",
				title: e.model_key,
				children: e.model_key
			}),
			/* @__PURE__ */ d("div", {
				className: `dlq-bar ${r ? "dlq-bar-indet" : ""} ${e.stalled ? "dlq-bar-stalled" : ""} dlq-bar-${e.status}`,
				children: /* @__PURE__ */ d("div", {
					className: "dlq-bar-fill",
					style: { width: r ? "40%" : `${n}%` }
				})
			}),
			/* @__PURE__ */ f("span", {
				className: "dlq-label",
				title: e.error || e.message || "",
				children: [
					e.status === "queued" && (e.message || "queued…"),
					e.status === "running" && (e.stalled ? "⚠ stalled — resuming…" : r ? `downloading… ${Zt(e.downloaded_bytes)}` : `${n}% · ${Zt(e.downloaded_bytes)} / ${Zt(e.total_bytes)}`),
					e.status === "running" && e.attempt > 1 && ` · try ${e.attempt}/${e.max_attempts}`,
					e.status === "running" && !e.stalled && i > 0 && ` · ${Zt(i)}/s`
				]
			}),
			/* @__PURE__ */ d("button", {
				className: "dlq-btn dlq-cancel",
				onClick: () => t(e.id),
				title: "Cancel download",
				children: "✕"
			})
		]
	});
}
function rn({ job: e, onRetry: t, onDiscard: n, onDiagnose: r }) {
	let i = en.has(e.status), a = i || e.status === "cancelled";
	return /* @__PURE__ */ f("div", {
		className: `dlq-row dlq-row-done dlq-${e.status}`,
		children: [
			/* @__PURE__ */ f("div", {
				className: "dlq-row-main",
				children: [
					/* @__PURE__ */ d("span", {
						className: "dlq-name",
						title: e.model_key,
						children: e.model_key
					}),
					/* @__PURE__ */ f("span", {
						className: "dlq-label",
						title: e.error || e.message || "",
						children: [
							e.status === "completed" && "✓ installed",
							e.status === "failed" && `✗ failed${e.error_reason ? ` [${e.error_reason}]` : ""}`,
							e.status === "expired" && "✗ expired — never ran",
							e.status === "cancelled" && "cancelled"
						]
					}),
					a && /* @__PURE__ */ d("button", {
						className: "dlq-btn dlq-retry",
						onClick: () => t(e.id),
						title: "Resume from where it stopped",
						children: "↻"
					}),
					i && !e.diagnosis && /* @__PURE__ */ d("button", {
						className: "dlq-btn dlq-diagnose",
						onClick: () => r(e.id),
						title: "Ask the hugpy keeper to diagnose this failure",
						children: "🩺"
					}),
					/* @__PURE__ */ d("button", {
						className: "dlq-btn dlq-clear",
						onClick: () => n(e.id),
						title: i ? "Discard this failure record (admin — removes it for everyone)" : "Discard this row",
						children: "✕"
					})
				]
			}),
			i && (e.error || e.message) && /* @__PURE__ */ f("div", {
				className: "dlq-detail",
				children: [e.error && /* @__PURE__ */ d("div", {
					className: "dlq-detail-error",
					children: e.error
				}), e.message && e.message !== e.error && /* @__PURE__ */ d("div", {
					className: "dlq-detail-msg",
					children: e.message
				})]
			}),
			i && e.diagnosis && /* @__PURE__ */ f("div", {
				className: "dlq-detail dlq-diagnosis",
				title: "Keeper diagnosis — pinned to this record",
				children: ["🩺 ", e.diagnosis]
			})
		]
	});
}
//#endregion
//#region src/components/HFSearch/HFSearch.jsx
var an = [
	["Multimodal", [
		"image-text-to-text",
		"visual-question-answering",
		"document-question-answering",
		"video-text-to-text",
		"image-to-text",
		"any-to-any"
	]],
	["Natural Language Processing", [
		"text-generation",
		"text2text-generation",
		"summarization",
		"translation",
		"text-classification",
		"token-classification",
		"fill-mask",
		"question-answering",
		"table-question-answering",
		"zero-shot-classification",
		"sentence-similarity",
		"feature-extraction"
	]],
	["Computer Vision", [
		"image-classification",
		"object-detection",
		"image-segmentation",
		"depth-estimation",
		"zero-shot-image-classification",
		"zero-shot-object-detection",
		"image-feature-extraction",
		"keypoint-detection",
		"mask-generation",
		"image-to-image",
		"image-to-video",
		"text-to-image",
		"unconditional-image-generation",
		"video-classification",
		"text-to-3d",
		"image-to-3d"
	]],
	["Audio", [
		"automatic-speech-recognition",
		"text-to-speech",
		"text-to-audio",
		"audio-classification",
		"audio-to-audio",
		"voice-activity-detection"
	]],
	["Tabular / Time Series", [
		"tabular-classification",
		"tabular-regression",
		"time-series-forecasting"
	]],
	["Reinforcement Learning & Other", [
		"reinforcement-learning",
		"robotics",
		"graph-ml"
	]]
];
an.flatMap(([, e]) => e);
var on = [
	"transformers",
	"gguf",
	"diffusers",
	"sentence-transformers",
	"timm"
], sn = "text-generation";
function cn(e) {
	return (e.tags || []).map((e) => String(e).toLowerCase()).some((e) => e.includes("gguf")) || e.library_name === "gguf" ? "gguf" : (e.library_name, "transformers");
}
function ln(e) {
	return e.pipeline_tag || "text-generation";
}
function un(e) {
	return e == null ? "–" : e >= 1e6 ? `${(e / 1e6).toFixed(1)}M` : e >= 1e3 ? `${Math.round(e / 1e3)}k` : String(e);
}
function dn(e) {
	if (e == null) return "–";
	let t = [
		"B",
		"KB",
		"MB",
		"GB",
		"TB"
	], n = Number(e), r = 0;
	for (; n >= 1024 && r < t.length - 1;) n /= 1024, r++;
	return `${n.toFixed(1)} ${t[r]}`;
}
function fn(e, t) {
	if (!Number.isFinite(t) || t <= 0) return null;
	let n = (e ?? []).filter((e) => e.fits_disk !== !1 && e.total_bytes != null).sort((e, t) => t.total_bytes - e.total_bytes), r = /* @__PURE__ */ new Set(), i = t * .9;
	for (let e of n) e.total_bytes <= i && (r.add(e.id), i -= e.total_bytes);
	return r;
}
function pn({ job: e, onCancel: t, onRetry: n }) {
	if (!e) return null;
	let r = Math.round((e.progress ?? 0) * 100), i = e.status === "running" && !e.total_bytes, a = e.status === "running" || e.status === "queued", o = e.status === "failed" || e.status === "cancelled" || e.status === "expired", s = e.bytes_per_second;
	return /* @__PURE__ */ f("div", {
		className: "dl-bar-wrap",
		children: [
			/* @__PURE__ */ d("div", {
				className: `dl-bar ${i ? "dl-bar-indet" : ""} ${e.stalled ? "dl-bar-stalled" : ""} dl-bar-${e.status}`,
				children: /* @__PURE__ */ d("div", {
					className: "dl-bar-fill",
					style: { width: i ? "40%" : `${r}%` }
				})
			}),
			/* @__PURE__ */ f("span", {
				className: "dl-bar-label",
				title: e.error || e.message || "",
				children: [
					e.status === "queued" && (e.message || "queued…"),
					e.status === "running" && (e.stalled ? "⚠ stalled — resuming…" : i ? `downloading… ${dn(e.downloaded_bytes)}` : `${r}% · ${dn(e.downloaded_bytes)} / ${dn(e.total_bytes)}`),
					e.status === "running" && e.attempt > 1 && ` · try ${e.attempt}/${e.max_attempts}`,
					e.status === "running" && !e.stalled && s > 0 && ` · ${dn(s)}/s`,
					e.status === "completed" && "✓ installed",
					e.status === "failed" && `✗ ${e.error_reason ? `[${e.error_reason}] ` : ""}${e.error ?? "failed"}`,
					e.status === "expired" && `✗ expired — ${e.message || "never ran"}`,
					e.status === "cancelled" && "cancelled"
				]
			}),
			a && /* @__PURE__ */ d("button", {
				className: "dl-cancel",
				onClick: () => t(e.id),
				title: "Cancel download",
				children: "✕ cancel"
			}),
			o && n && /* @__PURE__ */ d("button", {
				className: "dl-retry",
				onClick: () => n(e.id),
				title: "Resume from where it stopped",
				children: "↻ retry"
			})
		]
	});
}
var mn = /* @__PURE__ */ new Set(["queued", "running"]), hn = /* @__PURE__ */ new Set([
	"completed",
	"failed",
	"cancelled",
	"expired"
]);
function gn({ jobsByHub: e, onCancelJob: t, onRetryJob: n }) {
	let [r, i] = l(!0), a = Object.values(e ?? {});
	if (a.length === 0) return null;
	let o = a.filter((e) => mn.has(e.status)), s = a.filter((e) => hn.has(e.status));
	return /* @__PURE__ */ f("div", {
		className: "hf-downloads",
		children: [/* @__PURE__ */ f("button", {
			type: "button",
			className: "hf-downloads-head",
			onClick: () => i((e) => !e),
			title: r ? "Collapse downloads" : "Show current downloads",
			children: [
				/* @__PURE__ */ d("span", {
					className: "section-caret",
					children: r ? "▾" : "▸"
				}),
				/* @__PURE__ */ d("span", {
					className: "hf-downloads-title",
					children: "Downloads"
				}),
				o.length > 0 ? /* @__PURE__ */ f("span", {
					className: "hf-downloads-count hf-downloads-count-active",
					children: [o.length, " downloading"]
				}) : /* @__PURE__ */ d("span", {
					className: "hf-downloads-count hf-downloads-count-idle",
					children: "idle"
				}),
				s.length > 0 && /* @__PURE__ */ f("span", {
					className: "hf-downloads-count hf-downloads-recent",
					children: [s.length, " recent"]
				})
			]
		}), r && /* @__PURE__ */ d("div", {
			className: "hf-downloads-list",
			children: [...o, ...s].map((e) => /* @__PURE__ */ f("div", {
				className: "hf-downloads-row",
				children: [/* @__PURE__ */ d("a", {
					className: "hf-downloads-name",
					href: `https://huggingface.co/${e.hub_id}`,
					target: "_blank",
					rel: "noreferrer",
					title: e.hub_id || e.model_key,
					children: e.hub_id || e.model_key || "download"
				}), /* @__PURE__ */ d(pn, {
					job: e,
					onCancel: t,
					onRetry: n
				})]
			}, e.id))
		})]
	});
}
function _n(e) {
	return e ? e.queue_error ? e.queue_error : e.queued === !1 ? "The download queue did not accept this job." : null : null;
}
function vn({ r: e, fw: t, tk: n, onDisk: r, externallyQueued: a, job: o, onJobStarted: s, onCancelJob: c, onRetryJob: p }) {
	let [m, h] = l(!1), [g, _] = l(null), [v, y] = l(null), [b, x] = l(() => /* @__PURE__ */ new Set()), [S, C] = l(!1), [w, T] = l(!1), [E, D] = l(null), O = i(async () => {
		let t = !m;
		if (h(t), !(!t || g || S)) {
			C(!0);
			try {
				let t = await K(`/api/hf/spec?hub_id=${encodeURIComponent(e.hub_id)}`);
				_(t);
				let n = t?.options?.recommended ?? null;
				y(n), x(new Set(n == null ? [] : [n]));
			} catch (e) {
				_({ error: e.message });
			} finally {
				C(!1);
			}
		}
	}, [
		m,
		g,
		S,
		e.hub_id
	]), k = i(async () => {
		let t = g?.options;
		if (!t) return;
		let n = t.options.find((e) => e.id === v);
		if (n) {
			T(!0);
			try {
				let r = await K("/api/llm/repos/download", {
					method: "POST",
					headers: { "Content-Type": "application/json" },
					body: JSON.stringify({
						hub_id: e.hub_id,
						framework: n.framework,
						task: t.task,
						filename: n.filename ?? null,
						include: n.include ?? null,
						total_bytes: n.total_bytes ?? null,
						register: !0
					})
				});
				s?.({
					...r,
					hub_id: e.hub_id
				}), D(_n(r));
			} catch (e) {
				D(`Add failed: ${e.message}`);
			} finally {
				T(!1);
			}
		}
	}, [
		g,
		v,
		e.hub_id,
		s
	]), A = i(async () => {
		let t = g?.options;
		if (!t) return;
		let n = t.options.filter((e) => b.has(e.id));
		if (n.length === 0) return;
		T(!0);
		let r = null;
		try {
			for (let i of n) {
				let n = await K("/api/llm/repos/download", {
					method: "POST",
					headers: { "Content-Type": "application/json" },
					body: JSON.stringify({
						hub_id: e.hub_id,
						framework: i.framework,
						task: t.task,
						filename: i.filename ?? null,
						include: i.include ?? null,
						total_bytes: i.total_bytes ?? null,
						register: !0
					})
				});
				s?.({
					...n,
					hub_id: e.hub_id
				}), r ||= _n(n);
			}
			D(r);
		} catch (e) {
			D(`Add failed: ${e.message}`);
		} finally {
			T(!1);
		}
	}, [
		g,
		b,
		e.hub_id,
		s
	]), j = g?.options, M = j?.options?.find((e) => e.id === v), N = !!o, P = N && (o.status === "queued" || o.status === "running"), F = j?.options ?? [], I = g?.free_bytes ?? null, L = F.filter((e) => e.fits_disk !== !1), R = L.length > 0 && L.every((e) => b.has(e.id)), z = F.filter((e) => b.has(e.id)), B = z.length, V = z.reduce((e, t) => e + (t.total_bytes ?? 0), 0), ee = (e) => x((t) => {
		let n = new Set(t);
		return n.has(e) ? n.delete(e) : n.add(e), n;
	});
	return /* @__PURE__ */ f(u, { children: [/* @__PURE__ */ f("tr", {
		className: m ? "hf-row-open" : "",
		children: [
			/* @__PURE__ */ f("td", {
				className: "hf-col-repo",
				children: [
					/* @__PURE__ */ d("button", {
						className: "hf-expand",
						onClick: O,
						title: "Show install options",
						children: m ? "▾" : "▸"
					}),
					/* @__PURE__ */ d("a", {
						href: `https://huggingface.co/${e.hub_id}`,
						target: "_blank",
						rel: "noreferrer",
						title: e.hub_id,
						children: e.hub_id
					}),
					e.trust === "first-party" && /* @__PURE__ */ d("span", {
						className: "hf-trust hf-trust-first",
						title: "Curated trust tier (hugpy, not an official HF metric): canonical first-party publisher — the model's own vendor/org.",
						children: "✓ first-party"
					}),
					e.trust === "community" && /* @__PURE__ */ d("span", {
						className: "hf-trust hf-trust-community",
						title: "Curated trust tier (hugpy, not an official HF metric): reputable community repackager — a trusted re-uploader, not the canonical publisher.",
						children: "community"
					}),
					r === "installed" && /* @__PURE__ */ d("span", {
						className: "hf-ondisk hf-ondisk-yes",
						title: "Already on disk — this repo is installed on this box. Expanding still lets you add another variant / re-fetch.",
						children: "✓ on disk"
					}),
					r === "partial" && /* @__PURE__ */ d("span", {
						className: "hf-ondisk hf-ondisk-partial",
						title: "Partially on disk — some files are present but the install is incomplete.",
						children: "◐ partial"
					}),
					e.private && /* @__PURE__ */ d("span", {
						className: "hf-private",
						title: "private",
						children: " 🔒"
					})
				]
			}),
			/* @__PURE__ */ d("td", {
				className: "hf-col-task",
				children: n
			}),
			/* @__PURE__ */ d("td", {
				className: "hf-col-lib",
				children: /* @__PURE__ */ d("span", {
					className: `hf-fw fw-${t}`,
					children: t
				})
			}),
			/* @__PURE__ */ d("td", {
				className: "hf-col-num",
				title: t === "gguf" ? m && B > 0 ? `${B} quant${B === 1 ? "" : "s"} selected · ${dn(V)}. The whole repo (every quant) is ${dn(e.total_bytes)}.` : "Whole repo — every quantization. Expand to pick which (much smaller) variants to install." : void 0,
				children: t === "gguf" && m && B > 0 ? /* @__PURE__ */ f(u, { children: [dn(V), /* @__PURE__ */ f("em", {
					className: "hf-size-note",
					children: [
						" ",
						B,
						" quant",
						B === 1 ? "" : "s"
					]
				})] }) : /* @__PURE__ */ f(u, { children: [dn(e.total_bytes), t === "gguf" && /* @__PURE__ */ d("em", {
					className: "hf-size-note",
					children: " all quants"
				})] })
			}),
			/* @__PURE__ */ d("td", {
				className: "hf-col-num",
				children: un(e.downloads)
			}),
			/* @__PURE__ */ d("td", {
				className: "hf-col-num",
				children: un(e.likes)
			}),
			/* @__PURE__ */ d("td", {
				className: "hf-col-date",
				children: (e.created_at || e.last_modified || "").slice(0, 10) || "—"
			}),
			/* @__PURE__ */ d("td", {
				className: "hf-col-actions",
				children: /* @__PURE__ */ d("button", {
					className: "btn-pull",
					onClick: O,
					disabled: P || a,
					title: "Choose an install option",
					children: P ? "…" : a ? "queued" : "⬇ Options"
				})
			})
		]
	}), m && /* @__PURE__ */ d("tr", {
		className: "hf-spec-row",
		children: /* @__PURE__ */ f("td", {
			colSpan: 8,
			children: [
				S && /* @__PURE__ */ d("span", {
					className: "hf-spec-loading",
					children: "Loading specs…"
				}),
				g?.error && /* @__PURE__ */ d("span", {
					className: "hf-search-error",
					title: g.error,
					children: g.error
				}),
				j && /* @__PURE__ */ f("div", {
					className: "hf-options",
					children: [
						g.spec && /* @__PURE__ */ f("div", {
							className: "hf-spec-meta",
							children: [
								g.spec.context_length != null && /* @__PURE__ */ f("span", { children: ["ctx ", g.spec.context_length] }),
								g.spec.total_bytes != null && /* @__PURE__ */ d("span", { children: dn(g.spec.total_bytes) }),
								g.spec.license && /* @__PURE__ */ d("span", { children: g.spec.license }),
								g.spec.gated && /* @__PURE__ */ d("span", {
									className: "hf-gated",
									children: "gated"
								})
							]
						}),
						j.options.length === 0 && /* @__PURE__ */ d("div", {
							className: "hf-empty",
							children: "No installable weights found."
						}),
						t === "gguf" ? /* @__PURE__ */ f(u, { children: [
							j.options.length > 0 && /* @__PURE__ */ f("div", {
								className: "hf-opt-toolbar",
								children: [/* @__PURE__ */ f("label", {
									className: "hf-opt hf-opt-all",
									children: [
										/* @__PURE__ */ d("input", {
											type: "checkbox",
											checked: R,
											disabled: P || L.length === 0,
											onChange: () => x(R ? /* @__PURE__ */ new Set() : new Set(L.map((e) => e.id)))
										}),
										"Select all",
										L.length > 0 ? ` (${L.length})` : ""
									]
								}), /* @__PURE__ */ d("button", {
									type: "button",
									className: "hf-autofit",
									disabled: P || I == null || L.length === 0,
									onClick: () => {
										let e = fn(F, I);
										e && x(e);
									},
									title: I == null ? "Free-disk budget unavailable for this box" : `Select the largest set of quants that fits this box's free disk (${dn(I)} free, 10% kept in reserve)`,
									children: "⤓ Auto-fit to space"
								})]
							}),
							j.options.map((e) => /* @__PURE__ */ f("label", {
								className: `hf-opt ${e.fits_disk === !1 ? "hf-opt-toobig" : ""}`,
								children: [
									/* @__PURE__ */ d("input", {
										type: "checkbox",
										checked: b.has(e.id),
										disabled: e.fits_disk === !1 || P,
										onChange: () => ee(e.id)
									}),
									e.label,
									e.fits_disk === !1 && " · won’t fit"
								]
							}, e.id)),
							j.options.length > 0 && /* @__PURE__ */ d("div", {
								className: "hf-opt-summary",
								children: B > 0 ? /* @__PURE__ */ f(u, { children: [
									B,
									" quant",
									B === 1 ? "" : "s",
									" · ",
									dn(V)
								] }) : "nothing selected"
							})
						] }) : j.options.map((t) => /* @__PURE__ */ f("label", {
							className: `hf-opt ${t.fits_disk === !1 ? "hf-opt-toobig" : ""}`,
							children: [
								/* @__PURE__ */ d("input", {
									type: "radio",
									name: `opt-${e.hub_id}`,
									value: t.id,
									checked: v === t.id,
									disabled: t.fits_disk === !1 || P,
									onChange: () => y(t.id)
								}),
								t.label,
								t.fits_disk === !1 && " · won’t fit"
							]
						}, t.id)),
						E && /* @__PURE__ */ f("div", {
							className: "hf-queue-warn",
							role: "alert",
							children: [
								"⚠ ",
								E,
								/* @__PURE__ */ d("button", {
									className: "hf-queue-warn-dismiss",
									title: "Dismiss",
									onClick: () => D(null),
									children: "✕"
								})
							]
						}),
						N ? /* @__PURE__ */ d(pn, {
							job: o,
							onCancel: c,
							onRetry: p
						}) : j.options.length > 0 ? t === "gguf" ? /* @__PURE__ */ d("button", {
							className: "btn-pull",
							onClick: A,
							disabled: B === 0 || w || a,
							title: "Download and register every selected quant variant",
							children: w ? "…" : `⬇ Add ${B} to local`
						}) : /* @__PURE__ */ d("button", {
							className: "btn-pull",
							onClick: k,
							disabled: !M || w || a,
							title: "Download and register this variant",
							children: w ? "…" : "⬇ Add to local"
						}) : null
					]
				})
			]
		})
	})] });
}
var yn = [
	{
		key: "hub_id",
		label: "Repo",
		get: (e) => e.hub_id,
		type: "str"
	},
	{
		key: "task",
		label: "Task",
		get: (e) => ln(e),
		type: "str"
	},
	{
		key: "library",
		label: "Lib",
		get: (e) => cn(e),
		type: "str"
	},
	{
		key: "total_bytes",
		label: "Size",
		get: (e) => e.total_bytes ?? -1,
		type: "num"
	},
	{
		key: "downloads",
		label: "↓",
		get: (e) => e.downloads ?? -1,
		type: "num"
	},
	{
		key: "likes",
		label: "♥",
		get: (e) => e.likes ?? -1,
		type: "num"
	},
	{
		key: "published",
		label: "Published",
		get: (e) => (e.created_at || e.last_modified || "").slice(0, 10),
		type: "str"
	}
], bn = [
	{
		key: "name",
		label: "Model",
		get: (e) => e.name || "",
		type: "str"
	},
	{
		key: "base_model",
		label: "Base",
		get: (e) => e.base_model || "",
		type: "str"
	},
	{
		key: "total_bytes",
		label: "Size",
		get: (e) => e.total_bytes ?? -1,
		type: "num"
	},
	{
		key: "downloads",
		label: "↓",
		get: (e) => e.downloads ?? -1,
		type: "num"
	},
	{
		key: "likes",
		label: "♥",
		get: (e) => e.likes ?? -1,
		type: "num"
	},
	{
		key: "published",
		label: "Published",
		get: (e) => e.published || "",
		type: "str"
	}
];
function xn() {
	let [e, t] = Y("hugpy.sess.cv.query", ""), [n, r] = Y("hugpy.sess.cv.base", "SD 1.5"), [a, p] = Y("hugpy.sess.cv.sort", "Highest Rated"), [m, h] = l([]), [g, _] = l(!1), [v, y] = l(null), [b, x] = l({}), [S, C] = Y("hugpy.sess.cv.sortKey", "downloads"), [w, T] = Y("hugpy.sess.cv.sortDir", "desc"), E = c(0), D = s(() => {
		let e = bn.find((e) => e.key === S);
		if (!e) return m;
		let t = [...m];
		return t.sort((t, n) => {
			let r = e.get(t), i = e.get(n), a = e.type === "num" ? r - i : String(r).localeCompare(String(i));
			return w === "asc" ? a : -a;
		}), t;
	}, [
		m,
		S,
		w
	]), O = i((e) => {
		S === e ? T((e) => e === "asc" ? "desc" : "asc") : (C(e), T("desc"));
	}, [
		S,
		C,
		T
	]);
	o(() => {
		let t = ++E.current;
		_(!0);
		let r = setTimeout(() => {
			let r = new URLSearchParams({
				limit: "25",
				sort: a
			});
			e.trim() && r.set("query", e.trim()), n && r.set("base", n), K(`/api/civitai/search?${r.toString()}`).then((e) => {
				t === E.current && (h(Array.isArray(e) ? e : []), y(null));
			}).catch((e) => {
				t === E.current && (y(e.message), h([]));
			}).finally(() => {
				t === E.current && _(!1);
			});
		}, 400);
		return () => clearTimeout(r);
	}, [
		e,
		n,
		a
	]);
	let k = Object.values(b).some((e) => e.status === "downloading");
	o(() => {
		let e = () => K("/api/civitai/downloads").then(x).catch(() => {});
		if (e(), !k) return;
		let t = setInterval(e, 2e3);
		return () => clearInterval(t);
	}, [k]);
	let A = i(async (e) => {
		try {
			(await K("/api/civitai/download", {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({
					download_url: e.download_url,
					filename: e.filename,
					total_bytes: e.total_bytes,
					civitai_id: e.civitai_id,
					version_id: e.version_id,
					name: e.name,
					base_model: e.base_model
				})
			}))?.already && alert(`${e.filename} is already in /checkpoints`), x((t) => ({
				...t,
				[e.filename]: {
					status: "downloading",
					done_bytes: 0,
					total_bytes: e.total_bytes
				}
			}));
		} catch (e) {
			alert(`Download failed: ${e.message}`);
		}
	}, []);
	return /* @__PURE__ */ f(u, { children: [
		/* @__PURE__ */ f("div", {
			className: "hf-search-bar",
			children: [
				/* @__PURE__ */ d("input", {
					className: "hf-search-input",
					placeholder: "search Civitai checkpoints…",
					value: e,
					onChange: (e) => t(e.target.value)
				}),
				/* @__PURE__ */ f("select", {
					value: n,
					onChange: (e) => r(e.target.value),
					title: "Base model — SD 1.5 / SDXL are what the vanilla comfy template serves",
					children: [
						/* @__PURE__ */ d("option", {
							value: "",
							children: "Any base"
						}),
						/* @__PURE__ */ d("option", { children: "SD 1.5" }),
						/* @__PURE__ */ d("option", { children: "SDXL 1.0" }),
						/* @__PURE__ */ d("option", { children: "SD 2.1" })
					]
				}),
				/* @__PURE__ */ f("select", {
					value: a,
					onChange: (e) => p(e.target.value),
					children: [
						/* @__PURE__ */ d("option", { children: "Highest Rated" }),
						/* @__PURE__ */ d("option", { children: "Most Downloaded" }),
						/* @__PURE__ */ d("option", { children: "Newest" })
					]
				}),
				g && /* @__PURE__ */ d("span", {
					className: "hf-search-spinner",
					"aria-label": "searching"
				})
			]
		}),
		v && /* @__PURE__ */ d("div", {
			className: "hf-search-error",
			children: v
		}),
		m.length > 0 && /* @__PURE__ */ d("div", {
			className: "hf-results",
			children: /* @__PURE__ */ f("table", {
				className: "hf-results-table",
				children: [/* @__PURE__ */ d("thead", { children: /* @__PURE__ */ f("tr", { children: [bn.map((e) => /* @__PURE__ */ f("th", {
					className: "hf-sortable",
					onClick: () => O(e.key),
					children: [e.label, S === e.key && /* @__PURE__ */ d("span", {
						className: "hf-sort-arrow",
						children: w === "asc" ? " ▲" : " ▼"
					})]
				}, e.key)), /* @__PURE__ */ d("th", {})] }) }), /* @__PURE__ */ d("tbody", { children: D.map((e) => {
					let t = b[e.filename], n = t?.total_bytes ? Math.min(99, Math.round(100 * t.done_bytes / t.total_bytes)) : null;
					return /* @__PURE__ */ f("tr", { children: [
						/* @__PURE__ */ f("td", {
							className: "hf-col-repo",
							children: [/* @__PURE__ */ d("a", {
								href: e.page_url,
								target: "_blank",
								rel: "noreferrer",
								title: e.filename,
								children: e.name
							}), !e.comfy_ready && /* @__PURE__ */ d("span", {
								title: "base model outside the vanilla template's family (SD1.x/2.x/SDXL) — may not load",
								children: " ⚠"
							})]
						}),
						/* @__PURE__ */ d("td", {
							className: "hf-col-task",
							children: e.base_model
						}),
						/* @__PURE__ */ d("td", {
							className: "hf-col-num",
							children: dn(e.total_bytes)
						}),
						/* @__PURE__ */ d("td", {
							className: "hf-col-num",
							children: un(e.downloads)
						}),
						/* @__PURE__ */ d("td", {
							className: "hf-col-num",
							children: un(e.likes)
						}),
						/* @__PURE__ */ d("td", {
							className: "hf-col-date",
							children: e.published || "—"
						}),
						/* @__PURE__ */ d("td", {
							className: "hf-col-actions",
							children: t?.status === "done" ? /* @__PURE__ */ d("span", {
								title: "in /checkpoints — self-registered as a comfy model",
								children: "✓ registered"
							}) : t?.status === "failed" ? /* @__PURE__ */ d("span", {
								className: "hf-search-error",
								title: t.error,
								children: "failed"
							}) : t?.status === "downloading" ? /* @__PURE__ */ f("span", { children: ["⏳ ", n == null ? "…" : `${n}%`] }) : /* @__PURE__ */ d("button", {
								className: "btn-pull",
								onClick: () => A(e),
								title: "Stream into /checkpoints — registers as a comfy model automatically",
								children: "⬇ → comfy"
							})
						})
					] }, e.civitai_id);
				}) })]
			})
		})
	] });
}
function Sn({ models: e = [], onJobStarted: t, onCancelJob: n, onRetryJob: r, pendingByHub: a, jobsByHub: p, expanded: m, onToggleExpanded: h, embedded: g = !1 }) {
	let [_, v] = Y("hugpy.sess.hf.query", ""), [y, b] = Y("hugpy.sess.hf.task", sn), [x, S] = Y("hugpy.sess.hf.lib", ""), [C, w] = l([]), [T, E] = l(!1), [D, O] = l(null), [k, A] = Y("hugpy.sess.hf.sortKey.v2", "relevance"), [j, M] = Y("hugpy.sess.hf.sortDir.v2", "desc"), [N, P] = Y("hugpy.sess.hf.source", "hf"), F = c(0);
	o(() => {
		if (N !== "hf") return;
		let e = _.trim();
		if (!e && !y && !x) {
			w([]), O(null);
			return;
		}
		let t = ++F.current;
		E(!0);
		let n = setTimeout(() => {
			let n = new URLSearchParams({ limit: "40" });
			e && n.set("q", e), y && n.set("task", y), x && n.set("library", x), K(`/api/search?${n.toString()}`).then((e) => {
				t === F.current && (w(Array.isArray(e) ? e : []), O(null));
			}).catch((e) => {
				t === F.current && (O(e.message), w([]));
			}).finally(() => {
				t === F.current && E(!1);
			});
		}, 400);
		return () => clearTimeout(n);
	}, [
		N,
		_,
		y,
		x
	]);
	let I = s(() => {
		let e = yn.find((e) => e.key === k);
		if (!e) return C;
		let t = [...C];
		return t.sort((t, n) => {
			let r = e.get(t), i = e.get(n), a = e.type === "num" ? r - i : String(r).localeCompare(String(i));
			return j === "asc" ? a : -a;
		}), t;
	}, [
		C,
		k,
		j
	]), L = i((e) => {
		k === e ? M((e) => e === "asc" ? "desc" : "asc") : (A(e), M("desc"));
	}, [k]), R = s(() => {
		let t = /* @__PURE__ */ new Map();
		for (let n of e || []) {
			let e = (n.hub_id || "").toLowerCase();
			e && (n.status === "installed" ? t.set(e, "installed") : n.status === "partial" && !t.has(e) && t.set(e, "partial"));
		}
		return t;
	}, [e]), z = Object.values(p ?? {}).filter((e) => e.status === "queued" || e.status === "running").length, B = g || m;
	return /* @__PURE__ */ f("section", {
		className: `hf-search ${B ? "hf-search-expanded" : ""}`,
		children: [!g && /* @__PURE__ */ f("div", {
			className: "section-strip clickable",
			onClick: h,
			title: m ? "Collapse" : "Search the Hugging Face Hub and add models",
			children: [
				/* @__PURE__ */ d("span", {
					className: "section-caret",
					children: m ? "▾" : "▸"
				}),
				/* @__PURE__ */ f("span", {
					className: "section-title",
					children: ["Add models · ", N === "civitai" ? "Civitai" : N === "downloads" ? "Download queue" : "Hugging Face Hub"]
				}),
				!m && /* @__PURE__ */ d("span", {
					className: "hf-strip-hint",
					children: "search & download new models"
				}),
				z > 0 && /* @__PURE__ */ f("span", {
					className: "section-count hf-strip-active",
					children: [z, " downloading"]
				}),
				m && I.length > 0 && /* @__PURE__ */ f("span", {
					className: "section-count",
					children: [I.length, " results"]
				}),
				D && /* @__PURE__ */ d("span", {
					className: "hf-search-error",
					title: D,
					children: "!"
				})
			]
		}), B && /* @__PURE__ */ f(u, { children: [
			/* @__PURE__ */ f("div", {
				className: "hf-source-toggle",
				role: "tablist",
				"aria-label": "Model source",
				children: [
					/* @__PURE__ */ d("button", {
						type: "button",
						role: "tab",
						"aria-selected": N !== "civitai" && N !== "downloads",
						className: `hf-source-btn${N !== "civitai" && N !== "downloads" ? " hf-source-active" : ""}`,
						onClick: () => P("hf"),
						children: "🤗 Hugging Face"
					}),
					/* @__PURE__ */ d("button", {
						type: "button",
						role: "tab",
						"aria-selected": N === "civitai",
						className: `hf-source-btn${N === "civitai" ? " hf-source-active" : ""}`,
						onClick: () => P("civitai"),
						title: "SD-checkpoint habitat — downloads land in /checkpoints and self-register as comfy models",
						children: "🧩 Civitai"
					}),
					/* @__PURE__ */ d("button", {
						type: "button",
						role: "tab",
						"aria-selected": N === "downloads",
						className: `hf-source-btn${N === "downloads" ? " hf-source-active" : ""}`,
						onClick: () => P("downloads"),
						title: "Every model download on this box — queued, running and recent, from any session — with cancel/retry",
						children: "⬇ Download queue"
					})
				]
			}),
			N !== "downloads" && /* @__PURE__ */ d(gn, {
				jobsByHub: p,
				onCancelJob: n,
				onRetryJob: r
			}),
			N === "downloads" && /* @__PURE__ */ d(tn, { variant: "panel" }),
			N === "civitai" && /* @__PURE__ */ d(xn, {}),
			N !== "civitai" && N !== "downloads" && /* @__PURE__ */ f(u, { children: [
				/* @__PURE__ */ f("div", {
					className: "hf-search-bar",
					children: [
						/* @__PURE__ */ d("input", {
							className: "hf-search-input",
							placeholder: "filter by name (optional)…",
							value: _,
							onChange: (e) => v(e.target.value)
						}),
						/* @__PURE__ */ f("select", {
							value: y,
							onChange: (e) => b(e.target.value),
							children: [/* @__PURE__ */ d("option", {
								value: "",
								children: "Any task"
							}), an.map(([e, t]) => /* @__PURE__ */ d("optgroup", {
								label: e,
								children: t.map((e) => /* @__PURE__ */ d("option", {
									value: e,
									children: e
								}, e))
							}, e))]
						}),
						/* @__PURE__ */ f("select", {
							value: x,
							onChange: (e) => S(e.target.value),
							children: [/* @__PURE__ */ d("option", {
								value: "",
								children: "Any library"
							}), on.map((e) => /* @__PURE__ */ d("option", {
								value: e,
								children: e
							}, e))]
						}),
						T && /* @__PURE__ */ d("span", {
							className: "hf-search-loading",
							children: "…"
						})
					]
				}),
				I.length > 0 && /* @__PURE__ */ f(u, { children: [/* @__PURE__ */ f("div", {
					className: "hf-relevance-bar",
					children: [/* @__PURE__ */ d("span", {
						className: "hf-relevance-label",
						children: "Sort"
					}), k === "relevance" ? /* @__PURE__ */ d("span", {
						className: "hf-relevance-pill hf-relevance-active",
						title: "Results are in the server's relevance ranking (closest name + uploader trust). Click any column to re-sort client-side.",
						children: "✓ Relevance"
					}) : /* @__PURE__ */ d("button", {
						type: "button",
						className: "hf-relevance-pill hf-relevance-reset",
						onClick: () => {
							A("relevance"), M("desc");
						},
						title: "Back to the server's relevance ranking (closest name + uploader trust)",
						children: "↺ Relevance"
					})]
				}), /* @__PURE__ */ d("div", {
					className: "hf-results",
					children: /* @__PURE__ */ f("table", {
						className: "hf-results-table",
						children: [/* @__PURE__ */ d("thead", { children: /* @__PURE__ */ f("tr", { children: [yn.map((e) => /* @__PURE__ */ f("th", {
							className: "hf-sortable",
							onClick: () => L(e.key),
							children: [e.label, k === e.key && /* @__PURE__ */ d("span", {
								className: "hf-sort-arrow",
								children: j === "asc" ? " ▲" : " ▼"
							})]
						}, e.key)), /* @__PURE__ */ d("th", {})] }) }), /* @__PURE__ */ d("tbody", { children: I.map((e) => /* @__PURE__ */ d(vn, {
							r: e,
							fw: cn(e),
							tk: ln(e),
							onDisk: R.get((e.hub_id || "").toLowerCase()),
							externallyQueued: a?.[e.hub_id],
							job: p?.[e.hub_id],
							onJobStarted: t,
							onCancelJob: n,
							onRetryJob: r
						}, e.hub_id)) })]
					})
				})] }),
				!T && !D && I.length === 0 && /* @__PURE__ */ d("div", {
					className: "hf-empty",
					children: "No matches."
				})
			] })
		] })]
	});
}
//#endregion
//#region src/components/ModelTable/ServingControl.jsx
var Cn = [
	["off", "off (in-process)"],
	["systemd", "systemd (always-on)"],
	["swap", "swap (on-demand)"]
];
function wn({ modelKey: e, framework: t }) {
	let [n, r] = l(null), [a, s] = l(null), [c, u] = l(!1), [p, m] = l(""), h = i(async () => {
		m("");
		try {
			let t = await U(`/api/llm/serving/${encodeURIComponent(e)}`), n = await t.json();
			if (!t.ok) throw Error(n.error || `HTTP ${t.status}`);
			r(n), s({
				serve_mode: n.mode ?? "off",
				n_gpu_layers: n.n_gpu_layers ?? "",
				threads: n.threads ?? "",
				llama_ctx: n.ctx_size ?? ""
			});
		} catch (e) {
			m(String(e.message || e));
		}
	}, [e]);
	o(() => {
		h();
	}, [h]);
	let g = async (t) => {
		u(!0), m(t ? "applying…" : "saving…");
		try {
			let n = await U(`/api/llm/serving/${encodeURIComponent(e)}`, {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({
					...a,
					apply: t
				})
			}), i = await n.json();
			if (!n.ok) throw Error(i.error || `HTTP ${n.status}`);
			if (r(i), t) {
				let e = i.apply || {};
				m(e.applied ? "✓ applied (unit written + restarted)" : `saved — ${e.reason || "not applied"}`);
			} else m("✓ saved (apply to (re)write the unit)");
		} catch (e) {
			m(`✗ ${e.message || e}`);
		} finally {
			u(!1);
		}
	};
	if (t !== "gguf" && t !== "llama_cpp") return /* @__PURE__ */ d("div", {
		className: "mt-serve mt-serve-na",
		children: "Serving control applies to GGUF (llama.cpp) models; this one runs in-process."
	});
	if (!a) return /* @__PURE__ */ d("div", {
		className: "mt-serve",
		children: p || "loading serving…"
	});
	let _ = (e) => (t) => s((n) => ({
		...n,
		[e]: t.target.value
	}));
	return /* @__PURE__ */ f("div", {
		className: "mt-serve",
		children: [/* @__PURE__ */ f("div", {
			className: "mt-serve-fields",
			children: [
				/* @__PURE__ */ f("label", {
					className: "mt-serve-field",
					children: [/* @__PURE__ */ d("span", { children: "Mode" }), /* @__PURE__ */ d("select", {
						value: a.serve_mode,
						onChange: _("serve_mode"),
						children: Cn.map(([e, t]) => /* @__PURE__ */ d("option", {
							value: e,
							children: t
						}, e))
					})]
				}),
				/* @__PURE__ */ f("label", {
					className: "mt-serve-field",
					children: [/* @__PURE__ */ d("span", { children: "GPU layers" }), /* @__PURE__ */ d("input", {
						type: "number",
						value: a.n_gpu_layers,
						onChange: _("n_gpu_layers"),
						placeholder: "-1 = all"
					})]
				}),
				/* @__PURE__ */ f("label", {
					className: "mt-serve-field",
					children: [/* @__PURE__ */ d("span", { children: "CPU threads" }), /* @__PURE__ */ d("input", {
						type: "number",
						value: a.threads,
						onChange: _("threads")
					})]
				}),
				/* @__PURE__ */ f("label", {
					className: "mt-serve-field",
					children: [/* @__PURE__ */ d("span", { children: "Context" }), /* @__PURE__ */ d("input", {
						type: "number",
						value: a.llama_ctx,
						onChange: _("llama_ctx")
					})]
				})
			]
		}), /* @__PURE__ */ f("div", {
			className: "mt-serve-actions",
			children: [
				/* @__PURE__ */ d("button", {
					disabled: c,
					onClick: () => g(!1),
					children: "Save"
				}),
				/* @__PURE__ */ d("button", {
					disabled: c,
					onClick: () => g(!0),
					children: "Save & apply"
				}),
				n?.endpoint && /* @__PURE__ */ f("span", {
					className: "mt-serve-ep",
					children: ["→ ", n.endpoint]
				}),
				!n?.endpoint && /* @__PURE__ */ d("span", {
					className: "mt-serve-ep",
					children: "→ in-process"
				}),
				p && /* @__PURE__ */ d("span", {
					className: "mt-serve-msg",
					children: p
				})
			]
		})]
	});
}
//#endregion
//#region src/components/ModelTable/QuantControl.jsx
function Tn(e) {
	if (e == null || !isFinite(e)) return "";
	let t = [
		"B",
		"KB",
		"MB",
		"GB",
		"TB"
	], n = 0, r = Number(e);
	for (; r >= 1024 && n < t.length - 1;) r /= 1024, n++;
	return `${r >= 100 || n === 0 ? Math.round(r) : r.toFixed(1)} ${t[n]}`;
}
function En({ modelKey: e, framework: t, onChanged: n }) {
	let [r, a] = l([]), [s, c] = l(""), [u, p] = l(null), [m, h] = l(null), [g, _] = l("off"), [v, y] = l(!1), [b, x] = l(""), [S, C] = l(!1), w = i((e) => {
		a(Array.isArray(e.available_gguf_detail) ? e.available_gguf_detail : []), c(e.gguf_file || ""), p(e.effective_gguf ?? null), h(e.effective_bytes ?? null), _(e.mode ?? "off");
	}, []), T = i(async () => {
		try {
			let t = await U(`/api/llm/serving/${encodeURIComponent(e)}`), n = await t.json();
			if (!t.ok) throw Error(n.error || `HTTP ${t.status}`);
			w(n);
		} catch (e) {
			x(String(e.message || e));
		} finally {
			C(!0);
		}
	}, [e, w]);
	if (o(() => {
		T();
	}, [T]), t !== "gguf" && t !== "llama_cpp") return null;
	let E = async (t, r) => {
		y(!0), x(r ? "reloading…" : "saving…");
		try {
			let i = await U(`/api/llm/serving/${encodeURIComponent(e)}`, {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({
					gguf_file: t,
					apply: r
				})
			}), a = await i.json();
			if (!i.ok) throw Error(a.error || `HTTP ${i.status}`);
			if (w(a), r) {
				let e = a.apply || {};
				x(e.applied ? "✓ reloaded with this quant" : `saved — ${e.reason || "not applied"}`);
			} else x("✓ saved");
			n?.();
		} catch (e) {
			x(`✗ ${e.message || e}`);
		} finally {
			y(!1);
		}
	}, D = g === "systemd" || g === "swap";
	return /* @__PURE__ */ f("div", {
		className: "mt-quant",
		children: [/* @__PURE__ */ d("div", {
			className: "mt-serve-title",
			children: "Quantization (GGUF variant)"
		}), S ? r.length === 0 ? /* @__PURE__ */ d("div", {
			className: "mt-quant-none",
			children: "No .gguf variants downloaded for this model yet."
		}) : /* @__PURE__ */ f("div", {
			className: "mt-quant-row",
			children: [
				/* @__PURE__ */ f("select", {
					value: s,
					disabled: v,
					title: "Which downloaded quantization this model serves — global for the model. 'auto' lets the resolver pick (q4_k_m first).",
					onChange: (e) => E(e.target.value, !1),
					children: [/* @__PURE__ */ f("option", {
						value: "",
						children: ["auto", u ? ` → ${u}` : ""]
					}), r.map((e) => /* @__PURE__ */ f("option", {
						value: e.filename,
						children: [
							e.filename,
							e.bytes ? ` · ${Tn(e.bytes)}` : "",
							e.is_effective ? " ✓ effective" : ""
						]
					}, e.filename))]
				}),
				m != null && /* @__PURE__ */ f("span", {
					className: "mt-quant-eff",
					title: "Size the model actually serves — this quant plus its mmproj projector (vision).",
					children: ["serves ", Tn(m)]
				}),
				D && /* @__PURE__ */ d("button", {
					className: "mt-quant-apply",
					disabled: v,
					title: "Reload the running unit now with the selected quant (systemd / swap serving modes).",
					onClick: () => E(s, !0),
					children: "⟳ reload now"
				}),
				b && /* @__PURE__ */ d("span", {
					className: "mt-serve-msg",
					children: b
				})
			]
		}) : /* @__PURE__ */ d("div", {
			className: "mt-quant-none",
			children: "loading quants…"
		})]
	});
}
//#endregion
//#region src/components/ModelTable/PlacementControl.jsx
function Dn({ modelKey: e, workers: t = [] }) {
	let [n, r] = l(null), [a, s] = l(!1), [c, u] = l({}), [p, m] = l({
		prefs: [],
		polite: !1,
		byWorker: {}
	}), [h, g] = l(!1), [_, v] = l(""), y = (e) => {
		let t = Array.isArray(e.worker_prefs) ? e.worker_prefs : [], n = e.no_evict_by_worker && typeof e.no_evict_by_worker == "object" ? { ...e.no_evict_by_worker } : {};
		r(t), s(!!e.no_evict), u(n), m({
			prefs: t,
			polite: !!e.no_evict,
			byWorker: n
		});
	}, b = i(async () => {
		v("");
		try {
			let t = await U(`/api/llm/serving/${encodeURIComponent(e)}`), n = await t.json();
			if (!t.ok) throw Error(n.error || `HTTP ${t.status}`);
			y(n.override || {});
		} catch (e) {
			r([]), v(String(e.message || e));
		}
	}, [e]);
	o(() => {
		b();
	}, [b]);
	let x = (e) => t.find((t) => t.name === e || t.id === e), S = async () => {
		g(!0), v("saving…");
		try {
			for (let t of n) {
				let n = x(t);
				!n || (n.models || []).includes(e) || await U(`/api/llm/workers/${encodeURIComponent(n.id)}/assign`, {
					method: "POST",
					headers: { "Content-Type": "application/json" },
					body: JSON.stringify({ model_key: e })
				});
			}
			let t = await U(`/api/llm/serving/${encodeURIComponent(e)}`, {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({
					worker_prefs: n,
					no_evict: a,
					no_evict_by_worker: c
				})
			}), r = await t.json();
			if (!t.ok) throw Error(r.error || `HTTP ${t.status}`);
			y(r.override || {}), v("✓ saved");
		} catch (e) {
			v(`✗ ${e.message || e}`);
		} finally {
			g(!1);
		}
	};
	if (n === null) return /* @__PURE__ */ d("div", {
		className: "mt-place",
		children: "loading placement…"
	});
	let C = (e, t) => {
		let i = n.slice(), a = e + t;
		a < 0 || a >= i.length || ([i[e], i[a]] = [i[a], i[e]], r(i));
	}, w = (e) => r(n.filter((t, n) => n !== e)), T = (e) => {
		e && !n.includes(e) && r([...n, e]);
	}, E = t.filter((e) => !n.includes(e.name) && !n.includes(e.id)), D = JSON.stringify(n) !== JSON.stringify(p.prefs) || a !== p.polite || JSON.stringify(c) !== JSON.stringify(p.byWorker), O = (e) => Object.keys(c).find((t) => t.toLowerCase() === String(e).toLowerCase()), k = (e) => {
		let t = O(e);
		return t === void 0 ? null : !!c[t];
	}, A = (e) => {
		let t = k(e);
		return t === null ? a : t;
	}, j = (e) => {
		let t = k(e), n = O(e) || e, r = { ...c };
		t === null ? r[n] = !0 : t === !0 ? r[n] = !1 : delete r[n], u(r);
	}, M = [], N = /* @__PURE__ */ new Set(), P = (e) => {
		let t = String(e).toLowerCase();
		!e || N.has(t) || (N.add(t), M.push(e));
	};
	n.forEach(P), t.filter((t) => (t.models || []).includes(e)).filter((e) => !N.has(String(e.id || "").toLowerCase())).forEach((e) => P(e.name || e.id)), Object.keys(c).forEach((e) => {
		let t = x(e);
		t && (N.has(String(t.id || "").toLowerCase()) || N.has(String(t.name || "").toLowerCase())) || P(e);
	});
	let F = null, I = 0;
	for (let n of t) {
		let t = (n.model_call_stats || {})[e], r = t && t.last_call;
		r && r > I && (I = r, F = n.name || n.id);
	}
	let L = t.map((t) => [t, (t.load_reports || {})[e] || {}]).filter(([, e]) => e && e.ok === !1 && e.error);
	return /* @__PURE__ */ f("div", {
		className: "mt-place",
		children: [
			/* @__PURE__ */ f("div", {
				className: "mt-place-prefs",
				children: [n.length === 0 && /* @__PURE__ */ d("span", {
					className: "mt-place-none",
					children: "No preference — routing ranks by designation, residency then capability."
				}), n.map((e, t) => {
					let r = x(e), i = r?.gpus?.[0]?.memory_free;
					return /* @__PURE__ */ f("span", {
						className: "mt-place-chip",
						title: r ? `${e} — ${r.status || "unknown"}` : `${e} is not a registered worker right now; it stays on the list and is skipped until it comes back`,
						children: [
							/* @__PURE__ */ f("span", {
								className: "mt-place-rank",
								children: [t + 1, "."]
							}),
							/* @__PURE__ */ d("span", {
								className: r && r.status === "online" ? "" : "mt-place-off",
								children: e
							}),
							i != null && /* @__PURE__ */ f("span", {
								className: "mt-place-free",
								children: [(i / 2 ** 30).toFixed(1), " GiB free"]
							}),
							/* @__PURE__ */ d("button", {
								disabled: t === 0,
								title: "Try this worker earlier",
								onClick: () => C(t, -1),
								children: "↑"
							}),
							/* @__PURE__ */ d("button", {
								disabled: t === n.length - 1,
								title: "Try this worker later",
								onClick: () => C(t, 1),
								children: "↓"
							}),
							/* @__PURE__ */ d("button", {
								title: "Remove from the preference list",
								onClick: () => w(t),
								children: "✕"
							})
						]
					}, e);
				})]
			}),
			M.length > 0 && /* @__PURE__ */ f("table", {
				className: "mt-place-grid",
				children: [/* @__PURE__ */ d("thead", { children: /* @__PURE__ */ f("tr", { children: [/* @__PURE__ */ d("th", { children: "worker" }), /* @__PURE__ */ d("th", { children: "polite" })] }) }), /* @__PURE__ */ d("tbody", { children: M.map((e) => {
					let t = x(e), n = k(e), r = A(e);
					return /* @__PURE__ */ f("tr", { children: [/* @__PURE__ */ d("td", {
						className: t && t.status === "online" ? "" : "mt-place-off",
						title: t ? `${e} — ${t.status || "unknown"}` : `${e} is not a registered worker right now`,
						children: e
					}), /* @__PURE__ */ d("td", { children: /* @__PURE__ */ f("button", {
						className: `mt-place-tick${r ? " on" : ""}${n === null ? " inherited" : ""}`,
						onClick: () => j(e),
						title: n === null ? `follows the all-workers default (${a ? "yes" : "no"}) — click to pin this worker` : `pinned ${r ? "yes" : "no"} for this worker — click to ${r ? "pin no" : "follow the all-workers default"}`,
						children: [r ? "yes" : "no", n === null ? " *" : ""]
					}) })] }, e);
				}) })]
			}),
			M.some((e) => k(e) === null) && /* @__PURE__ */ d("div", {
				className: "mt-place-note",
				children: "* follows the all-workers default below"
			}),
			/* @__PURE__ */ f("div", {
				className: "mt-place-actions",
				children: [
					/* @__PURE__ */ f("select", {
						value: "",
						disabled: !E.length,
						title: E.length ? "Append a worker to the preference list" : "Every known worker is already on the list",
						onChange: (e) => {
							T(e.target.value), e.target.value = "";
						},
						children: [/* @__PURE__ */ d("option", {
							value: "",
							children: "＋ add worker…"
						}), E.map((e) => /* @__PURE__ */ f("option", {
							value: e.name,
							children: [e.name, e.status === "online" ? "" : " (offline)"]
						}, e.id))]
					}),
					/* @__PURE__ */ f("label", {
						className: "mt-place-toggle",
						title: "Polite load, ALL WORKERS: on every worker without its own verdict above, this model may take only genuinely free VRAM and will NEVER evict a resident to make space. If no candidate has room, the load is refused honestly instead of displacing someone.",
						children: [/* @__PURE__ */ d("input", {
							type: "checkbox",
							checked: a,
							onChange: (e) => s(e.target.checked)
						}), "all workers: load only into free room (never evict)"]
					}),
					/* @__PURE__ */ d("button", {
						disabled: h || !D,
						onClick: S,
						children: "Save placement"
					}),
					_ && /* @__PURE__ */ d("span", {
						className: "mt-serve-msg",
						children: _
					})
				]
			}),
			F && /* @__PURE__ */ f("div", {
				className: "mt-place-effective",
				children: [
					"last served by ",
					/* @__PURE__ */ d("strong", { children: F }),
					n.length > 0 && !n.includes(F) && " — not on the list (that call predates the current preference)"
				]
			}),
			L.map(([e, t]) => /* @__PURE__ */ f("div", {
				className: "mt-place-refusal",
				title: t.error,
				children: [
					"⚠ ",
					e.name,
					": ",
					String(t.error).slice(0, 220)
				]
			}, e.id))
		]
	});
}
//#endregion
//#region src/components/ModelTable/useModelGroups.js
var On = [
	"quality",
	"speed",
	"priority"
], kn = {
	quality: "Rule out degraded variants: the 4-bit class and below. Sets the floor of any ladder walk.",
	speed: "Rule out ram-only placement and spill: the chosen member must be fully GPU-resident.",
	priority: "May EVICT other residents to meet the ticked standards. Without it, quality and speed soften to preferences. (Confers no protection on the group's own residents.)"
}, An = "Operator credential required — the server refused this write.", jn = "Model groups are OFF. An operator enables them before ticks take effect; verdicts below are advisory.";
function Mn() {
	let [e, t] = l({
		enabled: !1,
		source: "default",
		groups: [],
		loading: !0,
		error: null
	}), [n, r] = l(null), a = c(!0);
	o(() => () => {
		a.current = !1;
	}, []);
	let u = i(async () => {
		try {
			let e = await K("/api/llm/groups");
			if (!a.current) return;
			t({
				enabled: !!e?.enabled,
				source: e?.source || "default",
				groups: Array.isArray(e?.groups) ? e.groups : [],
				loading: !1,
				error: e?.error || null
			});
		} catch (e) {
			if (!a.current) return;
			t((t) => ({
				...t,
				loading: !1,
				error: e?.message || String(e)
			}));
		}
	}, []);
	o(() => {
		u();
	}, [u]);
	let d = i(async (n, i, o) => {
		if (!On.includes(i)) return !1;
		let s = e.groups;
		r(null), t((e) => ({
			...e,
			groups: e.groups.map((e) => e.group_key === n ? {
				...e,
				ticks: {
					...e.ticks || {},
					[i]: o
				}
			} : e)
		}));
		try {
			let e = await U(`/api/settings/model_groups/${encodeURIComponent(n)}`, {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({ merge: { ticks: { [i]: o } } })
			});
			if (e.status === 401 || e.status === 403) throw Error(An);
			if (!e.ok) throw Error(`${e.status} ${e.statusText}`);
			return await u(), !0;
		} catch (e) {
			return a.current ? (t((e) => ({
				...e,
				groups: s
			})), r(e?.message || String(e)), !1) : !1;
		}
	}, [e.groups, u]), f = s(() => {
		let t = /* @__PURE__ */ new Map();
		for (let n of e.groups || []) for (let e of n.members || []) t.set(e.model_key, {
			group: n,
			member: e
		});
		return t;
	}, [e.groups]), p = s(() => (e.groups || []).filter((e) => (e.members || []).length > 1), [e.groups]);
	return {
		...e,
		byModel: f,
		multiMember: p,
		setTick: d,
		notice: n,
		clearNotice: () => r(null),
		offHint: e.enabled ? null : jn,
		refresh: u
	};
}
function Nn(e, t) {
	let n = [];
	for (let [r, i] of Object.entries(e?.verdicts || {})) {
		if (i?.preferred === t) {
			n.push({
				worker: r,
				tone: "ok",
				text: i.as ? `serves on ${r} as ${i.as}` : `serves on ${r}`,
				title: i.why || ""
			});
			continue;
		}
		let e = (i?.excluded || []).find((e) => e?.model_key === t);
		e && n.push({
			worker: r,
			tone: "muted",
			text: `${r}: ${e.reason}`,
			title: e.reason || ""
		});
	}
	return n;
}
//#endregion
//#region src/components/ModelTable/GroupHeaderRow.jsx
function Pn({ groupKey: e, tick: t, on: n, disabled: r, hint: i, onFlip: a }) {
	let o = [kn[t], i].filter(Boolean).join("\n\n");
	return /* @__PURE__ */ f("label", {
		className: `mt-tick${n ? " mt-tick-on" : ""}` + (r ? " mt-tick-disabled" : ""),
		title: o,
		children: [/* @__PURE__ */ d("input", {
			type: "checkbox",
			checked: !!n,
			disabled: !!r,
			"aria-label": `${t} tick for ${e}`,
			onChange: (e) => a(t, e.target.checked)
		}), /* @__PURE__ */ d("span", { children: t })]
	});
}
function Fn({ group: e, colSpan: t, enabled: n, offHint: r, onFlip: i, collapsed: a, onToggleCollapse: o }) {
	let s = e.ticks || {}, c = (e.members || []).length, l = On.some((e) => s[e]);
	return /* @__PURE__ */ d("tr", {
		className: "mt-group-row",
		children: /* @__PURE__ */ d("td", {
			colSpan: t,
			children: /* @__PURE__ */ f("div", {
				className: "mt-group-head",
				children: [
					/* @__PURE__ */ d("button", {
						type: "button",
						className: "mt-group-toggle",
						"aria-expanded": !a,
						title: a ? "Show members" : "Hide members",
						onClick: o,
						children: a ? "▸" : "▾"
					}),
					/* @__PURE__ */ d("span", {
						className: "mt-group-key",
						title: e.derived ? "Auto-derived from the members’ base name" : "Membership set by an operator override",
						children: e.group_key
					}),
					/* @__PURE__ */ f("span", {
						className: "mt-group-count",
						children: [
							c,
							" iteration",
							c === 1 ? "" : "s"
						]
					}),
					!e.derived && /* @__PURE__ */ d("span", {
						className: "badge badge-blue",
						title: "Operator override",
						children: "override"
					}),
					/* @__PURE__ */ d("span", {
						className: "mt-group-ticks",
						children: On.map((t) => /* @__PURE__ */ d(Pn, {
							groupKey: e.group_key,
							tick: t,
							on: s[t],
							disabled: !n,
							hint: n ? null : r,
							onFlip: i
						}, t))
					}),
					!n && l && /* @__PURE__ */ d("span", {
						className: "mt-group-inert",
						title: r,
						children: "ticks stored · inert while groups are off"
					})
				]
			})
		})
	});
}
function In({ group: e, modelKey: t }) {
	let n = Nn(e, t);
	return n.length ? /* @__PURE__ */ d("span", {
		className: "mt-verdicts",
		children: n.map((e) => /* @__PURE__ */ d("span", {
			className: `mt-worker-fit ${e.tone === "ok" ? "mt-fit-ok" : "mt-fit-partial"}`,
			title: e.title,
			children: e.text
		}, `${e.worker}:${e.tone}`))
	}) : null;
}
//#endregion
//#region src/components/ModelTable/ModelTable.jsx
function Ln(e) {
	return e == null ? "?" : (e = Number(e), e >= 1e6 ? `${(e / 1e6).toFixed(1)}M` : e >= 1e3 ? `${Math.round(e / 1e3)}k` : String(e));
}
function Rn(e) {
	if (e == null) return "–";
	let t = [
		"B",
		"KB",
		"MB",
		"GB",
		"TB"
	], n = Number(e), r = 0;
	for (; n >= 1024 && r < t.length - 1;) n /= 1024, r++;
	return `${n.toFixed(1)} ${t[r]}`;
}
var zn = /* @__PURE__ */ "iq1_s.iq1_m.iq2_xxs.iq2_xs.iq2_s.iq2_m.iq3_xxs.iq3_xs.iq3_s.iq3_m.iq4_xs.iq4_nl.q2_k.q3_k_l.q3_k_m.q3_k_s.q4_k_m.q4_k_s.q5_k_m.q5_k_s.q6_k.q8_0.q4_0.q4_1.q5_0.q5_1.bf16.f16.f32".split(".");
function Bn(e) {
	if (!e) return "";
	let t = String(e).toLowerCase();
	for (let e of zn) if (t.includes(e)) return e.toUpperCase();
	return String(e).replace(/\.gguf$/i, "");
}
function Vn(e) {
	return [
		e.model_key,
		e.key,
		e.name,
		e.hub_id,
		e.framework,
		Wn(e),
		e.status,
		...Array.isArray(e.tasks) ? e.tasks : [],
		...Array.isArray(e.tags) ? e.tags : []
	].filter(Boolean).join(" ").toLowerCase();
}
function Hn(e) {
	return e.framework ?? "";
}
function Un(e) {
	return e.status ?? "missing";
}
function Wn(e) {
	return e.primary_task ?? e.task ?? e.pipeline_tag ?? e.pipelineTag ?? e.tags?.find?.((e) => [
		"text-generation",
		"image-text-to-text",
		"automatic-speech-recognition",
		"feature-extraction",
		"summarization",
		"text-classification",
		"token-classification",
		"fill-mask",
		"zero-shot-classification",
		"image-classification",
		"object-detection"
	].includes(e)) ?? e.tasks?.[0] ?? "unknown";
}
function Gn(e) {
	return [.../* @__PURE__ */ new Set([...Array.isArray(e.tasks) ? e.tasks : [], Wn(e)])].filter(Boolean);
}
function Kn({ status: e }) {
	return e === "installed" ? /* @__PURE__ */ d("span", {
		className: "badge badge-green",
		children: "✓ ready"
	}) : e === "partial" ? /* @__PURE__ */ d("span", {
		className: "badge badge-yellow",
		children: "◐ partial"
	}) : /* @__PURE__ */ d("span", {
		className: "badge badge-red",
		children: "✗ missing"
	});
}
function qn({ job: e, onCancel: t, onRetry: n }) {
	let r = Math.round((e.progress ?? 0) * 100), i = e.status === "running" && !e.total_bytes, a = e.status === "running" || e.status === "queued", o = e.status === "failed" || e.status === "cancelled" || e.status === "expired", s = e.bytes_per_second;
	return /* @__PURE__ */ f("div", {
		className: "mt-dl",
		children: [
			/* @__PURE__ */ d("div", {
				className: `mt-dl-bar ${i ? "mt-dl-indet" : ""} ${e.stalled ? "mt-dl-stalled" : ""} mt-dl-${e.status}`,
				children: /* @__PURE__ */ d("div", {
					className: "mt-dl-fill",
					style: { width: i ? "40%" : `${r}%` }
				})
			}),
			/* @__PURE__ */ f("span", {
				className: "mt-dl-label",
				title: e.error || e.message || "",
				children: [
					e.status === "queued" && (e.message || "queued…"),
					e.status === "running" && (e.stalled ? "⚠ stalled — resuming…" : i ? `downloading… ${Rn(e.downloaded_bytes)}` : `${r}% · ${Rn(e.downloaded_bytes)} / ${Rn(e.total_bytes)}`),
					e.status === "running" && e.attempt > 1 && ` · try ${e.attempt}/${e.max_attempts}`,
					e.status === "running" && !e.stalled && s > 0 && ` · ${Rn(s)}/s`,
					e.status === "failed" && `✗ ${e.error_reason ? `[${e.error_reason}] ` : ""}${e.error ?? "failed"}`,
					e.status === "expired" && `✗ expired — ${e.message || "never ran"}`,
					e.status === "cancelled" && "cancelled"
				]
			}),
			a && /* @__PURE__ */ d("button", {
				className: "mt-dl-cancel",
				onClick: () => t(e.id),
				title: "Cancel download",
				children: "✕"
			}),
			o && n && /* @__PURE__ */ d("button", {
				className: "mt-dl-retry",
				onClick: () => n(e.id),
				title: "Resume from where it stopped",
				children: "↻"
			})
		]
	});
}
function Jn(e) {
	let t = String(e || "").trim();
	if (!t) return [];
	let n = /* @__PURE__ */ new Set([t, t.toLowerCase()]), r = t.split("/").pop();
	if (n.add(r), n.add(r.toLowerCase()), t.includes("~")) {
		let e = t.split("~").slice(1).join("~");
		e && (n.add(e), n.add(e.toLowerCase()));
	}
	return [...n];
}
var Yn = [
	{
		key: "index",
		label: "#",
		type: "num",
		get: (e, t) => t + 1
	},
	{
		key: "name",
		label: "Name",
		type: "str",
		get: (e) => {
			let t = e.model_key ?? e.key ?? "", n = e.name ?? t;
			return t.includes("~") ? `${n} (${t.split("~")[0]})` : n;
		}
	},
	{
		key: "media",
		label: "Media",
		type: "num",
		get: (e) => +!!e.media
	},
	{
		key: "framework",
		label: "Framework",
		type: "str",
		get: (e) => Hn(e)
	},
	{
		key: "task",
		label: "Task",
		type: "str",
		get: (e) => Wn(e)
	},
	{
		key: "ctx",
		label: "Ctx",
		type: "num",
		get: (e) => Number(e.model_max_length ?? -1)
	},
	{
		key: "status",
		label: "Status",
		type: "str",
		get: (e) => Un(e)
	},
	{
		key: "pgroup",
		label: "Group",
		type: "str",
		get: (e) => e.__pgroup ?? ""
	}
];
function Xn({ models: e, jobsByModel: t, activeChat: r, onDownload: a, onChat: u, onDelete: p, onPrune: m, onSetMedia: h, onSetMediaDefault: g, onCancel: _, onRetry: v, workers: y = [], onAssignWorker: b, onProbeWorker: x, onRefresh: S }) {
	let [C, w] = l({}), [T, E] = Y("hugpy.sess.mt.query", ""), [D, O] = Y("hugpy.sess.mt.fw", ""), [k, A] = Y("hugpy.sess.mt.task", ""), [j, M] = Y("hugpy.sess.mt.status", "installed"), [N, P] = Y("hugpy.sess.mt.assigned", !1), [F, I] = Y("hugpy.sess.mt.sortKey", "name"), [L, R] = Y("hugpy.sess.mt.sortDir", "asc"), [z, B] = l(null), V = Mn(), [ee, te] = Y("hugpy.sess.mt.groupBy", !0), [H, ne] = Y("hugpy.sess.mt.gcollapsed", {}), re = i((e) => {
		ne((t) => ({
			...t,
			[e]: !t?.[e]
		}));
	}, [ne]), [U, W] = l({}), G = c(/* @__PURE__ */ new Set()), ie = i((e, t = []) => {
		let n = [["", e], ...t.map((t) => [t, `${e}@${t}`])];
		for (let [t, r] of n) {
			if (G.current.has(r)) continue;
			G.current.add(r);
			let n = t ? `?worker=${encodeURIComponent(t)}` : "";
			K(`/api/models/${encodeURIComponent(e)}/meta${n}`).then((e) => W((t) => ({
				...t,
				[r]: e
			}))).catch(() => W((e) => ({
				...e,
				[r]: null
			})));
		}
	}, []), ae = /* @__PURE__ */ new Set(["text-generation", "image-text-to-text"]), oe = (e) => ae.has(Wn(e)), [se, ce] = l({
		running: !1,
		error: null
	}), le = c(null);
	o(() => () => clearTimeout(le.current), []);
	let ue = i(function e() {
		K("/api/models/discover").then((t) => {
			if (t.running) {
				le.current = setTimeout(e, 2500);
				return;
			}
			ce({
				running: !1,
				error: t.error || null
			}), t.error || S?.();
		}).catch((e) => ce({
			running: !1,
			error: String(e.message || e)
		}));
	}, [S]), de = i(() => {
		ce({
			running: !0,
			error: null
		}), K("/api/models/discover", { method: "POST" }).then(() => {
			le.current = setTimeout(ue, 2500);
		}).catch(() => {
			le.current = setTimeout(ue, 1e3);
		});
	}, [ue]), fe = s(() => [...new Set(e.map(Hn).filter(Boolean))].sort(), [e]), pe = s(() => [...new Set(e.flatMap(Gn))].sort(), [e]), me = s(() => [...new Set(e.map(Un).filter(Boolean))].sort(), [e]), he = s(() => new Set(y.flatMap((e) => e.models ?? [])), [y]), [ge, _e] = l([]), ve = i(() => {
		K("/api/llm/model-groups").then((e) => _e(e.groups || [])).catch(() => {});
	}, []);
	o(() => {
		ve();
	}, [ve]);
	let ye = s(() => {
		let e = /* @__PURE__ */ new Map();
		for (let t of ge) if (t.enabled) for (let n of t.members || []) for (let r of Jn(n)) e.has(r) || e.set(r, t);
		return e;
	}, [ge]), be = i((e) => {
		for (let t of Jn(e.model_key ?? e.key ?? "")) {
			let e = ye.get(t);
			if (e) return e;
		}
		return null;
	}, [ye]), xe = i((e, t) => {
		K("/api/llm/model-groups/member", {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({
				model_key: e,
				group_id: t || null
			})
		}).then(ve).catch((e) => window.alert(String(e.message || e)));
	}, [ve]), [Se, Ce] = l(null), we = i((e, t) => {
		let n = t ? { workers: [t] } : {};
		K(`/api/llm/model-groups/${encodeURIComponent(e.id)}/allocate`, {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify(n)
		}).then((t) => {
			let n = (t.outcomes || []).filter((e) => e.status !== "designated" && e.status !== "already");
			Ce({
				id: e.id,
				text: `${t.designated} designated` + (n.length ? `, ${n.length} skipped` : "")
			}), S?.();
		}).catch((e) => window.alert(String(e.message || e)));
	}, [S]), q = s(() => {
		let t = T.trim().toLowerCase(), n = e.filter((e) => !(t && !Vn(e).includes(t) || D && Hn(e) !== D || k && !Gn(e).includes(k) || j && Un(e) !== j || N && !he.has(e.model_key ?? e.key))).map((e) => ({
			...e,
			__pgroup: be(e)?.name || ""
		})), r = Yn.find((e) => e.key === F);
		return r ? [...n].sort((e, t) => {
			let n = r.get(e, 0), i = r.get(t, 0), a;
			return a = r.type === "num" ? Number(n) - Number(i) : String(n).localeCompare(String(i)), L === "asc" ? a : -a;
		}) : n;
	}, [
		e,
		T,
		D,
		k,
		j,
		N,
		he,
		F,
		L,
		be
	]), Te = s(() => {
		let e = q.map((e) => ({
			kind: "model",
			model: e
		})), t = new Set(q.map((e) => e.model_key ?? e.key)), n = new Map(q.map((e) => [e.model_key ?? e.key, e])), r = /* @__PURE__ */ new Set(), i = [], a = /* @__PURE__ */ new Map();
		for (let e of t) for (let t of Jn(e)) a.has(t) || a.set(t, e);
		for (let e of ge) {
			if (!e.enabled) continue;
			let t = [];
			for (let n of e.members || []) for (let e of Jn(n)) {
				let n = a.get(e);
				if (n && !r.has(n) && !t.includes(n)) {
					t.push(n);
					break;
				}
			}
			if (t.length) {
				if (i.push({
					kind: "pgroup",
					pgroup: e,
					count: t.length
				}), !H?.[`pg:${e.id}`]) for (let r of t) i.push({
					kind: "model",
					model: n.get(r),
					pgroup: e
				});
				t.forEach((e) => r.add(e));
			}
		}
		if (ee && V.multiMember.length) for (let e of V.multiMember) {
			let a = (e.members || []).map((e) => e.model_key).filter((e) => t.has(e) && !r.has(e));
			if (!(a.length < 2)) {
				if (i.push({
					kind: "group",
					group: e
				}), !H?.[e.group_key]) for (let t of a) i.push({
					kind: "model",
					model: n.get(t),
					group: e
				});
				a.forEach((e) => r.add(e));
			}
		}
		if (!i.length) return e;
		for (let t of e) r.has(t.model.model_key ?? t.model.key) || i.push(t);
		return i;
	}, [
		q,
		ee,
		V.multiMember,
		H,
		ge
	]), Ee = i((e) => {
		F === e ? R((e) => e === "asc" ? "desc" : "asc") : (I(e), R(e === "ctx" ? "desc" : "asc"));
	}, [F]), De = i(() => {
		E(""), O(""), A(""), M(""), P(!1);
	}, []);
	if (!e.length) return /* @__PURE__ */ d("div", {
		className: "empty",
		children: "No models available."
	});
	let Oe = [
		[
			"hub_id",
			"Hub ID",
			(e) => e
		],
		[
			"model_key",
			"Model key",
			(e) => e
		],
		[
			"key",
			"Key (legacy)",
			(e) => e
		],
		[
			"framework",
			"Framework",
			(e) => e
		],
		[
			"primary_task",
			"Primary task",
			(e) => e
		],
		[
			"tasks",
			"Tasks",
			(e) => Array.isArray(e) ? e.join(", ") : e
		],
		[
			"model_max_length",
			"Context",
			(e) => Ln(e)
		],
		[
			"status",
			"Status",
			(e) => e
		],
		[
			"folder",
			"Folder",
			(e) => e
		],
		[
			"filename",
			"Filename",
			(e) => e
		],
		[
			"parameter_count",
			"Parameters",
			(e) => e == null ? null : Number(e).toLocaleString()
		],
		[
			"license",
			"License",
			(e) => e
		],
		[
			"languages",
			"Languages",
			(e) => Array.isArray(e) ? e.join(", ") : e
		]
	], ke = new Set(Oe.map(([e]) => e));
	function Ae({ model: e, colSpan: n }) {
		let r = Oe.map(([t, n, r]) => [n, r(e[t])]).filter(([, e]) => e != null && e !== "" && !(Array.isArray(e) && !e.length)), i = Object.entries(e).filter(([e, t]) => !ke.has(e) && t != null && t !== "" && typeof t != "object"), o = e.model_key ?? e.key ?? e.hub_id ?? e.name, s = t?.[o] ?? t?.[e.key] ?? t?.[e.hub_id], c = e.status === "installed", l = c || e.status === "partial", h = s && (s.status === "queued" || s.status === "running"), g = c ? "↻ Re-download" : e.status === "partial" ? "⬇ Resume download" : "⬇ Download", v = y.filter((e) => e.status === "online");
		return /* @__PURE__ */ d("tr", {
			className: "mt-detail-row",
			children: /* @__PURE__ */ d("td", {
				colSpan: n,
				children: /* @__PURE__ */ f("div", {
					className: "mt-detail",
					children: [
						/* @__PURE__ */ f("div", {
							className: "mt-detail-actions",
							children: [
								/* @__PURE__ */ d("button", {
									className: "mt-act",
									disabled: !c || !oe(e),
									title: c ? oe(e) ? "Open chat" : "Not a chat model" : "Install model first",
									onClick: () => u(o),
									children: "💬 Chat"
								}),
								/* @__PURE__ */ d("button", {
									className: "mt-act",
									disabled: h,
									onClick: () => a(o),
									children: h ? "… downloading" : g
								}),
								h && /* @__PURE__ */ d("button", {
									className: "mt-act mt-act-danger",
									onClick: () => _(s.id),
									children: "✕ Cancel download"
								}),
								/* @__PURE__ */ d("button", {
									className: "mt-act mt-act-danger",
									disabled: !l || h,
									title: l ? "Remove downloaded files from disk" : "Nothing downloaded",
									onClick: () => p(o),
									children: "🗑 Delete files"
								}),
								!l && m && /* @__PURE__ */ d("button", {
									className: "mt-act mt-act-danger",
									disabled: h,
									title: "Remove this not-installed model from the registry (clears the ghost entry)",
									onClick: () => m(o),
									children: "⊘ Prune entry"
								})
							]
						}),
						(() => {
							let e = U[o];
							return e ? /* @__PURE__ */ f("div", {
								className: "mt-meta-strip",
								title: "From GET /models/<key>/meta — the single model-metadata source",
								children: [
									e.size_bytes != null && /* @__PURE__ */ f("span", {
										className: "mt-meta-chip",
										title: e.effective_gguf ? `Effective quant ${e.effective_gguf}${e.mmproj_bytes ? " + mmproj projector" : ""} — the one file that actually serves. Disk holds ${Rn(e.dir_bytes)} across all downloaded variants.` : "On-disk model footprint",
										children: [
											"💾 ",
											Rn(e.size_bytes),
											e.dir_bytes != null && e.dir_bytes > e.size_bytes * 1.05 && /* @__PURE__ */ f("em", {
												className: "mt-meta-sub",
												children: [
													" of ",
													Rn(e.dir_bytes),
													" on disk"
												]
											})
										]
									}),
									e.quant && /* @__PURE__ */ d("span", {
										className: "mt-meta-chip",
										children: e.quant
									}),
									e.params_b != null && /* @__PURE__ */ f("span", {
										className: "mt-meta-chip",
										children: [e.params_b, "B params"]
									}),
									e.ctx_max != null && /* @__PURE__ */ f("span", {
										className: "mt-meta-chip",
										children: ["ctx ", Ln(e.ctx_max)]
									}),
									e.recommended?.threads != null && /* @__PURE__ */ f("span", {
										className: "mt-meta-chip",
										children: ["threads ", e.recommended.threads]
									}),
									e.recommended?.need_bytes != null && /* @__PURE__ */ f("span", {
										className: "mt-meta-chip",
										title: "Estimated VRAM need incl. context/overhead",
										children: [
											"needs ≈",
											Rn(e.recommended.need_bytes),
											" VRAM"
										]
									})
								]
							}) : null;
						})(),
						v.length > 0 && /* @__PURE__ */ f("div", {
							className: "mt-worker-section",
							children: [/* @__PURE__ */ d("div", {
								className: "mt-serve-title",
								children: "🖧 Run on worker"
							}), v.map((t) => {
								let n = (t.models || []).includes(o), r = t.gpus?.[0]?.memory_free, i = e.effective_bytes ?? e.total_bytes, a = r != null && i != null && i > r, s = `${t.id}:${o}`, c = C[s], l = U[`${o}@${t.id}`]?.recommended;
								return /* @__PURE__ */ f("div", {
									className: "mt-worker-row",
									children: [
										/* @__PURE__ */ f("button", {
											className: n ? "mt-worker-on" : "",
											title: n ? "Already assigned — click to keep" : "Assign this model to this worker",
											onClick: () => {
												b?.(t, o);
											},
											children: [
												n ? "✓ " : "+ ",
												t.name,
												/* @__PURE__ */ f("span", {
													className: "mt-worker-vram",
													children: [r == null ? "GPU ?" : `${Rn(r)} free`, a && " ⚠"]
												})
											]
										}),
										l && /* @__PURE__ */ d("span", {
											className: `mt-worker-fit ${l.fits_vram ? "mt-fit-ok" : l.fits_vram === !1 ? "mt-fit-partial" : ""}`,
											title: l.reason || "",
											children: l.fits_vram ? "✓ fits in VRAM" : l.fits_vram === !1 ? `◐ ${l.gpu_fraction == null ? "partial offload" : Math.round(l.gpu_fraction * 100) + "% on GPU"}` : ""
										}),
										/* @__PURE__ */ d("button", {
											className: "mt-worker-probe",
											title: "Load the model on this GPU and report whether it fits",
											disabled: c === "probing",
											onClick: async () => {
												w((e) => ({
													...e,
													[s]: "probing"
												}));
												let e = await x?.(t, o);
												w((t) => ({
													...t,
													[s]: e || { ok: !1 }
												}));
											},
											children: c === "probing" ? "…" : c ? c.fit ? "✓ fits" : c.ok ? "◐ spills" : "✗" : "probe"
										}),
										c && c !== "probing" && /* @__PURE__ */ d("span", {
											className: "mt-worker-probe-detail",
											title: c.error || "",
											children: c.vram_used == null ? c.error ? "error" : "" : `used ${Rn(c.vram_used)}`
										})
									]
								}, t.id);
							})]
						}),
						/* @__PURE__ */ d("div", {
							className: "mt-serve-section mt-quant-block",
							children: /* @__PURE__ */ d(En, {
								modelKey: o,
								framework: e.framework,
								onChanged: S
							})
						}),
						/* @__PURE__ */ f("div", {
							className: "mt-serve-section",
							children: [/* @__PURE__ */ d("div", {
								className: "mt-serve-title",
								children: "🖧 Worker preference & polite load"
							}), /* @__PURE__ */ d(Dn, {
								modelKey: o,
								workers: y
							})]
						}),
						/* @__PURE__ */ f("div", {
							className: "mt-serve-section",
							children: [/* @__PURE__ */ d("div", {
								className: "mt-serve-title",
								children: "Serving (GPU / CPU / context)"
							}), /* @__PURE__ */ d(wn, {
								modelKey: e.model_key ?? e.key ?? e.hub_id,
								framework: e.framework
							})]
						}),
						/* @__PURE__ */ d("div", {
							className: "mt-detail-grid",
							children: r.map(([e, t]) => /* @__PURE__ */ f("div", {
								className: "mt-detail-item",
								children: [/* @__PURE__ */ d("span", {
									className: "mt-detail-label",
									children: e
								}), /* @__PURE__ */ d("span", {
									className: "mt-detail-value",
									children: String(t)
								})]
							}, e))
						}),
						e.tags?.length > 0 && /* @__PURE__ */ d("div", {
							className: "mt-detail-tags",
							children: e.tags.map((e) => /* @__PURE__ */ d("span", {
								className: "mt-detail-tag",
								children: e
							}, e))
						}),
						i.length > 0 && /* @__PURE__ */ f("details", {
							className: "mt-detail-raw",
							children: [/* @__PURE__ */ f("summary", { children: [
								"Other fields (",
								i.length,
								")"
							] }), /* @__PURE__ */ d("div", {
								className: "mt-detail-grid",
								children: i.map(([e, t]) => /* @__PURE__ */ f("div", {
									className: "mt-detail-item",
									children: [/* @__PURE__ */ d("span", {
										className: "mt-detail-label",
										children: e
									}), /* @__PURE__ */ d("span", {
										className: "mt-detail-value",
										children: String(t)
									})]
								}, e))
							})]
						})
					]
				})
			})
		});
	}
	return /* @__PURE__ */ f("div", {
		className: "model-table-panel",
		children: [/* @__PURE__ */ f("div", {
			className: "model-table-toolbar",
			children: [/* @__PURE__ */ f("div", {
				className: "model-table-filters",
				children: [
					/* @__PURE__ */ d("input", {
						className: "model-table-search",
						placeholder: "filter by name, repo, task, framework…",
						value: T,
						onChange: (e) => E(e.target.value)
					}),
					/* @__PURE__ */ d("button", {
						className: "btn-clear-filters",
						type: "button",
						onClick: De,
						title: "Clear the search and all filters",
						children: "clear"
					}),
					/* @__PURE__ */ f("span", {
						className: "model-table-count",
						children: [
							q.length,
							" / ",
							e.length
						]
					}),
					V.multiMember.length > 0 && /* @__PURE__ */ f("label", {
						className: "mt-groupby",
						title: "Pull iterations of the same base model under one group header" + (V.enabled ? "" : `\n\n${V.offHint}`),
						children: [/* @__PURE__ */ d("input", {
							type: "checkbox",
							checked: !!ee,
							onChange: (e) => te(e.target.checked)
						}), /* @__PURE__ */ f("span", { children: ["group", V.enabled ? "" : " (off)"] })]
					}),
					V.notice && /* @__PURE__ */ f("span", {
						className: "mt-group-notice",
						title: V.notice,
						onClick: V.clearNotice,
						role: "alert",
						children: ["🔑 ", V.notice]
					}),
					S && /* @__PURE__ */ d("button", {
						className: "model-table-refresh",
						type: "button",
						onClick: S,
						title: "Reload the model list (only needed for changes made outside this console)",
						"aria-label": "Reload model list",
						children: "↻"
					}),
					/* @__PURE__ */ d("button", {
						className: "model-table-discover",
						type: "button",
						onClick: de,
						disabled: se.running,
						title: "Re-scan model storage on disk and re-register anything missing from this list (can take a few minutes)",
						children: se.running ? "Discovering…" : "Discover models"
					}),
					se.error && /* @__PURE__ */ d("span", {
						className: "model-table-discover-err",
						title: se.error,
						children: "discover failed"
					})
				]
			}), /* @__PURE__ */ f("div", {
				className: "model-table-subfilters",
				children: [
					/* @__PURE__ */ f("select", {
						value: D,
						onChange: (e) => O(e.target.value),
						children: [/* @__PURE__ */ d("option", {
							value: "",
							children: "Any framework"
						}), fe.map((e) => /* @__PURE__ */ d("option", {
							value: e,
							children: e
						}, e))]
					}),
					/* @__PURE__ */ f("select", {
						value: k,
						onChange: (e) => A(e.target.value),
						children: [/* @__PURE__ */ d("option", {
							value: "",
							children: "All tasks"
						}), pe.map((e) => /* @__PURE__ */ d("option", {
							value: e,
							children: e
						}, e))]
					}),
					/* @__PURE__ */ f("select", {
						value: j,
						onChange: (e) => M(e.target.value),
						children: [/* @__PURE__ */ d("option", {
							value: "",
							children: "Any status"
						}), me.map((e) => /* @__PURE__ */ d("option", {
							value: e,
							children: e
						}, e))]
					}),
					/* @__PURE__ */ f("label", {
						className: "model-table-assigned",
						title: "Show only models currently assigned to a worker",
						children: [
							/* @__PURE__ */ d("input", {
								type: "checkbox",
								checked: N,
								onChange: (e) => P(e.target.checked)
							}),
							"assigned",
							he.size ? ` (${he.size})` : ""
						]
					})
				]
			})]
		}), /* @__PURE__ */ d("div", {
			className: "table-wrap",
			children: q.length === 0 ? /* @__PURE__ */ d("div", {
				className: "empty",
				children: "No models match the current filters."
			}) : /* @__PURE__ */ f("table", {
				className: "model-table",
				children: [/* @__PURE__ */ d("thead", { children: /* @__PURE__ */ d("tr", { children: Yn.map((e) => /* @__PURE__ */ f("th", {
					className: "mt-sortable",
					onClick: () => Ee(e.key),
					title: `Sort by ${e.label}`,
					children: [e.label, F === e.key && /* @__PURE__ */ d("span", {
						className: "mt-sort-arrow",
						children: L === "asc" ? " ▲" : " ▼"
					})]
				}, e.key)) }) }), /* @__PURE__ */ d("tbody", { children: Te.map((e, i) => {
					if (e.kind === "pgroup") {
						let t = e.pgroup, n = !!H?.[`pg:${t.id}`];
						return /* @__PURE__ */ d("tr", {
							className: "mt-pgroup-row",
							children: /* @__PURE__ */ d("td", {
								colSpan: Yn.length,
								children: /* @__PURE__ */ f("div", {
									className: "mt-pgroup-bar",
									children: [
										/* @__PURE__ */ d("button", {
											type: "button",
											className: "mt-pgroup-caret",
											"aria-expanded": !n,
											title: n ? "Expand group" : "Collapse group",
											onClick: () => re(`pg:${t.id}`),
											children: n ? "▸" : "▾"
										}),
										/* @__PURE__ */ f("span", {
											className: "mt-pgroup-name",
											title: `priority group ${t.id}`,
											children: ["▣ ", t.name]
										}),
										/* @__PURE__ */ f("span", {
											className: "mt-pgroup-count",
											children: [
												e.count,
												" model",
												e.count === 1 ? "" : "s"
											]
										}),
										(t.workers || []).length > 0 && /* @__PURE__ */ f("span", {
											className: "mt-pgroup-workers",
											title: "The group's worker allocation, in priority order",
											children: ["→ ", t.workers.join(" → ")]
										}),
										/* @__PURE__ */ d("span", { className: "mt-pgroup-spacer" }),
										Se?.id === t.id && /* @__PURE__ */ d("span", {
											className: "mt-pgroup-note",
											children: Se.text
										}),
										/* @__PURE__ */ f("select", {
											className: "mt-pgroup-select",
											value: "",
											title: "Designate every model in this group to one worker",
											onChange: (e) => {
												e.target.value && we(t, e.target.value), e.target.value = "";
											},
											children: [/* @__PURE__ */ d("option", {
												value: "",
												children: "assign group to worker…"
											}), y.map((e) => /* @__PURE__ */ d("option", {
												value: e.name || e.id,
												children: e.name || e.id
											}, e.id))]
										}),
										(t.workers || []).length > 0 && /* @__PURE__ */ d("button", {
											type: "button",
											className: "mt-pgroup-allocate",
											title: `Designate every model to ${t.workers.join(", ")}`,
											onClick: () => we(t, null),
											children: "allocate"
										})
									]
								})
							})
						}, `pgroup:${t.id}`);
					}
					if (e.kind === "group") return /* @__PURE__ */ d(Fn, {
						group: e.group,
						colSpan: Yn.length,
						enabled: V.enabled,
						offHint: V.offHint,
						collapsed: !!H?.[e.group.group_key],
						onToggleCollapse: () => re(e.group.group_key),
						onFlip: (t, n) => V.setTick(e.group.group_key, t, n)
					}, `group:${e.group.group_key}`);
					let a = e.model, o = a.model_key ?? a.key ?? a.hub_id ?? a.name ?? `model-${i}`, s = a.model_key ?? a.key ?? o, c = t?.[s] ?? t?.[o];
					return /* @__PURE__ */ f(n, { children: [/* @__PURE__ */ f("tr", {
						className: `${s === r || o === r ? "row-active" : ""}${z === o ? " row-expanded" : ""}` + (e.group ? " mt-group-member" : ""),
						children: [
							/* @__PURE__ */ d("td", {
								className: "col-num",
								children: /* @__PURE__ */ d("button", {
									type: "button",
									className: "mt-num-toggle",
									"aria-expanded": z === o,
									title: z === o ? "Collapse details" : "Expand details & actions",
									onClick: () => {
										let e = z !== o;
										B(e ? o : null), e && ie(s, y.filter((e) => e.status === "online").map((e) => e.id));
									},
									children: i + 1
								})
							}),
							/* @__PURE__ */ f("td", {
								className: "col-name",
								children: [
									/* @__PURE__ */ d("span", {
										className: `model-name${a.hub_id ? " model-name-link" : ""}`,
										title: a.hub_id ? `${a.hub_id} — double-click to open on Hugging Face` : a.name,
										onDoubleClick: () => {
											a.hub_id && window.open(`https://huggingface.co/${a.hub_id}`, "_blank", "noopener,noreferrer");
										},
										children: a.name ?? a.key
									}),
									/* @__PURE__ */ d("span", {
										className: "hub-id",
										children: a.hub_id
									}),
									e.group && /* @__PURE__ */ d(In, {
										group: e.group,
										modelKey: s
									}),
									a.effective_gguf && /* @__PURE__ */ f("button", {
										type: "button",
										className: "mt-quant-badge",
										title: `Serving quant: ${a.effective_gguf}` + (a.effective_bytes == null ? "" : ` · ${Rn(a.effective_bytes)}`) + ((a.gguf_variants?.length || 0) > 1 ? ` · ${a.gguf_variants.length} variants downloaded — click to choose` : " — click for details"),
										onClick: () => {
											let e = z !== o;
											B(e ? o : null), e && ie(s, y.filter((e) => e.status === "online").map((e) => e.id));
										},
										children: [
											"🧩 ",
											Bn(a.effective_gguf),
											a.effective_bytes == null ? "" : ` · ${Rn(a.effective_bytes)}`,
											(a.gguf_variants?.length || 0) > 1 ? " ▾" : ""
										]
									})
								]
							}),
							/* @__PURE__ */ d("td", {
								className: "col-media",
								children: oe(a) ? /* @__PURE__ */ f("span", {
									className: "media-cell",
									children: [/* @__PURE__ */ d("input", {
										type: "checkbox",
										className: "media-check",
										checked: !!a.media,
										title: "Offer this model in the media-intelligence chat dropdown",
										onChange: (e) => h?.(s, e.target.checked)
									}), a.media && /* @__PURE__ */ d("button", {
										type: "button",
										className: `media-default-btn${a.media_default ? " is-default" : ""}`,
										"aria-pressed": !!a.media_default,
										title: a.media_default ? "Default media model — shown first in the media chat list" : "Set as the default media model (first in the media chat list)",
										onClick: () => g?.(s, !a.media_default),
										children: a.media_default ? "★" : "☆"
									})]
								}) : /* @__PURE__ */ d("span", {
									className: "media-na",
									title: "Chat-capable models only",
									children: "—"
								})
							}),
							/* @__PURE__ */ d("td", { children: /* @__PURE__ */ d("span", {
								className: `fw-tag fw-${a.framework}`,
								children: a.framework
							}) }),
							/* @__PURE__ */ f("td", {
								className: "col-task",
								title: Gn(a).join(", "),
								children: [Wn(a), Gn(a).length > 1 ? ` +${Gn(a).length - 1}` : ""]
							}),
							/* @__PURE__ */ d("td", {
								className: "col-ctx",
								children: Ln(a.model_max_length)
							}),
							/* @__PURE__ */ d("td", {
								className: "col-status",
								children: c && c.status !== "completed" ? /* @__PURE__ */ d(qn, {
									job: c,
									onCancel: _,
									onRetry: v
								}) : /* @__PURE__ */ d(Kn, { status: a.status })
							}),
							/* @__PURE__ */ d("td", {
								className: "col-pgroup",
								children: /* @__PURE__ */ f("select", {
									className: "mt-pgroup-member-select",
									value: be(a)?.id || "",
									title: "Explicit priority group — change to move this model between groups",
									onChange: (e) => xe(s, e.target.value),
									children: [/* @__PURE__ */ d("option", {
										value: "",
										children: "—"
									}), ge.map((e) => /* @__PURE__ */ d("option", {
										value: e.id,
										children: e.name
									}, e.id))]
								})
							})
						]
					}), z === o && /* @__PURE__ */ d(Ae, {
						model: a,
						colSpan: Yn.length
					})] }, o);
				}) })]
			})
		})]
	});
}
//#endregion
//#region src/components/PeersBar/PeersBar.jsx
function Zn(e) {
	if (e == null) return "?";
	let t = [
		"B",
		"KB",
		"MB",
		"GB",
		"TB",
		"PB"
	], n = Number(e), r = 0;
	for (; n >= 1024 && r < t.length - 1;) n /= 1024, r++;
	return `${n.toFixed(1)} ${t[r]}`;
}
function Qn() {
	let [e, t] = l([]), [n, r] = l(null);
	return o(() => {
		let e = !1, n = () => {
			K("/api/llm/peers").then((n) => {
				e || (t(n), r(null));
			}).catch((t) => {
				e || r(t.message);
			});
		};
		n();
		let i = setInterval(n, 15e3);
		return () => {
			e = !0, clearInterval(i);
		};
	}, []), n ? /* @__PURE__ */ d("div", {
		className: "peers-bar peers-error",
		title: n,
		children: "peers offline"
	}) : e.length ? /* @__PURE__ */ f("div", {
		className: "peers-bar",
		children: [/* @__PURE__ */ d("span", {
			className: "section-title",
			children: "Peers"
		}), e.map((e) => {
			let t = e.disk?.free, n = e.disk?.total, r = t != null && n ? Math.round((1 - t / n) * 100) : null, i = e.storage_mounted ? "peer-mount-ok" : "peer-mount-bad";
			return /* @__PURE__ */ f("div", {
				className: `peer peer-${e.status}`,
				title: e.storage_root,
				children: [
					/* @__PURE__ */ d("span", { className: "peer-dot" }),
					/* @__PURE__ */ d("span", {
						className: "peer-name",
						children: e.name
					}),
					/* @__PURE__ */ d("span", {
						className: "peer-role",
						children: e.role
					}),
					/* @__PURE__ */ f("span", {
						className: `peer-mount ${i}`,
						children: [
							e.storage_mounted ? "✓" : "✗",
							" ",
							e.storage_root
						]
					}),
					r != null && /* @__PURE__ */ f("span", {
						className: "peer-disk",
						children: [
							Zn(t),
							" free · ",
							r,
							"% used"
						]
					})
				]
			}, e.name);
		})]
	}) : null;
}
//#endregion
//#region src/components/ModelPicker/ModelPicker.jsx
function $n(e) {
	if (e == null) return "—";
	let t = Number(e);
	return !Number.isFinite(t) || t <= 0 ? "—" : t >= 1024 ? `${Math.round(t / 1024)}k` : String(t);
}
function er(e) {
	if (e == null) return "—";
	let t = Number(e);
	if (!Number.isFinite(t) || t <= 0) return "—";
	let n = [
		"B",
		"KB",
		"MB",
		"GB",
		"TB"
	], r = 0, i = t;
	for (; i >= 1024 && r < n.length - 1;) i /= 1024, r++;
	return `${i >= 100 || r === 0 ? Math.round(i) : i.toFixed(1)} ${n[r]}`;
}
function tr(e) {
	let t = e.size_bytes == null ? e.effective_bytes : e.size_bytes;
	return t == null ? null : Number(t);
}
function nr(e) {
	return e.model_key ?? e.key;
}
function rr({ models: e = [], value: t = "", onPick: n, placeholder: r = "Pick a model…", disabled: i = !1, autoFocus: a = !1 }) {
	let [u, p] = l(!1), [m, h] = l(""), g = c(null), _ = c(null), v = s(() => e.find((e) => nr(e) === t) || null, [e, t]), y = s(() => {
		let t = m.trim().toLowerCase();
		return t ? e.filter((e) => `${e.name || ""} ${nr(e) || ""} ${(e.tasks || []).join(" ")} ${e.framework || ""}`.toLowerCase().includes(t)) : e;
	}, [e, m]);
	o(() => {
		if (!u) return;
		let e = (e) => {
			g.current && !g.current.contains(e.target) && p(!1);
		}, t = (e) => {
			e.key === "Escape" && p(!1);
		};
		return document.addEventListener("mousedown", e), document.addEventListener("keydown", t), () => {
			document.removeEventListener("mousedown", e), document.removeEventListener("keydown", t);
		};
	}, [u]), o(() => {
		u && _.current?.focus();
	}, [u]), o(() => {
		a && !i && p(!0);
	}, [a, i]);
	let b = (e) => {
		n?.(nr(e), e), p(!1), h("");
	};
	return /* @__PURE__ */ f("span", {
		className: "mp-root",
		ref: g,
		children: [/* @__PURE__ */ f("button", {
			type: "button",
			className: `mp-trigger${v ? "" : " mp-empty"}`,
			disabled: i,
			onClick: () => p((e) => !e),
			title: v ? nr(v) : r,
			children: [/* @__PURE__ */ d("span", {
				className: "mp-trigger-label",
				children: v ? v.name || nr(v) : r
			}), /* @__PURE__ */ d("span", {
				className: "mp-caret",
				children: u ? "▴" : "▾"
			})]
		}), u && /* @__PURE__ */ f("div", {
			className: "mp-pop",
			role: "listbox",
			children: [
				/* @__PURE__ */ d("input", {
					ref: _,
					className: "mp-filter",
					placeholder: "filter by name / task / lib…",
					value: m,
					onChange: (e) => h(e.target.value),
					onKeyDown: (e) => {
						e.key === "Enter" && y.length > 0 && b(y[0]);
					}
				}),
				/* @__PURE__ */ f("div", {
					className: "mp-grid mp-head",
					children: [
						/* @__PURE__ */ d("span", { children: "Model" }),
						/* @__PURE__ */ d("span", { children: "Task" }),
						/* @__PURE__ */ d("span", { children: "Lib" }),
						/* @__PURE__ */ d("span", { children: "Ctx" }),
						/* @__PURE__ */ d("span", { children: "Size" }),
						/* @__PURE__ */ d("span", { children: "Status" })
					]
				}),
				/* @__PURE__ */ f("div", {
					className: "mp-rows",
					children: [y.length === 0 && /* @__PURE__ */ d("div", {
						className: "mp-none",
						children: "no models match"
					}), y.map((e) => {
						let n = nr(e), r = e.primary_task || (e.tasks || [])[0] || "—", i = (e.tasks || []).filter((e) => e !== r);
						return /* @__PURE__ */ f("div", {
							className: `mp-grid mp-row${n === t ? " mp-current" : ""}`,
							role: "option",
							"aria-selected": n === t,
							onClick: () => b(e),
							title: n,
							children: [
								/* @__PURE__ */ d("span", {
									className: "mp-name",
									children: e.name || n
								}),
								/* @__PURE__ */ f("span", {
									className: "mp-task",
									title: (e.tasks || []).join(", "),
									children: [r, i.length > 0 && /* @__PURE__ */ f("em", {
										className: "mp-more",
										children: [" +", i.length]
									})]
								}),
								/* @__PURE__ */ d("span", {
									className: `mp-fw mp-fw-${e.framework || "unknown"}`,
									children: e.framework || "—"
								}),
								/* @__PURE__ */ d("span", {
									className: "mp-ctx",
									children: $n(e.model_max_length)
								}),
								(() => {
									let t = tr(e);
									return /* @__PURE__ */ d("span", {
										className: `mp-size${t == null ? " mp-size-none" : ""}`,
										title: t == null ? "size unknown (not on disk)" : e.framework === "gguf" ? `effective quant ${e.effective_gguf || ""} — ${er(t)}` : `${er(t)} on disk`,
										children: er(t)
									});
								})(),
								/* @__PURE__ */ d("span", {
									className: `mp-status mp-status-${e.status || "unknown"}`,
									children: e.status || "—"
								})
							]
						}, n);
					})]
				})
			]
		})]
	});
}
//#endregion
//#region src/components/FixDoc/FixDoc.jsx
var ir = {
	"engine-cpu-only": {
		href: "/docs#troubleshooting/engine-cpu-only",
		hint: "CPU-only engine, VRAM idle — the diagnostic, the two traps, and the CUDA/ROCm rebuild that fixes it"
	},
	"worker-admission": {
		href: "/docs#worker-troubleshooting/admission",
		hint: "Worker stuck pending/blocked — click Admit, or check the enrollment token / unblock it"
	},
	"worker-join": {
		href: "/docs#installation/workers",
		hint: "Add workers — point hugpy worker at a central address the box can actually reach (not localhost)"
	},
	"worker-over-budget": {
		href: "/docs#worker-troubleshooting/over-budget",
		hint: "Worker over its disk-cache budget — approve the eviction proposal, raise the ceiling, or free disk"
	},
	"worker-unreachable": {
		href: "/docs#worker-troubleshooting/unreachable",
		hint: "Worker shows ✗ unreachable — agent running? port 9100 reachable? one agent per port? registered URL dialable?"
	},
	"worker-model-missing": {
		href: "/docs#worker-troubleshooting/missing-files",
		hint: "Model shows ○ missing — normal for an assigned-but-never-called model (lazy download: weights transfer on first call). What to check when it stays missing AFTER a call (disk, mmproj)"
	},
	"registry-error": {
		href: "/docs#worker-troubleshooting/registry-error",
		hint: "Console shows registry error — the panel's poll of central failed; hover the chip, then check central/auth/proxy"
	}
};
function ar({ doc: e }) {
	let t = ir[e];
	return t ? /* @__PURE__ */ d("a", {
		className: "fixdoc",
		href: t.href,
		target: "_blank",
		rel: "noreferrer",
		title: `Docs: ${t.hint}`,
		onClick: (e) => e.stopPropagation(),
		children: "📖"
	}) : null;
}
//#endregion
//#region src/components/WorkersPanel/constants.js
var or = "hugpy.workers.servtable.layout.v2", sr = [
	"name",
	"memory",
	"alloc",
	"size",
	"ctx",
	"task",
	"framework",
	"fourbit",
	"moe",
	"state",
	"seat",
	"residency",
	"pin",
	"actions"
], cr = "name", lr = [
	"name",
	"state",
	"memory"
], ur = 2 ** 30, dr = [
	[
		"gpu-only",
		"🖥 GPU only",
		"all layers on the GPU, no spill — won’t fit the GPU (after evict) → refused"
	],
	[
		"ram-only",
		"🧠 RAM only",
		"all in host RAM, never the GPU (binds CPU even with a GPU present)"
	],
	[
		"max-gpu",
		"⚡ Max GPU",
		"as much GPU as fits, spill the rest to RAM — the DEFAULT (serves-and-spills, never OOMs)"
	],
	[
		"max-ram",
		"💾 Max RAM",
		"as much RAM as fits, spill the rest to the GPU"
	],
	[
		"explicit",
		"🎛 Explicit…",
		"target VRAM/RAM budgets + a leniency %% + a device priority — the only mode with knobs"
	]
], fr = /* @__PURE__ */ new Set([
	"gpu-only",
	"ram-only",
	"max-gpu",
	"max-ram"
]), pr = "explicit is GGUF-only — banded leniency has no transformers analogue";
//#endregion
//#region src/components/WorkersPanel/allocation.js
function mr(e) {
	let t = {
		mode: "auto",
		gib: "",
		ram: "",
		threads: "",
		gpuBand: "",
		ramBand: "",
		ctxPct: "",
		ctxBand: "",
		priority: ""
	};
	if (!e || !Object.keys(e).length) return t;
	let n = e.n_gpu_layers;
	return n === -1 || n === "-1" ? {
		...t,
		mode: "gpu"
	} : n === "off" || n === 0 || n === "0" ? {
		...t,
		mode: "cpu"
	} : {
		mode: "custom",
		gib: e.gpu_mem_gib == null ? "" : String(e.gpu_mem_gib),
		ram: e.cpu_mem_gib == null ? "" : String(e.cpu_mem_gib),
		threads: e.threads == null ? "" : String(e.threads),
		gpuBand: e.gpu_mem_gib_deviation_pct == null ? "" : String(e.gpu_mem_gib_deviation_pct),
		ramBand: e.cpu_mem_gib_deviation_pct == null ? "" : String(e.cpu_mem_gib_deviation_pct),
		ctxPct: e.ctx_pct == null ? "" : String(e.ctx_pct),
		ctxBand: e.ctx_deviation_pct == null ? "" : String(e.ctx_deviation_pct),
		priority: e.priority == null ? "" : String(e.priority)
	};
}
var hr = [
	"gpu_mem_gib",
	"cpu_mem_gib",
	"threads",
	"tensor_split",
	"gpu_mem_gib_deviation_pct",
	"cpu_mem_gib_deviation_pct",
	"ctx_pct",
	"ctx_deviation_pct",
	"priority",
	"leniency_pct",
	"priority_device"
];
function gr(e) {
	return !e || Object.keys(e).length === 0 ? !1 : hr.some((t) => t in e) ? !0 : String(e.alloc_mode || "").trim().toLowerCase() === "explicit";
}
function _r(e) {
	let { mode: t, gib: n, ram: r, threads: i, gpuBand: a, ramBand: o, ctxPct: s, ctxBand: c, priority: l } = mr(e);
	if (t === "gpu") return "GPU only";
	if (t === "cpu") return "RAM only";
	if (t === "custom") {
		let e = [];
		return n && e.push(`${n}G VRAM`), r && e.push(`${r}G RAM`), i && e.push(`${i} cores`), a && e.push(`±${a}%V`), o && e.push(`±${o}%R`), (s || c) && e.push(`ctx${s || "?"}±${c || 0}%`), l && e.push(`p${l}`), e.join(" · ") || "explicit";
	}
	return "max-gpu";
}
function vr(e) {
	let t = e || {}, n = t.alloc_mode == null ? "" : String(t.alloc_mode).trim().toLowerCase();
	if (n === "max-ram" || n === "explicit" || n === "gpu-only" || n === "ram-only" || n === "max-gpu") return n;
	if (n === "autofit") return "max-gpu";
	if (n === "cpu-only" || n === "cpu_only" || n === "ram_only") return "ram-only";
	if (n === "gpu_only") return "gpu-only";
	if (n === "max_ram") return "max-ram";
	if (n === "budget" || n === "bands") return "explicit";
	let r = t.n_gpu_layers;
	if (r != null) {
		let e = String(r).trim().toLowerCase();
		if (e === "-1") return "gpu-only";
		if (e === "0" || e === "off" || e === "cpu" || e === "none") return "ram-only";
	}
	for (let e of [
		"leniency_pct",
		"gpu_mem_gib",
		"cpu_mem_gib",
		"gpu_mem_gib_deviation_pct",
		"cpu_mem_gib_deviation_pct"
	]) if (t[e] != null) return "explicit";
	return "max-gpu";
}
function yr(e) {
	let t = dr.find((t) => t[0] === e);
	return t ? t[1] : e;
}
function br(e, t, n) {
	let r = e == null ? null : Number(e), i = t == null ? null : Number(t);
	if (r != null && Number.isFinite(r)) {
		if (r === 0) return {
			mode: "ram-only",
			split: null
		};
		if (r === -1) return {
			mode: "gpu-only",
			split: null
		};
		if (i != null && Number.isFinite(i) && i > 0) {
			if (r >= i) return {
				mode: "gpu-only",
				split: null
			};
			if (r > 0) return {
				mode: "max-gpu",
				split: `${r}/${i}`
			};
		}
		if (r > 0) return {
			mode: "max-gpu",
			split: null
		};
	}
	if (n != null && Number.isFinite(Number(n))) {
		let e = Number(n);
		return e <= 0 ? {
			mode: "ram-only",
			split: null
		} : e >= 100 ? {
			mode: "gpu-only",
			split: null
		} : {
			mode: "max-gpu",
			split: null
		};
	}
	return {
		mode: null,
		split: null
	};
}
function xr(e) {
	return e == null ? "?" : `${(Number(e) / 2 ** 30).toFixed(1)}GiB`;
}
function Sr(e, t, n) {
	let r = t && t.modelBytes, i = t && t.vramTotal, a = t && t.ramTotal;
	return e === "explicit" && n === !1 ? "explicit is GGUF-only — banded leniency has no transformers analogue" : r == null ? null : (e === "gpu-only" || e === "max-gpu" && n === !1) && i != null && r > .95 * i ? `model ${xr(r)} exceeds GPU ${xr(i)}` : e === "ram-only" && a != null && r > a ? `model ${xr(r)} exceeds RAM ${xr(a)}` : (e === "max-ram" || e === "explicit") && i != null && a != null && r > i + a ? `model ${xr(r)} exceeds GPU+RAM ${xr(i + a)}` : null;
}
//#endregion
//#region src/components/WorkersPanel/formatters.js
function Cr(e, t = 12, n = 14) {
	let r = String(e ?? "");
	return r.length <= t + n + 2 ? r : `${r.slice(0, t)}…${r.slice(-n)}`;
}
function X(e) {
	if (e == null) return "?";
	let t = [
		"B",
		"KiB",
		"MiB",
		"GiB",
		"TiB"
	], n = Number(e), r = 0;
	for (; n >= 1024 && r < t.length - 1;) n /= 1024, r++;
	return `${n.toFixed(1)} ${t[r]}`;
}
function wr(e) {
	if (!e) return "never served";
	let t = Math.max(0, Date.now() / 1e3 - Number(e));
	if (t < 45) return "just now";
	let n = t / 60;
	if (n < 60) return `${Math.round(n)}m ago`;
	let r = n / 60;
	if (r < 24) return `${Math.round(r)}h ago`;
	let i = r / 24;
	return i < 30 ? `${Math.round(i)}d ago` : `${Math.round(i / 30)}mo ago`;
}
//#endregion
//#region src/components/WorkersPanel/GroupAssignPanel.jsx
function Tr({ models: e, workers: t, onGroupAssign: n }) {
	let [r, i] = Y("hugpy.sess.wp.group.open", !1), [a, o] = l(""), [c, p] = l(() => /* @__PURE__ */ new Set()), [m, h] = l(!1), [g, _] = l(null), v = s(() => (t || []).map((e) => {
		let t = a ? (e.models || []).includes(a) : !1, n = e.status === "offline" || e.admission === "blocked" || e.admission === "pending";
		return {
			w: e,
			already: t,
			offline: n,
			eligible: !!a && !t && !n,
			reason: t ? "✓ already here" : n ? `(${e.admission === "blocked" ? "blocked" : e.status === "offline" ? "offline" : "not admitted"})` : ""
		};
	}), [t, a]), y = s(() => v.filter((e) => e.eligible).map((e) => e.w.id), [v]), b = y.length > 0 && y.every((e) => c.has(e)), x = s(() => v.filter((e) => e.eligible && c.has(e.w.id)).map((e) => e.w), [v, c]), S = (e) => p((t) => {
		let n = new Set(t);
		return n.has(e) ? n.delete(e) : n.add(e), n;
	});
	return /* @__PURE__ */ d("div", {
		className: "wp-group",
		children: r ? /* @__PURE__ */ f("div", {
			className: "wp-group-body",
			children: [
				/* @__PURE__ */ f("div", {
					className: "wp-group-head",
					children: [/* @__PURE__ */ d("strong", { children: "Group assign" }), /* @__PURE__ */ d("button", {
						className: "wp-group-x",
						onClick: () => {
							i(!1), o(""), p(/* @__PURE__ */ new Set()), _(null);
						},
						children: "×"
					})]
				}),
				/* @__PURE__ */ d(rr, {
					models: e,
					value: a,
					onPick: (e) => {
						o(e), p(/* @__PURE__ */ new Set()), _(null);
					},
					placeholder: "1. Pick a model to dedicate…"
				}),
				a && /* @__PURE__ */ f(u, { children: [
					/* @__PURE__ */ d("div", {
						className: "wp-group-selall",
						children: /* @__PURE__ */ f("label", {
							className: "wp-group-check",
							children: [/* @__PURE__ */ d("input", {
								type: "checkbox",
								checked: b,
								onChange: () => p(b ? /* @__PURE__ */ new Set() : new Set(y)),
								disabled: y.length === 0
							}), /* @__PURE__ */ f("span", { children: [
								"Select all eligible (",
								y.length,
								")"
							] })]
						})
					}),
					/* @__PURE__ */ f("div", {
						className: "wp-group-list",
						children: [v.length === 0 && /* @__PURE__ */ d("div", {
							className: "wp-none",
							children: "No workers in the pool."
						}), v.map(({ w: e, already: t, eligible: n, reason: r }) => /* @__PURE__ */ f("label", {
							className: `wp-group-row${n ? "" : " wp-group-locked"}`,
							title: n ? "" : r,
							children: [
								/* @__PURE__ */ d("input", {
									type: "checkbox",
									checked: t || c.has(e.id),
									disabled: !n,
									onChange: () => S(e.id)
								}),
								/* @__PURE__ */ d("span", {
									className: "wp-group-wname",
									children: e.name
								}),
								r && /* @__PURE__ */ d("span", {
									className: "wp-group-note",
									children: r
								}),
								e.disk && typeof e.disk.free_bytes == "number" && /* @__PURE__ */ f("span", {
									className: "wp-group-disk",
									children: [X(e.disk.free_bytes), " free"]
								})
							]
						}, e.id))]
					}),
					/* @__PURE__ */ d("div", {
						className: "wp-group-actions",
						children: /* @__PURE__ */ d("button", {
							className: "wp-group-go",
							disabled: m || x.length === 0,
							onClick: async () => {
								if (!(!a || x.length === 0)) {
									h(!0), _(null);
									try {
										let e = await n(a, x, !1);
										_(e);
										let t = new Set(e.filter((e) => !e.ok).map((e) => e.id));
										p((e) => new Set([...e].filter((e) => t.has(e))));
									} finally {
										h(!1);
									}
								}
							},
							title: "Preflight (VRAM+RAM+disk) then assign to each selected worker",
							children: m ? "Dedicating…" : `Dedicate to ${x.length} worker${x.length === 1 ? "" : "s"}`
						})
					}),
					g && /* @__PURE__ */ f("div", {
						className: "wp-group-results",
						children: [g.length === 0 && /* @__PURE__ */ d("span", {
							className: "wp-group-ok",
							children: "Nothing to do — all selected workers already had it."
						}), g.map((e) => /* @__PURE__ */ f("div", {
							className: e.ok ? "wp-group-ok" : "wp-group-bad",
							children: [
								e.ok ? "✓" : "✗",
								" ",
								e.name,
								e.note ? ` — ${e.note}` : ""
							]
						}, e.id))]
					})
				] })
			]
		}) : /* @__PURE__ */ d("button", {
			className: "wp-group-toggle",
			onClick: () => i(!0),
			title: "Dedicate one model to several workers at once",
			children: "⧉ Assign a model to a group of workers…"
		})
	});
}
//#endregion
//#region src/hooks/useColumnLayout.js
function Er(e, t) {
	let n = i((e) => {
		let n = new Set(t), r = [], i = /* @__PURE__ */ new Set();
		for (let t of e?.order || []) n.has(t) && !i.has(t) && (r.push(t), i.add(t));
		for (let e of t) i.has(e) || (r.push(e), i.add(e));
		let a = {};
		for (let [t, r] of Object.entries(e?.widths || {})) n.has(t) && Number.isFinite(r) && r > 0 && (a[t] = r);
		return {
			order: r,
			widths: a
		};
	}, [t]), [r, a] = l(() => {
		try {
			let t = localStorage.getItem(e);
			if (t != null) return n(JSON.parse(t));
		} catch {}
		return n(null);
	}), o = i((t) => {
		try {
			localStorage.setItem(e, JSON.stringify(t));
		} catch {}
		return t;
	}, [e]), s = i((e, t) => {
		e !== t && a((n) => {
			let r = n.order.filter((t) => t !== e), i = t == null ? r.length : r.indexOf(t);
			return r.splice(i < 0 ? r.length : i, 0, e), o({
				...n,
				order: r
			});
		});
	}, [o]), c = i((e, t) => {
		a((n) => o({
			...n,
			widths: {
				...n.widths,
				[e]: Math.max(48, Math.round(t))
			}
		}));
	}, [o]), u = i(() => {
		try {
			localStorage.removeItem(e);
		} catch {}
		a(n(null));
	}, [e, n]);
	return {
		order: r.order,
		widths: r.widths,
		moveColumn: s,
		setWidth: c,
		reset: u
	};
}
//#endregion
//#region src/hooks/usePriorityGroups.jsx
function Dr(e) {
	let t = String(e || "").trim();
	if (!t) return [];
	let n = /* @__PURE__ */ new Set([t, t.toLowerCase()]), r = t.split("/").pop();
	if (n.add(r), n.add(r.toLowerCase()), t.includes("~")) {
		let e = t.split("~").slice(1).join("~");
		e && (n.add(e), n.add(e.toLowerCase()));
	}
	return [...n];
}
function Or(e) {
	let t = /* @__PURE__ */ new Map();
	for (let n of e || []) if (n.enabled) for (let e of n.members || []) for (let r of Dr(e)) t.has(r) || t.set(r, n);
	return t;
}
var kr = null, Ar = null, jr = /* @__PURE__ */ new Set();
function Mr(e = !1) {
	return Ar || (kr && !e ? Promise.resolve(kr) : (Ar = K("/api/llm/model-groups").then((e) => (kr = e.groups || [], jr.forEach((e) => e(kr)), kr)).catch(() => kr || []).finally(() => {
		Ar = null;
	}), Ar));
}
function Nr() {
	let [e, t] = l(kr || []);
	return o(() => (jr.add(t), Mr().then(t), () => {
		jr.delete(t);
	}), []), {
		groups: e,
		reload: () => Mr(!0)
	};
}
//#endregion
//#region src/components/WorkersPanel/AllocationControls.jsx
function Pr({ text: e, value: t, onCommit: n, title: r }) {
	let [i, a] = l(!1), [o, s] = l("");
	return i ? /* @__PURE__ */ d("input", {
		type: "text",
		inputMode: "decimal",
		className: "wp-slider-edit",
		autoFocus: !0,
		value: o,
		title: r,
		onChange: (e) => s(e.target.value),
		onBlur: () => a(!1),
		onKeyDown: (e) => {
			e.key === "Enter" ? (n(o), a(!1)) : e.key === "Escape" && a(!1);
		}
	}) : /* @__PURE__ */ d("button", {
		type: "button",
		className: "wp-slider-readout",
		title: r || "Click to type an exact value",
		onClick: () => {
			s(t === "" || t == null ? "" : String(t)), a(!0);
		},
		children: e
	});
}
function Fr({ label: e, title: t, value: n, onChange: r, min: i, max: a, step: o = 1, formatValue: s, clampMax: c = !0, className: l = "" }) {
	let u = n === "" || n == null ? null : Math.max(i, c ? Math.min(a, Number(n)) : Number(n)), p = u == null ? i : Math.min(a, u), m = s ? s(u) : `${e} ${u ?? "—"}`;
	return /* @__PURE__ */ f("span", {
		className: `wp-plain-slider ${l}`,
		title: t,
		children: [
			e && /* @__PURE__ */ d("span", {
				className: "wp-slider-label",
				children: e
			}),
			/* @__PURE__ */ d("input", {
				type: "range",
				min: i,
				max: a,
				step: o,
				value: p,
				className: "wp-range",
				onChange: (e) => r(e.target.value)
			}),
			/* @__PURE__ */ d(Pr, {
				text: m,
				value: u ?? "",
				title: "Click to type an exact value",
				onCommit: (e) => {
					if (e === "") {
						r("");
						return;
					}
					let t = Number(e);
					Number.isFinite(t) && r(String(Math.round(Math.max(i, c ? Math.min(a, t) : t))));
				}
			})
		]
	});
}
function Ir({ spill: e, worker: t, need: n, onApply: r, onCancel: i }) {
	let a = n && n.gib > 0 ? n.gib : null, o = e && e.priority_device === "ram" ? "ram" : "gpu", [s, c] = l(a != null && e && e.gpu_mem_gib != null ? Math.max(0, Math.min(100, Math.round(e.gpu_mem_gib / a * 100))) : a != null && e && e.cpu_mem_gib != null ? Math.max(0, Math.min(100, 100 - Math.round(e.cpu_mem_gib / a * 100))) : o === "ram" ? 0 : 100), [u, p] = l(e && e.leniency_pct != null ? String(e.leniency_pct) : ""), [m, h] = l(o), g = 100 - s, _ = a == null ? null : +(a * s / 100).toFixed(1), v = a == null ? null : +(a * g / 100).toFixed(1), y = u === "" || u == null ? 0 : Math.max(0, Math.min(100, Number(u))), b = (() => {
		if (y <= 0) return "floor: exactly the split above (strict)";
		let e = m === "gpu" ? Math.max(0, s - y) : Math.min(100, s + y);
		return `floor: ${e}% GPU / ${100 - e}% RAM`;
	})();
	return /* @__PURE__ */ f("div", {
		className: "wp-allocmode-explicit",
		children: [
			/* @__PURE__ */ f("div", {
				className: "wp-allocmode-explicit-head",
				title: pr,
				children: ["🎛 Explicit — split of ", a == null ? "the model" : `${a.toFixed(1)} GiB model`]
			}),
			/* @__PURE__ */ d(Fr, {
				label: "on GPU",
				min: 0,
				max: 100,
				className: "wp-alloc-value",
				value: String(s),
				onChange: (e) => c(Math.max(0, Math.min(100, Math.round(Number(e) || 0)))),
				formatValue: (e) => `${e ?? 0}% GPU`,
				title: "Share of THIS MODEL's footprint placed on the GPU. The rest goes to host RAM — the two always sum to 100% of the model (not of any device)."
			}),
			/* @__PURE__ */ f("div", {
				className: "wp-allocmode-split",
				title: "The complementary split of the model's own footprint.",
				children: [
					/* @__PURE__ */ f("span", {
						className: "wp-allocmode-split-gpu",
						children: [
							s,
							"% GPU",
							_ == null ? "" : ` · ${_.toFixed(1)} GiB`
						]
					}),
					/* @__PURE__ */ d("span", {
						className: "wp-allocmode-split-sep",
						children: "/"
					}),
					/* @__PURE__ */ f("span", {
						className: "wp-allocmode-split-ram",
						children: [
							g,
							"% RAM",
							v == null ? "" : ` · ${v.toFixed(1)} GiB`
						]
					}),
					a != null && /* @__PURE__ */ f("span", {
						className: "wp-allocmode-split-need",
						children: [
							"= ",
							a.toFixed(1),
							" GiB model"
						]
					})
				]
			}),
			/* @__PURE__ */ d(Fr, {
				label: "leniency",
				min: 0,
				max: 100,
				className: "wp-alloc-solo",
				value: u,
				onChange: p,
				formatValue: (e) => e == null ? "leniency —" : `${e}%`,
				title: "Leniency: up to this percent of the model may shift off its preferred device before the load is refused. 0 = strict."
			}),
			/* @__PURE__ */ d("div", {
				className: "wp-allocmode-floor",
				title: "The worst-case placement the load will still accept before refusing.",
				children: b
			}),
			/* @__PURE__ */ f("span", {
				className: "wp-allocmode-prio",
				title: "Priority device: which side the split favors and which way leniency degrades. GPU default.",
				children: [
					/* @__PURE__ */ d("span", {
						className: "wp-slider-label",
						children: "priority"
					}),
					/* @__PURE__ */ d("button", {
						type: "button",
						className: `wp-allocmode-prio-btn${m === "gpu" ? " wp-allocmode-prio-on" : ""}`,
						onClick: () => h("gpu"),
						title: "Favor the GPU (default)",
						children: "GPU"
					}),
					/* @__PURE__ */ d("button", {
						type: "button",
						className: `wp-allocmode-prio-btn${m === "ram" ? " wp-allocmode-prio-on" : ""}`,
						onClick: () => h("ram"),
						title: "Favor host RAM",
						children: "RAM"
					})
				]
			}),
			/* @__PURE__ */ f("div", {
				className: "wp-allocmode-explicit-actions",
				children: [/* @__PURE__ */ d("button", {
					className: "wp-alloc-apply",
					onClick: () => {
						let e = { alloc_mode: "explicit" };
						_ != null && (e.gpu_mem_gib = _), v != null && (e.cpu_mem_gib = v), u !== "" && u != null && (e.leniency_pct = Number(u)), e.priority_device = m, r(e);
					},
					children: "Apply"
				}), /* @__PURE__ */ d("button", {
					className: "wp-alloc-cancel",
					onClick: i,
					title: "Cancel",
					children: "×"
				})]
			})
		]
	});
}
function Lr({ mode: e, spill: t, worker: n, need: r, engineGguf: i, feasible: a, feasibleCtx: s, derivedMode: p, anchorRef: m, onPick: h, onApplyExplicit: g, onRevertDerived: _, onClose: v }) {
	let y = c(null), [b, x] = l(!1), [S, C] = l(null);
	return o(() => {
		let e = () => {
			let e = m && m.current;
			if (!e) return;
			let t = e.getBoundingClientRect(), n = y.current && y.current.offsetWidth || 240, r = Math.max(4, (window.innerWidth || 0) - n - 4);
			C({
				top: t.bottom + 4,
				left: Math.max(4, Math.min(t.left, r))
			});
		};
		e();
		let t = typeof requestAnimationFrame == "function" ? requestAnimationFrame(e) : null;
		return window.addEventListener("scroll", e, !0), window.addEventListener("resize", e), () => {
			t != null && cancelAnimationFrame(t), window.removeEventListener("scroll", e, !0), window.removeEventListener("resize", e);
		};
	}, [m]), o(() => {
		let e = (e) => {
			y.current && y.current.contains(e.target) || m && m.current && m.current.contains(e.target) || v();
		};
		return document.addEventListener("mousedown", e), () => document.removeEventListener("mousedown", e);
	}, [v, m]), /* @__PURE__ */ d("div", {
		className: "wp-allocmode-menu",
		ref: y,
		role: "menu",
		style: S ? {
			position: "fixed",
			top: S.top,
			left: S.left
		} : void 0,
		onKeyDown: (e) => {
			e.key === "Escape" && v();
		},
		children: b ? /* @__PURE__ */ d(Ir, {
			spill: t,
			worker: n,
			need: r,
			onApply: (e) => {
				g(e);
			},
			onCancel: () => x(!1)
		}) : /* @__PURE__ */ f(u, { children: [_ && /* @__PURE__ */ f("button", {
			role: "menuitem",
			className: "wp-allocmode-opt wp-allocmode-opt-auto",
			title: "Revert to the DERIVED default — clears any pinned contract so this model tracks the derivation (and improves with it, as measured values land). This is the always-available revert-to-default.",
			onClick: () => _(),
			children: [/* @__PURE__ */ d("span", {
				className: "wp-allocmode-opt-label",
				children: "↺ Auto"
			}), /* @__PURE__ */ f("span", {
				className: "wp-allocmode-opt-desc",
				children: ["— derived: ", yr(p || e)]
			})]
		}, "__auto__"), dr.map(([t, n, r]) => {
			let o = Array.isArray(a) ? !a.includes(t) : i === !1 && !fr.has(t), c = o ? Sr(t, s, i) || "not feasible for this model on this worker" : null, l = o, u = t === e;
			return /* @__PURE__ */ f("button", {
				role: "menuitemradio",
				"aria-checked": u,
				disabled: l,
				className: `wp-allocmode-opt${u ? " wp-allocmode-opt-on" : ""}`,
				title: l ? c : u ? "current mode — click to close" : t === "explicit" ? "opens the explicit-budget knobs" : "applies immediately",
				onClick: () => {
					if (!l) {
						if (t === "explicit") {
							x(!0);
							return;
						}
						if (u) {
							v();
							return;
						}
						h(t);
					}
				},
				children: [
					/* @__PURE__ */ d("span", {
						className: "wp-allocmode-opt-label",
						children: n
					}),
					/* @__PURE__ */ f("span", {
						className: "wp-allocmode-opt-desc",
						children: ["— ", r]
					}),
					u && /* @__PURE__ */ d("span", {
						className: "wp-allocmode-opt-mark",
						children: "✓"
					})
				]
			}, t);
		})] })
	});
}
function Rr({ bulkKeys: e, getModelBytes: t, onApply: n, onCancel: r }) {
	let [i, a] = l(100), [o, s] = l(""), [c, u] = l("gpu"), p = 100 - i, m = (e || []).map((e) => {
		let n = t ? t(e) : null;
		return n && n.bytes != null ? n.bytes / ur : null;
	}).filter((e) => e != null && e > 0), h = (() => {
		if (!m.length) return null;
		let e = m.map((e) => e * i / 100), t = Math.min(...e), n = Math.max(...e);
		return {
			lo: +t.toFixed(1),
			hi: +n.toFixed(1)
		};
	})(), g = o === "" || o == null ? 0 : Math.max(0, Math.min(100, Number(o))), _ = (() => {
		if (g <= 0) return "floor: exactly the split above (strict)";
		let e = c === "gpu" ? Math.max(0, i - g) : Math.min(100, i + g);
		return `floor: ${e}% GPU / ${100 - e}% RAM`;
	})();
	return /* @__PURE__ */ f("div", {
		className: "wp-allocmode-explicit",
		children: [
			/* @__PURE__ */ d("div", {
				className: "wp-allocmode-explicit-head",
				title: pr,
				children: "🎛 Explicit — split of EACH selected model (% of-the-model; GGUF members only)"
			}),
			/* @__PURE__ */ d(Fr, {
				label: "on GPU",
				min: 0,
				max: 100,
				className: "wp-alloc-value",
				value: String(i),
				onChange: (e) => a(Math.max(0, Math.min(100, Math.round(Number(e) || 0)))),
				formatValue: (e) => `${e ?? 0}% GPU`,
				title: "Share of EACH selected model's own footprint placed on the GPU. The rest goes to host RAM — the two always sum to 100% of that model (resolved against each member's own size at apply, so the GiB differs per model)."
			}),
			/* @__PURE__ */ f("div", {
				className: "wp-allocmode-split",
				title: "The complementary split, applied per model against its own size.",
				children: [
					/* @__PURE__ */ f("span", {
						className: "wp-allocmode-split-gpu",
						children: [
							i,
							"% GPU",
							h == null ? "" : ` · ${h.lo === h.hi ? `${h.lo.toFixed(1)} GiB` : `${h.lo.toFixed(1)}–${h.hi.toFixed(1)} GiB`}`
						]
					}),
					/* @__PURE__ */ d("span", {
						className: "wp-allocmode-split-sep",
						children: "/"
					}),
					/* @__PURE__ */ f("span", {
						className: "wp-allocmode-split-ram",
						children: [p, "% RAM"]
					}),
					/* @__PURE__ */ d("span", {
						className: "wp-allocmode-split-need",
						children: h == null ? "per model (sizes unknown → % intent only)" : "per model — GiB range across the selection"
					})
				]
			}),
			/* @__PURE__ */ d(Fr, {
				label: "leniency",
				min: 0,
				max: 100,
				className: "wp-alloc-solo",
				value: o,
				onChange: s,
				formatValue: (e) => e == null ? "leniency —" : `${e}%`,
				title: "Leniency: up to this percent of each model may shift off its preferred device before that load is refused. 0 = strict."
			}),
			/* @__PURE__ */ d("div", {
				className: "wp-allocmode-floor",
				title: "The worst-case placement each load will still accept before refusing.",
				children: _
			}),
			/* @__PURE__ */ f("span", {
				className: "wp-allocmode-prio",
				title: "Priority device: which side each split favors and which way leniency degrades. GPU default.",
				children: [
					/* @__PURE__ */ d("span", {
						className: "wp-slider-label",
						children: "priority"
					}),
					/* @__PURE__ */ d("button", {
						type: "button",
						className: `wp-allocmode-prio-btn${c === "gpu" ? " wp-allocmode-prio-on" : ""}`,
						onClick: () => u("gpu"),
						title: "Favor the GPU (default)",
						children: "GPU"
					}),
					/* @__PURE__ */ d("button", {
						type: "button",
						className: `wp-allocmode-prio-btn${c === "ram" ? " wp-allocmode-prio-on" : ""}`,
						onClick: () => u("ram"),
						title: "Favor host RAM",
						children: "RAM"
					})
				]
			}),
			/* @__PURE__ */ f("div", {
				className: "wp-allocmode-explicit-actions",
				children: [/* @__PURE__ */ f("button", {
					className: "wp-alloc-apply",
					onClick: () => {
						let r = {};
						for (let n of e || []) {
							let e = t ? t(n) : null, a = e && e.bytes != null ? e.bytes / ur : null, s = {
								alloc_mode: "explicit",
								priority_device: c
							};
							a != null && a > 0 && (s.gpu_mem_gib = +(a * i / 100).toFixed(1), s.cpu_mem_gib = +(a * p / 100).toFixed(1)), o !== "" && o != null && (s.leniency_pct = Number(o)), r[n] = s;
						}
						n(null, r);
					},
					children: ["Apply to ", (e || []).length]
				}), /* @__PURE__ */ d("button", {
					className: "wp-alloc-cancel",
					onClick: r,
					title: "Cancel",
					children: "×"
				})]
			})
		]
	});
}
function zr({ count: e, bulkKeys: t, getModelBytes: n, onApply: r, onCancel: i }) {
	let [a, o] = l(!1);
	return a ? /* @__PURE__ */ d("div", {
		className: "wp-alloc wp-bulkalloc",
		children: /* @__PURE__ */ d(Rr, {
			bulkKeys: t,
			getModelBytes: n,
			onApply: r,
			onCancel: () => o(!1)
		})
	}) : /* @__PURE__ */ f("div", {
		className: "wp-alloc wp-bulkalloc",
		children: [/* @__PURE__ */ f("div", {
			className: "wp-allocmode-menu wp-bulkalloc-menu",
			role: "menu",
			children: [/* @__PURE__ */ f("button", {
				role: "menuitem",
				className: "wp-allocmode-opt wp-allocmode-opt-auto",
				title: "Default (derived) — clears any pinned contract on every selected model so each tracks its derived default (and improves with it as measured values land). The bulk revert-to-default.",
				onClick: () => r({}, null),
				children: [/* @__PURE__ */ d("span", {
					className: "wp-allocmode-opt-label",
					children: "↺ Default (derived)"
				}), /* @__PURE__ */ d("span", {
					className: "wp-allocmode-opt-desc",
					children: "— clear overrides; track the derivation"
				})]
			}, "__default__"), dr.map(([e, t, n]) => {
				let i = !fr.has(e);
				return /* @__PURE__ */ f("button", {
					role: "menuitem",
					className: "wp-allocmode-opt",
					title: e === "explicit" ? "opens the explicit-budget knobs (a % split of each model — GGUF members only)" : i ? `${n} — GGUF-only; transformers/comfy members in the selection are skipped with a reason` : `${n} — applies to every selected model immediately`,
					onClick: () => {
						if (e === "explicit") {
							o(!0);
							return;
						}
						r({ alloc_mode: e }, null);
					},
					children: [
						/* @__PURE__ */ d("span", {
							className: "wp-allocmode-opt-label",
							children: t
						}),
						/* @__PURE__ */ f("span", {
							className: "wp-allocmode-opt-desc",
							children: ["— ", n]
						}),
						i && /* @__PURE__ */ d("span", {
							className: "wp-allocmode-opt-mark",
							title: "GGUF-only mode",
							children: "GGUF"
						})
					]
				}, e);
			})]
		}), /* @__PURE__ */ f("div", {
			className: "wp-bulkalloc-foot",
			children: [/* @__PURE__ */ f("span", {
				className: "wp-alloc-note",
				title: "explicit is a GGUF-only concept — its banded leniency floor has no transformers analogue. In a mixed selection the backend applies it to the GGUF members and skips the rest with an honest reason. Default / Max GPU / GPU only / RAM only / Max RAM apply to every engine.",
				children: [
					"applies to ",
					e,
					" selected · explicit skips transformers members"
				]
			}), i && /* @__PURE__ */ d("button", {
				className: "wp-alloc-cancel",
				onClick: i,
				title: "Cancel",
				children: "×"
			})]
		})]
	});
}
//#endregion
//#region src/components/WorkersPanel/ExternalLeases.jsx
function Br({ worker: e }) {
	let [t, n] = l(null), [r, a] = l(null), [s, u] = l(null), p = c(!0), m = i(async () => {
		try {
			let t = await K(`/api/llm/workers/${encodeURIComponent(e.id)}/external`);
			if (!p.current) return;
			n(t.external || []);
		} catch {
			p.current && n([]);
		}
	}, [e.id]);
	o(() => {
		p.current = !0, m();
		let e = setInterval(m, 3e4);
		return () => {
			p.current = !1, clearInterval(e);
		};
	}, [m]);
	let h = i(async (t, n) => {
		a(t.model_key), u(null);
		try {
			await K(`/api/llm/workers/${encodeURIComponent(e.id)}/external-set`, {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({
					model_key: t.model_key,
					...n
				})
			});
		} catch (e) {
			u(`${t.model_key}: ${e.message}`);
		} finally {
			a(null), m();
		}
	}, [e.id, m]);
	return !t || t.length === 0 ? null : /* @__PURE__ */ f("div", {
		className: "wp-caps wp-leases",
		children: [
			/* @__PURE__ */ d("span", {
				className: "wp-cap-chip",
				title: "External gpu_lease residents: batch jobs sharing this card as pseudo-models (hugpy-lease). Both toggles apply live via the worker’s /ops/external/set — no job restart. They revert to the lease’s launch flags if the lease service itself restarts.",
				children: "🎟 leases:"
			}),
			t.map((e) => {
				let t = e.evictable !== !1, n = e.resume !== "disabled", i = r === e.model_key;
				return /* @__PURE__ */ f("span", {
					className: "wp-cap-chip wp-lease-chip",
					children: [
						/* @__PURE__ */ d("strong", {
							title: e.note || e.model_key,
							children: e.model_key
						}),
						e.vram_gib != null && /* @__PURE__ */ f("em", { children: [
							" ",
							e.vram_gib,
							"GiB"
						] }),
						e.state && /* @__PURE__ */ f("em", { children: [" · ", e.state] }),
						/* @__PURE__ */ d("button", {
							className: `wp-lease-toggle${t ? "" : " wp-lease-off"}`,
							disabled: i,
							title: t ? "Evictable: any demand path (model load, image gen) may pause this job to reclaim VRAM. Click to protect it (external twin of 🔒 static — only force-evict touches it)." : "Protected (non-evictable): only force-evict can pause this job. Click to make it evictable again.",
							onClick: () => h(e, { evictable: !t }),
							children: t ? "⚡ evictable" : "🔒 protected"
						}),
						/* @__PURE__ */ d("button", {
							className: `wp-lease-toggle${n ? "" : " wp-lease-off"}`,
							disabled: i,
							title: n ? "Resume enabled: after an eviction pause the lease waits, re-claims VRAM and relaunches the job where it left off. Click to make a pause final." : "Resume disabled: a pause ENDS the run (the supervisor unregisters and exits). Click to re-enable relaunch-after-pause.",
							onClick: () => h(e, { resume: n ? "disabled" : "enabled" }),
							children: n ? "🔁 resume" : "⏹ one-shot"
						})
					]
				}, e.model_key);
			}),
			s && /* @__PURE__ */ f("span", {
				className: "wp-lease-err",
				title: s,
				children: ["⚠ ", s]
			})
		]
	});
}
//#endregion
//#region src/components/WorkersPanel/ResidencyMenu.jsx
var Vr = [[
	"on-demand",
	"⏲ on-demand",
	"loads on call; holds its slot until another model needs the seat (default)"
], [
	"static",
	"🔒 static",
	"always kept on this worker: downloaded eagerly, never evicted (the only tier that keeps files on disk)"
]];
function Hr({ mode: e, onPick: t, onClose: n }) {
	let r = c(null);
	return o(() => {
		let e = (e) => {
			r.current && !r.current.contains(e.target) && n();
		};
		return document.addEventListener("mousedown", e), () => document.removeEventListener("mousedown", e);
	}, [n]), /* @__PURE__ */ d("div", {
		className: "wp-res-menu",
		ref: r,
		role: "menu",
		onKeyDown: (e) => {
			e.key === "Escape" && n();
		},
		children: Vr.map(([r, i, a]) => /* @__PURE__ */ f("button", {
			role: "menuitemradio",
			"aria-checked": r === e,
			className: `wp-res-opt${r === e ? " wp-res-opt-on" : ""}`,
			title: r === e ? "current state — click to close" : "applies via a ~5s agent restart",
			onClick: () => r === e ? n() : t(r),
			children: [
				/* @__PURE__ */ d("span", {
					className: "wp-res-opt-label",
					children: i
				}),
				/* @__PURE__ */ f("span", {
					className: "wp-res-opt-desc",
					children: ["— ", a]
				}),
				r === e && /* @__PURE__ */ d("span", {
					className: "wp-res-opt-mark",
					children: "✓"
				})
			]
		}, r))
	});
}
//#endregion
//#region src/components/WorkersPanel/workerMetrics.js
function Ur(e) {
	return !!e && (e.vram_bytes != null && e.vram_bytes > 0 || e.rss_bytes != null && e.rss_bytes > 0 || e.device === "cpu" && e.device_source !== "inferred");
}
function Wr(e) {
	return !e || e.materialized === !1 ? !1 : Ur(e);
}
function Gr(e) {
	return e.kind === "slot" ? e.healthy && e.busy ? {
		state: "answering",
		glyph: "⚡ answering",
		title: "actively processing a request right now"
	} : e.healthy ? {
		state: "serving",
		glyph: "🔥 serving",
		title: "hosted in a slot on this worker — routable"
	} : {
		state: "warming",
		glyph: "⏳ warming",
		title: "seated but not yet healthy — warming"
	} : Wr(e) ? {
		state: "serving",
		glyph: "🔥 serving",
		title: "measured residency — the worker sees this model occupying VRAM/RAM right now"
	} : e.materialized === !1 || e.serving === !0 ? {
		state: "allocated",
		glyph: "◌ allocated",
		title: "an allocation is attributed to this model but nothing measured its residency — it may never have loaded"
	} : {
		state: "idle",
		glyph: "○ idle",
		title: "a runner exists but holds no measured VRAM/RAM — not actually resident"
	};
}
function Kr(e, t, n) {
	if (e?.vram_bytes != null) return {
		bytes: e.vram_bytes,
		estimated: !1
	};
	let r = t?.loaded_detail?.[e?.model_key] || {}, i = n?.get?.(e?.model_key), a = e?.weight_bytes ?? e?.model_bytes ?? r.weight_bytes ?? r.model_bytes ?? (i && i.eff != null ? i.eff : null);
	if (a == null) return null;
	let o = e?.n_gpu_layers, s = e?.total_layers ?? r.total_layers, c = (e) => ({
		bytes: e,
		estimated: !0
	});
	return o === -1 ? c(a) : o != null && o > 0 ? s ? c(a * o / s) : null : e?.gpu_pct != null && e.gpu_pct > 0 ? c(a * e.gpu_pct / 100) : null;
}
//#endregion
//#region src/components/WorkersPanel/BudgetBars.jsx
function qr({ label: e, total: t, used: n, reserve: r, reserveGib: i, note: a }) {
	let o = t ? Math.min(Math.max((n || 0) / t * 100, 0), 100) : 0, s = r && t ? Math.min(r / t * 100, 100) : 0, c = n == null ? null : Math.max(t - n, 0);
	return /* @__PURE__ */ f("div", {
		className: "wp-budget-row",
		title: `${e}: ${X(n || 0)} used / ${X(t)}` + (c == null ? "" : ` · ${X(c)} free`) + (a ? ` — ${a}` : ""),
		children: [/* @__PURE__ */ d("span", {
			className: "wp-budget-label",
			children: e
		}), /* @__PURE__ */ f("div", {
			className: "wp-budget-bar",
			children: [/* @__PURE__ */ d("div", {
				className: "wp-budget-fill",
				style: { width: `${o}%` }
			}), s > 0 && /* @__PURE__ */ d("div", {
				className: "wp-budget-reserve",
				style: { width: `${s}%` },
				title: `reserve: ${i} GiB held out of the pool (not allocatable)`
			})]
		})]
	});
}
//#endregion
//#region src/components/WorkersPanel/storageBadge.js
function Jr(e, t) {
	return e.refused ? {
		pill: "wp-pill-refused",
		glyph: "⊘ missing",
		title: `won't fit on this worker — the download was REFUSED before it started (nothing was deleted, no partial file). ${e.refused.reason || ""}\n\nRaise this worker's storage allocation (disk_cache_gib), free a protected model, or route this model to another box.`
	} : t ? {
		pill: "wp-pill-evict",
		glyph: "🗑 will free",
		title: "in the eviction proposal — freed on approval. 📌 A pinned model can appear here: pin keeps its allocation/routing, not its files."
	} : e.store === "shared" ? {
		pill: "wp-pill-shared",
		glyph: "🔗 shared",
		title: "on the SHARED central catalog (the fleet's source-of-truth copies, read through from here). Never evicted from this worker and never counted against its storage budget — it is not this box's cache."
	} : e.store === "unreapable" ? {
		pill: "wp-pill-shared",
		glyph: "🔗 unreapable store",
		title: "on a model store this box has not declared local & disposable (HUGPY_MODEL_STORE_REAPABLE unset), so nothing here can be reaped. Shown for visibility; never counted against this worker's storage budget."
	} : e.why === "static" ? {
		pill: "wp-pill-loaded",
		glyph: "🔒 static",
		title: "static residency — a locked seat, never evicted, files kept on disk (the only tier that blocks eviction/reaping)"
	} : e.loaded ? {
		pill: "wp-pill-serving",
		glyph: "🔥 loaded",
		title: "resident/serving right now — protected"
	} : e.loading ? {
		pill: "wp-pill-heating",
		glyph: "🔶 heating",
		title: "weights loading — protected"
	} : e.provisioning ? {
		pill: "wp-pill-pulling",
		glyph: "⏳ pulling",
		title: "files transferring right now — protected while the bytes land"
	} : e.assigned ? {
		pill: "wp-pill-idle",
		glyph: "📎 assigned",
		title: "designated to this worker — protected in the operator-gated bulk reaper"
	} : e.pinned ? {
		pill: "wp-pill-loaded",
		glyph: "📌 pinned",
		title: "this allocation survives restarts (routing to this worker is durable). Does NOT download the model and does NOT protect its files from eviction — bytes arrive on call and can be evicted to make room (routing is unaffected). Only 🔒 static keeps files on disk."
	} : e.protected ? {
		pill: "wp-pill-idle",
		glyph: "🛡 protected",
		title: e.why || "protected"
	} : {
		pill: "wp-pill-cold",
		glyph: "○ evictable",
		title: "on disk, unassigned & cold — reclaimable"
	};
}
//#endregion
//#region src/components/WorkersPanel/WorkerStorageBar.jsx
function Yr({ worker: e, onApproveEvictions: t, sizeByKey: n, detailsOnly: r = !1 }) {
	let i = 2 ** 30, a = e.storage;
	if (!a || !a.reported) return null;
	let o = Array.isArray(a.models) ? a.models : [], s = o.filter((e) => e.counts_toward_budget === !1), c = o.filter((e) => e.counts_toward_budget !== !1), l = a.unbudgeted_bytes == null ? s.reduce((e, t) => e + (t.bytes || 0), 0) : a.unbudgeted_bytes, u = a.gauge_used_bytes == null ? a.resident_bytes == null ? a.cache_used_bytes || 0 : a.resident_bytes : a.gauge_used_bytes, p = a.budget_basis === "cap", m = !!a.over_budget, h = Array.isArray(a.proposed_evictions) ? a.proposed_evictions : [], g = new Set(h.map((e) => e.model_key)), _ = Math.max(p ? a.budget || 0 : u + (a.disk_free || 0), u, 1), v = p ? 0 : a.reserve || 0, y = `${X(u)} / ${a.budget == null ? "?" : X(a.budget)} on disk · ${c.length} model${c.length === 1 ? "" : "s"} resident`, b = a.allocated_over_budget_bytes || 0, x = a.allocated_unknown_count || 0, S = a.allocated_total_bytes || 0, C = b > 0, w = a.orphaned_bytes || 0, T = a.orphaned_count || 0, E = Array.isArray(a.orphaned_items) ? a.orphaned_items : [], D = T === 0 ? "" : `${X(w)} across ${T} item(s) sit on this worker's disk but are attributed to NO current model — leftover dirs or stalled partial downloads (*.part) from an abandoned pull. This is NOT in the assigned set and NOT a resident model; it is residue eating the drive.\n\n` + E.slice(0, 12).map((e) => `  • ${e.path} — ${X(e.bytes)}${e.kind === "partial" ? " (stalled .part)" : ""}`).join("\n") + (E.length > 12 ? `\n  …and ${E.length - 12} more` : ""), O = `${x ? "≥" : ""}${X(S)}`, k = `The ${a.allocated_count} model(s) ASSIGNED to this worker total ${O}${x ? ` (${x} of unknown size — this total is a floor)` : ""}, against a ${X(a.budget)} budget — ${X(b)} OVER.\n\nThis is the assignment set, not what is on disk: models download lazily, on first call. The set as a whole cannot fit, so eviction cannot save it — some call will eventually be refused no matter which model asks.\n\nUnassign models from this worker, raise its allocation (disk_cache_gib), or route some of them to another box.`, A = s.length === 0 ? "" : `${X(l)} across ${s.length} model(s) sit on a store this worker may never delete from — the SHARED central catalog (${a.store_root_shared ? "this box's model root IS that catalog" : "mounted and read through from here"}), or a store the box never declared local & disposable.\n\nThey are shown for visibility but count ZERO toward "${X(u)} on disk" and toward the over-budget math: the eviction economy is this worker's OWN cache. They can never be proposed for eviction, and every delete-time guard refuses them independently.`, j = a.refused && typeof a.refused == "object" ? a.refused : {}, M = Object.entries(j).map(([e, t]) => ({
		model_key: e,
		bytes: 0,
		refused: t,
		protected: !1,
		last_picked: null
	})), N = [...M, ...c].sort((e, t) => {
		let n = e.refused ? -1 : g.has(e.model_key) ? 0 : e.protected ? 2 : 1, r = t.refused ? -1 : g.has(t.model_key) ? 0 : t.protected ? 2 : 1;
		return n === r ? (e.last_picked || 0) - (t.last_picked || 0) : n - r;
	});
	return /* @__PURE__ */ f("div", {
		className: `wp-storage${m ? " wp-storage-over" : ""}`,
		children: [
			!r && /* @__PURE__ */ f("div", {
				className: "wp-storage-head",
				title: `Model weights RESIDENT (on disk) on ${e.disk?.root || "the model-root volume"}: ${X(u)} across ${c.length} model(s) — NOT the assigned set (models download lazily, so attribution ≠ disk usage). ` + (p ? `explicit cap ${X(a.budget)}` : `budget ${X(a.budget)} (keeps ${X(v)} disk free in reserve)`) + ". Loaded / 🔒static / assigned models are protected and never proposed. 📌 Pinned models are NOT protected — pin keeps the allocation/routing, not the files, so a pinned model can be proposed for eviction (its bytes re-pull on next call).",
				children: [
					/* @__PURE__ */ d("span", {
						className: "wp-storage-icon",
						children: "💾 storage"
					}),
					/* @__PURE__ */ d("span", {
						className: "wp-storage-figs",
						children: y
					}),
					/* @__PURE__ */ d("span", {
						className: "wp-storage-basis",
						children: p ? "cap" : "reserve"
					}),
					m && /* @__PURE__ */ f("span", {
						className: "wp-storage-warn",
						title: `Over budget by ${X(a.need_bytes)} — the review below proposes evicting the coldest unprotected models to get back under.`,
						children: [
							"⚠ over budget · ",
							X(a.need_bytes),
							" over",
							/* @__PURE__ */ d(ar, { doc: "worker-over-budget" })
						]
					}),
					C && /* @__PURE__ */ f("span", {
						className: "wp-storage-warn",
						title: k,
						children: [
							"⚠ assigned ",
							O,
							" · ",
							X(b),
							" over allocation"
						]
					}),
					T > 0 && /* @__PURE__ */ f("span", {
						className: "wp-storage-orphan",
						title: D,
						children: [
							"🧹 unattributed on disk: ",
							X(w),
							" · ",
							T,
							" item",
							T === 1 ? "" : "s"
						]
					}),
					s.length > 0 && /* @__PURE__ */ f("span", {
						className: "wp-storage-shared-chip",
						title: A,
						children: [
							"🔗 shared catalog: ",
							X(l),
							" · ",
							s.length,
							" model",
							s.length === 1 ? "" : "s",
							" (never evicted)"
						]
					})
				]
			}),
			!r && a.budget != null && /* @__PURE__ */ d(qr, {
				label: "DISK",
				total: _,
				used: u,
				reserve: v,
				reserveGib: v ? Math.round(v / i) : 0,
				note: p ? "model cache vs. explicit per-worker cap" : "model cache; hatched tail = disk reserve kept free"
			}),
			r && m && /* @__PURE__ */ f("div", {
				className: "wp-storage-warn wp-storage-warn-solo",
				title: "Over budget — the proposal below evicts the coldest unprotected models to get back under.",
				children: [
					"⚠ over budget · ",
					X(a.need_bytes),
					" over",
					/* @__PURE__ */ d(ar, { doc: "worker-over-budget" })
				]
			}),
			r && N.length === 0 && s.length === 0 && /* @__PURE__ */ d("div", {
				className: "wp-res-empty",
				children: "No model files cached on this worker yet."
			}),
			r && N.length === 0 && s.length > 0 && /* @__PURE__ */ d("div", {
				className: "wp-res-empty",
				title: A,
				children: "No model files in this worker's own cache — everything below is on the shared catalog."
			}),
			M.length > 0 && /* @__PURE__ */ f("div", {
				className: "wp-storage-warn wp-storage-warn-solo",
				title: "These models were REQUESTED but could not be downloaded: even after evicting every cold, unprotected model, they would not fit under this worker's storage allocation. The pulls were refused BEFORE they started, so no partial files and no wasted disk. Hover a row for its exact numbers.",
				children: [
					"⊘ ",
					M.length,
					" model",
					M.length === 1 ? "" : "s",
					" missing — won't fit under this worker's storage budget"
				]
			}),
			N.length > 0 && /* @__PURE__ */ d("div", {
				className: "wp-storage-list",
				children: N.map((e) => {
					let t = g.has(e.model_key), r = Jr(e, t), i = n && n.get(e.model_key), a = i && e.bytes > i.eff * 1.05;
					return /* @__PURE__ */ f("div", {
						className: `wp-storage-row${e.protected ? " wp-storage-protected" : ""}${t ? " wp-storage-proposed" : ""}${e.refused ? " wp-storage-refused" : ""}`,
						children: [
							/* @__PURE__ */ d("span", {
								className: `wp-state-pill ${r.pill}`,
								title: r.title,
								children: r.glyph
							}),
							/* @__PURE__ */ d("span", {
								className: "wp-storage-name",
								title: e.model_key,
								children: e.model_key
							}),
							e.refused ? /* @__PURE__ */ d("span", {
								className: "wp-storage-size",
								title: r.title,
								children: /* @__PURE__ */ f("em", {
									className: "wp-storage-quant",
									children: ["needs ", X(e.refused.needs_bytes || 0)]
								})
							}) : /* @__PURE__ */ f("span", {
								className: "wp-storage-size",
								title: a ? `${X(e.bytes)} on disk (all variants); effective quant ${i.effGguf || ""} ${X(i.eff)}` : void 0,
								children: [X(e.bytes), a && /* @__PURE__ */ f("em", {
									className: "wp-storage-quant",
									children: [" · quant ", X(i.eff)]
								})]
							}),
							/* @__PURE__ */ d("span", {
								className: "wp-storage-served",
								title: e.refused ? "never served — the files were never downloaded" : "Last time central routed a request to this (worker, model)",
								children: e.refused ? "—" : wr(e.last_picked)
							})
						]
					}, e.model_key);
				})
			}),
			s.length > 0 && /* @__PURE__ */ f("div", {
				className: "wp-storage-shared",
				children: [/* @__PURE__ */ f("div", {
					className: "wp-storage-shared-head",
					title: A,
					children: [
						"🔗 shared catalog (never evicted) — ",
						X(l),
						" across ",
						s.length,
						" model",
						s.length === 1 ? "" : "s",
						", on a store this worker may not delete from. Not counted toward the budget above."
					]
				}), /* @__PURE__ */ d("div", {
					className: "wp-storage-list",
					children: s.slice().sort((e, t) => (t.bytes || 0) - (e.bytes || 0)).map((e) => {
						let t = Jr(e, !1);
						return /* @__PURE__ */ f("div", {
							className: "wp-storage-row wp-storage-protected",
							children: [
								/* @__PURE__ */ d("span", {
									className: `wp-state-pill ${t.pill}`,
									title: t.title,
									children: t.glyph
								}),
								/* @__PURE__ */ d("span", {
									className: "wp-storage-name",
									title: e.model_key,
									children: e.model_key
								}),
								/* @__PURE__ */ d("span", {
									className: "wp-storage-size",
									title: "on the shared store — not billed to this worker's budget",
									children: X(e.bytes)
								}),
								/* @__PURE__ */ d("span", {
									className: "wp-storage-served",
									title: "Last time central routed a request to this (worker, model)",
									children: wr(e.last_picked)
								})
							]
						}, e.model_key);
					})
				})]
			}),
			m && h.length > 0 && /* @__PURE__ */ f("div", {
				className: "wp-storage-review",
				children: [
					/* @__PURE__ */ f("div", {
						className: "wp-storage-review-head",
						children: [
							"Eviction proposal — frees ",
							X(a.proposed_free_bytes),
							" by deleting ",
							h.length,
							" cold, unprotected model",
							h.length === 1 ? "" : "s",
							" (least-recently-served first):"
						]
					}),
					/* @__PURE__ */ d("div", {
						className: "wp-storage-review-list",
						children: h.map((e) => /* @__PURE__ */ f("div", {
							className: "wp-storage-review-row",
							children: [
								/* @__PURE__ */ d("span", {
									className: "wp-storage-name",
									title: e.model_key,
									children: e.model_key
								}),
								/* @__PURE__ */ d("span", {
									className: "wp-storage-size",
									children: X(e.bytes)
								}),
								/* @__PURE__ */ d("span", {
									className: "wp-storage-served",
									children: wr(e.last_picked)
								})
							]
						}, e.model_key))
					}),
					/* @__PURE__ */ f("button", {
						className: "wp-storage-approve",
						onClick: () => t && t(e),
						title: "Delete these files now. Central re-checks the proposal at approval time and the worker re-proves every guard per model before deleting — protected (loaded / 🔒static / assigned) files are never touched. 📌 Pinned files ARE eligible: pin keeps the allocation, not the bytes, which re-pull on next call.",
						children: ["✓ Approve & free ", X(a.proposed_free_bytes)]
					})
				]
			})
		]
	});
}
//#endregion
//#region src/components/WorkersPanel/ResourceStrip.jsx
function Xr({ gpus: e }) {
	return !e || !e.length ? /* @__PURE__ */ d("span", {
		className: "wp-nogpu",
		children: "no GPU reported"
	}) : /* @__PURE__ */ d("div", {
		className: "wp-gpus",
		children: e.map((e, t) => {
			let n = e.memory_total, r = e.memory_free, i = n != null && r != null ? Math.max(n - r, 0) : null, a = i != null && n ? Math.min(i / n * 100, 100) : 0;
			return /* @__PURE__ */ f("div", {
				className: "wp-gpu",
				title: e.name || "",
				children: [/* @__PURE__ */ f("div", {
					className: "wp-gpu-head",
					children: [
						"🖥 ",
						e.name || `GPU ${e.index ?? t}`,
						i != null && /* @__PURE__ */ f("em", { children: [
							" · ",
							X(i),
							" used / ",
							X(n)
						] })
					]
				}), i != null && /* @__PURE__ */ d("div", {
					className: "wp-vram-bar",
					title: `${X(r)} free`,
					children: /* @__PURE__ */ d("div", {
						className: "wp-vram-fill",
						style: { width: `${a}%` }
					})
				})]
			}, t);
		})
	});
}
function Zr({ worker: e, onEvict: t }) {
	let n = e && e.pid_registry, [r, i] = l(null);
	if (!n) return null;
	let a = Array.isArray(n.models) ? n.models : [], o = Array.isArray(n.unattributed) ? n.unattributed : [];
	if (!a.length && !o.length) return null;
	let s = async (n) => {
		if (!(!t || r)) {
			i(n);
			try {
				await t(e, n);
			} finally {
				i(null);
			}
		}
	}, c = 1024 * 1024, p = a.reduce((e, t) => e + (Number(t.vram_bytes) || 0), 0), m = o.reduce((e, t) => e + (Number(t.mib) || 0) * c, 0), h = p + m, g = (e) => e.model_key || e.label || e.display_label || (e.host_mode === "cuda_context" ? "agent CUDA context" : e.host_mode === "comfy" ? "ComfyUI" : e.host_mode || "unknown");
	return /* @__PURE__ */ f("div", {
		className: "wp-pidreg",
		children: [
			a.length > 0 && /* @__PURE__ */ f(u, { children: [/* @__PURE__ */ f("div", {
				className: "wp-res-detail-label",
				children: [
					"GPU process registry — model → pid → measured VRAM (mirrors nvidia-smi · ",
					a.length,
					")"
				]
			}), /* @__PURE__ */ d("div", {
				className: "wp-pidreg-list",
				children: a.map((e, n) => {
					let i = !!e.model_key, a = i && r === e.model_key, o = e.alive !== !1, l = e.vram_bytes_planned, u = e.vram_bytes, p = l != null && u != null && Math.abs(Number(l) - Number(u)) > c;
					return /* @__PURE__ */ f("div", {
						className: "wp-pidreg-row",
						title: `${g(e)} · pid ${e.pid} · ${e.host_mode || "unknown host"}`,
						children: [
							/* @__PURE__ */ d("span", {
								className: `wp-pidreg-dot ${o ? "wp-pidreg-alive" : "wp-pidreg-dead"}`,
								title: o ? "process alive" : "process gone (stale entry)",
								children: o ? "●" : "○"
							}),
							/* @__PURE__ */ d("span", {
								className: "wp-pidreg-name",
								children: g(e)
							}),
							/* @__PURE__ */ f("span", {
								className: "wp-pidreg-meta",
								children: ["pid ", e.pid]
							}),
							/* @__PURE__ */ d("span", {
								className: "wp-pidreg-mode",
								title: `host mode: ${e.host_mode || "unknown"}`,
								children: e.host_mode || "—"
							}),
							/* @__PURE__ */ f("span", {
								className: "wp-pidreg-vram",
								title: p ? `measured ${X(u)} (nvidia-smi) vs planned ${X(l)} (declared weights) — the gap is CUDA context / KV` : "measured VRAM from nvidia-smi per-PID",
								children: [u == null ? "—" : X(u), p && /* @__PURE__ */ f("span", {
									className: "wp-fact-est",
									children: [
										" (plan ~",
										X(l),
										")"
									]
								})]
							}),
							/* @__PURE__ */ d("button", {
								className: "wp-model-x wp-pidreg-x",
								disabled: a || !t || !i,
								title: i ? a ? "evicting…" : `Evict ${e.model_key} — free its VRAM/RAM (reloads on next request)` : "worker infrastructure / external — not an evictable model",
								onClick: () => i && s(e.model_key),
								children: a ? "⏳" : "×"
							})
						]
					}, `${e.model_key || e.host_mode}-${e.pid}-${n}`);
				})
			})] }),
			o.length > 0 && /* @__PURE__ */ f(u, { children: [/* @__PURE__ */ f("div", {
				className: "wp-res-detail-label wp-pidreg-foreign-label",
				title: "Unattributed GPU processes this worker did NOT spawn and could not recognize — hugpy cannot manage their lifecycle. Item I's 22 GiB zombie would surface HERE.",
				children: [
					"Unattributed / foreign GPU processes (",
					o.length,
					")"
				]
			}), /* @__PURE__ */ d("div", {
				className: "wp-pidreg-list wp-pidreg-foreign",
				children: o.map((e, t) => /* @__PURE__ */ f("div", {
					className: "wp-pidreg-row wp-pidreg-foreign-row",
					title: "Unattributed — this worker did not spawn it and could not recognize it. Its VRAM folds into the GPU's used total but it is not a pool resident.",
					children: [
						/* @__PURE__ */ d("span", {
							className: "wp-pidreg-dot wp-pidreg-foreign-dot",
							children: "◈"
						}),
						/* @__PURE__ */ d("span", {
							className: "wp-pidreg-name",
							children: e.name || "unknown process"
						}),
						/* @__PURE__ */ f("span", {
							className: "wp-pidreg-meta",
							children: ["pid ", e.pid]
						}),
						/* @__PURE__ */ d("span", {
							className: "wp-pidreg-vram",
							title: "measured VRAM from nvidia-smi per-PID",
							children: e.mib == null ? "—" : X(Number(e.mib) * c)
						}),
						/* @__PURE__ */ d("button", {
							className: "wp-model-x wp-pidreg-x",
							disabled: !0,
							title: "foreign process — kill requires a privileged worker helper (not yet enabled)",
							children: "×"
						})
					]
				}, `${e.pid}-${t}`))
			})] }),
			/* @__PURE__ */ f("div", {
				className: "wp-pidreg-total",
				title: "Sum of every attributed model row + unattributed row, each sized by its measured nvidia-smi per-PID mib. Mirrors nvidia-smi compute-apps byte-for-byte.",
				children: [
					"Σ measured = ",
					X(h),
					m > 0 && /* @__PURE__ */ f("span", {
						className: "wp-fact-est",
						children: [
							" (",
							X(p),
							" attributed + ",
							X(m),
							" unattributed)"
						]
					}),
					" — mirrors nvidia-smi"
				]
			})
		]
	});
}
var Qr = "estimated from declared placement × file size — not measured";
function $r({ items: e, loadedSet: t, worker: n, resource: r, emptyLabel: i, sizeByKey: a }) {
	return !e || !e.length ? /* @__PURE__ */ d("div", {
		className: "wp-res-empty",
		children: i
	}) : /* @__PURE__ */ d("div", {
		className: "wp-models wp-res-models",
		children: e.map((e, i) => {
			let o = Gr(e, t), s = n.config?.residency?.[e.model_key], c, l = !1;
			if (r === "vram") {
				let t = n.loaded_detail?.[e.model_key] || {}, r = e.n_gpu_layers, i = e.total_layers ?? t.total_layers, o = Kr(e, n, a), s = o ? o.bytes : null, u = !!(o && o.estimated), d = s == null ? null : `${u ? "~" : ""}${X(s)} VRAM`, f = [], p = br(r, i, e.gpu_pct), m = vr(n.spill_by_model?.[e.model_key]);
				if (p.mode) {
					let e = m && m !== p.mode;
					f.push(e ? `⚠ ${yr(p.mode)} (configured ${yr(m)})` : yr(p.mode));
				}
				if (r === -1) f.push(d || "all layers on GPU");
				else if (r != null && r > 0) {
					let e = i == null ? `${r} layer${r === 1 ? "" : "s"} on GPU` : `${r}/${i} layers`;
					f.push(d ? `${d} · ${e}` : e);
				} else e.gpu_pct != null && e.gpu_pct > 0 ? f.push(d ? `${d} · ${Math.round(e.gpu_pct)}% on GPU` : `${Math.round(e.gpu_pct)}% on GPU`) : f.push("host RAM — not in VRAM");
				e.ctx != null && f.push(`ctx ${e.ctx}`), c = f.join(" · "), l = u && s != null;
			} else {
				let t = a?.get?.(e.model_key), n = t && t.eff != null ? t.eff : e.model_bytes ?? e.weight_bytes, r = [];
				if (e.rss_anon_bytes != null) {
					let t = e.rss_file_bytes;
					r.push(t != null && t > 0 ? `${X(e.rss_anon_bytes)} RAM + ${X(t)} cache` : `${X(e.rss_anon_bytes)} RAM`);
				} else e.ram_resident_bytes == null ? e.rss_bytes == null ? o.state === "allocated" ? r.push(n == null ? "allocated (unmeasured)" : `allocated (unmeasured) · ${X(n)} on disk`) : o.state === "idle" ? r.push(n == null ? "idle" : `idle · ${X(n)} on disk`) : r.push(n == null ? "resident · RAM size not measured" : `resident · RAM size not measured · ${X(n)} on disk`) : r.push(`${X(e.rss_bytes)}${e.kind === "slot" ? " RSS" : ""}`) : r.push(`${X(e.ram_resident_bytes)} RAM resident`);
				e.ctx != null && r.push(`ctx ${e.ctx}`), c = r.join(" · ");
			}
			return /* @__PURE__ */ f("span", {
				className: `wp-model wp-st-${o.state}`,
				children: [
					/* @__PURE__ */ d("span", {
						className: `wp-state-pill wp-pill-${o.state}`,
						title: o.title,
						children: o.glyph
					}),
					/* @__PURE__ */ d("span", {
						className: "wp-model-name",
						children: e.model_key
					}),
					s === "static" && /* @__PURE__ */ d("span", {
						className: "wp-slot-res",
						title: "occupant residency: static — locked in, never swapped out",
						children: "🔒"
					}),
					/* @__PURE__ */ d("span", {
						className: `wp-model-facts${l ? " wp-fact-est" : ""}`,
						title: l ? Qr : void 0,
						children: c
					})
				]
			}, (e.model_key || "") + i);
		})
	});
}
function ei({ worker: e, gpuRes: t, comfy: n, sizeByKey: r }) {
	let i = e.vram_used;
	if (i == null || i <= 0) return null;
	let a = 0, o = 0, s = 0, c = !0;
	for (let n of t) {
		let t = Kr(n, e, r);
		t == null ? c = !1 : t.estimated ? (o += t.bytes, s += 1) : a += t.bytes;
	}
	let l = Math.max(i - a - o, 0);
	if (!t.length) return /* @__PURE__ */ f("div", {
		className: "wp-vram-reconcile",
		title: "VRAM is in use but no model is reported as GPU-resident here — in-process models load onto the GPU without saying so. The worker's per-process (nvidia-smi) read will attribute this per model.",
		children: [/* @__PURE__ */ f("span", {
			className: "wp-vram-used",
			children: [X(i), " used"]
		}), " — per-model VRAM pending the worker per-process read"]
	});
	let u = e.vram_attributed_bytes, d = e.vram_unattributed_bytes;
	if (u != null) {
		let e = Math.max(i - u - (d || 0), 0);
		return /* @__PURE__ */ f("div", {
			className: "wp-vram-reconcile",
			title: "Ledger from the worker's per-process (nvidia-smi + pid_registry) read: attributed = hugpy's own served models and CUDA context; unattributed = foreign / non-hugpy GPU use (the honest overhead the bar counts as external); residual = KV cache / activations not tied to a process.",
			children: [
				/* @__PURE__ */ f("span", {
					className: "wp-vram-attributed",
					children: [X(u), " attributed"]
				}),
				d > 0 && /* @__PURE__ */ f("span", {
					className: "wp-vram-overhead",
					children: [
						" + ",
						X(d),
						" unattributed / foreign"
					]
				}),
				e > 0 && /* @__PURE__ */ f("span", {
					className: "wp-vram-overhead",
					children: [
						" + ",
						X(e),
						" KV cache / activations"
					]
				}),
				/* @__PURE__ */ f("span", {
					className: "wp-vram-used",
					children: [
						" = ",
						X(i),
						" used"
					]
				})
			]
		});
	}
	return /* @__PURE__ */ f("div", {
		className: "wp-vram-reconcile",
		title: "Measured weights come from the worker's per-process read. " + (s > 0 ? `${s} model${s === 1 ? " is" : "s are"} shown as a separate estimated term (${Qr}) — never added into the measured sum. ` : "") + "The remainder is KV cache, CUDA context, and any other GPU use — a residual, not an independently measured number.",
		children: [
			`${c ? "" : "~"}${X(a)} measured weights`,
			s > 0 && /* @__PURE__ */ f("span", {
				className: "wp-vram-est",
				children: [
					" + ~",
					X(o),
					" estimated (",
					s,
					" model",
					s === 1 ? "" : "s",
					")"
				]
			}),
			l > 0 && /* @__PURE__ */ f("span", {
				className: "wp-vram-overhead",
				children: [
					" + ",
					X(l),
					" KV cache / context / other"
				]
			}),
			/* @__PURE__ */ f("span", {
				className: "wp-vram-used",
				children: [
					" = ",
					X(i),
					" used"
				]
			})
		]
	});
}
function ti({ icon: e, name: t, used: n, total: r, count: i, active: a, onClick: o, disabled: s, physicalTotal: c, bar: l, extraTerm: u }) {
	let p = l?.semantics, m = !!l && p !== "legacy", h = m && l.used != null ? l.used : n, g = m && l.total != null ? l.total : r, _ = m && l.rawUsed != null ? l.rawUsed : h, v = m && !!l.overLimit, y = m && l.encroachment || 0, b = g != null && g > 0 && h != null, x = b ? Math.min(Math.max(h / g * 100, 0), 100) : 0, S = m && l.remaining != null ? l.remaining : b ? Math.max(g - h, 0) : null, C = p === "central" || c != null && g != null && g < c, w = !!l && p === "legacy", T = b ? `${t}: ${X(h)} used / ${X(g)}` + (C ? ` central limit${c == null ? "" : ` (${X(c)} physical)`}` : " physical") + (S == null ? "" : ` · ${X(S)} free`) + (y > 0 ? `\n\n⚠ ${X(y)} of the worker's budget is encroached by external (non-hugpy) usage that exceeded the physical headroom above the limit.` : "") + (v ? `\n\n⛔ OVER LIMIT by ${X(l.overBy || Math.max(_ - g, 0))} — raw usage ${X(_)} exceeds the ${X(g)} budget. The bar is pinned full; admission is paused until it drains.` : "") + (w ? "\n\n(legacy estimate — this worker predates the honest budget-bar inputs; the figure mixes physical and limit universes.)" : "") + " — click for resident detail" : `${t}: no data`;
	return /* @__PURE__ */ f("button", {
		type: "button",
		disabled: s,
		className: `wp-gpu wp-res-chip${a ? " wp-res-active" : ""}${s ? " wp-res-disabled" : ""}${v ? " wp-res-overlimit" : ""}`,
		onClick: o,
		title: T,
		children: [
			/* @__PURE__ */ f("div", {
				className: "wp-gpu-head",
				children: [
					/* @__PURE__ */ f("span", {
						className: "wp-res-name",
						children: [
							e,
							" ",
							t
						]
					}),
					b ? /* @__PURE__ */ f("em", { children: [
						" · ",
						X(h),
						" / ",
						X(g)
					] }) : /* @__PURE__ */ d("em", {
						className: "wp-res-nodata",
						children: " · —"
					}),
					C && !w && /* @__PURE__ */ d("span", {
						className: "wp-res-limit",
						title: `Central limit ${X(g)}${c == null ? "" : ` of ${X(c)} physical`} — the budget the fleet respects.`,
						children: "limit"
					}),
					w && /* @__PURE__ */ d("span", {
						className: "wp-res-legacy",
						title: "This worker predates the honest budget-bar inputs (t13/t14). The bar is a legacy estimate mixing physical and central-limit figures — update the worker to get the true bar.",
						children: "legacy est."
					}),
					v && /* @__PURE__ */ d("span", {
						className: "wp-res-warn",
						title: `Over the ${X(g)} budget by ${X(l.overBy || Math.max(_ - g, 0))} — raw usage ${X(_)}. Admission is paused until it drains.`,
						children: "⛔ over"
					}),
					!v && y > 0 && /* @__PURE__ */ f("span", {
						className: "wp-res-warn wp-res-encroach",
						title: `${X(y)} of this worker's budget is taken by external (non-hugpy) usage that spilled past the physical headroom above the limit.`,
						children: ["⚠ encroached ", X(y)]
					}),
					i != null && i > 0 && /* @__PURE__ */ d("span", {
						className: "wp-res-count",
						children: i
					}),
					!s && /* @__PURE__ */ d("span", {
						className: "wp-res-caret",
						children: a ? "▾" : "▸"
					})
				]
			}),
			/* @__PURE__ */ f("div", {
				className: `wp-vram-bar${v ? " wp-vram-bar-over" : ""}`,
				children: [b && /* @__PURE__ */ d("div", {
					className: `wp-vram-fill${v ? " wp-vram-fill-over" : ""}`,
					style: { width: `${x}%` }
				}), b && y > 0 && !v && /* @__PURE__ */ d("div", {
					className: "wp-vram-encroach",
					style: { width: `${Math.min(Math.max(y / g * 100, 0), 100)}%` }
				})]
			}),
			u && /* @__PURE__ */ d("div", {
				className: "wp-res-extraterm",
				children: u
			})
		]
	});
}
function ni({ worker: e, models: t, onApproveEvictions: n, onEvict: r }) {
	let [i, a] = l(null), o = Array.isArray(e.gpus) ? e.gpus : [], s = e.storage || {}, c = /* @__PURE__ */ new Map();
	for (let e of t || []) {
		let t = e && (e.model_key || e.key), n = String(e && e.framework || "").toLowerCase();
		t && (n === "gguf" || n === "llama_cpp") && e.effective_bytes != null && c.set(t, {
			eff: e.effective_bytes,
			effGguf: e.effective_gguf
		});
	}
	let u = 2 ** 30, p = e.limits || {}, m = Array.isArray(e.allocations) ? e.allocations.filter((e) => e && e.model_key) : [], h = new Set(e.loaded_models || []), g = (e) => e.device === "cuda" || e.vram_bytes != null && e.vram_bytes > 0 || e.kind === "slot" && e.n_gpu_layers != null && e.n_gpu_layers !== 0 || e.gpu_pct != null && e.gpu_pct > 0, _ = (e) => e.kind === "slot" ? !0 : e.device === "cuda" ? !1 : e.gpu_pct == null || e.gpu_pct < 99, v = m.filter(g), y = m.filter(_), b = !!e.comfy?.available, x = o.length === 1 ? o[0].name || "VRAM" : o.length > 1 ? `VRAM · ${o.length} GPUs` : "VRAM", S = e.vram_total != null && e.vram_total > 0 || o.length > 0, C = p.ram_max_gib == null ? e.ram_total : p.ram_max_gib * u, w = p.gpu_mem_gib == null ? e.vram_total : p.gpu_mem_gib * u, T = s.reported ? Math.max((s.disk_total ?? 0) - (s.disk_free ?? 0), 0) : null, E = s.reported ? s.disk_total ?? null : null, D = (t) => {
		let n = e[`${t}bar_semantics`];
		if (n != null) return {
			semantics: n,
			used: e[`${t}bar_used`],
			total: e[`${t}bar_total`],
			remaining: e[`${t}bar_remaining`],
			rawUsed: e[`${t}bar_raw_used`],
			encroachment: e[`${t}bar_encroachment`] || 0,
			overLimit: !!e[`${t}bar_over_limit`],
			overBy: e[`${t}bar_over_by`] || 0
		};
	}, O = D("ram_"), k = D("vram_"), A = e.vram_unattributed_bytes, j = k && k.semantics !== "legacy" && A != null && A > 0 ? `+ ${X(A)} unattributed / overhead` : null, M = (e) => a((t) => t === e ? null : e);
	return /* @__PURE__ */ f("div", {
		className: "wp-resources-wrap",
		children: [
			/* @__PURE__ */ f("div", {
				className: "wp-resources",
				children: [
					/* @__PURE__ */ d(ti, {
						icon: "🖥",
						name: x,
						used: e.vram_used,
						total: w,
						physicalTotal: e.vram_total,
						count: v.length,
						active: i === "vram",
						bar: k,
						extraTerm: j,
						disabled: !S,
						onClick: () => M("vram")
					}),
					/* @__PURE__ */ d(ti, {
						icon: "🧠",
						name: "RAM",
						used: e.ram_used,
						total: C,
						physicalTotal: e.ram_total,
						count: y.length,
						active: i === "ram",
						bar: O,
						onClick: () => M("ram")
					}),
					/* @__PURE__ */ d(ti, {
						icon: "💾",
						name: "Storage",
						used: T,
						total: E,
						count: s.reported ? s.models?.length ?? 0 : null,
						active: i === "storage",
						disabled: !s.reported,
						onClick: () => M("storage")
					})
				]
			}),
			i === "vram" && /* @__PURE__ */ f("div", {
				className: "wp-res-detail",
				children: [
					o.length > 1 && /* @__PURE__ */ d(Xr, { gpus: o }),
					/* @__PURE__ */ f("div", {
						className: "wp-res-detail-label",
						children: [
							"Models in VRAM — allocation each (",
							v.length,
							")"
						]
					}),
					/* @__PURE__ */ d($r, {
						items: v,
						loadedSet: h,
						worker: e,
						resource: "vram",
						sizeByKey: c,
						emptyLabel: "No models are using this GPU's VRAM right now."
					}),
					/* @__PURE__ */ d(ei, {
						worker: e,
						gpuRes: v,
						comfy: b,
						sizeByKey: c
					}),
					/* @__PURE__ */ d(Zr, {
						worker: e,
						onEvict: r
					})
				]
			}),
			i === "ram" && /* @__PURE__ */ f("div", {
				className: "wp-res-detail",
				children: [
					/* @__PURE__ */ f("div", {
						className: "wp-res-detail-label",
						children: [
							"Models in host RAM (",
							y.length,
							")"
						]
					}),
					/* @__PURE__ */ d($r, {
						items: y,
						loadedSet: h,
						worker: e,
						resource: "ram",
						sizeByKey: c,
						emptyLabel: "No models resident in host RAM right now."
					}),
					b && /* @__PURE__ */ d("div", {
						className: "wp-budget-comfy",
						title: "ComfyUI is an adopted, externally-owned process. Its checkpoints load on-demand inside ComfyUI and are NOT worker pool residents; its memory folds into the used/total above as an out-of-pool reservation.",
						children: "🧩 ComfyUI attached — external (out-of-pool)"
					})
				]
			}),
			i === "storage" && /* @__PURE__ */ d("div", {
				className: "wp-res-detail",
				children: s.reported ? /* @__PURE__ */ d(Yr, {
					worker: e,
					onApproveEvictions: n,
					sizeByKey: c,
					detailsOnly: !0
				}) : /* @__PURE__ */ d("div", {
					className: "wp-res-empty",
					children: "This worker hasn’t reported a storage survey yet."
				})
			})
		]
	});
}
//#endregion
//#region src/components/WorkersPanel/SpillBadge.jsx
function ri({ spill: e }) {
	if (!e || !e.mode) return null;
	let t = e.free_vram_bytes;
	return /* @__PURE__ */ f("span", {
		className: "wp-spill",
		title: "GPU/CPU split mode reported by the worker",
		children: [
			"spill: ",
			e.mode === "auto" ? "autofit" : e.mode,
			t != null && /* @__PURE__ */ f("em", { children: [
				" · ",
				X(t),
				" VRAM free"
			] })
		]
	});
}
//#endregion
//#region src/components/WorkersPanel/useNarrowContainer.js
function ii(e, t) {
	let [n, r] = l(!1);
	return o(() => {
		let n = e.current;
		if (!n || typeof ResizeObserver > "u") return;
		let i = new ResizeObserver((e) => {
			for (let i of e) {
				let e = i.contentRect ? i.contentRect.width : n.clientWidth;
				r(e > 0 && e < t);
			}
		});
		return i.observe(n), () => i.disconnect();
	}, [e, t]), n;
}
//#endregion
//#region src/components/WorkersPanel/WorkerLoadTable.jsx
function ai({ models: e, allocation: t, workerId: n, worker: r, onAllocate: a, onCancel: p }) {
	let m = (e) => e.model_key ?? e.key, h = (e) => {
		let t = e.size_bytes == null ? e.effective_bytes : e.size_bytes;
		return t == null ? null : Number(t);
	}, g = (e) => {
		let t = String(e.framework || "").toLowerCase();
		return t === "gguf" || t === "llama_cpp";
	}, [_, v] = l(() => /* @__PURE__ */ new Set()), [y, b] = Y("hugpy.sess.wp.load.sort", "name"), [x, S] = Y("hugpy.sess.wp.load.dir", "asc"), [C, w] = Y("hugpy.sess.wp.load.task", ""), [T, E] = l(!1), [D, O] = l(""), [k, A] = l(!1), [j, M] = l({}), [N, P] = l([]), [F, I] = l(() => /* @__PURE__ */ new Set()), [L, R] = l({}), z = c(!0), B = c(/* @__PURE__ */ new Set()), V = i((e) => (t[m(e)] || []).filter((e) => e.id !== n), [t, n]), ee = i(async () => {
		try {
			let e = await K("/api/llm/central-provisioning");
			z.current && M(e && typeof e == "object" ? e : {});
		} catch {}
	}, []), te = i(async () => {
		try {
			let e = await K("/api/jobs");
			if (!z.current) return;
			let t = Array.isArray(e) ? e : [];
			P(t);
			let n = new Set(t.filter((e) => e && (e.status === "queued" || e.status === "running")).map((e) => e.model_key)), r = !1;
			B.current.forEach((e) => {
				n.has(e) || (r = !0);
			}), B.current = n, r && ee();
		} catch {}
	}, [ee]);
	o(() => {
		z.current = !0;
		let e = [], t = (t, n) => {
			let r = () => {
				Promise.resolve(t()).finally(() => {
					z.current && e.push(setTimeout(r, n));
				});
			};
			r();
		};
		return t(te, 2500), t(ee, 1e4), () => {
			z.current = !1, e.forEach(clearTimeout);
		};
	}, [ee, te]);
	let H = i((e) => N.find((t) => t && t.model_key === e && (t.status === "queued" || t.status === "running")) || null, [N]), ne = i((e) => {
		let t = m(e), n = H(t);
		if (n) return {
			state: "downloading",
			job: n,
			reason: null
		};
		let r = j[t];
		return r ? {
			state: r.state || "absent",
			reason: r.reason ?? null
		} : {
			state: "absent",
			reason: null
		};
	}, [j, H]), re = async (e) => {
		I((t) => new Set(t).add(e)), R((t) => {
			let n = { ...t };
			return delete n[e], n;
		});
		try {
			await K(`/api/models/${encodeURIComponent(e)}/download`, { method: "POST" }), await te();
		} catch (t) {
			z.current && R((n) => ({
				...n,
				[e]: t.message || "download failed"
			}));
		} finally {
			z.current && I((t) => {
				let n = new Set(t);
				return n.delete(e), n;
			});
		}
	}, U = s(() => [...new Set(e.flatMap(Gn))].sort(), [e]), W = [
		{
			key: "name",
			label: "Model",
			get: (e) => e.name || m(e)
		},
		{
			key: "central",
			label: "Central",
			get: (e) => ne(e).state
		},
		{
			key: "task",
			label: "Task",
			get: (e) => Wn(e) || "—"
		},
		{
			key: "framework",
			label: "Engine",
			get: (e) => e.framework || "—"
		},
		{
			key: "size",
			label: "Size",
			get: (e) => h(e) ?? -1,
			num: !0
		},
		{
			key: "ctx",
			label: "Ctx",
			get: (e) => e.model_max_length || 0,
			num: !0
		},
		{
			key: "elsewhere",
			label: "Allocated on",
			get: (e) => V(e).map((e) => e.name).join(", ")
		}
	], G = s(() => {
		let t = D.trim().toLowerCase(), n = e;
		t && (n = n.filter((e) => (e.name || m(e)).toLowerCase().includes(t))), C && (n = n.filter((e) => Gn(e).includes(C)));
		let r = W.find((e) => e.key === y) || W[0], i = x === "asc" ? 1 : -1;
		return [...n].sort((e, t) => {
			let n = r.get(e), a = r.get(t);
			return r.num ? (Number(n) - Number(a)) * i : String(n).localeCompare(String(a)) * i;
		});
	}, [
		e,
		D,
		C,
		y,
		x,
		t,
		n
	]), ie = i((e) => ne(e).state === "ready" && (T || V(e).length === 0), [
		ne,
		T,
		V
	]), ae = s(() => G.filter(ie).map(m), [G, ie]), oe = ae.length > 0 && ae.every((e) => _.has(e)), se = (e) => v((t) => {
		let n = new Set(t);
		return n.has(e) ? n.delete(e) : n.add(e), n;
	}), ce = () => v(oe ? /* @__PURE__ */ new Set() : new Set(ae)), le = (e) => {
		y === e ? S(x === "asc" ? "desc" : "asc") : (b(e), S("asc"));
	}, ue = (e) => y === e ? x === "asc" ? " ▲" : " ▼" : "";
	return /* @__PURE__ */ f("div", {
		className: "wp-loadtable",
		children: [
			/* @__PURE__ */ f("div", {
				className: "wp-loadtable-bar",
				children: [
					/* @__PURE__ */ d("input", {
						className: "wp-loadtable-q",
						placeholder: "filter models…",
						value: D,
						onChange: (e) => O(e.target.value),
						autoFocus: !0
					}),
					/* @__PURE__ */ f("select", {
						className: "wp-loadtable-task",
						value: C,
						onChange: (e) => w(e.target.value),
						title: "Filter by task — matches ANY task a model advertises, not just its primary",
						children: [/* @__PURE__ */ d("option", {
							value: "",
							children: "All tasks"
						}), U.map((e) => /* @__PURE__ */ d("option", {
							value: e,
							children: e
						}, e))]
					}),
					/* @__PURE__ */ f("label", {
						className: "wp-breaker",
						title: "Anti-duplicate is ON by default: models already on another worker are locked. Flip this to deliberately allocate a duplicate (replicate across workers — e.g. scene fan-out).",
						children: [/* @__PURE__ */ d("input", {
							type: "checkbox",
							checked: T,
							onChange: (e) => {
								E(e.target.checked), v(/* @__PURE__ */ new Set());
							}
						}), "⚡ allow duplicate allocation"]
					}),
					/* @__PURE__ */ d("button", {
						className: "wp-load-cancel",
						title: "Cancel",
						onClick: p,
						children: "×"
					})
				]
			}),
			/* @__PURE__ */ d("div", {
				className: "wp-loadtable-scroll",
				children: /* @__PURE__ */ f("table", {
					className: "wp-loadtable-t",
					children: [/* @__PURE__ */ d("thead", { children: /* @__PURE__ */ f("tr", { children: [/* @__PURE__ */ d("th", {
						className: "wp-lt-check",
						children: /* @__PURE__ */ d("input", {
							type: "checkbox",
							checked: oe,
							onChange: ce,
							disabled: ae.length === 0,
							title: "Select all eligible"
						})
					}), W.map((e) => /* @__PURE__ */ f("th", {
						className: `wp-lt-sortable${e.num ? " wp-lt-num" : ""}`,
						onClick: () => le(e.key),
						children: [e.label, ue(e.key)]
					}, e.key))] }) }), /* @__PURE__ */ f("tbody", { children: [G.length === 0 && /* @__PURE__ */ d("tr", { children: /* @__PURE__ */ d("td", {
						colSpan: W.length + 1,
						className: "wp-none",
						children: "No models to allocate."
					}) }), G.map((e) => {
						let t = m(e), n = V(e), r = n.length > 0 && !T, i = ne(e);
						return /* @__PURE__ */ f("tr", {
							className: r ? "wp-lt-locked" : "",
							children: [
								/* @__PURE__ */ d("td", {
									className: "wp-lt-check",
									children: /* @__PURE__ */ d("input", {
										type: "checkbox",
										checked: _.has(t),
										disabled: !ie(e),
										onChange: () => se(t)
									})
								}),
								/* @__PURE__ */ d("td", { children: e.name || t }),
								/* @__PURE__ */ f("td", {
									className: "wp-cprov-cell",
									children: [
										i.state === "ready" && /* @__PURE__ */ d("span", {
											className: "wp-state-pill wp-cprov-ready",
											title: "fully on central disk — allocatable",
											children: "✓ on disk"
										}),
										i.state === "downloading" && (() => {
											let e = i.job || {}, t = e.total_bytes && e.total_bytes > 0 && e.progress != null ? Math.round(e.progress * 100) : null;
											return /* @__PURE__ */ f("span", {
												className: "wp-state-pill wp-cprov-downloading",
												title: e.total_bytes ? `downloading to central — ${X(e.downloaded_bytes)} / ${X(e.total_bytes)}` : "downloading to central…",
												children: ["⏳ ", t == null ? "…" : `${t}%`]
											});
										})(),
										i.state === "incomplete" && /* @__PURE__ */ f(u, { children: [/* @__PURE__ */ d("span", {
											className: "wp-state-pill wp-cprov-incomplete",
											title: "directory present but files incomplete — download to finish",
											children: "◐ incomplete"
										}), /* @__PURE__ */ d("button", {
											className: "wp-cprov-dl",
											disabled: F.has(t),
											onClick: () => re(t),
											title: "Finish downloading this model to central disk",
											children: F.has(t) ? "…" : "⬇ Download"
										})] }),
										i.state === "absent" && /* @__PURE__ */ f(u, { children: [/* @__PURE__ */ d("span", {
											className: "wp-state-pill wp-cprov-absent",
											title: "not on central disk / not in the manifest",
											children: "○ not on central"
										}), /* @__PURE__ */ d("button", {
											className: "wp-cprov-dl",
											disabled: F.has(t),
											onClick: () => re(t),
											title: "Download this model to central disk",
											children: F.has(t) ? "…" : "⬇ Download"
										})] }),
										i.state === "error" && /* @__PURE__ */ d("span", {
											className: "wp-state-pill wp-cprov-error",
											title: i.reason || "central could not resolve this model",
											children: "⚠ error"
										}),
										L[t] && /* @__PURE__ */ d("span", {
											className: "wp-cprov-err",
											title: L[t],
											children: "failed"
										})
									]
								}),
								/* @__PURE__ */ f("td", {
									className: "wp-lt-muted",
									title: Gn(e).join(", "),
									children: [Wn(e), Gn(e).length > 1 ? ` +${Gn(e).length - 1}` : ""]
								}),
								/* @__PURE__ */ d("td", {
									className: "wp-lt-muted",
									children: e.framework || "—"
								}),
								/* @__PURE__ */ d("td", {
									className: "wp-lt-num wp-lt-size",
									title: h(e) == null ? "size unknown — not on disk / not reported by the feed" : g(e) ? `effective quant${e.effective_gguf ? ` ${e.effective_gguf}` : ""} — ${X(h(e))} (the ONE quant that serves, not the all-quants dir sum)` : `${X(h(e))} on disk`,
									children: h(e) == null ? "—" : X(h(e))
								}),
								/* @__PURE__ */ d("td", {
									className: "wp-lt-num",
									children: e.model_max_length || "—"
								}),
								/* @__PURE__ */ d("td", {
									className: n.length && T ? "wp-lt-dup" : "wp-lt-muted",
									children: n.length ? `${T ? "⚡ " : ""}${n.map((e) => e.name).join(", ")}` : "—"
								})
							]
						}, t);
					})] })]
				})
			}),
			r && (r.vram_total != null || r.ram_total != null) && (() => {
				let t = (Array.isArray(r.allocations) ? r.allocations.filter((e) => e && e.model_key) : []).reduce((e, t) => e + (t.vram_bytes != null && t.vram_bytes > 0 ? t.vram_bytes : 0), 0), n = [..._].reduce((t, n) => {
					let r = e.find((e) => m(e) === n);
					return t + (r && h(r) || 0);
				}, 0);
				return /* @__PURE__ */ f("div", {
					className: "wp-load-headroom",
					title: "Measured headroom on this worker — what a model is fit-checked against before it loads. VRAM free is nvidia-smi; RAM free is the worker's measured free (reserve-adjusted). 'resident' sums measured per-model VRAM, never file sizes. Per-model RAM RSS for in-process models is not attributable yet (page-cache/arena gap), so RAM is shown at the worker level.",
					children: [
						/* @__PURE__ */ d("span", {
							className: "wp-load-headroom-label",
							children: "Headroom (measured):"
						}),
						r.vram_total != null && /* @__PURE__ */ f("span", {
							className: "wp-load-headroom-item",
							title: "Free VRAM from nvidia-smi (measured)",
							children: [
								"🖥 ",
								X(r.vram_free),
								" free / ",
								X(r.vram_total),
								" VRAM",
								t > 0 && /* @__PURE__ */ f("em", { children: [
									" · ",
									X(t),
									" resident"
								] })
							]
						}),
						r.ram_total != null && /* @__PURE__ */ f("span", {
							className: "wp-load-headroom-item",
							title: "Worker's measured free RAM (reserve-adjusted). Per-model RAM attribution is not available yet.",
							children: [
								"🧠 ",
								X(r.free_ram),
								" free / ",
								X(r.ram_total),
								" RAM"
							]
						}),
						_.size > 0 && n > 0 && /* @__PURE__ */ f("span", {
							className: "wp-load-headroom-sel",
							title: "Total on-disk size of the checked models (effective quant for GGUF). Resident memory use is typically at or below this; the fit-guard makes the final call per model.",
							children: [
								"selected ",
								_.size,
								": ",
								X(n),
								" on disk"
							]
						})
					]
				});
			})(),
			/* @__PURE__ */ f("div", {
				className: "wp-loadtable-actions",
				children: [/* @__PURE__ */ d("span", {
					className: "wp-load-hint",
					title: "Each model is checked against this worker's free VRAM + RAM + disk before it loads — a model that won't fit is refused, not OOM'd",
					children: "✓ fit-guarded"
				}), /* @__PURE__ */ d("button", {
					className: "wp-loadtable-go",
					disabled: k || _.size === 0,
					onClick: async () => {
						let e = [..._];
						if (e.length !== 0) {
							A(!0);
							try {
								await a(e, T), v(/* @__PURE__ */ new Set());
							} finally {
								A(!1);
							}
						}
					},
					children: k ? "Allocating…" : `Allocate ${_.size} model${_.size === 1 ? "" : "s"} to this worker`
				})]
			})
		]
	});
}
//#endregion
//#region src/components/WorkersPanel/WorkerRow.jsx
function oi({ worker: e, models: t, allocation: r, onAssign: a, onLoad: p, onUnassign: m, onRemove: h, onFree: g, onFreeAll: _, onFreeRam: v, onRestart: y, onUpdate: b = null, onAdmit: x, onBlock: S, onSetPool: C, onSetLimits: w, onSetConfig: T, onSetResidency: E, onSetResidencyMany: D, onSetAllocMany: O, onTogglePin: k, onPinAll: A, onUnpinAll: j, onReap: M, onApproveEvictions: N, onEvict: P, onAllocateMany: F, onRefresh: I = null, applying: L = !1, restarting: R = !1, updating: z = !1, blockedKeys: B = null, onToggleBlock: V = null }) {
	let [ee, te] = l(""), [H, ne] = l({}), [re, U] = l(null), W = c(null), [G, ie] = l({}), [ae, oe] = l({}), se = s(() => Object.keys(e.bnb_available || {}), [e.bnb_available]), [ce, le] = l({}), ue = i(async (t, n) => {
		le((e) => ({
			...e,
			[t]: n === null ? void 0 : n
		}));
		try {
			await K(`/api/llm/workers/${encodeURIComponent(e.id)}/moe`, {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({
					model_key: t,
					value: n
				})
			});
		} catch (e) {
			le((e) => {
				let n = { ...e };
				return delete n[t], n;
			}), alert(`MoE split failed: ${e.message}`);
		}
		typeof I == "function" && I();
	}, [e.id, I]), de = s(() => Object.keys(e.moe_capable || {}), [e.moe_capable]), fe = i(async (t) => {
		if (de.length) {
			try {
				await K(`/api/llm/workers/${encodeURIComponent(e.id)}/moe`, {
					method: "POST",
					headers: { "Content-Type": "application/json" },
					body: JSON.stringify({
						all: !0,
						value: t
					})
				});
			} catch (e) {
				alert(`MoE bulk failed: ${e.message}`);
			}
			le({}), typeof I == "function" && I();
		}
	}, [
		e.id,
		de,
		I
	]), pe = i(async (t) => {
		let n = Object.keys(e.bnb_available || {});
		if (n.length && !(t && !confirm(`Load ${n.length} model(s) on ${e.name || "this worker"} at 4-bit (bitsandbytes nf4)?

Each is re-priced at ~30% of its fp16 size, so their allocations re-derive — several may move from RAM onto the GPU. Quantization costs some output quality.`))) {
			oe((e) => {
				let r = { ...e };
				for (let e of n) r[e] = t;
				return r;
			});
			try {
				await K(`/api/llm/workers/${encodeURIComponent(e.id)}/bnb`, {
					method: "POST",
					headers: { "Content-Type": "application/json" },
					body: JSON.stringify({
						all: !0,
						enabled: t
					})
				});
			} catch (e) {
				oe({}), alert(`4-bit bulk failed: ${e.message}`);
			}
			typeof I == "function" && I();
		}
	}, [
		e.id,
		e.name,
		e.bnb_available,
		I
	]), me = i(async (t, n) => {
		oe((e) => ({
			...e,
			[t]: n
		}));
		try {
			await K(`/api/llm/workers/${encodeURIComponent(e.id)}/bnb`, {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({
					model_key: t,
					enabled: n
				})
			}), typeof I == "function" && I();
		} catch (e) {
			oe((e) => {
				let n = { ...e };
				return delete n[t], n;
			}), alert(`Specialization failed: ${e.message}`);
		}
	}, [e.id, I]), [he, ge] = l({}), _e = c(/* @__PURE__ */ new Set()), ve = i((e) => {
		_e.current.has(e) || (_e.current.add(e), ge((t) => ({
			...t,
			[e]: { loading: !0 }
		})), K(`/api/llm/serving/${encodeURIComponent(e)}`).then((t) => ge((n) => ({
			...n,
			[e]: t || {}
		}))).catch((t) => {
			_e.current.delete(e), ge((n) => ({
				...n,
				[e]: { error: t.message }
			}));
		}));
	}, []), [ye, be] = l(null), [xe, Se] = l(!1), [Ce, we] = l({
		ram_max_gib: "",
		gpu_mem_gib: "",
		disk_cache_gib: "",
		threads: ""
	}), [q, Te] = l(null), [Ee, De] = l(!1), [Oe, ke] = l(null), [Ae, je] = l(""), [Me, Ne] = l(""), [Pe, Fe] = Y("hugpy.sess.wp.serving.sort", "name"), [Ie, Le] = Y("hugpy.sess.wp.serving.dir", "asc"), [Re, ze] = l(() => /* @__PURE__ */ new Set()), Be = i((e) => {
		ze((t) => {
			let n = new Set(t);
			return n.has(e) ? n.delete(e) : n.add(e), n;
		});
	}, []), Ve = i(() => ze(/* @__PURE__ */ new Set()), []), [He, Ue] = l(!1), J = i((t, n) => {
		U(null), ie((e) => ({
			...e,
			[t]: n
		})), Promise.resolve(a(e, t, n)).finally(() => ie((e) => {
			let n = { ...e };
			return delete n[t], n;
		}));
	}, [a, e]), We = i(async () => {
		Te("checking");
		try {
			let t = await K(`/api/llm/workers/${encodeURIComponent(e.id)}/health`);
			Te(t);
		} catch (e) {
			Te({
				reachable: !1,
				error: e.message
			});
		}
	}, [e.id]), Ge = s(() => {
		let n = new Set(e.models || []);
		return t.filter((e) => !n.has(e.model_key ?? e.key));
	}, [t, e.models]), Ke = i((e) => t.find((t) => (t.model_key ?? t.key) === e)?.name || e, [t]), qe = i((e) => {
		let n = t.find((t) => (t.model_key ?? t.key) === e);
		if (!n) return null;
		let r = String(n.framework || "").toLowerCase(), i = n.size_bytes == null ? n.effective_bytes : n.size_bytes;
		return {
			bytes: i == null ? null : Number(i),
			isGguf: r === "gguf" || r === "llama_cpp",
			effGguf: n.effective_gguf
		};
	}, [t]), Je = new Set(e.loaded_models || []), Ye = new Set(e.provisioning || []), Xe = new Set(e.loading || []), Ze = Array.isArray(e.allocations) ? Object.fromEntries(e.allocations.filter((e) => e && e.model_key).map((e) => [e.model_key, e])) : null, Qe = new Set((e.slots || []).filter((e) => e && e.model_key && e.healthy).map((e) => e.model_key)), $e = new Set((e.slots || []).filter((e) => e && e.model_key && e.busy).map((e) => e.model_key)), et = e.models_local ? new Set(e.models_local) : null, tt = (n) => {
		let r = Ye.has(n), i = Xe.has(n), a = e.spill_by_model?.[n], o = e.model_alloc_modes?.[n] || null, s = !!e.config?.pinned?.[n], c = Ze ? Ze[n] : void 0, l, u, d, f = !1;
		if (Ze) {
			l = !!c && c.kind === "slot" && !!c.healthy, u = l && !!c.busy;
			let e = !!c && c.kind === "ram" && Wr(c);
			d = l || e, f = !!c && c.kind === "ram" && !e;
		} else d = Je.has(n), l = Qe.has(n), u = l && $e.has(n);
		let p = e.provision_progress?.[n], m = p && p.total_bytes > 0 ? Math.min(99, Math.round(100 * (p.done_bytes ?? p.frac * p.total_bytes) / p.total_bytes)) : null, h = t.find((e) => (e.model_key ?? e.key) === n), g = !!h && (h.status === "installed" || (h.dir_bytes ?? 0) > 0), _ = et != null && et.has(n), v = !r && !i && !d && !f && !_ && !g, y = r ? "pulling" : i ? "heating" : d ? l ? u ? "answering" : "serving" : "loaded" : f ? "idle" : _ ? "hot" : g ? "cold" : "missing", b = r ? `downloading files from central/HF${m == null ? "" : ` — ${m}%`}` : i ? "weights loading into VRAM/RAM right now" : d ? l ? u ? "actively processing a request right now" : "hosted in a slot on this worker — routable, crash-isolated server child" : "resident in this worker's own process — dedicated to this machine, not in a slot" : f ? "a runner is cached on this worker but holds NO measured VRAM/RAM and hasn't served recently — not actually resident (a reconcile warm may have just instantiated it, or its weights were freed). Its measured residency is the truth here, not the runner-cache membership." : y === "hot" ? "files on THIS worker's drive, not loaded — weights lift into VRAM/RAM on the first request or the next warm pass" : y === "cold" ? "files on CENTRAL storage (llm_storage), not on this worker's drive yet — the normal resting state for discovered models. Assignment is attribution, not a transfer order: the weights copy to the worker on the FIRST CALL (lazy download). Nothing is transferring right now." : "files found NEITHER on this worker NOR on central storage — a stale catalog row, a phantom assignment, or a download that never happened. Serving will fail until the files are downloaded (or the key is unassigned).", x = e.load_reports?.[n], S = !!x && (y === "hot" || y === "cold" || y === "missing" || y === "idle" || y === "heating"), C = S && (x?.ok === !1 || x?.fit === !1), w = S && !C && x?.ok === !0;
		return {
			isPulling: r,
			isHeating: i,
			override: a,
			isPinned: s,
			alloc: c,
			inSlot: l,
			derivedMode: o,
			isAnswering: u,
			isServing: d,
			isIdleResident: f,
			pct: m,
			isMissing: v,
			state: y,
			stateTitle: b,
			loadReport: x,
			loadFailed: C,
			loadStale: w
		};
	}, nt = {
		answering: 8,
		serving: 7,
		loaded: 6,
		heating: 5,
		pulling: 4,
		idle: 3,
		hot: 2,
		cold: 1,
		missing: 0
	}, rt = {
		task: {
			label: "Task",
			sortable: !0,
			cls: "wp-lt-muted",
			title: ({ m: e }) => e ? Gn(e).join(", ") : "",
			render: ({ m: e }) => e ? `${Wn(e)}${Gn(e).length > 1 ? ` +${Gn(e).length - 1}` : ""}` : "—"
		},
		framework: {
			label: "Engine",
			sortable: !0,
			cls: "wp-lt-muted",
			render: ({ m: e }) => e?.framework || "—"
		},
		seat: {
			label: "Seat",
			sortable: !1,
			cls: "wp-servtable-seat",
			render: ({ d: e }) => e.alloc ? /* @__PURE__ */ d("span", {
				className: `wp-alloc-kind wp-alloc-${e.alloc.kind}`,
				title: e.alloc.kind === "slot" ? "allocated a slot (a resource seat) on this worker" : "resident in the worker’s own process (RAM allocation)",
				children: e.alloc.kind === "slot" ? "🎰 slot" : "🧠 RAM"
			}) : /* @__PURE__ */ d("span", {
				className: "wp-lt-muted",
				children: "—"
			})
		},
		residency: {
			label: "Residency",
			sortable: !1,
			cls: "",
			render: ({ key: t }) => e.config && E ? (() => {
				let n = e.config.residency?.[t] === "static" ? "static" : "on-demand", r = n === "static" ? " (locked seat — never swapped out; permanent with 📌 pin)" : " (default — loads on call; holds its slot until another model needs the seat)";
				return /* @__PURE__ */ d("button", {
					className: `wp-residency wp-residency-${n}`,
					disabled: L,
					title: L ? "Agent is restarting to apply the previous change — retry in a few seconds." : `Residency policy: ${n}${r} — click to choose (a change applies via a ~5s agent restart)`,
					onClick: () => be(ye === t ? null : t),
					children: n === "static" ? "🔒 static" : "⏲ on-demand"
				});
			})() : /* @__PURE__ */ d("span", {
				className: "wp-lt-muted",
				children: "—"
			})
		},
		pin: {
			label: "📌",
			sortable: !1,
			cls: "",
			render: ({ key: t, d: n }) => e.config && k ? /* @__PURE__ */ f("button", {
				className: `wp-pin${n.isPinned ? " wp-pin-on" : ""}`,
				disabled: L,
				title: L ? "Agent is restarting to apply the previous change — retry in a few seconds." : n.isPinned ? "Pinned: this allocation survives restarts — routing to this worker is durable and unassign is refused (409). Does not download the model and does not protect its files from eviction (bytes re-pull on call). Click to unpin." : "Pin: make this allocation survive restarts — routing to this worker becomes durable and unassign is refused (409). Does not download the model or protect its files from eviction.",
				onClick: () => k(e, t, !n.isPinned),
				children: ["📌", n.isPinned ? "" : "?"]
			}) : /* @__PURE__ */ d("span", {
				className: "wp-lt-muted",
				children: "—"
			})
		},
		name: {
			label: "Model",
			sortable: !0,
			cls: "wp-servtable-name",
			title: ({ key: e }) => e,
			render: ({ key: e, isBlocked: t, compact: n }) => /* @__PURE__ */ f(u, { children: [n ? Cr(Ke(e), 8, 14) : Cr(Ke(e)), t && /* @__PURE__ */ d("span", {
				className: "wp-blocked-chip",
				title: "⛔ Blocked from the serving pool by the operator — not routed to, assigned, warmed, or used as a fallback anywhere. This designation stays recorded but inert (block outranks pin). Files are untouched. Use the ⛔ action to unblock.",
				children: "⛔ blocked"
			})] })
		},
		size: {
			label: "Size",
			sortable: !0,
			num: !0,
			cls: "wp-lt-num wp-lt-size",
			title: ({ key: t }) => {
				let n = qe(t), r = e.loaded_detail?.[t], i = n && n.bytes != null ? n.bytes : r?.model_bytes;
				return i == null ? "size unknown — not on disk / not reported by the feed" : `${X(i)} on disk${n && n.isGguf ? ` (effective quant${n.effGguf ? ` ${n.effGguf}` : ""}, not the all-quants dir sum)` : ""}`;
			},
			render: ({ key: t }) => {
				let n = qe(t), r = e.loaded_detail?.[t], i = n && n.bytes != null ? n.bytes : r?.model_bytes;
				return i == null ? "—" : X(i);
			}
		},
		ctx: {
			label: "Ctx",
			sortable: !0,
			num: !0,
			cls: "wp-lt-num",
			render: ({ m: e }) => e?.model_max_length || "—"
		},
		state: {
			label: "State",
			sortable: !0,
			cls: "wp-servtable-state",
			render: ({ d: e }) => {
				let { isPulling: t, isHeating: n, isServing: r, inSlot: i, isAnswering: a, isIdleResident: o, isMissing: s, pct: c, state: l, stateTitle: p, loadFailed: m, loadReport: h, loadStale: g } = e;
				return /* @__PURE__ */ f(u, { children: [
					/* @__PURE__ */ f("span", {
						className: `wp-state-pill wp-pill-${l}`,
						title: p,
						children: [t ? `⏳ pulling${c == null ? "" : ` ${c}%`}` : n ? "🔶 heating" : r ? i ? a ? "⚡ answering" : "🔥 serving" : "📌 loaded" : o ? "◍ idle" : l === "hot" ? "🌡 hot" : l === "cold" ? "○ cold" : "○ missing", s && /* @__PURE__ */ d(ar, { doc: "worker-model-missing" })]
					}),
					m && /* @__PURE__ */ d("span", {
						className: "wp-loadwhy wp-loadwhy-bad",
						title: `${h?.error || (h?.fit === !1 ? "probe: fit=false" : "warm failed — the model never became resident")}${h?.ts ? ` · ${wr(h?.ts)}` : ""}`,
						children: "⚠"
					}),
					g && /* @__PURE__ */ d("span", {
						className: "wp-loadwhy wp-loadwhy-ok",
						title: `last warm succeeded${h?.ts ? ` ${wr(h?.ts)}` : ""} — not resident now`,
						children: "ⓘ"
					})
				] });
			}
		},
		moe: {
			label: "MoE",
			sortable: !1,
			cls: "",
			render: ({ key: t }) => {
				if (!e.moe_capable?.[t]) return /* @__PURE__ */ d("span", {
					className: "wp-4bit-na",
					title: "No expert structure — this model is dense, so there is nothing to split.",
					children: "—"
				});
				let n = e.moe_by_model?.[t], r = n != null, i = t in ce ? ce[t] : r ? !!n : !!e.moe_effective?.[t];
				return /* @__PURE__ */ f("span", {
					className: "wp-moe",
					children: [/* @__PURE__ */ d("label", {
						title: r ? `Expert split PINNED ${i ? "on" : "off"} by you. Click to flip; ⟲ restores auto.` : i ? "Expert split ACTIVE, derived automatically — experts in RAM, everything else on the GPU. Untick to force it off." : "Capable of an expert split, but the derivation did not apply one here (transformers MoE has no split path yet). Tick to force it on.",
						children: /* @__PURE__ */ d("input", {
							type: "checkbox",
							className: "wp-moe-box",
							checked: i,
							disabled: L,
							onChange: (e) => ue(t, e.target.checked)
						})
					}), r && /* @__PURE__ */ d("button", {
						className: "wp-moe-auto",
						disabled: L,
						title: "Restore AUTO — follow the derivation for this model.",
						onClick: () => ue(t, null),
						children: "⟲"
					})]
				});
			}
		},
		fourbit: {
			label: "4-bit",
			sortable: !1,
			cls: "",
			render: ({ key: t }) => {
				if (!e.bnb_available?.[t]) return /* @__PURE__ */ d("span", {
					className: "wp-4bit-na",
					title: "No bitsandbytes specialization here: GGUF models carry their own quantization, the 4-bit kernels need a CUDA worker, and an already-quantized repo cannot be re-quantized.",
					children: "—"
				});
				let n = t in ae ? ae[t] : !!e.bnb_by_model?.[t];
				return /* @__PURE__ */ f("label", {
					className: "wp-4bit",
					title: n ? "bitsandbytes 4-bit (nf4) ON — the model is priced at ~30% of its fp16 size, so its allocation re-derives. Untick to restore full precision." : "Load this model with bitsandbytes 4-bit (nf4). It is priced at ~30% of its fp16 size, so the Alloc column re-derives — often turning a RAM-only model into one that fits the GPU. Costs some quality.",
					children: [/* @__PURE__ */ d("input", {
						type: "checkbox",
						className: "wp-4bit-box",
						checked: n,
						disabled: L,
						onChange: (e) => me(t, e.target.checked)
					}), /* @__PURE__ */ d("span", {
						className: "wp-4bit-tag",
						children: n ? "4-bit" : ""
					})]
				});
			}
		},
		alloc: {
			label: "Alloc",
			sortable: !1,
			cls: "",
			render: ({ key: t, m: n, d: r, need: i }) => {
				let a = t in G ? G[t] : r.override, o = a && Object.keys(a).length > 0 ? vr(a) : r.derivedMode || vr(a), s = /^(gguf|llama_cpp)$/.test(String(n?.framework || "").toLowerCase()), c = re === t, l = he[t], u = l && typeof l.alloc_mode_derived == "boolean" && !(t in G) ? l.alloc_mode_derived : !(a && Object.keys(a).length > 0), p = l && l.alloc_by_worker && l.alloc_by_worker[e.id] || null, m = l && Array.isArray(l.alloc_modes_feasible) ? l.alloc_modes_feasible : null, h = p && Array.isArray(p.feasible) ? p.feasible : m, g = p && p.derived_default || l && l.alloc_mode_derived && l.alloc_mode || null;
				return /* @__PURE__ */ f("span", {
					className: "wp-allocmode-anchor",
					children: [/* @__PURE__ */ f("button", {
						className: `wp-alloc-edit${u ? " wp-alloc-derived" : ""}`,
						disabled: L,
						ref: c ? W : void 0,
						title: L ? "Agent is applying the previous change — retry in a few seconds." : u ? `Allocation: ${yr(o)} — DERIVED (no pinned contract; tracks the default as it improves). Click to change or pin.` : `Allocation: ${yr(o)} — pinned. Click to change or revert to the derived default.`,
						onClick: () => {
							U(c ? null : t), c || ve(t);
						},
						children: [yr(o), u && /* @__PURE__ */ d("span", {
							className: "wp-alloc-auto-affix",
							children: " · auto"
						})]
					}), c && /* @__PURE__ */ d(Lr, {
						mode: o,
						spill: a,
						worker: e,
						need: i,
						engineGguf: s,
						feasible: h,
						feasibleCtx: {
							modelBytes: i?.bytes ?? null,
							vramTotal: e.vram_total ?? null,
							ramTotal: e.ram_total ?? null
						},
						derivedMode: g,
						anchorRef: W,
						onPick: (e) => J(t, { alloc_mode: e }),
						onApplyExplicit: (e) => J(t, e),
						onRevertDerived: () => J(t, {}),
						onClose: () => U(null)
					})]
				});
			}
		},
		memory: {
			label: "Memory",
			sortable: !1,
			cls: "wp-servtable-mem",
			render: ({ key: t, d: n }) => {
				let r = e.loaded_detail?.[t], i = n.alloc || null, a = i && i.vram_bytes != null ? i.vram_bytes : r && r.vram_bytes != null ? r.vram_bytes : null, o = i && i.rss_anon_bytes != null ? i.rss_anon_bytes : i && i.ram_resident_bytes != null ? i.ram_resident_bytes : r && r.rss_anon_bytes != null ? r.rss_anon_bytes : r && r.ram_resident_bytes != null ? r.ram_resident_bytes : null, s = i && i.n_gpu_layers != null ? i.n_gpu_layers : r && r.n_gpu_layers != null ? r.n_gpu_layers : null, c = i && i.total_layers != null ? i.total_layers : r && r.total_layers != null ? r.total_layers : null, l = a, u = qe(t), p = u && u.bytes != null ? u.bytes : r?.model_bytes;
				if (p == null && r?.gpu_pct == null && l == null) return /* @__PURE__ */ d("span", {
					className: "wp-lt-muted",
					children: "—"
				});
				let m = a != null || o != null ? (a || 0) + (o || 0) : null, h = s == null ? "" : `${s === -1 ? "all" : s}${c ? `/${c}` : ""} layers on GPU`, g = [
					p == null ? "" : `${X(p)} on disk${u && u.isGguf ? ` (effective quant${u.effGguf ? ` ${u.effGguf}` : ""}, not the all-quants dir sum)` : ""}`,
					m == null ? "" : `${X(m)} resident = ${a == null ? "0" : X(a)} VRAM + ${o == null ? "0" : X(o)} anon RAM`,
					h,
					a != null || o != null ? "VRAM/RAM figures are MEASURED (nvidia-smi / rss_anon / ram_resident); VRAM 0 = running on CPU" : r?.gpu_pct == null ? "" : "GPU split is DECLARED by the loader, not a measured VRAM read"
				].filter(Boolean).join(" · "), _ = e.planned_split?.[t];
				if (m == null && l == null && _ && (_.gpu_bytes != null || _.ram_bytes != null)) {
					let e = _.gpu_bytes, t = _.ram_bytes, n = [];
					return e && n.push(`${X(e)} VRAM`), t && n.push(`${X(t)} RAM`), /* @__PURE__ */ f("span", {
						className: "wp-model-facts wp-fact-planned",
						title: (_.split ? `PLANNED expert split: ${X(e || 0)} of non-expert tensors on the GPU, ${X(t || 0)} of experts in RAM. ` : `PLANNED placement under '${_.mode}': `) + (_.split ? "" : `${X(e || t || 0)} on ${e ? "the GPU" : "the CPU"}` + (_.mode === "max-gpu" ? " (spills whatever will not fit at load time)" : "") + ". ") + "Not resident yet — this is what the current Alloc mode and the 4-bit / MoE switches add up to, not a measurement.",
						children: [n.join(" + "), /* @__PURE__ */ d("span", {
							className: "wp-fact-planned-tag",
							children: " planned"
						})]
					});
				}
				return /* @__PURE__ */ f("span", {
					className: "wp-model-facts",
					title: g,
					children: [p != null && /* @__PURE__ */ f("span", {
						className: "wp-fact-disk",
						children: [X(p), " disk"]
					}), m == null ? l == null ? r?.gpu_pct == null ? "" : ` · ~${r.gpu_pct}% GPU / ${100 - r.gpu_pct}% spill` : l > 0 ? ` · ${X(l)} VRAM` : " · 0 VRAM · on CPU" : /* @__PURE__ */ f("span", {
						className: "wp-fact-resident",
						children: [
							" · ",
							X(m),
							" resident",
							/* @__PURE__ */ f("span", {
								className: "wp-fact-split",
								children: [
									" (",
									a == null ? "0" : X(a),
									" VRAM + ",
									o == null ? "0" : X(o),
									" RAM)"
								]
							}),
							c != null && s != null && /* @__PURE__ */ f("span", {
								className: "wp-fact-layers",
								children: [
									" · ",
									s === -1 ? c : s,
									"/",
									c,
									" layers"
								]
							})
						]
					})]
				});
			}
		},
		actions: {
			label: "Actions",
			sortable: !1,
			cls: "wp-servtable-actions",
			render: ({ key: t, d: n, isBlocked: r }) => {
				let { state: i, isServing: a, isIdleResident: o, isPinned: s } = n;
				return /* @__PURE__ */ f(u, { children: [
					p && (i === "hot" || i === "cold") && /* @__PURE__ */ d("button", {
						className: "wp-activate",
						disabled: Oe === t,
						title: Oe === t ? "seating this model on the worker…" : "Activate: allocate this worker’s resources to the model now so it serves immediately. Files transfer first if not local yet.",
						onClick: async () => {
							ke(t);
							try {
								await p(e, t);
							} finally {
								ke(null);
							}
						},
						children: Oe === t ? "⏳ activating…" : "▶ activate"
					}),
					(a || o) && /* @__PURE__ */ d("button", {
						className: "wp-free",
						title: o ? "Clear this idle runner shell from the worker (stays assigned)" : "Unload from VRAM (stays assigned)",
						onClick: () => g(e, t),
						children: "⏏"
					}),
					/* @__PURE__ */ d("button", {
						className: "wp-model-x",
						disabled: s,
						title: s ? "pinned — unpin first" : "Unassign",
						onClick: () => m(e, t),
						children: "×"
					}),
					V && /* @__PURE__ */ d("button", {
						className: `wp-model-block${r ? " wp-model-block-on" : ""}`,
						title: r ? "Blocked from the serving pool — click to UNBLOCK (return it to routing). Block outranks pin; the designation is unchanged." : "Block this model from the serving pool (global): never routed to / assigned / warmed / a fallback default anywhere. Files stay; designations stay (inert). Reversible.",
						onClick: () => V(t, !r),
						children: "⛔"
					})
				] });
			}
		}
	}, { order: it, widths: at, moveColumn: ot, setWidth: st, reset: ct } = Er(or, sr), lt = it.map((e) => ({
		key: e,
		...rt[e]
	})).filter((e) => e.label != null), ut = c(null), dt = ii(ut, 720), ft = dt ? lt.filter((e) => lr.includes(e.key)) : lt, pt = dt ? lt.filter((e) => !lr.includes(e.key) && e.key !== "actions") : [], mt = rt.actions, ht = 1 + ft.length, [gt, _t] = l(null);
	o(() => {
		dt || _t(null);
	}, [dt]);
	let vt = (e) => (t) => {
		if (!dt) return;
		let n = t.target;
		n && typeof n.closest == "function" && n.closest("button, input, select, textarea, label, a, .wp-servtable-selcol") || _t((t) => t === e ? null : e);
	}, [yt, bt] = l(null), [xt, St] = l(null), Ct = (e, t) => {
		e.preventDefault(), e.stopPropagation();
		let n = e.currentTarget.closest("th"), r = e.clientX, i = n ? n.getBoundingClientRect().width : at[t] || 120, a = (e) => st(t, i + (e.clientX - r)), o = () => {
			window.removeEventListener("pointermove", a), window.removeEventListener("pointerup", o);
		};
		window.addEventListener("pointermove", a), window.addEventListener("pointerup", o);
	}, wt = s(() => {
		let n = new Set(e.models || []), r = t.filter((e) => n.has(e.model_key ?? e.key));
		return [...new Set(r.flatMap(Gn))].sort();
	}, [t, e.models]), Tt = (t) => {
		let n = qe(t), r = e.loaded_detail?.[t], i = n && n.bytes != null ? n.bytes : r?.model_bytes;
		return i == null ? -1 : Number(i);
	}, Et = (e.models || []).map((e) => ({
		key: e,
		m: t.find((t) => (t.model_key ?? t.key) === e),
		d: tt(e)
	})), Dt = Ae.trim().toLowerCase(), Ot = Et;
	Dt && (Ot = Ot.filter(({ key: e, m: t }) => (t?.name || e).toLowerCase().includes(Dt) || e.toLowerCase().includes(Dt))), Me && (Ot = Ot.filter(({ m: e }) => e && Gn(e).includes(Me)));
	let kt = ({ key: e, m: t, d: n }) => {
		switch (Pe) {
			case "task": return t && Wn(t) || "";
			case "framework": return t?.framework || "";
			case "size": return Tt(e);
			case "ctx": return t?.model_max_length || 0;
			case "state": return nt[n.state] ?? -1;
			default: return t?.name || e;
		}
	}, At = Pe === "size" || Pe === "ctx" || Pe === "state", jt = Ie === "asc" ? 1 : -1;
	Ot = [...Ot].sort((e, t) => {
		let n = kt(e), r = kt(t);
		return At ? (Number(n) - Number(r)) * jt : String(n).localeCompare(String(r)) * jt;
	});
	let Mt = (e) => {
		Pe === e ? Le(Ie === "asc" ? "desc" : "asc") : (Fe(e), Le("asc"));
	}, Nt = (e) => Pe === e ? Ie === "asc" ? " ▲" : " ▼" : "", { groups: Pt } = Nr(), Ft = s(() => Or(Pt), [Pt]), [It, Lt] = Y(`hugpy.sess.wp.pgcollapsed.${e.id}`, {}), Rt = (() => {
		if (!Ft.size) return Ot;
		let e = (e) => {
			for (let t of Dr(e)) {
				let e = Ft.get(t);
				if (e) return e;
			}
			return null;
		}, t = /* @__PURE__ */ new Map();
		for (let n of Ot) {
			let r = e(n.key);
			r && (t.has(r.id) || t.set(r.id, {
				g: r,
				rows: []
			}), t.get(r.id).rows.push(n));
		}
		if (!t.size) return Ot;
		let n = [], r = /* @__PURE__ */ new Set();
		for (let i of Ot) {
			let a = e(i.key);
			if (!a || r.has(a.id)) continue;
			r.add(a.id);
			let o = t.get(a.id);
			n.push({
				kind: "pgheader",
				g: a,
				count: o.rows.length
			}), It?.[a.id] || n.push(...o.rows);
		}
		for (let t of Ot) e(t.key) || n.push(t);
		return n;
	})(), zt = Ot.map((e) => e.key), Bt = zt.filter((e) => Re.has(e)), Vt = Bt.length, Ht = zt.length > 0 && Vt === zt.length, Ut = Vt > 0 && !Ht, Wt = () => ze((e) => {
		let t = new Set(e);
		return Ht ? zt.forEach((e) => t.delete(e)) : zt.forEach((e) => t.add(e)), t;
	}), Gt = !!e.config && !!D, Kt = !!O;
	return /* @__PURE__ */ f("div", {
		className: `wp-worker wp-${e.status} wp-adm-${e.admission || "approved"}`,
		children: [
			/* @__PURE__ */ f("div", {
				className: "wp-worker-head",
				children: [
					/* @__PURE__ */ d("span", { className: "wp-dot" }),
					/* @__PURE__ */ d("span", {
						className: "wp-name",
						children: e.name
					}),
					/* @__PURE__ */ d("span", {
						className: "wp-status",
						children: e.status
					}),
					e.pkg_version && /* @__PURE__ */ f("span", {
						className: `wp-ver ${e.version_ok === !0 ? "wp-ver-ok" : e.version_ok === !1 ? "wp-ver-skew" : "wp-ver-unknown"}`,
						title: e.version_ok === !0 ? `abstract_hugpy_dev ${e.pkg_version} — in sync with central` : e.version_ok === !1 ? `worker on ${e.pkg_version}, central requires ${e.required_pkg_version || "?"} — use ⬆ Update to converge now (or it self-updates on its next heartbeat)` : `abstract_hugpy_dev ${e.pkg_version} — central reported no required version to compare against`,
						children: ["v", e.pkg_version]
					}),
					e.engine_build && /* @__PURE__ */ f("span", {
						className: "wp-ver wp-ver-engine",
						title: `native llama-server engine build ${e.engine_build} — surfaced so engine skew across the fleet is visible (item L). The fat-arch rebuild is tracked separately.`,
						children: ["⚙ ", e.engine_build]
					}),
					/* @__PURE__ */ f("span", {
						className: `wp-adm wp-adm-pill-${e.admission || "approved"}`,
						title: "Operator admission gate — only approved workers serve traffic",
						children: [e.admission || "approved", e.admission && e.admission !== "approved" && /* @__PURE__ */ d(ar, { doc: "worker-admission" })]
					}),
					/* @__PURE__ */ f("span", {
						className: `wp-pool ${e.pool ? "wp-pool-set" : ""}`,
						role: "button",
						title: "Dedicated pool — this worker serves ONLY requests tagged for it (general traffic never lands here). Click to set/clear.",
						onClick: () => C(e),
						children: ["🏷 ", e.pool || "general"]
					}),
					e.role === "rpc" && /* @__PURE__ */ d("span", {
						className: "wp-role",
						title: "shard backend (lends GPU via rpc-server)",
						children: "rpc"
					}),
					e.comfy?.available && /* @__PURE__ */ d("span", {
						className: "wp-role",
						title: `ComfyUI running on this worker${e.comfy.version ? ` (v${e.comfy.version})` : ""} at ${e.comfy.url} — comfy-templated generation routes here (engine slice B)`,
						children: "🧩 comfy"
					}),
					/* @__PURE__ */ d("span", {
						className: "wp-url",
						title: e.url,
						children: e.url
					}),
					/* @__PURE__ */ d(ri, { spill: e.spill }),
					(e.gpus || []).length > 0 && e.engine?.supports_gpu_offload === !1 && /* @__PURE__ */ f("span", {
						className: "wp-cpu-only",
						title: "This worker's llama-cpp-python is a CPU-only build: GGUF models run on CPU and n_gpu_layers is silently ignored, so VRAM stays idle. Rebuilding it with GPU support has real traps (missing nvcc, missing CUDA runtime libs, AVX512 SIGILL) — click 📖 for the diagnostic and the rebuild recipe that actually works.",
						children: ["⚠ CPU-only engine", /* @__PURE__ */ d(ar, { doc: "engine-cpu-only" })]
					}),
					e.install && e.install.canonical === !1 && /* @__PURE__ */ d("span", {
						className: "wp-noncanon",
						title: `Non-canonical install — this worker did NOT come from the standard bootstrap/installer.\nunit: ${e.install.unit || (e.install.via_systemd ? "systemd (name unknown)" : "not a systemd unit")}\nvenv: ${e.install.venv || "unknown"}\nCanonical = a hugpy-worker.service (or legacy abstract-hugpy-worker.service) user unit running from ~/hugpy-worker/venv. See WORKER-SETUP.md §1.`,
						children: "⚠ non-canonical install"
					}),
					q && q !== "checking" && /* @__PURE__ */ f("span", {
						className: `wp-ping ${q.reachable ? "wp-ping-ok" : "wp-ping-bad"}`,
						title: q.reachable ? "central can reach this worker" : q.error || "unreachable",
						children: [q.reachable ? "✓ reachable" : "✗ unreachable", !q.reachable && /* @__PURE__ */ d(ar, { doc: "worker-unreachable" })]
					}),
					/* @__PURE__ */ d("button", {
						className: "wp-ping-btn",
						title: "Ping the worker's /health from central",
						onClick: We,
						disabled: q === "checking",
						children: q === "checking" ? "…" : "ping"
					}),
					Je.size > 0 && /* @__PURE__ */ d("button", {
						className: "wp-free-all",
						title: "Unload every model from this GPU (stays assigned)",
						onClick: () => _(e),
						children: "⏏ free VRAM"
					}),
					/* @__PURE__ */ d("button", {
						className: "wp-free-ram",
						title: "Return reclaimable host RAM to the OS — non-destructive: loaded models stay resident",
						onClick: () => v(e),
						children: "🧹 Free RAM"
					}),
					/* @__PURE__ */ d("button", {
						className: "wp-restart",
						title: "Restart the worker agent — drops all loaded models and re-execs the agent process",
						onClick: () => y(e),
						disabled: R,
						children: R ? "↻ restarting…" : "↻ Restart"
					}),
					b && /* @__PURE__ */ d("button", {
						className: `wp-update${e.version_ok === !1 ? " wp-update-skew" : " wp-update-noop"}`,
						title: e.version_ok === !1 ? `worker on ${e.pkg_version || "?"}, central requires ${e.required_pkg_version || "?"} — update pip-installs the required version and restarts the agent` : "already at central’s required version — update is a no-op",
						onClick: () => b(e),
						disabled: z || R,
						children: z ? "⬆ updating…" : "⬆ Update"
					}),
					e.admission === "approved" ? /* @__PURE__ */ d("button", {
						className: "wp-block",
						title: "Block: stop serving; the agent exits on its next contact and won't respawn",
						onClick: () => S(e),
						children: "⛔ block"
					}) : /* @__PURE__ */ f("button", {
						className: "wp-admit",
						title: e.admission === "blocked" ? "Unblock and allow serving" : "Admit: allow this worker to serve",
						onClick: () => x(e),
						children: ["✓ ", e.admission === "blocked" ? "unblock" : "admit"]
					}),
					/* @__PURE__ */ d("button", {
						className: "wp-remove",
						title: "Remove worker (forget — a live agent re-appears as pending; use Block to evict)",
						onClick: () => h(e),
						children: "✕"
					})
				]
			}),
			/* @__PURE__ */ d(ni, {
				worker: e,
				models: t,
				onApproveEvictions: N,
				onEvict: P
			}),
			/* @__PURE__ */ d(Br, { worker: e }),
			/* @__PURE__ */ f("div", {
				className: "wp-caps",
				children: [
					e.caps && Object.keys(e.caps).length > 0 && /* @__PURE__ */ f("span", {
						className: "wp-cap-chip",
						title: "Configured on the worker box itself (unit env) — central can only set limits at or below these.",
						children: [
							"box caps:",
							e.caps.ram_max_gib != null && ` RAM ${e.caps.ram_max_gib}GiB`,
							e.caps.gpu_mem_gib != null && ` · VRAM ${e.caps.gpu_mem_gib}GiB`,
							e.caps.disk_cache_gib != null && ` · disk ${e.caps.disk_cache_gib}GiB`,
							e.caps.threads != null && ` · ${e.caps.threads} threads`
						]
					}),
					e.limits && Object.keys(e.limits).length > 0 && /* @__PURE__ */ f("span", {
						className: "wp-cap-chip wp-limit-chip",
						title: "Central-set limits (≤ box caps); the worker adopts them on its next heartbeat.",
						children: [
							"central limits:",
							e.limits.ram_max_gib != null && ` RAM ${e.limits.ram_max_gib}GiB`,
							e.limits.gpu_mem_gib != null && ` · VRAM ${e.limits.gpu_mem_gib}GiB`,
							e.limits.disk_cache_gib != null && ` · disk ${e.limits.disk_cache_gib}GiB`,
							e.limits.threads != null && ` · ${e.limits.threads} threads`
						]
					}),
					e.config?.slot_count != null && /* @__PURE__ */ f("span", {
						className: `wp-cap-chip${L ? " wp-chip-applying" : ""}`,
						role: "button",
						title: L ? "The agent is restarting (~5s) to apply the previous config change — controls unlock when the new config arrives in a heartbeat." : `Worker slot pool: ${e.config.slot_count} slot(s) — source: ${e.config.slot_count_source || "?"}. Click to change (persists in the agent's runtime settings; applies via a ~5s agent restart).`,
						onClick: () => !L && T && T(e),
						children: [
							"🎛 slots: ",
							e.config.slot_count,
							e.config.slot_count_source && e.config.slot_count_source !== "settings" && /* @__PURE__ */ f("em", { children: [
								" (",
								e.config.slot_count_source,
								")"
							] })
						]
					}),
					L && /* @__PURE__ */ d("span", {
						className: "wp-applying",
						title: "The last pin/residency/slot-count change was accepted; the agent re-execs (~5s) to apply it and this worker's config controls are paused until the new config shows up in a heartbeat.",
						children: "⏳ applying…"
					}),
					w && /* @__PURE__ */ d("button", {
						className: "wp-limits-edit",
						title: "Worker budget — this box's resource ceiling for ALL models combined (VRAM / RAM / disk / threads), clamped to its box caps. This is NOT a per-model allocation: per-model VRAM/RAM placement is the Alloc column on each serving row. Central-set (≤ box caps).",
						onClick: () => {
							we({
								ram_max_gib: e.limits?.ram_max_gib ?? "",
								gpu_mem_gib: e.limits?.gpu_mem_gib ?? "",
								disk_cache_gib: e.limits?.disk_cache_gib ?? "",
								threads: e.limits?.threads ?? ""
							}), Se((e) => !e);
						},
						children: "⚙ worker budget"
					}),
					xe && /* @__PURE__ */ f("span", {
						className: "wp-limits-form",
						children: [
							/* @__PURE__ */ d("span", {
								className: "wp-limits-title",
								title: "This box's resource ceiling across ALL models it hosts — not a per-model budget.",
								children: "worker budget — this box's resource ceiling (all models combined):"
							}),
							/* @__PURE__ */ d("input", {
								type: "number",
								step: "1",
								min: "0",
								placeholder: "RAM GiB",
								value: Ce.ram_max_gib,
								onChange: (e) => we((t) => ({
									...t,
									ram_max_gib: e.target.value
								}))
							}),
							/* @__PURE__ */ d("input", {
								type: "number",
								step: "1",
								min: "0",
								placeholder: "VRAM GiB",
								value: Ce.gpu_mem_gib,
								onChange: (e) => we((t) => ({
									...t,
									gpu_mem_gib: e.target.value
								}))
							}),
							/* @__PURE__ */ d("input", {
								type: "number",
								step: "1",
								min: "0",
								placeholder: "disk cache GiB",
								title: "Local model-cache ceiling for this worker. Over it, cold local models become eviction candidates in the storage proposal. Clamped to the box's own caps.disk_cache_gib — the worker's stated delegation wins.",
								value: Ce.disk_cache_gib,
								onChange: (e) => we((t) => ({
									...t,
									disk_cache_gib: e.target.value
								}))
							}),
							/* @__PURE__ */ d("input", {
								type: "number",
								step: "1",
								min: "1",
								placeholder: "threads",
								value: Ce.threads,
								onChange: (e) => we((t) => ({
									...t,
									threads: e.target.value
								}))
							}),
							/* @__PURE__ */ d("button", {
								className: "wp-alloc-apply",
								onClick: () => {
									let t = {};
									for (let e of [
										"ram_max_gib",
										"gpu_mem_gib",
										"disk_cache_gib",
										"threads"
									]) Ce[e] !== "" && Ce[e] != null && (t[e] = Number(Ce[e]));
									w(e, t), Se(!1);
								},
								children: "Set"
							}),
							/* @__PURE__ */ d("button", {
								className: "wp-alloc-cancel",
								title: "Clear all central limits",
								onClick: () => {
									w(e, {}), Se(!1);
								},
								children: "clear"
							})
						]
					})
				]
			}),
			/* @__PURE__ */ f("div", {
				className: `wp-models wp-servtable${dt ? " wp-servtable-compact" : ""}`,
				ref: ut,
				children: [/* @__PURE__ */ f("div", {
					className: "wp-servtable-bar",
					children: [
						/* @__PURE__ */ d("span", {
							className: "wp-models-label",
							children: "Serving:"
						}),
						(e.models || []).length > 0 && /* @__PURE__ */ f(u, { children: [
							/* @__PURE__ */ d("input", {
								className: "wp-servtable-q",
								placeholder: "filter models…",
								value: Ae,
								onChange: (e) => je(e.target.value)
							}),
							/* @__PURE__ */ f("select", {
								className: "wp-servtable-task",
								value: Me,
								onChange: (e) => Ne(e.target.value),
								title: "Filter by task — matches ANY task a model advertises, not just its primary",
								children: [/* @__PURE__ */ d("option", {
									value: "",
									children: "All tasks"
								}), wt.map((e) => /* @__PURE__ */ d("option", {
									value: e,
									children: e
								}, e))]
							}),
							dt && /* @__PURE__ */ f("span", {
								className: "wp-servtable-sortsel",
								children: [/* @__PURE__ */ d("select", {
									value: Pe,
									onChange: (e) => Fe(e.target.value),
									title: "Sort the serving list (the sortable columns are hidden in this narrow layout)",
									children: lt.filter((e) => e.sortable).map((e) => /* @__PURE__ */ f("option", {
										value: e.key,
										children: ["sort: ", e.label]
									}, e.key))
								}), /* @__PURE__ */ d("button", {
									className: "wp-servtable-sortdir",
									title: Ie === "asc" ? "Ascending — click for descending" : "Descending — click for ascending",
									onClick: () => Le(Ie === "asc" ? "desc" : "asc"),
									children: Ie === "asc" ? "▲" : "▼"
								})]
							}),
							!dt && (it.join(",") !== sr.join(",") || Object.keys(at).length > 0) && /* @__PURE__ */ d("button", {
								className: "wp-servtable-resetcols",
								onClick: ct,
								title: "Reset the serving-table columns (order + widths) to the default layout. Drag a header to reorder, or a header's right edge to resize — your layout is remembered per browser.",
								children: "⟲ reset columns"
							})
						] }),
						(Gt || Kt) && Vt > 0 && /* @__PURE__ */ f("span", {
							className: "wp-bulkres wp-servtable-bulkres",
							children: [
								/* @__PURE__ */ f("span", {
									className: "wp-bulkres-count",
									title: "models selected in the current filter",
									children: [Vt, " selected"]
								}),
								Gt && /* @__PURE__ */ f(u, { children: [
									/* @__PURE__ */ d("span", {
										className: "wp-bulkres-label",
										children: "set residency →"
									}),
									/* @__PURE__ */ d("button", {
										className: "wp-bulkres-btn wp-bulkres-ondemand",
										disabled: L,
										title: L ? "Agent is restarting to apply the previous change — retry in a few seconds." : `Set the ${Vt} selected model${Vt === 1 ? "" : "s"} to ⏲ on-demand (the default — loads on call, yields its seat under contention). Confirms first; one ~5s agent restart.`,
										onClick: () => D(e, Bt, "on-demand"),
										children: "⏲ on-demand"
									}),
									/* @__PURE__ */ d("button", {
										className: "wp-bulkres-btn wp-bulkres-static",
										disabled: L,
										title: L ? "Agent is restarting to apply the previous change — retry in a few seconds." : `Set the ${Vt} selected model${Vt === 1 ? "" : "s"} to 🔒 static (locked seat — kept on this worker, never evicted). Confirms first; one ~5s agent restart.`,
										onClick: () => D(e, Bt, "static"),
										children: "🔒 static"
									})
								] }),
								Kt && /* @__PURE__ */ f(u, { children: [
									Gt && /* @__PURE__ */ d("span", {
										className: "wp-bulkres-sep",
										"aria-hidden": "true",
										children: "·"
									}),
									/* @__PURE__ */ d("span", {
										className: "wp-bulkres-label",
										children: "set alloc →"
									}),
									/* @__PURE__ */ f("button", {
										className: `wp-bulkres-btn wp-bulkres-alloc${He ? " wp-bulkres-alloc-open" : ""}`,
										title: `Set the GPU allocation (Default / Max GPU / GPU only / RAM only / Max RAM / Explicit) for the ${Vt} selected model${Vt === 1 ? "" : "s"}. A registry contract applied on next load — no agent restart. Confirms first.`,
										onClick: () => Ue((e) => !e),
										children: ["⚙ allocation ", He ? "▾" : "▸"]
									})
								] }),
								/* @__PURE__ */ d("button", {
									className: "wp-bulkres-clear",
									title: "Clear the selection",
									onClick: () => {
										Ve(), Ue(!1);
									},
									children: "clear"
								})
							]
						}),
						Kt && Vt > 0 && He && /* @__PURE__ */ f("div", {
							className: "wp-bulkalloc-editor",
							children: [/* @__PURE__ */ f("span", {
								className: "wp-bulkalloc-hint",
								children: [
									"Apply this allocation to the ",
									Vt,
									" selected model",
									Vt === 1 ? "" : "s",
									":"
								]
							}), /* @__PURE__ */ d(zr, {
								count: Vt,
								bulkKeys: Bt,
								getModelBytes: qe,
								onApply: (t, n) => {
									Ue(!1), O(e, Bt, t, n);
								},
								onCancel: () => Ue(!1)
							})]
						}),
						e.config && (A || j) && (e.models || []).length > 0 && /* @__PURE__ */ f("span", {
							className: "wp-bulkpin wp-servtable-bulk",
							children: [
								A && /* @__PURE__ */ d("button", {
									className: "wp-bulkpin-btn",
									disabled: L,
									title: L ? "Agent is restarting to apply the previous change — retry in a few seconds." : "Pin ALL of this worker’s models — permanent attribution; each then blocks unassign until unpinned. Confirms first.",
									onClick: () => A(e),
									children: "📌 pin all"
								}),
								j && /* @__PURE__ */ d("button", {
									className: "wp-bulkpin-btn wp-bulkunpin-btn",
									disabled: L,
									title: L ? "Agent is restarting to apply the previous change — retry in a few seconds." : "Unpin ALL of this worker’s models — the undo for Pin all; lets them be unassigned again.",
									onClick: () => j(e),
									children: "📌✕ unpin all"
								}),
								de.length > 0 && /* @__PURE__ */ f("button", {
									className: "wp-bulkpin-btn wp-bulkmoe-btn",
									disabled: L,
									title: `Restore AUTO expert-split handling on all ${de.length} MoE-capable model(s) — clears any forced on/off and lets the derivation decide.`,
									onClick: () => fe(null),
									children: [
										"⟲ MoE auto (",
										de.length,
										")"
									]
								}),
								se.length > 0 && /* @__PURE__ */ f(u, { children: [/* @__PURE__ */ f("button", {
									className: "wp-bulkpin-btn wp-bulk4bit-btn",
									disabled: L,
									title: `Load all ${se.length} eligible model(s) at bitsandbytes 4-bit (nf4). Each is then priced at ~30% of its fp16 size, so their Alloc modes re-derive — models too big for the GPU can become GPU-resident. Costs some quality.`,
									onClick: () => pe(!0),
									children: [
										"▦ 4-bit all (",
										se.length,
										")"
									]
								}), /* @__PURE__ */ d("button", {
									className: "wp-bulkpin-btn wp-bulkunpin-btn",
									disabled: L,
									title: "Restore full precision on every model that currently has the 4-bit specialization — the undo for 4-bit all.",
									onClick: () => pe(!1),
									children: "▦✕ clear 4-bit"
								})] })
							]
						})
					]
				}), (e.models || []).length === 0 ? /* @__PURE__ */ d("span", {
					className: "wp-none",
					children: "— nothing assigned —"
				}) : /* @__PURE__ */ d("div", {
					className: "wp-servtable-scroll",
					children: /* @__PURE__ */ f("table", {
						className: "wp-servtable-t",
						children: [
							/* @__PURE__ */ f("colgroup", { children: [/* @__PURE__ */ d("col", { className: "wp-servtable-selcol-col" }), ft.map((e) => {
								let t = e.key === cr;
								return /* @__PURE__ */ d("col", {
									className: t ? "wp-servtable-namecol" : void 0,
									style: !t && at[e.key] ? { width: at[e.key] } : void 0
								}, e.key);
							})] }),
							/* @__PURE__ */ d("thead", { children: /* @__PURE__ */ f("tr", { children: [/* @__PURE__ */ d("th", {
								className: "wp-servtable-h wp-servtable-selcol",
								children: Gt ? /* @__PURE__ */ d("input", {
									type: "checkbox",
									className: "wp-servtable-selall",
									checked: Ht,
									ref: (e) => {
										e && (e.indeterminate = Ut);
									},
									disabled: zt.length === 0,
									onChange: Wt,
									title: "Select all models in the current filter (for a bulk residency change)"
								}) : null
							}), ft.map((e) => {
								let t = e.key === cr, n = !t && !dt;
								return /* @__PURE__ */ f("th", {
									className: `wp-col-th${e.sortable ? " wp-lt-sortable" : " wp-servtable-h"}${e.num ? " wp-lt-num" : ""}${t ? " wp-servtable-stickycol wp-servtable-stickyname" : ""}${n && xt === e.key && yt && yt !== e.key ? " wp-col-dragover" : ""}${yt === e.key ? " wp-col-dragging" : ""}`,
									draggable: n,
									onDragStart: n ? (t) => {
										bt(e.key), t.dataTransfer.effectAllowed = "move";
									} : void 0,
									onDragOver: n ? (t) => {
										yt && yt !== e.key && (t.preventDefault(), St(e.key));
									} : void 0,
									onDrop: n ? (t) => {
										t.preventDefault(), yt && yt !== e.key && ot(yt, e.key), bt(null), St(null);
									} : void 0,
									onDragEnd: n ? () => {
										bt(null), St(null);
									} : void 0,
									title: t ? "Click to sort · this column is pinned left (frozen while you scroll sideways) · its width is fluid — long names are shortened in the MIDDLE, hover for the full key" : n ? e.sortable ? "Click to sort · drag to reorder · drag the right edge to resize" : "Drag to reorder · drag the right edge to resize" : e.sortable ? "Click to sort" : void 0,
									onClick: e.sortable ? () => Mt(e.key) : void 0,
									children: [
										e.label,
										e.sortable ? Nt(e.key) : "",
										n && /* @__PURE__ */ d("span", {
											className: "wp-col-resize",
											draggable: !1,
											onDragStart: (e) => e.preventDefault(),
											onClick: (e) => e.stopPropagation(),
											onPointerDown: (t) => Ct(t, e.key)
										})
									]
								}, e.key);
							})] }) }),
							/* @__PURE__ */ f("tbody", { children: [Ot.length === 0 && /* @__PURE__ */ d("tr", { children: /* @__PURE__ */ f("td", {
								colSpan: ht,
								className: "wp-servtable-nomatch",
								children: [
									"no match — clear the filter to see all ",
									(e.models || []).length,
									" assigned"
								]
							}) }), Rt.map((t) => {
								if (t.kind === "pgheader") {
									let e = t.g, n = !!It?.[e.id], r = (e.members || []).length;
									return /* @__PURE__ */ d("tr", {
										className: "wp-pgroup-row",
										children: /* @__PURE__ */ d("td", {
											colSpan: ht,
											children: /* @__PURE__ */ f("button", {
												type: "button",
												className: "wp-pgroup-bar",
												"aria-expanded": !n,
												title: `priority group ${e.id} — click to ${n ? "expand" : "collapse"}`,
												onClick: () => Lt((t) => ({
													...t,
													[e.id]: !t?.[e.id]
												})),
												children: [
													/* @__PURE__ */ d("span", {
														className: "wp-pgroup-caret",
														children: n ? "▸" : "▾"
													}),
													/* @__PURE__ */ f("span", {
														className: "wp-pgroup-name",
														children: ["▣ ", e.name]
													}),
													/* @__PURE__ */ d("span", {
														className: "wp-pgroup-count",
														children: t.count === r ? `${t.count} models` : `${t.count} of ${r} models here`
													}),
													(e.workers || []).length > 0 && /* @__PURE__ */ f("span", {
														className: "wp-pgroup-workers",
														children: ["→ ", e.workers.join(" → ")]
													})
												]
											})
										})
									}, `pg:${e.id}`);
								}
								let { key: r, m: i, d: a } = t, o = a.state, s = Re.has(r), c = !!(B && B.has(r)), l = {
									key: r,
									m: i,
									d: a,
									isSel: s,
									isBlocked: c,
									need: (() => {
										let e = qe(r);
										return e && e.isGguf && e.bytes != null ? {
											bytes: e.bytes,
											gib: e.bytes / ur
										} : null;
									})(),
									compact: dt
								}, u = dt && gt === r;
								return /* @__PURE__ */ f(n, { children: [
									/* @__PURE__ */ f("tr", {
										className: `wp-servtable-row wp-st-${o}${s ? " wp-servtable-sel" : ""}${c ? " wp-servtable-blocked" : ""}${u ? " wp-servtable-rowopen" : ""}`,
										onClick: vt(r),
										children: [/* @__PURE__ */ d("td", {
											className: "wp-servtable-selcol",
											children: Gt ? /* @__PURE__ */ d("input", {
												type: "checkbox",
												className: "wp-servtable-selrow",
												checked: s,
												onChange: () => Be(r),
												title: "Select this model for a bulk residency change"
											}) : null
										}), ft.map((e) => /* @__PURE__ */ d("td", {
											className: `${e.cls || ""}${e.key === "name" ? " wp-servtable-stickycol wp-servtable-stickyname" : ""}`.trim() || void 0,
											title: e.title ? e.title(l) : void 0,
											children: e.render(l)
										}, e.key))]
									}),
									u && /* @__PURE__ */ d("tr", {
										className: "wp-servtable-expand wp-servtable-drawer",
										children: /* @__PURE__ */ d("td", {
											colSpan: ht,
											children: /* @__PURE__ */ f("div", {
												className: "wp-servdrawer",
												children: [
													/* @__PURE__ */ d("div", {
														className: "wp-servdrawer-name",
														title: r,
														children: Ke(r)
													}),
													Ke(r) !== r && /* @__PURE__ */ d("div", {
														className: "wp-servdrawer-key",
														children: r
													}),
													/* @__PURE__ */ d("div", {
														className: "wp-servdrawer-chips",
														children: pt.map((e) => /* @__PURE__ */ f("div", {
															className: "wp-servdrawer-chip",
															title: e.title ? e.title(l) : void 0,
															children: [/* @__PURE__ */ d("span", {
																className: "wp-servdrawer-chip-label",
																children: e.label
															}), /* @__PURE__ */ d("span", {
																className: "wp-servdrawer-chip-body",
																children: e.render(l)
															})]
														}, e.key))
													}),
													/* @__PURE__ */ d("div", {
														className: "wp-servdrawer-actions",
														children: mt.render(l)
													})
												]
											})
										})
									}),
									ye === r && e.config && E && /* @__PURE__ */ d("tr", {
										className: "wp-servtable-expand",
										children: /* @__PURE__ */ d("td", {
											colSpan: ht,
											children: /* @__PURE__ */ d(Hr, {
												mode: e.config.residency?.[r] === "static" ? "static" : "on-demand",
												onClose: () => be(null),
												onPick: (t) => {
													be(null), E(e, r, t);
												}
											})
										})
									})
								] }, r);
							})] })
						]
					})
				})]
			}),
			Ee ? /* @__PURE__ */ d(ai, {
				models: Ge,
				allocation: r || {},
				workerId: e.id,
				worker: e,
				onAllocate: async (t, n) => {
					await F(e, t, n), De(!1);
				},
				onCancel: () => De(!1)
			}) : /* @__PURE__ */ d("button", {
				className: "wp-load-toggle",
				onClick: () => De(!0),
				title: "Load another model onto this worker",
				children: "＋ load a model"
			})
		]
	});
}
//#endregion
//#region src/components/WorkersPanel/WorkersPanel.jsx
function si({ models: e = [], embedded: t = !1 }) {
	let [n, r] = l([]), [a, c] = l(null), [p, m] = Y("hugpy.sess.wp.open", !1), [h, g] = l({
		name: "",
		url: "",
		models: ""
	}), [_, v] = l(!1), [y, b] = l([]), [x, S] = l(null), [C, w] = l(!1), [T, E] = l(""), [D, O] = l(!1), [k, A] = l(null), [j, M] = l({}), [N, P] = l({}), [F, I] = Y("hugpy.sess.wp.tab", ""), L = s(() => F && n.some((e) => e.id === F) ? F : n[0]?.id || "", [F, n]), R = s(() => n.length > 1 ? n.filter((e) => e.id === L) : n, [n, L]), z = s(() => new Set((e || []).filter((e) => e && e.blocked).map((e) => e.model_key ?? e.key)), [e]), [B, V] = l(z);
	o(() => {
		V(z);
	}, [z]);
	let [ee, te] = Y("hugpy.sess.wp.setupOpen", !1), [H, re] = l({}), U = i((e, t) => {
		re((n) => ({
			...n,
			[e]: {
				t: Date.now(),
				...t
			}
		}));
	}, []);
	o(() => {
		if (Object.keys(H).length === 0) return;
		let e = (e, t) => {
			if (!e || !e.config) return !1;
			if (t.kind === "residency") {
				let n = e.config.residency?.[t.model] ?? null;
				return t.value === "static" ? n === "static" : n !== "static";
			}
			if (t.kind === "pinned") return !!e.config.pinned?.[t.model] == !!t.value;
			if (t.kind === "pin_all") {
				let n = e.config.pinned || {};
				return (t.models || []).every((e) => !!n[e] == !!t.value);
			}
			if (t.kind === "residency_all") {
				let n = e.config.residency || {};
				return (t.models || []).every((e) => t.value === "static" ? n[e] === "static" : n[e] !== "static");
			}
			return t.kind === "slot_count" && e.config.slot_count === t.value;
		}, t = () => re((t) => {
			let r = !1, i = { ...t };
			for (let [a, o] of Object.entries(t)) {
				let t = n.find((e) => e.id === a);
				(Date.now() - o.t > 1e4 || e(t, o)) && (delete i[a], r = !0);
			}
			return r ? i : t;
		});
		t();
		let r = setTimeout(t, 10500);
		return () => clearTimeout(r);
	}, [n, H]);
	let W = i(() => {
		let e = K("/api/llm/workers").then((e) => {
			Array.isArray(e) ? (r(e), c(null)) : c("worker list unavailable (kept the last known roster)");
		}).catch((e) => c(e.message)), t = K("/api/llm/slots").then((e) => {
			e && typeof e == "object" && S(e);
		}).catch(() => {});
		return Promise.all([e, t]);
	}, []), G = i(() => {
		K("/api/llm/enroll-tokens").then((e) => b(Array.isArray(e) ? e : [])).catch(() => {});
	}, []);
	o(() => {
		let e = !1, t = null, n = () => {
			Promise.resolve(W()).finally(() => {
				e || (t = setTimeout(n, 1e4));
			});
		};
		return n(), () => {
			e = !0, t && clearTimeout(t);
		};
	}, [W]), o(() => {
		G();
	}, [G]);
	let ie = i(async (e) => {
		O(!0);
		try {
			let t = await K("/api/llm/slots/load", {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({ model_key: e })
			});
			t && t.loaded === !1 && alert(`Not loaded: ${t.reason || "no free slot"}`), w(!1), E(""), W();
		} catch (e) {
			alert(`Load failed: ${e.message}`);
		} finally {
			O(!1);
		}
	}, [W]), ae = i(async (e) => {
		try {
			await K("/api/llm/slots/unload", {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({ control: e })
			}), W();
		} catch (e) {
			alert(`Unload failed: ${e.message}`);
		}
	}, [W]), oe = i(async () => {
		let e = prompt("Label for this enrollment token (e.g. gpu-box-2):", "");
		if (e !== null) try {
			let t = await K("/api/llm/enroll-tokens", {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({ label: e })
			});
			A(t), G();
		} catch (e) {
			alert(`Could not issue token: ${e.message}`);
		}
	}, [G]), se = i(async (e) => {
		if (confirm(`Revoke token "${e.label || e.id}"? Workers using it are refused and their agents stop.`)) try {
			await K(`/api/llm/enroll-tokens/${encodeURIComponent(e.id)}`, { method: "DELETE" }), G();
		} catch (e) {
			alert(`Revoke failed: ${e.message}`);
		}
	}, [G]), ce = i(async (e) => {
		if (e.preventDefault(), h.name.trim()) {
			v(!0);
			try {
				let e = {
					name: h.name.trim(),
					models: h.models.split(",").map((e) => e.trim()).filter(Boolean)
				};
				h.url.trim() && (e.url = h.url.trim()), await K("/api/llm/workers/register", {
					method: "POST",
					headers: { "Content-Type": "application/json" },
					body: JSON.stringify(e)
				}), g({
					name: "",
					url: "",
					models: ""
				}), W();
			} catch (e) {
				alert(`Could not register worker: ${e.message}`);
			} finally {
				v(!1);
			}
		}
	}, [h, W]), le = i(async (e, t, n) => {
		try {
			let r = { model_key: t };
			n && (r.spill = n), await K(`/api/llm/workers/${encodeURIComponent(e.id)}/assign`, {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify(r)
			}), await W();
		} catch (e) {
			alert(`Assign failed: ${e.message}`);
		}
	}, [W]), ue = i(async (e, t, n, r) => {
		try {
			let i = { model_key: t };
			n && (i.spill = n), r && (i.force = !0);
			let a = await K(`/api/llm/workers/${encodeURIComponent(e.id)}/load`, {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify(i)
			});
			a && a.preflight && a.preflight.gpu_resident === !1 && a.preflight.reason && alert(`Placed — note: ${a.preflight.reason}`), W();
		} catch (r) {
			confirm(`${r.message}\n\nForce-load anyway? (may OOM the worker)`) && ue(e, t, n, !0);
		}
	}, [W]), de = i(async (e, t, n) => {
		let r = t.filter((t) => !(t.models || []).includes(e)), i = await Promise.all(r.map(async (t) => {
			try {
				let r = { model_key: e };
				n && (r.force = !0);
				let i = await K(`/api/llm/workers/${encodeURIComponent(t.id)}/load`, {
					method: "POST",
					headers: { "Content-Type": "application/json" },
					body: JSON.stringify(r)
				}), a = i && i.preflight && i.preflight.gpu_resident === !1;
				return {
					id: t.id,
					name: t.name,
					ok: !0,
					note: a ? "CPU-spilled" : ""
				};
			} catch (e) {
				return {
					id: t.id,
					name: t.name,
					ok: !1,
					note: e.message
				};
			}
		}));
		return W(), i;
	}, [W]), fe = i(async (e, t, n) => {
		let r = await Promise.all((t || []).map(async (t) => {
			try {
				let n = await K(`/api/llm/workers/${encodeURIComponent(e.id)}/load`, {
					method: "POST",
					headers: { "Content-Type": "application/json" },
					body: JSON.stringify({ model_key: t })
				});
				return {
					mk: t,
					ok: !0,
					note: n && n.preflight && n.preflight.gpu_resident === !1 ? "CPU-spilled" : ""
				};
			} catch (e) {
				return {
					mk: t,
					ok: !1,
					note: e.message
				};
			}
		}));
		W();
		let i = r.filter((e) => e.ok).length, a = r.filter((e) => !e.ok), o = `Allocated ${i}/${r.length} to ${e.name}${n ? " (duplicate-allocation breaker was ON)" : ""}.`;
		a.length && (o += "\n\nRefused:\n" + a.map((e) => `  • ${e.mk} — ${e.note}`).join("\n")), (a.length || i !== r.length) && alert(o);
	}, [W]), pe = i(async (e, t) => {
		try {
			await K(`/api/llm/workers/${encodeURIComponent(e.id)}/unassign`, {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({ model_key: t })
			}), W();
		} catch (e) {
			alert(`Unassign failed: ${e.message}`);
		}
	}, [W]), me = i(async (e, t) => {
		if (!(t && !confirm(`Block "${e}" from the serving pool?\n\nIt will no longer be routed to, assigned, warmed, or used as a fallback default anywhere — files stay on disk and existing designations stay recorded (inert). Reversible.`))) {
			V((n) => {
				let r = new Set(n);
				return t ? r.add(e) : r.delete(e), r;
			});
			try {
				await K(`/api/llm/models/${encodeURIComponent(e)}/${t ? "block" : "unblock"}`, {
					method: "POST",
					headers: { "Content-Type": "application/json" },
					body: JSON.stringify({})
				}), W();
			} catch (n) {
				V((n) => {
					let r = new Set(n);
					return t ? r.delete(e) : r.add(e), r;
				}), alert(`${t ? "Block" : "Unblock"} failed: ${n.message}`);
			}
		}
	}, [W]), he = i(async (e, t) => {
		try {
			let n = await K(`/api/llm/workers/${encodeURIComponent(e.id)}/unload`, {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({ model_key: t })
			});
			n && n.ok === !1 && alert(`Free failed: ${n.error || "unknown error"}`), W();
		} catch (e) {
			alert(`Free failed: ${e.message}`);
		}
	}, [W]), ge = i(async (e, t) => {
		try {
			let n = await K(`/api/llm/workers/${encodeURIComponent(e.id)}/evict`, {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({ model_key: t })
			});
			return n && n.ok === !1 ? alert(`Evict failed: ${n.error || n.reason || "unknown error"}`) : n && n.evicted === !1 && alert(`Not evicted: ${n.reason || "model is not resident on this worker"}`), W(), n;
		} catch (e) {
			alert(`Evict failed: ${e.message}`);
		}
	}, [W]), _e = i(async (e) => {
		if (confirm(`Unload all models from ${e.name}'s GPU? They stay assigned and reload on demand.`)) try {
			let t = await K(`/api/llm/workers/${encodeURIComponent(e.id)}/unload`, {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({ all: !0 })
			});
			t && t.ok === !1 && alert(`Free failed: ${t.error || "unknown error"}`), W();
		} catch (e) {
			alert(`Free failed: ${e.message}`);
		}
	}, [W]), ve = i(async (e) => {
		try {
			let t = await K(`/api/llm/workers/${encodeURIComponent(e.id)}/free-ram`, {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({})
			});
			if (t && t.ok === !1) {
				alert(`Free RAM failed: ${t.error || "unknown error"}`);
				return;
			}
			let n = t && typeof t.ram_freed == "number" ? t.ram_freed : null;
			alert(n && n > 0 ? `Freed ${(n / 1073741824).toFixed(1)} GiB RAM on ${e.name}` : `No reclaimable RAM on ${e.name} right now.`), W();
		} catch (e) {
			alert(`Free RAM failed: ${e.message}`);
		}
	}, [W]), ye = i(async (e) => {
		if (!confirm(`Restart ${e.name}'s worker agent? It drops all loaded models and re-execs the agent.`)) return;
		M((t) => ({
			...t,
			[e.id]: !0
		}));
		let t = () => M((t) => {
			let n = { ...t };
			return delete n[e.id], n;
		});
		try {
			await K(`/api/llm/workers/${encodeURIComponent(e.id)}/restart`, {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({})
			});
		} catch (e) {
			if (!/\b503\b|AgentRestarting/i.test(e.message || "")) {
				alert(`Restart failed: ${e.message}`), t();
				return;
			}
		}
		setTimeout(t, 12e3), W();
	}, [W]), be = i(async (e) => {
		if (!confirm(`Update ${e.name} to central's required version (${e.required_pkg_version || "unknown"})? The worker pip-installs and restarts itself.`)) return;
		P((t) => ({
			...t,
			[e.id]: !0
		}));
		let t = () => P((t) => {
			let n = { ...t };
			return delete n[e.id], n;
		});
		try {
			await K(`/api/llm/workers/${encodeURIComponent(e.id)}/update`, {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({})
			});
		} catch (e) {
			if (!/\b503\b|AgentRestarting/i.test(e.message || "")) {
				alert(`Update failed: ${e.message}`), t();
				return;
			}
		}
		setTimeout(t, 2e4), W();
	}, [W]), xe = i(async (e) => {
		try {
			let t = await K(`/api/llm/workers/${encodeURIComponent(e.id)}/reap`, {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({ dry_run: !0 })
			}), n = t && t.reclaimable || [];
			if (n.length === 0) {
				alert(`Nothing to reclaim on ${e.name}. Everything on disk is assigned, loaded, or 📌pinned.`);
				return;
			}
			let r = (t.reclaimable_bytes || 0) / 1073741824, i = n.slice(0, 12).map((e) => `  • ${e.model_key} (${(e.bytes / 1073741824).toFixed(1)} GiB)`).join("\n"), a = n.length > 12 ? `\n  …and ${n.length - 12} more` : "";
			if (!confirm(`Reclaim ${r.toFixed(1)} GiB from ${e.name} by deleting ${n.length} unassigned model(s)?\n\n${i}${a}\n\nProtected (assigned / loaded / 🔒static) files are left untouched. 📌 Pin does not protect files — it keeps the allocation/routing only.`)) return;
			let o = await K(`/api/llm/workers/${encodeURIComponent(e.id)}/reap`, {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({ all: !0 })
			}), s = (o && o.freed_bytes || 0) / 1073741824, c = (o && o.results || []).filter((e) => !e.ok), l = `Reclaimed ${s.toFixed(1)} GiB from ${e.name}.`;
			c.length && (l += `\n\nSkipped ${c.length}:\n` + c.map((e) => `  • ${e.model_key} — ${e.reason}`).join("\n")), alert(l), W();
		} catch (e) {
			alert(`Reclaim failed: ${e.message}`);
		}
	}, [W]), Se = i(async (e) => {
		let t = e.storage && e.storage.proposed_evictions || [];
		if (t.length === 0) {
			alert(`Nothing proposed for eviction on ${e.name} — it isn't over budget.`);
			return;
		}
		let n = t.map((e) => e.model_key), r = (e.storage && e.storage.proposed_free_bytes || 0) / 1e9, i = t.slice(0, 12).map((e) => `  • ${e.model_key} (${(e.bytes / 1e9).toFixed(1)} GB)`).join("\n"), a = t.length > 12 ? `\n  …and ${t.length - 12} more` : "";
		if (!confirm(`Approve eviction on ${e.name}?\n\nFrees ~${r.toFixed(1)} GB by deleting ${n.length} cold, unprotected model(s):\n\n${i}${a}\n\nLoaded / 🔒static / assigned files are never touched. 📌 Pinned files ARE eligible — pin keeps the allocation, not the bytes (they re-pull on next call). Central re-checks this list and the worker re-proves each model before deleting.`)) return;
		let o = `/api/llm/workers/${encodeURIComponent(e.id)}/reap-approve`, s = {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({ model_keys: n })
		};
		try {
			let t;
			try {
				t = await K(o, s);
			} catch (e) {
				throw /\b404\b|not found/i.test(e.message || "") ? Error("reap-approve is not available on this central/worker (needs 0.1.137+). Refusing to fall back to the un-guarded reaper.") : e;
			}
			let n = (t && t.freed_bytes || 0) / 1e9, r = (t && t.results || []).filter((e) => !e.ok), i = `Freed ${n.toFixed(1)} GB from ${e.name}.`;
			t && t.note && (i += `\n\n${t.note}`), r.length && (i += `\n\nSkipped ${r.length}:\n` + r.map((e) => `  • ${e.model_key} — ${e.reason}`).join("\n")), alert(i), W();
		} catch (e) {
			alert(`Eviction failed: ${e.message}`);
		}
	}, [W]), Ce = i(async (e) => {
		if (confirm(`Remove worker ${e.name} from the pool?`)) try {
			await K(`/api/llm/workers/${encodeURIComponent(e.id)}`, { method: "DELETE" }), W();
		} catch (e) {
			alert(`Remove failed: ${e.message}`);
		}
	}, [W]), we = i(async (e) => {
		try {
			await K(`/api/llm/workers/${encodeURIComponent(e.id)}/admit`, { method: "POST" }), W();
		} catch (e) {
			alert(`Admit failed: ${e.message}`);
		}
	}, [W]), q = i(async (e) => {
		if (confirm(`Block ${e.name}? It stops serving and its agent exits on next contact.`)) try {
			await K(`/api/llm/workers/${encodeURIComponent(e.id)}/block`, { method: "POST" }), W();
		} catch (e) {
			alert(`Block failed: ${e.message}`);
		}
	}, [W]), Te = i(async (e) => {
		let t = window.prompt(`Dedicated pool for "${e.name}" (blank = general). Requests tagged for this pool route here; general traffic won't.`, e.pool || "");
		if (t !== null) try {
			await K(`/api/llm/workers/${encodeURIComponent(e.id)}/pool`, {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({ pool: t.trim() })
			}), W();
		} catch (e) {
			alert(`Set pool failed: ${e.message}`);
		}
	}, [W]), Ee = i(async (e) => {
		let t = e.config?.slot_count, n = window.prompt(`Slot count for "${e.name}" (0–16; currently ${t ?? "?"}${e.config?.slot_count_source ? ` from ${e.config.slot_count_source}` : ""}).\n0 = no slots (in-process only). Applies via a ~5s agent restart.`, t == null ? "" : String(t));
		if (n === null || n.trim() === "") return;
		let r = Number(n);
		if (!Number.isInteger(r) || r < 0 || r > 16) {
			alert("slot count must be an integer 0–16");
			return;
		}
		try {
			let t = await K(`/api/llm/workers/${encodeURIComponent(e.id)}/config`, {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({ slot_count: r })
			});
			t && t.ok === !1 ? alert(`Config failed: ${t.error?.message || "unknown"}`) : t && t.restarting && U(e.id, {
				kind: "slot_count",
				value: r
			}), W();
		} catch (t) {
			alert(H[e.id] ? "The agent is restarting to apply the previous change — retry in a few seconds." : `Config failed: ${t.message}`);
		}
	}, [
		W,
		U,
		H
	]), De = i(async (e, t, n) => {
		try {
			let r = await K(`/api/llm/workers/${encodeURIComponent(e.id)}/config`, {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({ residency: { [t]: n === "static" ? "static" : null } })
			});
			r && r.ok === !1 ? alert(`Residency failed: ${r.error?.message || "unknown"}`) : r && r.restarting && U(e.id, {
				kind: "residency",
				model: t,
				value: n
			}), W();
		} catch (t) {
			alert(H[e.id] ? "The agent is restarting to apply the previous change — retry in a few seconds." : `Residency failed: ${t.message}`);
		}
	}, [
		W,
		U,
		H
	]), Oe = i(async (e, t, n) => {
		let r = (t || []).filter((t) => (e.models || []).includes(t));
		if (r.length === 0) {
			alert("No models selected.");
			return;
		}
		let i = n === "static" ? "🔒 static" : "⏲ on-demand", a = n === "static" ? "Static is a locked seat: the model is kept on this worker and never evicted (the only tier that keeps files on disk). " : "On-demand is the default: the model loads on call and yields its seat when another model needs it. ";
		if (confirm(`Set residency → ${i} for ${r.length} selected model${r.length === 1 ? "" : "s"} on ${e.name}?\n\n` + a + "The worker agent restarts once (~5s) to apply all of them together.")) try {
			let t = await K(`/api/llm/workers/${encodeURIComponent(e.id)}/residency-all`, {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({
					model_keys: r,
					mode: n
				})
			});
			if (t && t.ok === !1 && !t.results) {
				alert(`Residency change failed: ${t.error?.message || "unknown"}`);
				return;
			}
			t && t.restarting && U(e.id, {
				kind: "residency_all",
				models: r,
				value: n === "static" ? "static" : "on-demand"
			});
			let a = t?.counts?.ok ?? 0, o = t?.counts?.error ?? 0;
			if (o > 0) {
				let n = Object.entries(t.results || {}).filter(([, e]) => e !== "ok");
				alert(`Set residency → ${i}: ${a}/${a + o} on ${e.name}.\n\nFailed:\n` + n.map(([e, t]) => `  • ${e} — ${t}`).join("\n"));
			}
			W();
		} catch (t) {
			alert(H[e.id] ? "The agent is restarting to apply the previous change — retry in a few seconds." : `Set residency failed: ${t.message}`);
		}
	}, [
		W,
		U,
		H
	]), ke = i(async (t, n, r, i) => {
		let a = (n || []).filter((e) => (t.models || []).includes(e));
		if (a.length === 0) {
			alert("No models selected.");
			return;
		}
		let o = !!(i && Object.keys(i).length > 0), s = r && r.alloc_mode ? String(r.alloc_mode) : null, c = !o && (!r || Object.keys(r).length === 0), l = o ? "Explicit" : c ? "Default (derived)" : s ? yr(s) : _r(r), u = (t) => String(e.find((e) => (e.model_key ?? e.key) === t)?.framework || "").toLowerCase(), d = (e) => e === "gguf" || e === "llama_cpp", f = a.filter((e) => d(u(e))), p = s === "max-ram" || s === "explicit" || o && Object.values(i).some((e) => e && e.alloc_mode === "explicit") || gr(r), m = c ? `Revert ${a.length} selected model${a.length === 1 ? "" : "s"} to the DERIVED default on ${t.name}?\n\nClears any pinned allocation contract so each model tracks its derived default again (and improves with it as measured values land). Applies on next load — no agent restart.` : `Set GPU allocation → ${l} for ${a.length} selected model${a.length === 1 ? "" : "s"} on ${t.name}?\n\nThe allocation is the model's resource contract on this worker; it applies the next time each model loads (no agent restart).`;
		if (p) {
			let e = a.length - f.length;
			if (m = `Set GPU allocation → ${l} (GGUF-only) on ${t.name}?\n\nApplies to ${f.length} GGUF model${f.length === 1 ? "" : "s"}; ${e} transformers/comfy model${e === 1 ? "" : "s"} skipped with a reason (${l} is a GGUF concept; Default / Max GPU / GPU only / RAM only would apply to all).\n\nApplies on next load — no agent restart.`, f.length === 0) {
				alert(`None of the ${a.length} selected models are GGUF — "${l}" is GGUF-only and would touch nothing. Use Default, Max GPU, GPU only, or RAM only to affect non-GGUF models.`);
				return;
			}
		}
		if (o && (m += "\n\n(An explicit % split is in play — each model resolves it against its OWN size, so the actual GiB numbers differ per model.)"), confirm(m)) try {
			let e = o ? {
				model_keys: a,
				spills: Object.fromEntries(a.map((e) => [e, i[e] || {}]))
			} : {
				model_keys: a,
				spill: r || {}
			}, n = await K(`/api/llm/workers/${encodeURIComponent(t.id)}/alloc-all`, {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify(e)
			});
			if (n && n.ok === !1 && !n.results) {
				alert(`Allocation change failed: ${n.error?.message || "unknown"}`);
				return;
			}
			let s = n?.counts?.ok ?? 0, c = n?.counts?.error ?? 0, u = n?.counts?.skipped ?? 0, d = Object.entries(n.results || {}).filter(([, e]) => e !== "ok");
			d.length > 0 && alert(`Set alloc → ${l}: ${s} applied` + (u ? `, ${u} skipped` : "") + (c ? `, ${c} failed` : "") + ` on ${t.name}.\n\n` + d.map(([e, t]) => `  • ${e} — ${t}`).join("\n")), W();
		} catch (e) {
			alert(`Set alloc failed: ${e.message}`);
		}
	}, [W, e]), Ae = i(async (e, t, n) => {
		try {
			let r = await K(`/api/llm/workers/${encodeURIComponent(e.id)}/config`, {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({ pinned: { [t]: n ? !0 : null } })
			});
			r && r.ok === !1 ? alert(`Pin failed: ${r.error?.message || "unknown"}`) : r && r.restarting && U(e.id, {
				kind: "pinned",
				model: t,
				value: n
			}), W();
		} catch (t) {
			alert(H[e.id] ? "The agent is restarting to apply the previous change — retry in a few seconds." : `Pin failed: ${t.message}`);
		}
	}, [
		W,
		U,
		H
	]), je = i(async (e) => {
		let t = e.models || [];
		if (t.length === 0) {
			alert(`${e.name} has no assigned models to pin.`);
			return;
		}
		if (t.filter((t) => !e.config?.pinned?.[t]).length === 0) {
			alert(`All ${t.length} model${t.length === 1 ? "" : "s"} on ${e.name} are already pinned.`);
			return;
		}
		if (confirm(`📌 Pin all ${t.length} model${t.length === 1 ? "" : "s"} on ${e.name}?\n\nPinning is PERMANENT attribution: each pinned model then refuses unassign ("unpin first") until you Unpin all. The worker agent restarts (~5s) to apply.`)) try {
			let n = await K(`/api/llm/workers/${encodeURIComponent(e.id)}/pin-all`, {
				method: "POST",
				headers: { "Content-Type": "application/json" }
			});
			n && n.restarting && U(e.id, {
				kind: "pin_all",
				models: t,
				value: !0
			});
			let r = n?.counts?.ok ?? 0, i = n?.counts?.error ?? 0;
			if (i > 0) {
				let t = Object.entries(n.results || {}).filter(([, e]) => e !== "ok");
				alert(`Pinned ${r}/${r + i} on ${e.name}.\n\nFailed:\n` + t.map(([e, t]) => `  • ${e} — ${t}`).join("\n"));
			}
			W();
		} catch (t) {
			alert(H[e.id] ? "The agent is restarting to apply the previous change — retry in a few seconds." : `Pin all failed: ${t.message}`);
		}
	}, [
		W,
		U,
		H
	]), Me = i(async (e) => {
		let t = e.models || [], n = t.filter((t) => e.config?.pinned?.[t]);
		if (n.length === 0) {
			alert(`No pinned models on ${e.name}.`);
			return;
		}
		if (confirm(`Unpin all ${n.length} pinned model${n.length === 1 ? "" : "s"} on ${e.name}?\n\nThis is the undo for Pin all — the models can be unassigned again afterward. The worker agent restarts (~5s) to apply.`)) try {
			let n = await K(`/api/llm/workers/${encodeURIComponent(e.id)}/unpin-all`, {
				method: "POST",
				headers: { "Content-Type": "application/json" }
			});
			n && n.restarting && U(e.id, {
				kind: "pin_all",
				models: t,
				value: !1
			});
			let r = n?.counts?.ok ?? 0, i = n?.counts?.error ?? 0;
			if (i > 0) {
				let t = Object.entries(n.results || {}).filter(([, e]) => e !== "ok");
				alert(`Unpinned ${r}/${r + i} on ${e.name}.\n\nFailed:\n` + t.map(([e, t]) => `  • ${e} — ${t}`).join("\n"));
			}
			W();
		} catch (t) {
			alert(H[e.id] ? "The agent is restarting to apply the previous change — retry in a few seconds." : `Unpin all failed: ${t.message}`);
		}
	}, [
		W,
		U,
		H
	]), Ne = i(async (e, t) => {
		try {
			await K(`/api/llm/workers/${encodeURIComponent(e.id)}/limits`, {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({ limits: t })
			}), W();
		} catch (e) {
			alert(`Set limits failed: ${e.message}`);
		}
	}, [W]), Pe = i(async () => {
		let e = n.filter((e) => e.status !== "online");
		e.length && confirm(`Remove ${e.length} offline worker(s) from the pool?`) && (await Promise.all(e.map((e) => K(`/api/llm/workers/${encodeURIComponent(e.id)}`, { method: "DELETE" }).catch(() => {}))), W());
	}, [n, W]), Fe = n.filter((e) => e.status === "online").length, Ie = n.length - Fe, Le = n.filter((e) => e.admission === "pending").length, Re = s(() => {
		let e = 0, t = 0, r = 0, i = !1;
		for (let a of n) if (a.status === "online") {
			r += (a.loaded_models || []).length;
			for (let n of a.gpus || []) n.memory_total != null && (e += n.memory_total, i = !0), n.memory_free != null && (t += n.memory_free);
		}
		return {
			total: e,
			free: t,
			used: Math.max(e - t, 0),
			serving: r,
			hasVram: i
		};
	}, [n]), ze = n.length === 0, Be = ne(), Ve = s(() => {
		try {
			let e = new URL(Be).hostname.replace(/^\[|\]$/g, "");
			return e === "localhost" || e === "::1" || e === "0.0.0.0" || e.startsWith("127.");
		} catch {
			return !1;
		}
	}, [Be]), [He, Ue] = l(null);
	o(() => {
		Ve && K("/api/llm/workers/central-address").then((e) => {
			e && e.lan_ip && e.base_url && Ue(e.base_url);
		}).catch(() => {});
	}, [Ve]);
	let J = `curl -fsSL ${Ve && He || Be}/api/llm/workers/install.sh | bash`, We = h.models.split(",").map((e) => e.trim()).filter(Boolean), Ge = (e) => g((t) => {
		let n = t.models.split(",").map((e) => e.trim()).filter(Boolean);
		return n.includes(e) ? t : {
			...t,
			models: [...n, e].join(",")
		};
	}), Ke = (e) => g((t) => ({
		...t,
		models: t.models.split(",").map((e) => e.trim()).filter(Boolean).filter((t) => t !== e).join(",")
	})), qe = s(() => {
		let e = {};
		for (let t of n) for (let n of t.models || []) (e[n] ??= []).push({
			id: t.id,
			name: t.name
		});
		return e;
	}, [n]);
	return /* @__PURE__ */ f("div", {
		className: "workers-panel",
		children: [/* @__PURE__ */ f("div", {
			className: `wp-bar${t ? " wp-bar-static" : ""}`,
			onClick: t ? void 0 : () => m((e) => !e),
			children: [
				/* @__PURE__ */ d("span", {
					className: "wp-title",
					children: "🖧 GPU Workers"
				}),
				/* @__PURE__ */ f("span", {
					className: "wp-count",
					children: [
						Fe,
						" online / ",
						n.length,
						" total"
					]
				}),
				Re.hasVram && /* @__PURE__ */ f("span", {
					className: "wp-fleet",
					title: "VRAM used / total across online workers",
					children: [
						"VRAM ",
						X(Re.used),
						" / ",
						X(Re.total),
						" · ",
						X(Re.free),
						" free"
					]
				}),
				Re.serving > 0 && /* @__PURE__ */ f("span", {
					className: "wp-fleet-loaded",
					title: "models serving (resident in VRAM) across the fleet",
					children: [
						"🔥 ",
						Re.serving,
						" serving"
					]
				}),
				Le > 0 && /* @__PURE__ */ f("span", {
					className: "wp-pending-chip",
					title: "Workers awaiting your approval — they don't serve until admitted",
					children: [
						"⏳ ",
						Le,
						" pending"
					]
				}),
				a && /* @__PURE__ */ f("span", {
					className: "wp-err",
					title: a,
					children: ["registry error", /* @__PURE__ */ d(ar, { doc: "registry-error" })]
				}),
				Ie > 0 && /* @__PURE__ */ f("button", {
					className: "wp-prune",
					title: "Remove all offline workers",
					onClick: (e) => {
						e.stopPropagation(), Pe();
					},
					children: [
						"clear ",
						Ie,
						" offline"
					]
				}),
				!t && /* @__PURE__ */ d("span", {
					className: "wp-toggle",
					children: p ? "▾" : "▸"
				})
			]
		}), (t || p) && /* @__PURE__ */ f("div", {
			className: "wp-body",
			children: [
				n.length === 0 && /* @__PURE__ */ d("div", {
					className: "wp-empty",
					children: "No workers have joined the pool yet."
				}),
				n.length > 1 && /* @__PURE__ */ d("div", {
					className: "wp-tabs",
					role: "tablist",
					children: n.map((e) => /* @__PURE__ */ f("button", {
						role: "tab",
						"aria-selected": e.id === L,
						className: `wp-tab wp-tab-${e.status || "unknown"}${e.id === L ? " wp-tab-on" : ""}`,
						onClick: () => I(e.id),
						title: `${e.name || e.id} — ${e.status || "unknown"}, ${(e.models || []).length} assigned`,
						children: [
							/* @__PURE__ */ d("span", { className: "wp-tab-dot" }),
							/* @__PURE__ */ d("span", {
								className: "wp-tab-name",
								children: e.name || e.id.slice(0, 8)
							}),
							/* @__PURE__ */ d("span", {
								className: "wp-tab-count",
								children: (e.models || []).length
							})
						]
					}, e.id))
				}),
				R.map((t) => /* @__PURE__ */ d(oi, {
					worker: t,
					models: e,
					allocation: qe,
					onAssign: le,
					onRefresh: W,
					onLoad: ue,
					onUnassign: pe,
					onRemove: Ce,
					onFree: he,
					onFreeAll: _e,
					onFreeRam: ve,
					onRestart: ye,
					restarting: !!j[t.id],
					onUpdate: be,
					updating: !!N[t.id],
					onAdmit: we,
					onBlock: q,
					onSetPool: Te,
					onSetLimits: Ne,
					onSetConfig: Ee,
					onSetResidency: De,
					onSetResidencyMany: Oe,
					onSetAllocMany: ke,
					onTogglePin: Ae,
					onPinAll: je,
					onUnpinAll: Me,
					onReap: xe,
					onApproveEvictions: Se,
					onEvict: ge,
					onAllocateMany: fe,
					applying: !!H[t.id],
					blockedKeys: B,
					onToggleBlock: me
				}, t.id)),
				x && /* @__PURE__ */ d("div", {
					className: "wp-central-footer",
					children: /* @__PURE__ */ f("div", {
						className: "wp-worker wp-online wp-central",
						children: [
							/* @__PURE__ */ f("div", {
								className: "wp-worker-head",
								children: [
									/* @__PURE__ */ d("span", { className: "wp-dot" }),
									/* @__PURE__ */ d("span", {
										className: "wp-name",
										children: "central"
									}),
									/* @__PURE__ */ d("span", {
										className: "wp-status",
										children: "online"
									}),
									/* @__PURE__ */ d("span", {
										className: "wp-adm wp-adm-pill-approved",
										title: "The console's own server",
										children: "this host"
									}),
									/* @__PURE__ */ f("span", {
										className: "wp-url",
										title: "Serves via its local slot pool; also the fallback when no worker can take a model",
										children: ["local slot pool", x.enabled === !1 ? " (disabled)" : ""]
									})
								]
							}),
							x.resources && /* @__PURE__ */ f("div", {
								className: "wp-ram",
								title: "Central's own RAM — what the slot preflight budgets against (minus reserves)",
								children: [
									"🧠 RAM ",
									X(x.resources.available_bytes ?? x.resources.free_bytes),
									" free",
									x.resources.total_bytes != null && /* @__PURE__ */ f(u, { children: [" of ", X(x.resources.total_bytes)] }),
									x.resources.cpu_count != null && /* @__PURE__ */ f(u, { children: [
										" · ",
										x.resources.cpu_count,
										" cores"
									] })
								]
							}),
							/* @__PURE__ */ f("div", {
								className: "wp-models",
								children: [
									/* @__PURE__ */ d("span", {
										className: "wp-models-label",
										children: "Slots:"
									}),
									(x.slots || []).length === 0 && /* @__PURE__ */ d("span", {
										className: "wp-none",
										children: "— no slots —"
									}),
									(x.slots || []).map((e) => {
										let t = e.model_key ? e.healthy ? "serving" : "warming" : "idle";
										return /* @__PURE__ */ f("span", {
											className: `wp-model wp-st-${t}`,
											children: [
												/* @__PURE__ */ d("span", {
													className: `wp-state-pill wp-pill-${t}`,
													children: e.model_key ? e.healthy ? "🔥 serving" : "⏳ warming" : `○ slot ${e.slot_id}`
												}),
												/* @__PURE__ */ d("span", {
													className: "wp-model-name",
													children: e.model_key || "— idle —"
												}),
												/* @__PURE__ */ f("span", {
													className: "wp-model-facts",
													children: [
														e.rss_bytes ? `${X(e.rss_bytes)} RSS` : "",
														e.n_gpu_layers == null ? "" : ` · ngl ${e.n_gpu_layers}`,
														e.ctx == null ? "" : ` · ctx ${e.ctx}`
													]
												}),
												e.model_key && /* @__PURE__ */ d("button", {
													className: "wp-free",
													title: "Unload this slot (frees its RAM/VRAM)",
													onClick: () => ae(e._control),
													children: "⏏"
												})
											]
										}, e.slot_id);
									})
								]
							}),
							C ? /* @__PURE__ */ f("div", {
								className: "wp-assign-row",
								children: [
									/* @__PURE__ */ d(rr, {
										models: e,
										value: T,
										onPick: E,
										placeholder: "Load a model into a free central slot…",
										autoFocus: !0
									}),
									/* @__PURE__ */ d("button", {
										className: "wp-alloc-apply",
										disabled: !T || D,
										onClick: () => ie(T),
										children: D ? "…" : "+ Load"
									}),
									/* @__PURE__ */ d("button", {
										className: "wp-load-cancel",
										title: "Cancel",
										onClick: () => {
											w(!1), E("");
										},
										children: "×"
									})
								]
							}) : /* @__PURE__ */ d("button", {
								className: "wp-load-toggle",
								onClick: () => w(!0),
								title: "Load a model into a free central slot",
								children: "＋ load a model"
							})
						]
					})
				}),
				n.length > 0 && /* @__PURE__ */ d(Tr, {
					models: e,
					workers: n,
					onGroupAssign: de
				}),
				/* @__PURE__ */ f("details", {
					className: "wp-setup",
					open: ee || ze,
					onToggle: (e) => {
						ze || te(e.currentTarget.open);
					},
					children: [
						/* @__PURE__ */ d("summary", {
							className: "wp-setup-summary",
							children: "＋ Add & manage workers — install command, enrollment tokens, manual add"
						}),
						/* @__PURE__ */ f("div", {
							className: "wp-install",
							children: [
								/* @__PURE__ */ d("span", {
									className: "wp-install-label",
									children: "Add a GPU box — run on the worker:"
								}),
								/* @__PURE__ */ d("code", {
									className: "wp-install-cmd",
									title: "Click to copy",
									onClick: () => navigator.clipboard?.writeText(J),
									children: J
								}),
								/* @__PURE__ */ d("span", {
									className: "wp-install-note",
									children: "Central fills in its own address and this box's reachable IP — no per-worker config."
								}),
								Ve && !He && /* @__PURE__ */ f("span", {
									className: "wp-install-warn",
									children: [
										"⚠ You're browsing central at ",
										Be,
										" — a worker box can't reach that address. Replace it with one the worker can reach (e.g. central's LAN IP).",
										/* @__PURE__ */ d(ar, { doc: "worker-join" })
									]
								})
							]
						}),
						/* @__PURE__ */ f("details", {
							className: "wp-tokens",
							children: [/* @__PURE__ */ f("summary", { children: [
								"Enrollment tokens (",
								y.filter((e) => !e.revoked).length,
								" active) — admit machines to the fleet"
							] }), /* @__PURE__ */ f("div", {
								className: "wp-tokens-body",
								children: [
									/* @__PURE__ */ d("button", {
										className: "wp-token-issue",
										onClick: oe,
										children: "+ Issue enrollment token"
									}),
									k && /* @__PURE__ */ f("div", {
										className: "wp-token-new",
										children: [
											/* @__PURE__ */ d("strong", { children: "Copy this token now — it is shown only once:" }),
											/* @__PURE__ */ d("code", {
												className: "wp-token-secret",
												title: "Click to copy",
												onClick: () => navigator.clipboard?.writeText(k.token),
												children: k.token
											}),
											/* @__PURE__ */ d("span", {
												className: "wp-install-note",
												children: "Run on the worker (bakes central + token into its unit):"
											}),
											/* @__PURE__ */ d("code", {
												className: "wp-install-cmd",
												title: "Click to copy",
												onClick: () => navigator.clipboard?.writeText(`WORKER_ENROLL_TOKEN=${k.token} ${J}`),
												children: `WORKER_ENROLL_TOKEN=${k.token} ${J}`
											}),
											/* @__PURE__ */ d("button", {
												className: "wp-token-dismiss",
												onClick: () => A(null),
												children: "Done"
											})
										]
									}),
									y.length === 0 && /* @__PURE__ */ d("div", {
										className: "wp-none",
										children: "No enrollment tokens issued."
									}),
									y.map((e) => /* @__PURE__ */ f("div", {
										className: `wp-token-row${e.revoked ? " wp-token-revoked" : ""}`,
										children: [
											/* @__PURE__ */ d("span", {
												className: "wp-token-label",
												children: e.label || "(no label)"
											}),
											/* @__PURE__ */ d("span", {
												className: "wp-token-id",
												title: "token id",
												children: e.id
											}),
											e.revoked ? /* @__PURE__ */ d("span", {
												className: "wp-token-state",
												children: "revoked"
											}) : /* @__PURE__ */ d("button", {
												className: "wp-token-revoke",
												title: "Revoke — its workers are refused and stop",
												onClick: () => se(e),
												children: "revoke"
											})
										]
									}, e.id))
								]
							})]
						}),
						/* @__PURE__ */ f("details", {
							className: "wp-manual",
							children: [/* @__PURE__ */ d("summary", { children: "Add manually" }), /* @__PURE__ */ f("form", {
								className: "wp-register",
								onSubmit: ce,
								children: [
									/* @__PURE__ */ d("input", {
										placeholder: "worker name (e.g. gpu-box-1)",
										value: h.name,
										onChange: (e) => g((t) => ({
											...t,
											name: e.target.value
										}))
									}),
									/* @__PURE__ */ d("input", {
										placeholder: "worker URL (optional — central uses source IP)",
										value: h.url,
										onChange: (e) => g((t) => ({
											...t,
											url: e.target.value
										}))
									}),
									/* @__PURE__ */ f("div", {
										className: "wp-models-pick",
										children: [/* @__PURE__ */ d(rr, {
											models: e.filter((e) => !We.includes(e.model_key ?? e.key)),
											value: "",
											onPick: Ge,
											placeholder: e.length ? "models to serve (optional)…" : "no models registered yet",
											disabled: !e.length
										}), We.length > 0 && /* @__PURE__ */ d("div", {
											className: "wp-models-chips",
											children: We.map((t) => /* @__PURE__ */ f("span", {
												className: "wp-models-chip",
												children: [e.find((e) => (e.model_key ?? e.key) === t)?.name || t, /* @__PURE__ */ d("button", {
													type: "button",
													className: "wp-chip-x",
													title: "Remove",
													onClick: () => Ke(t),
													children: "×"
												})]
											}, t))
										})]
									}),
									/* @__PURE__ */ d("button", {
										type: "submit",
										disabled: _,
										children: "+ Add worker"
									})
								]
							})]
						})
					]
				})
			]
		})]
	});
}
//#endregion
//#region src/components/PriorityGroupsPanel/PriorityGroupsPanel.jsx
var ci = {
	chosen: "✓",
	blocked: "⛔",
	missing: "∅",
	"no-worker": "○",
	"lower-priority": "↓"
};
function li({ mk: e, i: t, n, onMove: r, onRemove: i }) {
	let a = String(e).toLowerCase().startsWith("group:");
	return /* @__PURE__ */ f("li", {
		className: `pg-member${a ? " pg-member-ref" : ""}`,
		children: [
			/* @__PURE__ */ d("span", {
				className: "pg-rank",
				title: "priority — 1 is tried first",
				children: t + 1
			}),
			/* @__PURE__ */ d("span", {
				className: "pg-member-key",
				title: e,
				children: a ? /* @__PURE__ */ f(u, { children: [
					"▣ ",
					String(e).slice(6),
					" ",
					/* @__PURE__ */ d("em", { children: "(group)" })
				] }) : e
			}),
			/* @__PURE__ */ f("span", {
				className: "pg-member-ctl",
				children: [
					/* @__PURE__ */ d("button", {
						type: "button",
						className: "pg-arrow",
						disabled: t === 0,
						title: "Move up (tried earlier)",
						onClick: () => r(t, t - 1),
						children: "▲"
					}),
					/* @__PURE__ */ d("button", {
						type: "button",
						className: "pg-arrow",
						disabled: t === n - 1,
						title: "Move down (tried later)",
						onClick: () => r(t, t + 1),
						children: "▼"
					}),
					/* @__PURE__ */ d("button", {
						type: "button",
						className: "pg-remove",
						title: "Remove from the group",
						onClick: () => i(t),
						children: "✕"
					})
				]
			})
		]
	});
}
function ui({ initial: e, models: t, allGroups: n, busy: r, onSave: a, onCancel: c }) {
	let [u, p] = l(e?.name || ""), [m, h] = l(e?.members || []), [g, _] = l(e?.workers || []), [v, y] = l(!e || !!e.enabled), [b, x] = l([]);
	o(() => {
		K("/api/llm/workers").then((e) => x((e || []).map((e) => e.name || e.id).filter(Boolean))).catch(() => x([]));
	}, []);
	let S = i((e) => (t, n) => {
		e((e) => {
			if (n < 0 || n >= e.length) return e;
			let r = e.slice(), [i] = r.splice(t, 1);
			return r.splice(n, 0, i), r;
		});
	}, []), C = s(() => S(h), [S]), w = s(() => S(_), [S]), T = i((e) => {
		h((t) => t.filter((t, n) => n !== e));
	}, []), E = i((e) => {
		_((t) => t.filter((t, n) => n !== e));
	}, []), D = i((e) => {
		e && h((t) => t.includes(e) ? t : [...t, e]);
	}, []), O = i((e) => {
		e && _((t) => t.some((t) => t.toLowerCase() === e.toLowerCase()) ? t : [...t, e]);
	}, []), k = u.trim() && m.length > 0;
	return /* @__PURE__ */ f("div", {
		className: "pg-editor",
		children: [
			/* @__PURE__ */ f("div", {
				className: "pg-row",
				children: [/* @__PURE__ */ d("input", {
					className: "pg-name",
					placeholder: "Group name (e.g. Qwen2.5-VL-7B-Instruct)",
					value: u,
					onChange: (e) => p(e.target.value)
				}), /* @__PURE__ */ f("label", {
					className: "pg-enable",
					title: "A disabled group routes nothing. Only enabled groups claim a model key.",
					children: [/* @__PURE__ */ d("input", {
						type: "checkbox",
						checked: v,
						onChange: (e) => y(e.target.checked)
					}), "enabled"]
				})]
			}),
			m.length === 0 && /* @__PURE__ */ d("div", {
				className: "pg-hint",
				children: "Add models in the order they should be tried. #1 serves whenever it is usable; the rest are fallbacks."
			}),
			/* @__PURE__ */ d("ol", {
				className: "pg-members",
				children: m.map((e, t) => /* @__PURE__ */ d(li, {
					mk: e,
					i: t,
					n: m.length,
					onMove: C,
					onRemove: T
				}, e))
			}),
			/* @__PURE__ */ f("div", {
				className: "pg-row",
				children: [/* @__PURE__ */ d(rr, {
					models: t,
					onPick: D,
					placeholder: "+ add model…"
				}), /* @__PURE__ */ f("select", {
					className: "pg-worker-pick",
					value: "",
					title: "Mount another group here as a module: its own ordered members expand at this position, and it inherits this group's workers unless it has its own",
					onChange: (e) => {
						e.target.value && D(`group:${e.target.value}`), e.target.value = "";
					},
					children: [/* @__PURE__ */ d("option", {
						value: "",
						children: "+ add group…"
					}), (n || []).filter((t) => t.id !== e?.id && !m.some((e) => String(e).toLowerCase() === `group:${t.id}`.toLowerCase())).map((e) => /* @__PURE__ */ f("option", {
						value: e.id,
						children: ["▣ ", e.name]
					}, e.id))]
				})]
			}),
			/* @__PURE__ */ d("div", {
				className: "pg-workers-head",
				title: "Where the group's models live, in priority order. #1 is preferred; members never land on a worker off this list. Empty = no placement statement (routing unchanged).",
				children: "workers (allocation, priority order)"
			}),
			g.length === 0 && /* @__PURE__ */ d("div", {
				className: "pg-hint",
				children: "Optional: allocate the group to workers. #1 is preferred; models in this group only land on listed workers. Leave empty to route as today."
			}),
			/* @__PURE__ */ d("ol", {
				className: "pg-members",
				children: g.map((e, t) => /* @__PURE__ */ d(li, {
					mk: e,
					i: t,
					n: g.length,
					onMove: w,
					onRemove: E
				}, e))
			}),
			/* @__PURE__ */ f("div", {
				className: "pg-row",
				children: [
					/* @__PURE__ */ f("select", {
						className: "pg-worker-pick",
						value: "",
						onChange: (e) => {
							O(e.target.value), e.target.value = "";
						},
						children: [/* @__PURE__ */ d("option", {
							value: "",
							children: "+ add worker…"
						}), b.filter((e) => !g.some((t) => t.toLowerCase() === e.toLowerCase())).map((e) => /* @__PURE__ */ d("option", {
							value: e,
							children: e
						}, e))]
					}),
					/* @__PURE__ */ d("span", { className: "pg-spacer" }),
					/* @__PURE__ */ d("button", {
						type: "button",
						className: "pg-cancel",
						onClick: c,
						disabled: r,
						children: "cancel"
					}),
					/* @__PURE__ */ d("button", {
						type: "button",
						className: "pg-save",
						disabled: !k || r,
						onClick: () => a({
							id: e?.id,
							name: u.trim(),
							members: m,
							workers: g,
							enabled: v
						}),
						children: r ? "saving…" : e ? "save" : "create group"
					})
				]
			})
		]
	});
}
function di({ result: e }) {
	if (!e) return null;
	let t = e.candidates || [];
	return /* @__PURE__ */ f("div", {
		className: "pg-preview",
		children: [
			/* @__PURE__ */ f("div", {
				className: "pg-preview-head",
				children: [
					/* @__PURE__ */ d("span", {
						className: "pg-preview-key",
						children: e.requested
					}),
					/* @__PURE__ */ d("span", {
						className: "pg-preview-arrow",
						children: "→"
					}),
					/* @__PURE__ */ d("span", {
						className: `pg-preview-chosen ${e.chosen ? "" : "pg-none"}`,
						children: e.chosen || "unchanged"
					}),
					e.group && /* @__PURE__ */ f("span", {
						className: "pg-preview-group",
						title: "the enabled priority group that claims this key",
						children: ["group ", e.group.name]
					})
				]
			}),
			/* @__PURE__ */ d("div", {
				className: "pg-why",
				children: e.why
			}),
			t.length > 0 && /* @__PURE__ */ d("ol", {
				className: "pg-cands",
				children: t.map((e) => /* @__PURE__ */ f("li", {
					className: `pg-cand pg-st-${e.status}`,
					children: [
						/* @__PURE__ */ d("span", {
							className: "pg-rank",
							children: e.position
						}),
						/* @__PURE__ */ d("span", {
							className: "pg-cand-icon",
							title: e.status,
							children: ci[e.status] || "·"
						}),
						/* @__PURE__ */ d("span", {
							className: "pg-cand-key",
							title: e.catalog_key || e.model_key,
							children: e.model_key
						}),
						e.via && /* @__PURE__ */ f("span", {
							className: "pg-cand-via",
							title: `expanded from the nested group ${e.via}`,
							children: ["via ▣ ", e.via]
						}),
						/* @__PURE__ */ d("span", {
							className: "pg-cand-why",
							children: e.reason
						})
					]
				}, `${e.position}-${e.model_key}`))
			})
		]
	});
}
function fi({ models: e = [] }) {
	let [t, n] = Y("hugpy.priorityGroups.open", !1), [r, a] = l([]), [c, p] = l(null), [m, h] = l(!1), [g, _] = l(null), [v, y] = l(""), [b, x] = l(null), [S, C] = l(null), w = i(() => {
		K("/api/llm/model-groups").then((e) => {
			a(e.groups || []), p(e.error || null);
		}).catch((e) => p(String(e.message || e)));
	}, []);
	o(() => {
		t && w();
	}, [t, w]);
	let T = i((e) => {
		h(!0), p(null), K(e.id ? `/api/llm/model-groups/${encodeURIComponent(e.id)}` : "/api/llm/model-groups", {
			method: e.id ? "PUT" : "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({
				name: e.name,
				members: e.members,
				workers: e.workers,
				enabled: e.enabled
			})
		}).then(() => {
			_(null), w();
		}).catch((e) => p(String(e.message || e))).finally(() => h(!1));
	}, [w]), E = i((e) => {
		(e.workers || []).length && window.confirm(`Allocate "${e.name}": designate ${e.members.length} model(s) to ${e.workers.join(" → ")}?`) && (h(!0), p(null), K(`/api/llm/model-groups/${encodeURIComponent(e.id)}/allocate`, {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({})
		}).then((t) => {
			let n = (t.outcomes || []).filter((e) => e.status !== "designated" && e.status !== "already");
			C({
				id: e.id,
				text: `${t.designated} designated` + (n.length ? `, ${n.length} skipped (${[...new Set(n.map((e) => e.status))].join(", ")})` : "")
			}), w();
		}).catch((e) => p(String(e.message || e))).finally(() => h(!1)));
	}, [w]), D = i((e, t) => {
		h(!0), p(null), K(`/api/llm/model-groups/${encodeURIComponent(e.id)}`, {
			method: "PATCH",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({ enabled: t })
		}).then(w).catch((e) => p(String(e.message || e))).finally(() => h(!1));
	}, [w]), O = i((e) => {
		window.confirm(`Delete priority group "${e.name}"?`) && (h(!0), p(null), K(`/api/llm/model-groups/${encodeURIComponent(e.id)}`, { method: "DELETE" }).then(w).catch((e) => p(String(e.message || e))).finally(() => h(!1)));
	}, [w]), k = i((e) => {
		let t = (e || "").trim();
		t && (y(t), K(`/api/llm/model-groups/resolve?key=${encodeURIComponent(t)}`).then(x).catch((e) => p(String(e.message || e))));
	}, []), A = s(() => r.filter((e) => e.enabled).length, [r]);
	return /* @__PURE__ */ f("section", {
		className: "priority-groups-panel",
		children: [/* @__PURE__ */ f("div", {
			className: "pg-bar",
			onClick: () => n((e) => !e),
			children: [
				/* @__PURE__ */ d("span", {
					className: "pg-title",
					children: "Priority groups"
				}),
				/* @__PURE__ */ d("span", {
					className: "pg-count",
					children: r.length ? `${r.length} · ${A} enabled` : "none"
				}),
				c && /* @__PURE__ */ d("span", {
					className: "pg-err",
					children: c
				}),
				/* @__PURE__ */ d("span", {
					className: "pg-toggle",
					children: t ? "▾" : "▸"
				})
			]
		}), t && /* @__PURE__ */ f("div", {
			className: "pg-body",
			children: [
				/* @__PURE__ */ f("p", {
					className: "pg-intro",
					children: [
						"An ",
						/* @__PURE__ */ d("strong", { children: "explicit, ordered" }),
						" fallback list. Calling any member serves the first member that is usable right now. A ",
						/* @__PURE__ */ d("strong", { children: "blocked" }),
						" ",
						"model is skipped, never served — a group cannot bypass a block. If nothing in the group is usable the request fails exactly as it does without the group."
					]
				}),
				r.map((t) => /* @__PURE__ */ f("div", {
					className: `pg-group ${t.enabled ? "" : "pg-off"}`,
					children: [
						/* @__PURE__ */ f("div", {
							className: "pg-group-head",
							children: [
								/* @__PURE__ */ d("span", {
									className: "pg-group-name",
									children: t.name
								}),
								/* @__PURE__ */ d("span", {
									className: "pg-group-id",
									children: t.id
								}),
								/* @__PURE__ */ d("span", { className: "pg-spacer" }),
								/* @__PURE__ */ f("label", {
									className: "pg-enable",
									title: "Disable to stop this group routing anything",
									children: [/* @__PURE__ */ d("input", {
										type: "checkbox",
										checked: !!t.enabled,
										disabled: m,
										onChange: (e) => D(t, e.target.checked)
									}), "enabled"]
								}),
								/* @__PURE__ */ d("button", {
									type: "button",
									className: "pg-edit",
									disabled: m,
									onClick: () => _(g?.id === t.id ? null : t),
									children: "edit"
								}),
								/* @__PURE__ */ d("button", {
									type: "button",
									className: "pg-del",
									disabled: m,
									onClick: () => O(t),
									children: "delete"
								})
							]
						}),
						/* @__PURE__ */ d("ol", {
							className: "pg-members pg-members-ro",
							children: (t.members || []).map((e, t) => {
								let n = String(e).toLowerCase().startsWith("group:");
								return /* @__PURE__ */ f("li", {
									className: `pg-member${n ? " pg-member-ref" : ""}`,
									children: [
										/* @__PURE__ */ d("span", {
											className: "pg-rank",
											children: t + 1
										}),
										/* @__PURE__ */ d("span", {
											className: "pg-member-key",
											children: n ? /* @__PURE__ */ f(u, { children: [
												"▣ ",
												String(e).slice(6),
												" ",
												/* @__PURE__ */ d("em", { children: "(group)" })
											] }) : e
										}),
										!n && /* @__PURE__ */ d("button", {
											type: "button",
											className: "pg-probe",
											title: "Preview how this key resolves right now",
											onClick: () => k(e),
											children: "resolve"
										})
									]
								}, e);
							})
						}),
						(t.workers || []).length > 0 && /* @__PURE__ */ f("div", {
							className: "pg-alloc-row",
							title: "The group's worker allocation, in priority order. Members only land on these workers.",
							children: [
								/* @__PURE__ */ d("span", {
									className: "pg-alloc-label",
									children: "workers:"
								}),
								/* @__PURE__ */ d("span", {
									className: "pg-alloc-workers",
									children: t.workers.join(" → ")
								}),
								/* @__PURE__ */ d("span", { className: "pg-spacer" }),
								S?.id === t.id && /* @__PURE__ */ d("span", {
									className: "pg-alloc-note",
									children: S.text
								}),
								/* @__PURE__ */ d("button", {
									type: "button",
									className: "pg-allocate",
									disabled: m,
									title: "Designate every member to every listed worker (no load/warm)",
									onClick: () => E(t),
									children: "allocate"
								})
							]
						}),
						g?.id === t.id && /* @__PURE__ */ d(ui, {
							initial: t,
							models: e,
							allGroups: r,
							busy: m,
							onSave: T,
							onCancel: () => _(null)
						})
					]
				}, t.id)),
				g === "new" ? /* @__PURE__ */ d(ui, {
					initial: null,
					models: e,
					allGroups: r,
					busy: m,
					onSave: T,
					onCancel: () => _(null)
				}) : /* @__PURE__ */ d("button", {
					type: "button",
					className: "pg-new",
					disabled: m,
					onClick: () => _("new"),
					children: "+ new priority group"
				}),
				/* @__PURE__ */ f("div", {
					className: "pg-resolve",
					children: [
						/* @__PURE__ */ d("span", {
							className: "pg-title",
							children: "Resolve preview"
						}),
						/* @__PURE__ */ f("div", {
							className: "pg-row",
							children: [/* @__PURE__ */ d("input", {
								className: "pg-probe-input",
								placeholder: "model key…",
								value: v,
								onChange: (e) => y(e.target.value),
								onKeyDown: (e) => {
									e.key === "Enter" && k(v);
								}
							}), /* @__PURE__ */ d("button", {
								type: "button",
								className: "pg-probe",
								onClick: () => k(v),
								children: "resolve"
							})]
						}),
						/* @__PURE__ */ d(di, { result: b })
					]
				})
			]
		})]
	});
}
//#endregion
//#region src/components/TemplatesPanel/TemplatesPanel.jsx
function pi({ initial: e, groups: t, workers: n, busy: r, onSave: a, onCancel: o }) {
	let [s, c] = l(e?.name || ""), [u, p] = l(e?.groups || []), [m, h] = l(e?.workers || []), g = i((e) => {
		p((t) => t.includes(e) ? t.filter((t) => t !== e) : [...t, e]);
	}, []), _ = i((e) => {
		e && h((t) => t.some((t) => t.toLowerCase() === e.toLowerCase()) ? t : [...t, e]);
	}, []), v = s.trim() && u.length > 0;
	return /* @__PURE__ */ f("div", {
		className: "tp-editor",
		children: [
			/* @__PURE__ */ d("div", {
				className: "tp-row",
				children: /* @__PURE__ */ d("input", {
					className: "tp-name",
					placeholder: "Template name (e.g. cinema night-shoot)",
					value: s,
					onChange: (e) => c(e.target.value)
				})
			}),
			/* @__PURE__ */ d("div", {
				className: "tp-pick-head",
				children: "module groups (the cast)"
			}),
			/* @__PURE__ */ d("div", {
				className: "tp-groups",
				children: t.map((e) => /* @__PURE__ */ f("label", {
					className: "tp-group-pick",
					children: [
						/* @__PURE__ */ d("input", {
							type: "checkbox",
							checked: u.includes(e.id),
							onChange: () => g(e.id)
						}),
						"▣ ",
						e.name
					]
				}, e.id))
			}),
			/* @__PURE__ */ d("div", {
				className: "tp-pick-head",
				title: "Optional: leave empty to derive from the groups' own worker allocations at activation time",
				children: "workers to reserve (optional — empty derives from the groups)"
			}),
			m.length > 0 && /* @__PURE__ */ d("div", {
				className: "tp-workers",
				children: m.map((e) => /* @__PURE__ */ f("span", {
					className: "tp-worker-chip",
					children: [e, /* @__PURE__ */ d("button", {
						type: "button",
						title: "remove",
						onClick: () => h((t) => t.filter((t) => t !== e)),
						children: "✕"
					})]
				}, e))
			}),
			/* @__PURE__ */ f("div", {
				className: "tp-row",
				children: [
					/* @__PURE__ */ f("select", {
						className: "tp-worker-pick",
						value: "",
						onChange: (e) => {
							_(e.target.value), e.target.value = "";
						},
						children: [/* @__PURE__ */ d("option", {
							value: "",
							children: "+ add worker…"
						}), n.filter((e) => !m.some((t) => t.toLowerCase() === (e.name || e.id).toLowerCase())).map((e) => /* @__PURE__ */ d("option", {
							value: e.name || e.id,
							children: e.name || e.id
						}, e.id))]
					}),
					/* @__PURE__ */ d("span", { className: "tp-spacer" }),
					/* @__PURE__ */ d("button", {
						type: "button",
						className: "tp-cancel",
						onClick: o,
						disabled: r,
						children: "cancel"
					}),
					/* @__PURE__ */ d("button", {
						type: "button",
						className: "tp-save",
						disabled: !v || r,
						onClick: () => a({
							id: e?.id,
							name: s.trim(),
							groups: u,
							workers: m
						}),
						children: r ? "saving…" : e ? "save" : "create template"
					})
				]
			})
		]
	});
}
function mi({ workers: e = [] }) {
	let [t, n] = l(!1), [r, a] = l([]), [s, c] = l(null), [u, p] = l(!1), [m, h] = l(null), [g, _] = l(null), { groups: v } = Nr(), y = i(() => {
		K("/api/llm/templates").then((e) => {
			a(e.templates || []), c(e.error || null);
		}).catch((e) => c(String(e.message || e)));
	}, []);
	o(() => {
		t && y();
	}, [t, y]);
	let b = i((e) => {
		p(!0), c(null), K(e.id ? `/api/llm/templates/${encodeURIComponent(e.id)}` : "/api/llm/templates", {
			method: e.id ? "PUT" : "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({
				name: e.name,
				groups: e.groups,
				workers: e.workers
			})
		}).then(() => {
			h(null), y();
		}).catch((e) => c(String(e.message || e))).finally(() => p(!1));
	}, [y]), x = i((e, t) => {
		let n = t === "activate", r = (e.workers?.length ? e.workers : e.derived_workers) || [];
		n && !window.confirm(`Activate "${e.name}": reserve ${r.join(", ") || "(derived workers)"} for this template's pool and allocate ${e.groups.join(", ")}? Reserved workers stop serving general traffic until deactivated.`) || (p(!0), c(null), K(`/api/llm/templates/${encodeURIComponent(e.id)}/${t}`, {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: "{}"
		}).then((t) => {
			_({
				id: e.id,
				text: n ? `reserved ${(t.reserved || []).join(", ")} · ${t.designated} designated` : `released ${(t.released || []).join(", ") || "nothing (no tags held)"}`
			}), y();
		}).catch((e) => c(String(e.message || e))).finally(() => p(!1)));
	}, [y]), S = i((e) => {
		window.confirm(`Delete template "${e.name}"?`) && (p(!0), c(null), K(`/api/llm/templates/${encodeURIComponent(e.id)}`, { method: "DELETE" }).then(y).catch((e) => c(String(e.message || e))).finally(() => p(!1)));
	}, [y]);
	return /* @__PURE__ */ f("section", {
		className: "templates-panel",
		children: [/* @__PURE__ */ f("div", {
			className: "tp-bar",
			onClick: () => n((e) => !e),
			children: [
				/* @__PURE__ */ d("span", {
					className: "tp-title",
					children: "Task templates"
				}),
				/* @__PURE__ */ d("span", {
					className: "tp-count",
					children: r.length ? `${r.length} · ${r.filter((e) => e.active).length} active` : "none"
				}),
				s && /* @__PURE__ */ d("span", {
					className: "tp-err",
					children: s
				}),
				/* @__PURE__ */ d("span", {
					className: "tp-toggle",
					children: t ? "▾" : "▸"
				})
			]
		}), t && /* @__PURE__ */ f("div", {
			className: "tp-body",
			children: [
				/* @__PURE__ */ f("p", {
					className: "tp-intro",
					children: [
						"A template maps a TASK to its ",
						/* @__PURE__ */ d("strong", { children: "module groups" }),
						" (what models it needs, in their own priority orders) and the",
						" ",
						/* @__PURE__ */ d("strong", { children: "workers reserved" }),
						" for it. Activating pools those workers under the template id — general traffic can no longer land on them; requests tagged with the pool can — and allocates every group to them. Deactivating releases the reservation; designations stay so reactivation is instant."
					]
				}),
				r.map((t) => /* @__PURE__ */ f("div", {
					className: `tp-template ${t.active ? "tp-active" : ""}`,
					children: [
						/* @__PURE__ */ f("div", {
							className: "tp-template-head",
							children: [
								/* @__PURE__ */ d("span", {
									className: "tp-template-name",
									children: t.name
								}),
								t.active && /* @__PURE__ */ d("span", {
									className: "tp-badge",
									children: "● reserved"
								}),
								/* @__PURE__ */ d("span", { className: "tp-spacer" }),
								g?.id === t.id && /* @__PURE__ */ d("span", {
									className: "tp-note",
									children: g.text
								}),
								/* @__PURE__ */ d("button", {
									type: "button",
									disabled: u,
									onClick: () => x(t, t.active ? "deactivate" : "activate"),
									children: t.active ? "deactivate" : "activate"
								}),
								/* @__PURE__ */ d("button", {
									type: "button",
									disabled: u,
									onClick: () => h(m?.id === t.id ? null : t),
									children: "edit"
								}),
								/* @__PURE__ */ d("button", {
									type: "button",
									disabled: u || t.active,
									className: "tp-del",
									title: t.active ? "deactivate first" : "delete",
									onClick: () => S(t),
									children: "delete"
								})
							]
						}),
						/* @__PURE__ */ f("div", {
							className: "tp-template-detail",
							children: [/* @__PURE__ */ f("span", { children: ["groups: ", t.groups.map((e) => `▣ ${e}`).join("  ")] }), /* @__PURE__ */ f("span", {
								className: "tp-workers-line",
								children: [
									"workers: ",
									(t.workers?.length ? t.workers : t.derived_workers)?.join(" → ") || "—",
									!t.workers?.length && t.derived_workers?.length ? " (derived)" : ""
								]
							})]
						}),
						m?.id === t.id && /* @__PURE__ */ d(pi, {
							initial: t,
							groups: v,
							workers: e,
							busy: u,
							onSave: b,
							onCancel: () => h(null)
						})
					]
				}, t.id)),
				m === "new" ? /* @__PURE__ */ d(pi, {
					initial: null,
					groups: v,
					workers: e,
					busy: u,
					onSave: b,
					onCancel: () => h(null)
				}) : /* @__PURE__ */ d("button", {
					type: "button",
					className: "tp-new",
					disabled: u,
					onClick: () => h("new"),
					children: "+ new template"
				})
			]
		})]
	});
}
//#endregion
//#region src/components/PhoneBrickPanel/PhoneBrickPanel.jsx
var hi = {
	AGR: {
		label: "agrees",
		cls: "pb-agr"
	},
	DIS: {
		label: "disagrees",
		cls: "pb-dis"
	},
	NOD: {
		label: "no detection",
		cls: "pb-nod"
	}
}, gi = [
	"done",
	"error",
	"cancelled"
];
function _i({ phone: e, selected: t, onToggle: n, onRemove: r }) {
	let [a, o] = l(null), s = e.live || {}, c = i(async () => {
		o("checking");
		try {
			o(await K(`/api/phone-brick/phones/${encodeURIComponent(e.id)}/health`));
		} catch (e) {
			o({
				reachable: !1,
				error: e.message
			});
		}
	}, [e.id]);
	return /* @__PURE__ */ f("div", {
		className: `pb-phone pb-${e.status}`,
		children: [
			/* @__PURE__ */ d("label", {
				className: "pb-phone-pick",
				title: "Include this phone in the next run",
				children: /* @__PURE__ */ d("input", {
					type: "checkbox",
					checked: t,
					onChange: () => n(e.id),
					disabled: e.status !== "online"
				})
			}),
			/* @__PURE__ */ d("span", {
				className: "pb-dot",
				style: { background: e.color }
			}),
			/* @__PURE__ */ d("span", {
				className: "pb-name",
				children: e.name
			}),
			/* @__PURE__ */ d("span", {
				className: "pb-status",
				children: e.status
			}),
			/* @__PURE__ */ f("span", {
				className: "pb-url",
				title: e.url,
				children: [
					e.host,
					":",
					e.port
				]
			}),
			s.model_loaded != null && /* @__PURE__ */ d("span", {
				className: `pb-model ${s.model_loaded ? "pb-model-on" : "pb-model-off"}`,
				title: s.model_path || "",
				children: s.model_loaded ? "🧠 model loaded" : "○ no model"
			}),
			s.queue_size != null && /* @__PURE__ */ f("span", {
				className: "pb-queue",
				children: ["queue ", s.queue_size]
			}),
			a && a !== "checking" && /* @__PURE__ */ d("span", {
				className: `pb-ping ${a.reachable ? "pb-ping-ok" : "pb-ping-bad"}`,
				title: a.reachable ? "central can reach this phone" : a.error || "unreachable",
				children: a.reachable ? "✓ reachable" : "✗ unreachable"
			}),
			/* @__PURE__ */ d("button", {
				className: "pb-ping-btn",
				onClick: c,
				disabled: a === "checking",
				title: "Ping the phone's /status from central",
				children: a === "checking" ? "…" : "ping"
			}),
			/* @__PURE__ */ d("button", {
				className: "pb-remove",
				title: "Remove phone",
				onClick: () => r(e),
				children: "✕"
			})
		]
	});
}
function vi({ rows: e }) {
	return e.length ? /* @__PURE__ */ f("table", {
		className: "pb-phases",
		children: [/* @__PURE__ */ d("thead", { children: /* @__PURE__ */ f("tr", { children: [
			/* @__PURE__ */ d("th", { children: "phone" }),
			/* @__PURE__ */ d("th", { children: "top class" }),
			/* @__PURE__ */ d("th", { children: "conf" }),
			/* @__PURE__ */ d("th", { children: "consensus" }),
			/* @__PURE__ */ d("th", { children: "dets" })
		] }) }), /* @__PURE__ */ d("tbody", { children: e.map((e, t) => {
			let n = e.consensus ? hi[e.consensus] || {
				label: e.consensus,
				cls: ""
			} : {
				label: "…",
				cls: "pb-nod"
			};
			return /* @__PURE__ */ f("tr", { children: [
				/* @__PURE__ */ d("td", { children: e.phone }),
				/* @__PURE__ */ d("td", { children: e.top_cls }),
				/* @__PURE__ */ f("td", { children: [e.top_conf_pct, "%"] }),
				/* @__PURE__ */ d("td", {
					className: n.cls,
					children: n.label
				}),
				/* @__PURE__ */ d("td", { children: e.detections })
			] }, t);
		}) })]
	}) : null;
}
function yi({ run: e, liveProgress: t, currentPhone: n, onCancel: r }) {
	if (!e) return null;
	let i = !gi.includes(e.status), a = e.status === "done" && (e.phases || []).length ? e.phases : t;
	return /* @__PURE__ */ f("div", {
		className: "pb-run",
		children: [
			/* @__PURE__ */ f("div", {
				className: "pb-run-head",
				children: [
					/* @__PURE__ */ d("span", {
						className: `pb-run-status pb-rs-${e.status}`,
						children: e.status
					}),
					/* @__PURE__ */ d("span", {
						className: "pb-run-image",
						children: e.image
					}),
					i && /* @__PURE__ */ d("button", {
						className: "pb-cancel",
						onClick: () => r(e.id),
						title: "Cancel this run",
						children: "✕ cancel"
					})
				]
			}),
			i && /* @__PURE__ */ f("div", {
				className: "pb-run-busy",
				children: [n ? /* @__PURE__ */ f(u, { children: [
					"Asking ",
					/* @__PURE__ */ d("b", { children: n }),
					"… "
				] }) : "Fanning across phones… ", /* @__PURE__ */ d("span", { className: "pb-spin" })]
			}),
			e.status === "error" && /* @__PURE__ */ f("div", {
				className: "pb-run-error",
				children: ["Run failed: ", e.error || "unknown error"]
			}),
			e.status === "cancelled" && /* @__PURE__ */ d("div", {
				className: "pb-run-cancelled",
				children: "Run cancelled."
			}),
			e.status === "done" && e.output_rel && /* @__PURE__ */ d("img", {
				className: "pb-run-img",
				alt: "annotated result",
				src: H(`/api/phone-brick/runs/${encodeURIComponent(e.id)}/image`)
			}),
			/* @__PURE__ */ d(vi, { rows: a })
		]
	});
}
function bi({ embedded: e = !1 }) {
	let [t, n] = l([]), [r, a] = l(null), [s, u] = l(!1), [p, m] = l({}), [h, g] = l(null), [_, v] = l(null), [y, b] = l([]), [x, S] = l(null), [C, w] = l(!1), T = c(null), E = i(() => {
		K("/api/phone-brick/phones").then((e) => {
			n(Array.isArray(e) ? e : []), a(null);
		}).catch((e) => a(e.message));
	}, []);
	o(() => {
		E();
		let e = setInterval(E, 1e4);
		return () => clearInterval(e);
	}, [E]), o(() => {
		if (!_ || gi.includes(_.status)) return;
		let e = new EventSource(H(`/api/phone-brick/runs/${encodeURIComponent(_.id)}/stream`), { withCredentials: B().credentials === "include" });
		return e.onmessage = (t) => {
			let n;
			try {
				n = JSON.parse(t.data);
			} catch {
				return;
			}
			n.type === "progress" ? b((e) => [...e, n.phase]) : n.type === "current" ? S(n.phone) : n.type === "status" ? v((e) => e && {
				...e,
				status: n.status
			}) : gi.includes(n.type) && (v(n.run), S(null), e.close());
		}, e.onerror = () => e.close(), () => e.close();
	}, [_?.id]);
	let D = i((e) => m((t) => ({
		...t,
		[e]: !t[e]
	})), []), O = i(async (e) => {
		if (confirm(`Remove phone ${e.name} from the pool?`)) try {
			await K(`/api/phone-brick/phones/${encodeURIComponent(e.id)}`, { method: "DELETE" }), E();
		} catch (e) {
			alert(`Remove failed: ${e.message}`);
		}
	}, [E]), k = i(async () => {
		if (!h) {
			alert("Choose an image to analyze first.");
			return;
		}
		let e = t.filter((e) => p[e.id] && e.status === "online").map((e) => e.id), n = e.length ? e : t.filter((e) => e.status === "online").map((e) => e.id);
		if (!n.length) {
			alert("No online phones to run on.");
			return;
		}
		let r = new FormData();
		r.append("image", h), r.append("phone_ids", n.join(",")), w(!0);
		try {
			let e = await K("/api/phone-brick/run", {
				method: "POST",
				body: r
			});
			b([]), S(null), v(e), g(null), T.current && (T.current.value = "");
		} catch (e) {
			alert(`Run failed: ${e.message}`);
		} finally {
			w(!1);
		}
	}, [
		h,
		t,
		p
	]), A = i(async (e) => {
		try {
			await K(`/api/phone-brick/runs/${encodeURIComponent(e)}/cancel`, { method: "POST" });
		} catch (e) {
			alert(`Cancel failed: ${e.message}`);
		}
	}, []), j = t.filter((e) => e.status === "online").length, M = t.length - j;
	return /* @__PURE__ */ f("div", {
		className: "phonebrick-panel",
		children: [/* @__PURE__ */ f("div", {
			className: `pb-bar${e ? " pb-bar-static" : ""}`,
			onClick: e ? void 0 : () => u((e) => !e),
			children: [
				/* @__PURE__ */ d("span", {
					className: "pb-title",
					children: "📱 Phone Brick — video analytics pool"
				}),
				/* @__PURE__ */ f("span", {
					className: "pb-count",
					children: [
						j,
						" online / ",
						t.length,
						" total"
					]
				}),
				r && /* @__PURE__ */ d("span", {
					className: "pb-err",
					title: r,
					children: "registry error"
				}),
				!e && /* @__PURE__ */ d("span", {
					className: "pb-toggle",
					children: s ? "▾" : "▸"
				})
			]
		}), (e || s) && /* @__PURE__ */ f("div", {
			className: "pb-body",
			children: [
				/* @__PURE__ */ f("div", {
					className: "pb-howto",
					children: [
						"Phones join automatically when started with",
						" ",
						/* @__PURE__ */ d("code", { children: "PHONE_BRICK_CENTRAL" }),
						" pointed at this console."
					]
				}),
				t.length === 0 && /* @__PURE__ */ d("div", {
					className: "pb-empty",
					children: "No phones have joined the pool yet."
				}),
				t.map((e) => /* @__PURE__ */ d(_i, {
					phone: e,
					selected: !!p[e.id],
					onToggle: D,
					onRemove: O
				}, e.id)),
				/* @__PURE__ */ f("div", {
					className: "pb-run-row",
					children: [
						/* @__PURE__ */ d("input", {
							ref: T,
							type: "file",
							accept: "image/*",
							onChange: (e) => g(e.target.files?.[0] || null)
						}),
						/* @__PURE__ */ d("button", {
							className: "pb-run-btn",
							onClick: k,
							disabled: C || !j,
							children: C ? "Starting…" : "▶ Run detection"
						}),
						/* @__PURE__ */ d("span", {
							className: "pb-run-hint",
							children: j ? "Runs on ticked phones, or all online if none ticked." : "No online phones."
						})
					]
				}),
				/* @__PURE__ */ d(yi, {
					run: _,
					liveProgress: y,
					currentPhone: x,
					onCancel: A
				}),
				M > 0 && /* @__PURE__ */ f("div", {
					className: "pb-offline-note",
					children: [M, " phone(s) offline (missed heartbeats)."]
				})
			]
		})]
	});
}
//#endregion
//#region src/components/AgentNodesPanel/AgentNodesPanel.jsx
var xi = ["done", "error"], Si = 90, Ci = 1e4, wi = 2500;
function Ti(e) {
	if (!e) return "never";
	let t = Math.max(0, Date.now() / 1e3 - Number(e));
	if (t < 5) return "just now";
	if (t < 90) return `${Math.round(t)}s ago`;
	let n = t / 60;
	if (n < 60) return `${Math.round(n)}m ago`;
	let r = n / 60;
	return r < 24 ? `${Math.round(r)}h ago` : `${Math.round(r / 24)}d ago`;
}
function Ei(e) {
	return e.last_seen != null && Date.now() / 1e3 - Number(e.last_seen) <= Si;
}
function Di(e) {
	let t = e.current_task;
	return t === "" ? {
		text: "idle",
		cls: "an-ct-idle",
		title: "node reported no current task (current_task === \"\")"
	} : t == null ? {
		text: "no report",
		cls: "an-ct-unknown",
		title: "no current_task reported yet (null = heartbeat kept prior value — not necessarily idle)"
	} : {
		text: `▶ task ${t}`,
		cls: "an-ct-busy",
		title: `node reports it is working task seq ${t}`
	};
}
function Oi(e) {
	return e === "idle" ? "an-st-idle" : e === "busy" || e === "running" ? "an-st-busy" : e === "enrolled" ? "an-st-enrolled" : e === "offline" ? "an-st-offline" : "an-st-other";
}
function ki(e) {
	return e === "done" ? "an-ts-done" : e === "error" ? "an-ts-error" : "an-ts-queued";
}
function Ai({ rec: e }) {
	let t = e.view || {}, n = t.status || "queued", r = !xi.includes(n);
	return /* @__PURE__ */ f("div", {
		className: `an-task an-task-${n}`,
		children: [
			/* @__PURE__ */ f("div", {
				className: "an-task-head",
				children: [
					/* @__PURE__ */ f("span", {
						className: `an-ts-pill ${ki(n)}`,
						children: [r && /* @__PURE__ */ d("span", { className: "an-spin" }), n]
					}),
					/* @__PURE__ */ f("span", {
						className: "an-task-seq",
						title: "dispatch sequence (this node's monotonic task cursor)",
						children: ["seq ", e.seq]
					}),
					/* @__PURE__ */ d("span", {
						className: "an-task-prompt",
						title: e.prompt,
						children: e.prompt
					}),
					t.finished_at != null && /* @__PURE__ */ f("span", {
						className: "an-task-fin",
						title: "finished_at (from central)",
						children: ["· ", Ti(t.finished_at)]
					})
				]
			}),
			r && /* @__PURE__ */ d("div", {
				className: "an-task-wait",
				children: "Dispatched — waiting for the node to pull, run, and report…"
			}),
			n === "done" && /* @__PURE__ */ d("pre", {
				className: "an-task-result",
				children: t.result != null && t.result !== "" ? t.result : "(empty result)"
			}),
			n === "error" && /* @__PURE__ */ d("pre", {
				className: "an-task-result an-task-result-err",
				children: t.result || "node reported an error (no detail)"
			})
		]
	});
}
function ji({ node: e, tasks: t, onDispatch: n }) {
	let [r, a] = l(""), [o, s] = l(!1), [c, u] = l(null), p = Ei(e), m = Di(e), h = i(async () => {
		let t = r.trim();
		if (!(!t || o)) {
			s(!0), u(null);
			try {
				await n(e, t), a("");
			} catch (e) {
				u(e.message || "dispatch failed");
			} finally {
				s(!1);
			}
		}
	}, [
		r,
		o,
		e,
		n
	]);
	return /* @__PURE__ */ f("div", {
		className: `an-node${p ? "" : " an-node-stale"}${e.revoked ? " an-node-revoked" : ""}`,
		children: [
			/* @__PURE__ */ f("div", {
				className: "an-node-head",
				children: [
					/* @__PURE__ */ d("span", {
						className: `an-dot ${p ? "an-dot-live" : "an-dot-stale"}`,
						title: p ? "heartbeat fresh" : "no recent heartbeat (stale / offline)"
					}),
					/* @__PURE__ */ d("span", {
						className: "an-name",
						title: e.id,
						children: e.name || e.id
					}),
					/* @__PURE__ */ d("span", {
						className: `an-status ${Oi(e.status)}`,
						children: e.status || "unknown"
					}),
					/* @__PURE__ */ d("span", {
						className: `an-ct ${m.cls}`,
						title: m.title,
						children: m.text
					}),
					e.version && /* @__PURE__ */ f("span", {
						className: "an-ver",
						title: "node version",
						children: ["v", e.version]
					}),
					/* @__PURE__ */ f("span", {
						className: "an-seen",
						title: "last heartbeat",
						children: ["seen ", Ti(e.last_seen)]
					}),
					e.revoked && /* @__PURE__ */ d("span", {
						className: "an-revoked",
						title: "node credential revoked",
						children: "revoked"
					})
				]
			}),
			(e.host || e.capabilities && e.capabilities.length > 0) && /* @__PURE__ */ f("div", {
				className: "an-node-meta",
				children: [e.host && /* @__PURE__ */ d("span", {
					className: "an-host",
					title: "reported host",
					children: e.host
				}), Array.isArray(e.capabilities) && e.capabilities.map((e) => /* @__PURE__ */ d("span", {
					className: "an-cap",
					children: e
				}, e))]
			}),
			/* @__PURE__ */ f("div", {
				className: "an-dispatch",
				children: [/* @__PURE__ */ d("input", {
					className: "an-dispatch-input",
					type: "text",
					value: r,
					placeholder: "Prompt to dispatch to this node…",
					disabled: o || e.revoked,
					onChange: (e) => a(e.target.value),
					onKeyDown: (e) => {
						e.key === "Enter" && h();
					}
				}), /* @__PURE__ */ d("button", {
					className: "an-dispatch-btn",
					onClick: h,
					disabled: o || e.revoked || !r.trim(),
					title: "POST /agent/<id>/dispatch {task:{prompt}}",
					children: o ? "Dispatching…" : "▶ Dispatch"
				})]
			}),
			c && /* @__PURE__ */ d("div", {
				className: "an-dispatch-err",
				children: c
			}),
			t.length > 0 && /* @__PURE__ */ d("div", {
				className: "an-task-list",
				children: t.map((e) => /* @__PURE__ */ d(Ai, { rec: e }, e.key))
			})
		]
	});
}
function Mi({ embedded: e = !1 }) {
	let [t, n] = l([]), [r, a] = l(!1), [s, p] = l({
		kind: "loading",
		msg: null
	}), [m, h] = l([]), g = c(!0), _ = i(async () => {
		let e;
		try {
			e = await U("/api/agent/nodes");
		} catch (e) {
			g.current && p({
				kind: "error",
				msg: e.message || "network error"
			});
			return;
		}
		let t = await e.text();
		if (g.current) {
			if (e.ok) {
				let e = [];
				try {
					e = JSON.parse(t);
				} catch {
					e = [];
				}
				n(Array.isArray(e) ? e : []), p({
					kind: "ok",
					msg: null
				});
				return;
			}
			if (e.status === 404 || e.status === 501) p({
				kind: "unavailable",
				msg: "agent routes not deployed on this central yet"
			});
			else if (e.status === 401 || e.status === 403) p({
				kind: "auth",
				msg: "operator authentication required to view agent nodes"
			});
			else {
				let n = t.trim();
				try {
					let e = JSON.parse(t);
					n = e.error || e.detail || e.message || n;
				} catch {}
				p({
					kind: "error",
					msg: n || `HTTP ${e.status}`
				});
			}
		}
	}, []);
	o(() => {
		g.current = !0, _();
		let e = setInterval(_, Ci);
		return () => {
			g.current = !1, clearInterval(e);
		};
	}, [_]), o(() => {
		let e = m.filter((e) => !xi.includes(e.view && e.view.status));
		if (!e.length) return;
		let t = setInterval(async () => {
			await Promise.all(e.map(async (e) => {
				try {
					let t = await K(`/api/agent/${encodeURIComponent(e.nodeId)}/tasks/${encodeURIComponent(e.seq)}`);
					h((n) => n.map((n) => n.key === e.key ? {
						...n,
						view: t
					} : n));
				} catch {}
			}));
		}, wi);
		return () => clearInterval(t);
	}, [m]);
	let v = i(async (e, t) => {
		let n = await K(`/api/agent/${encodeURIComponent(e.id)}/dispatch`, {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({ task: { prompt: t } })
		});
		return h((r) => [{
			key: `${e.id}:${n.seq}`,
			nodeId: e.id,
			nodeName: e.name || e.id,
			seq: n.seq,
			prompt: t,
			view: n,
			dispatchedAt: Date.now()
		}, ...r]), n;
	}, []), y = i((e) => m.filter((t) => t.nodeId === e), [m]), b = t.filter(Ei).length, x = {
		loading: null,
		ok: null,
		unavailable: "routes not deployed",
		auth: "operator auth required",
		error: "roster error"
	}[s.kind];
	return /* @__PURE__ */ f("div", {
		className: "agentnodes-panel",
		children: [/* @__PURE__ */ f("div", {
			className: `an-bar${e ? " an-bar-static" : ""}`,
			onClick: e ? void 0 : () => a((e) => !e),
			children: [
				/* @__PURE__ */ d("span", {
					className: "an-title",
					children: "🕸 Agent Nodes — dispatch & watch"
				}),
				/* @__PURE__ */ f("span", {
					className: "an-count",
					children: [
						b,
						" live / ",
						t.length,
						" total"
					]
				}),
				x && /* @__PURE__ */ d("span", {
					className: `an-headnote an-headnote-${s.kind}`,
					title: s.msg || "",
					children: x
				}),
				!e && /* @__PURE__ */ d("span", {
					className: "an-toggle",
					children: r ? "▾" : "▸"
				})
			]
		}), (e || r) && /* @__PURE__ */ f("div", {
			className: "an-body",
			children: [
				/* @__PURE__ */ f("div", {
					className: "an-howto",
					children: [
						"Nodes enroll by running ",
						/* @__PURE__ */ d("code", { children: "hugpy-agent serve --node" }),
						" with",
						" ",
						/* @__PURE__ */ d("code", { children: "HUGPY_AGENT_CENTRAL" }),
						" pointed at this console. Dispatched prompts are pulled and run by the node; results stream back here as each task finalizes."
					]
				}),
				s.kind === "unavailable" && /* @__PURE__ */ f("div", {
					className: "an-banner an-banner-unavailable",
					children: [
						"The ",
						/* @__PURE__ */ d("code", { children: "/agent/*" }),
						" routes are not deployed on this central yet.",
						s.msg ? /* @__PURE__ */ f(u, { children: [
							" (",
							s.msg,
							")"
						] }) : null,
						" The panel will populate once the P3.1/P3.1b blueprint is live."
					]
				}),
				s.kind === "auth" && /* @__PURE__ */ f("div", {
					className: "an-banner an-banner-auth",
					children: [s.msg, ". Sign in as the operator (or set the operator token) to view and dispatch to agent nodes."]
				}),
				s.kind === "error" && /* @__PURE__ */ f("div", {
					className: "an-banner an-banner-error",
					title: s.msg || "",
					children: ["Could not read the node roster: ", s.msg || "unknown error"]
				}),
				s.kind === "loading" && /* @__PURE__ */ d("div", {
					className: "an-banner an-banner-loading",
					children: "Loading node roster…"
				}),
				s.kind === "ok" && t.length === 0 && /* @__PURE__ */ d("div", {
					className: "an-empty",
					children: "No agent nodes have enrolled yet."
				}),
				t.map((e) => /* @__PURE__ */ d(ji, {
					node: e,
					tasks: y(e.id),
					onDispatch: v
				}, e.id))
			]
		})]
	});
}
//#endregion
//#region src/components/AssessmentPanel/AssessmentPanel.jsx
function Ni(e) {
	if (e == null) return "—";
	let t = [
		"B",
		"KB",
		"MB",
		"GB",
		"TB"
	], n = Number(e), r = 0;
	for (; n >= 1024 && r < t.length - 1;) n /= 1024, r++;
	return `${n.toFixed(+(n < 10 && r > 0))} ${t[r]}`;
}
function Pi(e) {
	return e ? e >= 1e9 ? `${(e / 1e9).toFixed(1)}B` : e >= 1e6 ? `${Math.round(e / 1e6)}M` : String(e) : null;
}
var Fi = [
	{
		id: "all",
		label: "All"
	},
	{
		id: "media",
		label: "Media"
	},
	{
		id: "worker",
		label: "Worker"
	},
	{
		id: "slot",
		label: "Slot"
	},
	{
		id: "api",
		label: "API"
	},
	{
		id: "active",
		label: "Active"
	}
], Ii = [
	"media",
	"worker",
	"slot",
	"api",
	"processing"
], Li = [
	{
		id: "name",
		label: "Model",
		w: 200,
		cls: "as-c-name",
		sort: (e) => e.name.toLowerCase()
	},
	{
		id: "loc",
		label: "Location",
		w: 120,
		cls: "as-c-loc",
		sort: (e) => e.loc || ""
	},
	{
		id: "pending",
		label: "Pending",
		w: 78,
		cls: "as-c-m",
		num: !0,
		sort: (e) => e.qWaiting
	},
	{
		id: "active",
		label: "Active",
		w: 78,
		cls: "as-c-m",
		num: !0,
		sort: (e) => e.qActive
	},
	{
		id: "vram",
		label: "VRAM",
		w: 90,
		cls: "as-c-m",
		num: !0,
		sort: (e) => e.vram ?? -1
	},
	{
		id: "ram",
		label: "RAM",
		w: 90,
		cls: "as-c-m",
		num: !0,
		sort: (e) => e.ram ?? -1
	},
	{
		id: "cores",
		label: "Cores",
		w: 78,
		cls: "as-c-m",
		num: !0,
		sort: (e) => e.cores ?? -1
	},
	{
		id: "cats",
		label: "Categories",
		w: 220,
		cls: "as-c-cats",
		sort: (e) => Ii.filter((t) => e.cats[t]).length
	}
];
function Ri(e, t, n) {
	switch (e) {
		case "name": return /* @__PURE__ */ f(u, { children: [/* @__PURE__ */ d("span", {
			className: "as-caret",
			children: n ? "▾" : "▸"
		}), t.name] });
		case "cats": return /* @__PURE__ */ d(Bi, { r: t });
		case "loc": return t.loc || "—";
		case "pending": return t.qWaiting || "·";
		case "active": return t.qActive || "·";
		case "vram": return t.vram == null ? "—" : Ni(t.vram);
		case "ram": return t.ram == null ? "—" : Ni(t.ram);
		case "cores": return t.cores == null ? "—" : `${t.cores} thr`;
		default: return null;
	}
}
function zi({ models: e = [], workers: t = [] }) {
	let [n, r] = l({
		enabled: null,
		slots: [],
		resources: null
	}), [a, c] = l({
		active: [],
		counts: {}
	}), [u, p] = l([]), [m, h] = l(/* @__PURE__ */ new Map()), [g, _] = l("all"), [v, y] = l({}), [b, x] = l({
		col: "loc",
		dir: -1
	}), [S, C] = l(() => Li.map((e) => e.w));
	o(() => {
		let e = !0, t = () => {
			K("/api/llm/slots").then((t) => e && t && r(t)).catch(() => {}), K("/api/llm/queue").then((t) => e && c(t || {
				active: [],
				counts: {}
			})).catch(() => {}), K("/api/llm/serving").then((t) => e && p(Array.isArray(t) ? t : [])).catch(() => {}), K("/api/v1/models").then((t) => e && h(new Map((t?.data || []).map((e) => [e.id, e])))).catch(() => {});
		};
		t();
		let n = setInterval(t, 3e3);
		return () => {
			e = !1, clearInterval(n);
		};
	}, []);
	let w = i((e) => y((t) => ({
		...t,
		[e]: !t[e]
	})), []), T = s(() => {
		let r = a.active || [], i = {};
		for (let e of n.slots || []) e.model_key && (i[e.model_key] ||= []).push(e);
		let o = /* @__PURE__ */ new Map();
		for (let t of e) o.set(t.model_key ?? t.key, t);
		let s = (e) => {
			e && !o.has(e) && o.set(e, {
				model_key: e,
				name: e,
				status: "remote"
			});
		};
		for (let e of t) (e.models || []).forEach(s), (e.loaded_models || []).forEach(s), (e.provisioning || []).forEach(s);
		for (let e of n.slots || []) s(e.model_key);
		for (let e of r) s(e.model_key);
		for (let e of m.keys()) s(e);
		return [...o.values()].map((e) => {
			let n = e.model_key ?? e.key, a = t.filter((e) => (e.models || []).includes(n)), o = t.filter((e) => (e.loaded_models || []).includes(n)), s = t.filter((e) => (e.provisioning || []).includes(n)), c = i[n] || [], l = c.some((e) => e.healthy), u = c.some((e) => e.model_key && !e.healthy), d = m.get(n) || null, f = r.filter((e) => e.model_key === n && e.state === "waiting"), p = r.filter((e) => e.model_key === n && e.state === "active"), h = {
				media: !!e.media,
				worker: a.length > 0 || o.length > 0 || s.length > 0,
				loaded: o.length > 0,
				slot: c.length > 0,
				serving: l,
				api: !!d,
				processing: p.length > 0
			}, g = s.length > 0 || u, _ = 0, v = !1;
			for (let e of c) e.expected_bytes != null && (_ += e.expected_bytes, v = !0);
			for (let e of a) {
				let t = e.spill_by_model?.[n]?.gpu_mem_gib;
				t != null && (_ += t * 1073741824, v = !0);
			}
			let y = 0, b = !1;
			for (let e of c) {
				let t = e.rss_anon_bytes ?? e.rss_bytes;
				t && (y += t, b = !0);
			}
			for (let e of t) for (let t of e.allocations || []) {
				if (!t || t.model_key !== n || t.kind !== "ram") continue;
				let e = t.rss_anon_bytes ?? t.ram_resident_bytes;
				e && (y += e, b = !0);
			}
			let x = 0, S = !1;
			for (let e of c) e.threads != null && (x += e.threads, S = !0);
			let C = [.../* @__PURE__ */ new Set([
				...o,
				...s,
				...a
			])], w = [...c.map((e) => `slot ${e.slot_id}`), ...C.map((e) => e.name)], T = w.length ? w[0] + (w.length > 1 ? ` +${w.length - 1}` : "") : null;
			return {
				key: n,
				name: e.name || n,
				model: e,
				cats: h,
				loading: g,
				qWaiting: f.length,
				qActive: p.length,
				qActiveRows: p,
				vram: v ? _ : null,
				ram: b ? y : null,
				cores: S ? x : null,
				loc: T,
				mSlots: c,
				assignedOn: a,
				loadedOn: o,
				pendingOn: s,
				workersForModel: C,
				apiEntry: d
			};
		}).filter((e) => e.cats.media || e.cats.worker || e.cats.slot || e.cats.api || e.cats.processing);
	}, [
		e,
		t,
		n,
		a,
		m
	]), E = s(() => T.filter((e) => g === "all" ? !0 : g === "active" ? e.cats.processing : e.cats[g]), [T, g]), D = s(() => {
		if (!b.col) return E;
		let e = Li.find((e) => e.id === b.col);
		return e ? [...E].sort((t, n) => {
			let r = e.sort(t), i = e.sort(n);
			return r < i ? -b.dir : r > i ? b.dir : 0;
		}) : E;
	}, [E, b]), O = i((e) => x((t) => t.col === e ? {
		col: e,
		dir: -t.dir
	} : {
		col: e,
		dir: 1
	}), []), k = i((e, t) => {
		t.preventDefault(), t.stopPropagation();
		let n = t.clientX, r = S[e], i = (t) => C((i) => {
			let a = [...i];
			return a[e] = Math.max(48, r + (t.clientX - n)), a;
		}), a = () => {
			window.removeEventListener("mousemove", i), window.removeEventListener("mouseup", a), document.body.style.cursor = "";
		};
		window.addEventListener("mousemove", i), window.addEventListener("mouseup", a), document.body.style.cursor = "col-resize";
	}, [S]), A = S.map((e) => `${e}px`).join(" ");
	return /* @__PURE__ */ f("div", {
		className: "as-panel",
		children: [/* @__PURE__ */ d("div", {
			className: "as-filters",
			children: Fi.map((e) => /* @__PURE__ */ f("button", {
				className: `as-chip${g === e.id ? " as-chip-on" : ""}`,
				onClick: () => _(e.id),
				children: [e.label, e.id !== "all" && /* @__PURE__ */ d("span", {
					className: "as-chip-n",
					children: T.filter((t) => e.id === "active" ? t.cats.processing : t.cats[e.id]).length
				})]
			}, e.id))
		}), /* @__PURE__ */ f("div", {
			className: "as-list",
			style: { "--as-cols": A },
			children: [
				/* @__PURE__ */ d("div", {
					className: "as-row as-head",
					children: Li.map((e, t) => /* @__PURE__ */ f("div", {
						className: `as-hcell ${e.cls}${e.num ? " as-num" : ""}${b.col === e.id ? " as-sorted" : ""}`,
						onClick: () => O(e.id),
						title: "Click to sort · drag the edge to resize",
						children: [
							/* @__PURE__ */ d("span", {
								className: "as-hlabel",
								children: e.label
							}),
							/* @__PURE__ */ d("span", {
								className: "as-sort-ind",
								children: b.col === e.id ? b.dir > 0 ? "▲" : "▼" : ""
							}),
							t < Li.length - 1 && /* @__PURE__ */ d("span", {
								className: "as-resize",
								onMouseDown: (e) => k(t, e),
								onClick: (e) => e.stopPropagation()
							})
						]
					}, e.id))
				}),
				D.length === 0 && /* @__PURE__ */ d("div", {
					className: "as-empty",
					children: "No models match — nothing is in media, on a worker, in a slot, on the API, or processing."
				}),
				D.map((e) => /* @__PURE__ */ f("div", {
					className: `as-item${v[e.key] ? " as-item-open" : ""}`,
					children: [/* @__PURE__ */ d("button", {
						className: "as-row as-data",
						onClick: () => w(e.key),
						children: Li.map((t) => /* @__PURE__ */ d("span", {
							className: `${t.cls}${t.id === "pending" && e.qWaiting ? " as-m-warn" : ""}${t.id === "active" && e.qActive ? " as-m-live" : ""}`,
							children: Ri(t.id, e, v[e.key])
						}, t.id))
					}), v[e.key] && /* @__PURE__ */ d(Vi, {
						r: e,
						serving: u
					})]
				}, e.key))
			]
		})]
	});
}
function Bi({ r: e }) {
	let t = [];
	return e.cats.media && t.push([
		"media",
		"media",
		"🎬"
	]), e.cats.slot && t.push([
		"slot",
		e.cats.serving ? "slot" : "slot·loading",
		"🧩"
	]), e.cats.worker && t.push([
		"worker",
		e.cats.loaded ? "loaded" : e.loading ? "loading" : "worker",
		e.cats.loaded ? "🔥" : e.loading ? "⏳" : "○"
	]), e.cats.api && t.push([
		"api",
		"API",
		"🔌"
	]), e.cats.processing && t.push([
		"proc",
		"active",
		"⚡"
	]), /* @__PURE__ */ d(u, { children: t.map(([e, t, n]) => /* @__PURE__ */ f("span", {
		className: `as-tag as-tag-${e}`,
		children: [
			n,
			" ",
			t
		]
	}, e + t)) });
}
function Vi({ r: e, serving: t }) {
	let n = t.find((t) => t.key === e.key);
	return /* @__PURE__ */ f("div", {
		className: "as-detail",
		children: [
			/* @__PURE__ */ f(Hi, {
				k: "Model",
				children: [
					Pi(e.model.parameter_count) && /* @__PURE__ */ d(Z, {
						label: "params",
						children: Pi(e.model.parameter_count)
					}),
					e.model.framework && /* @__PURE__ */ d(Z, {
						label: "framework",
						children: e.model.framework
					}),
					e.model.primary_task && /* @__PURE__ */ d(Z, {
						label: "task",
						children: e.model.primary_task
					}),
					e.model.total_bytes && /* @__PURE__ */ d(Z, {
						label: "on disk",
						children: Ni(e.model.total_bytes)
					}),
					/* @__PURE__ */ d(Z, {
						label: "status",
						children: e.model.status
					}),
					e.cats.media && /* @__PURE__ */ d(Z, {
						label: "media",
						children: "selected"
					})
				]
			}),
			/* @__PURE__ */ f(Hi, {
				k: "Queue",
				children: [
					/* @__PURE__ */ d(Z, {
						label: "pending",
						children: e.qWaiting
					}),
					/* @__PURE__ */ d(Z, {
						label: "active",
						children: e.qActive
					}),
					e.qActiveRows.map((e) => /* @__PURE__ */ f(Z, {
						label: e.kind || "req",
						children: [
							e.tokens == null ? "" : `${e.tokens} tok`,
							" ",
							e.elapsed == null ? "" : `· ${e.elapsed}s`
						]
					}, e.request_id)),
					!e.qWaiting && !e.qActive && /* @__PURE__ */ d("span", {
						className: "as-dim",
						children: "idle — no in-flight requests"
					})
				]
			}),
			e.workersForModel.length > 0 && /* @__PURE__ */ d(Hi, {
				k: "Workers",
				children: e.workersForModel.map((t) => {
					let n = (t.loaded_models || []).includes(e.key), r = (t.provisioning || []).includes(e.key), i = t.spill_by_model?.[e.key]?.gpu_mem_gib, a = (t.allocations || []).find((t) => t && t.model_key === e.key && t.kind === "ram"), o = a ? a.rss_anon_bytes ?? a.ram_resident_bytes : null;
					return /* @__PURE__ */ f(Z, {
						label: t.name,
						title: o == null ? void 0 : "measured resident host RAM (worker smaps read)",
						children: [
							r ? "⏳ loading" : n ? "🔥 loaded" : "○ assigned",
							o == null ? "" : ` · ${Ni(o)} RAM resident`,
							i == null ? "" : ` · ${i} GiB GPU budget`,
							` · ${t.gpus?.[0]?.name || (t.gpus?.length ? "GPU" : "CPU")}`
						]
					}, t.id);
				})
			}),
			e.mSlots.map((e) => /* @__PURE__ */ f(Hi, {
				k: `Slot ${e.slot_id}`,
				children: [
					/* @__PURE__ */ d(Z, {
						label: "state",
						children: e.healthy ? "serving" : e.model_key ? "loading" : "idle"
					}),
					e.expected_bytes != null && /* @__PURE__ */ d(Z, {
						label: "VRAM",
						children: Ni(e.expected_bytes)
					}),
					e.rss_anon_bytes == null ? e.rss_bytes ? /* @__PURE__ */ f(Z, {
						label: "RAM (VmRSS)",
						title: "VmRSS — includes mmap'd file pages, overstates pinned RAM",
						children: ["~", Ni(e.rss_bytes)]
					}) : null : /* @__PURE__ */ f(Z, {
						label: "RAM",
						children: [Ni(e.rss_anon_bytes), e.rss_file_bytes > 0 ? ` + ${Ni(e.rss_file_bytes)} cache` : ""]
					}),
					e.n_gpu_layers != null && /* @__PURE__ */ f(Z, {
						label: "GPU layers",
						children: [String(e.n_gpu_layers), e.total_layers == null ? "" : `/${e.total_layers}`]
					}),
					e.threads != null && /* @__PURE__ */ d(Z, {
						label: "threads",
						children: e.threads
					}),
					(e.allowed_cpus || e.cpus) && /* @__PURE__ */ d(Z, {
						label: "cores",
						children: e.allowed_cpus || e.cpus
					}),
					e.gpu != null && e.gpu !== "" && /* @__PURE__ */ d(Z, {
						label: "GPU",
						children: String(e.gpu)
					}),
					e.ctx != null && /* @__PURE__ */ d(Z, {
						label: "context",
						children: e.ctx
					}),
					e.free_vram_bytes != null && /* @__PURE__ */ d(Z, {
						label: "VRAM free",
						children: Ni(e.free_vram_bytes)
					})
				]
			}, e.slot_id)),
			(e.cats.api || n) && /* @__PURE__ */ f(Hi, {
				k: "API",
				children: [
					e.apiEntry && /* @__PURE__ */ d(Z, {
						label: "exposed",
						children: "/api/v1/models"
					}),
					e.apiEntry?.context_length && /* @__PURE__ */ d(Z, {
						label: "context",
						children: e.apiEntry.context_length
					}),
					n && /* @__PURE__ */ d(Z, {
						label: "serve mode",
						children: n.always_on ? "always-on" : n.mode || "swap"
					}),
					n?.endpoint && /* @__PURE__ */ d(Z, {
						label: "endpoint",
						children: n.endpoint
					}),
					n?.ttl_seconds && /* @__PURE__ */ f(Z, {
						label: "ttl",
						children: [n.ttl_seconds, "s"]
					})
				]
			})
		]
	});
}
function Hi({ k: e, children: t }) {
	return /* @__PURE__ */ f("div", {
		className: "as-drow",
		children: [/* @__PURE__ */ d("span", {
			className: "as-dk",
			children: e
		}), /* @__PURE__ */ d("div", {
			className: "as-dv",
			children: t
		})]
	});
}
function Z({ label: e, title: t, children: n }) {
	return /* @__PURE__ */ f("span", {
		className: "as-field",
		title: t,
		children: [/* @__PURE__ */ d("span", {
			className: "as-field-k",
			children: e
		}), n]
	});
}
//#endregion
//#region src/components/StatusBar/FleetResidency.jsx
var Ui = [
	["answering", "answering"],
	["loading", "loading"],
	["serving", "serving"],
	["idle", "idle (resident)"]
];
function Wi(e) {
	if (e == null || !isFinite(e)) return "—";
	let t = [
		"B",
		"KiB",
		"MiB",
		"GiB",
		"TiB"
	], n = Number(e), r = 0;
	for (; n >= 1024 && r < t.length - 1;) n /= 1024, r++;
	return `${n.toFixed(+(n < 10 && r > 0))} ${t[r]}`;
}
var Gi = (e) => typeof e == "number" && isFinite(e) && e > 0 ? e : 0;
function Ki(e, t, n, r) {
	return e && e.busy === !0 || n.has(t) ? "answering" : r.has(t) ? "loading" : e && (e.serving === !0 || e.healthy === !0) ? "serving" : "idle";
}
function qi(e) {
	if (!e) return null;
	let t = e.n_gpu_layers, n = e.total_layers;
	return typeof t != "number" || typeof n != "number" || n <= 0 ? null : t >= n ? `all ${n} layers on GPU` : `${t}/${n} layers on GPU`;
}
function Ji(e, t, n) {
	let r = e.reduce((e, t) => e + t.bytes, 0), i = Math.max(0, Gi(t) - r), a = Math.max(Gi(n), r + i, 1), o = Math.max(0, a - r - i);
	return {
		denom: a,
		sum: r,
		other: i,
		free: o,
		overflow: r > Gi(t) && Gi(t) > 0,
		segs: e.map((e) => ({
			...e,
			pct: e.bytes / a * 100
		})),
		otherPct: i / a * 100,
		freePct: o / a * 100
	};
}
function Yi({ workers: e = [], queue: t = {} }) {
	let n = s(() => {
		let n = /* @__PURE__ */ new Set();
		for (let e of t.active || []) if (e && e.state === "active") {
			let t = e.model_key || e.model;
			t && n.add(t);
		}
		return (e || []).filter((e) => e && e.status === "online").map((e) => {
			let t = new Set([...Array.isArray(e.loading) ? e.loading : [], ...Array.isArray(e.provisioning) ? e.provisioning : []].filter(Boolean)), r = e.provision_progress || {}, i = Array.isArray(e.gpus) ? e.gpus : [], a = i.reduce((e, t) => e + Gi(t.memory_total), 0), o = i.reduce((e, t) => e + Gi(t.memory_free), 0), s = Math.max(0, a - o), c = i.map((e) => e.name).filter(Boolean).join(" + "), l = Gi(e.bar_total) || Gi(e.free_ram) + Gi(e.bar_used), u = Gi(e.bar_used) || Math.max(0, l - Gi(e.free_ram)), d = Array.isArray(e.allocations) ? e.allocations : null, f = d != null, p = [], m = [];
			for (let e of d || []) {
				let r = e && e.model_key;
				if (!r) continue;
				let i = Ki(e, r, n, t), a = qi(e), o = Gi(e.vram_bytes);
				o && p.push({
					key: r,
					bytes: o,
					state: i,
					layers: a,
					where: "vram"
				});
				let s = Gi(e.ram_resident_bytes) || Gi(e.rss_anon_bytes), c = s || Gi(e.model_bytes);
				c && m.push({
					key: r,
					bytes: c,
					state: i,
					layers: a,
					where: "ram",
					basis: s ? "measured" : "file"
				});
			}
			if (p.sort((e, t) => t.bytes - e.bytes), m.sort((e, t) => t.bytes - e.bytes), m.some((e) => e.basis === "measured") && m.some((e) => e.basis === "file")) for (let e of m) e.mixed = !0;
			let h = new Set([...p, ...m].map((e) => e.key)), g = [...t].filter((e) => !h.has(e)).map((e) => ({
				key: e,
				frac: r[e] && typeof r[e].frac == "number" ? r[e].frac : null
			}));
			return {
				id: e.name || e.worker_id || e.id || e.host || "worker",
				name: e.name || e.worker_id || e.host || "worker",
				gpuName: c,
				detailed: f,
				pending: g,
				vram: Ji(p, s, a),
				vramUsed: s,
				vramTotal: a,
				ram: Ji(m, u, l),
				ramUsed: u,
				ramTotal: l
			};
		});
	}, [e, t]);
	return /* @__PURE__ */ f("div", {
		className: "fr-panel",
		children: [/* @__PURE__ */ f("div", {
			className: "fr-legend",
			children: [
				Ui.map(([e, t]) => /* @__PURE__ */ f("span", {
					className: "fr-legend-item",
					children: [/* @__PURE__ */ d("i", { className: `fr-dot fr-s-${e}` }), t]
				}, e)),
				/* @__PURE__ */ f("span", {
					className: "fr-legend-item",
					children: [/* @__PURE__ */ d("i", { className: "fr-dot fr-s-other" }), "other / unattributed"]
				}),
				/* @__PURE__ */ f("span", {
					className: "fr-legend-item",
					children: [/* @__PURE__ */ d("i", { className: "fr-dot fr-s-free" }), "free"]
				})
			]
		}), n.length === 0 ? /* @__PURE__ */ d("div", {
			className: "fr-empty",
			children: "no online workers"
		}) : n.map((e) => /* @__PURE__ */ f("div", {
			className: "fr-worker",
			children: [
				/* @__PURE__ */ f("div", {
					className: "fr-worker-head",
					children: [
						/* @__PURE__ */ d("span", {
							className: "fr-worker-name",
							children: e.name
						}),
						e.gpuName && /* @__PURE__ */ d("span", {
							className: "fr-worker-gpu",
							children: e.gpuName
						}),
						!e.detailed && /* @__PURE__ */ d("span", {
							className: "fr-nodetail",
							children: "no per-model detail"
						})
					]
				}),
				/* @__PURE__ */ d(Zi, {
					label: "VRAM",
					bar: e.vram,
					used: e.vramUsed,
					total: e.vramTotal,
					detailed: e.detailed,
					empty: "no GPU-resident models"
				}),
				/* @__PURE__ */ d(Zi, {
					label: "RAM",
					bar: e.ram,
					used: e.ramUsed,
					total: e.ramTotal,
					detailed: e.detailed,
					empty: "no RAM-parked models"
				}),
				e.pending.length > 0 && /* @__PURE__ */ d("div", {
					className: "fr-list",
					children: e.pending.map((e) => /* @__PURE__ */ f("span", {
						className: "fr-chip",
						title: `${e.key} — loading${e.frac == null ? "" : ` ${Math.round(e.frac * 100)}%`}`,
						children: [
							/* @__PURE__ */ d("i", { className: "fr-dot fr-s-loading" }),
							/* @__PURE__ */ d("span", {
								className: "fr-chip-key",
								children: e.key
							}),
							/* @__PURE__ */ f("span", {
								className: "fr-chip-size",
								children: ["loading", e.frac == null ? "" : ` ${Math.round(e.frac * 100)}%`]
							})
						]
					}, `ld-${e.key}`))
				})
			]
		}, e.id))]
	});
}
function Xi(e, t) {
	let n = [
		e.key,
		`${Wi(e.bytes)} ${t}`,
		e.state
	];
	return e.layers && n.push(e.layers), e.mixed && n.push(e.basis === "measured" ? "measured resident" : "file size — upper bound"), n.join(" — ");
}
function Zi({ label: e, bar: t, used: n, total: r, detailed: i, empty: a }) {
	return r ? /* @__PURE__ */ f("div", {
		className: "fr-row",
		children: [/* @__PURE__ */ d("span", {
			className: "fr-row-label",
			children: e
		}), /* @__PURE__ */ f("div", {
			className: "fr-row-body",
			children: [/* @__PURE__ */ f("div", {
				className: "fr-track",
				title: `${e}: ${Wi(n)} used of ${Wi(r)}`,
				children: [
					t.segs.map((t, n) => /* @__PURE__ */ d("div", {
						className: `fr-seg fr-s-${t.state}`,
						style: { width: `${t.pct}%` },
						title: Xi(t, e)
					}, `${t.key}-${n}`)),
					t.otherPct > 0 && /* @__PURE__ */ d("div", {
						className: "fr-seg fr-s-other",
						style: { width: `${t.otherPct}%` },
						title: `other / unattributed — ${Wi(t.other)} ${e}`
					}),
					t.freePct > 0 && /* @__PURE__ */ d("div", {
						className: "fr-seg fr-s-free",
						style: { width: `${t.freePct}%` },
						title: `free — ${Wi(t.free)} ${e}`
					})
				]
			}), /* @__PURE__ */ f("div", {
				className: "fr-list",
				children: [t.segs.length === 0 ? /* @__PURE__ */ d("span", {
					className: "fr-note",
					children: i ? a : "no per-model detail from this worker"
				}) : t.segs.map((t, n) => /* @__PURE__ */ f("span", {
					className: "fr-chip",
					title: Xi(t, e),
					children: [
						/* @__PURE__ */ d("i", { className: `fr-dot fr-s-${t.state}` }),
						/* @__PURE__ */ d("span", {
							className: "fr-chip-key",
							children: t.key
						}),
						/* @__PURE__ */ d("span", {
							className: "fr-chip-size",
							children: Wi(t.bytes)
						}),
						t.layers && /* @__PURE__ */ d("span", {
							className: "fr-chip-layers",
							children: t.layers
						})
					]
				}, `${t.key}-c-${n}`)), /* @__PURE__ */ f("span", {
					className: "fr-note fr-note-tot",
					children: [
						Wi(n),
						" used / ",
						Wi(r)
					]
				})]
			})]
		})]
	}) : null;
}
//#endregion
//#region src/components/StatusBar/StatusBar.jsx
function Qi(e) {
	if (e == null) return "—";
	let t = [
		"B",
		"KB",
		"MB",
		"GB",
		"TB"
	], n = Number(e), r = 0;
	for (; n >= 1024 && r < t.length - 1;) n /= 1024, r++;
	return `${n.toFixed(+(n < 10 && r > 0))} ${t[r]}`;
}
function $i({ models: e = [], workers: t = [] }) {
	let [n, r] = l({
		enabled: null,
		slots: [],
		resources: null
	}), [i, a] = l({
		active: [],
		counts: {}
	}), [c, u] = l([]), [p, m] = l([]), [h, g] = Y("hugpy.sess.sb.residency", !1);
	o(() => {
		let e = !0, t = () => {
			K("/api/llm/slots").then((t) => e && t && r(t)).catch(() => {}), K("/api/llm/queue").then((t) => e && a(t || {
				active: [],
				counts: {}
			})).catch(() => {}), K("/api/llm/jobs").then((t) => e && u(Array.isArray(t?.jobs) ? t.jobs : [])).catch(() => {}), K("/api/phone-brick/phones").then((t) => e && m(Array.isArray(t) ? t : [])).catch(() => {});
		};
		t();
		let n = setInterval(t, 3e3);
		return () => {
			e = !1, clearInterval(n);
		};
	}, []);
	let _ = s(() => {
		let r = n.slots || [], a = /* @__PURE__ */ new Set();
		for (let e of t) if (e.status === "online") for (let t of e.loaded_models || []) a.add(t);
		for (let e of r) e.model_key && e.healthy && a.add(e.model_key);
		let o = /* @__PURE__ */ new Set([
			"pending",
			"processing",
			"streaming"
		]), s = c.filter((e) => e.kind !== "download" && o.has(e.status)), l = new Set((i.active || []).map((e) => e.request_id)), u = s.filter((e) => !l.has(e.id)), d = (i.counts?.active ?? (i.active || []).filter((e) => e.state === "active").length) + u.filter((e) => e.status !== "pending").length, f = (i.counts?.waiting ?? (i.active || []).filter((e) => e.state === "waiting").length) + u.filter((e) => e.status === "pending").length, m = {};
		for (let e of u) {
			let t = e.transport || e.kind || "?";
			m[t] = (m[t] || 0) + 1;
		}
		let h = (i.active || []).length;
		h && (m.web = (m.web || 0) + h);
		let g = t.filter((e) => e.status === "online").length, _ = t.filter((e) => e.admission === "pending").length, v = t.length - g, y = r.filter((e) => e.model_key).length, b = r.filter((e) => e.error).length, x = p.filter((e) => e.status === "online").length, S = p.length - x, C = 0, w = 0, T = !1;
		for (let e of t) if (e.status === "online") for (let t of e.gpus || []) t.memory_total != null && (C += t.memory_total, T = !0), t.memory_free != null && (w += t.memory_free);
		let E = n.resources, D = [];
		return _ && D.push({
			k: "wpend",
			cls: "warn",
			text: `${_} worker pending`
		}), v && D.push({
			k: "woff",
			cls: "bad",
			text: `${v} worker offline`
		}), b && D.push({
			k: "sdown",
			cls: "bad",
			text: `${b} slot down`
		}), S && D.push({
			k: "poff",
			cls: "bad",
			text: `${S} phone offline`
		}), {
			installed: e.filter((e) => e.status === "installed").length,
			registered: e.length,
			inService: a.size,
			active: d,
			pending: f,
			byTransport: m,
			wOnline: g,
			wTotal: t.length,
			slotsEnabled: n.enabled,
			slotsBusy: y,
			slotsTotal: r.length,
			phOnline: x,
			phTotal: p.length,
			vramUsed: Math.max(C - w, 0),
			vramTotal: C,
			hasVram: T,
			ramUsed: E?.used_bytes ?? null,
			ramTotal: E?.total_bytes ?? null,
			alerts: D
		};
	}, [
		e,
		t,
		n,
		i,
		c,
		p
	]);
	return /* @__PURE__ */ f("div", {
		className: "sb-bar",
		children: [
			/* @__PURE__ */ f("div", {
				className: "sb-tiles",
				children: [
					/* @__PURE__ */ d(ea, {
						k: "Installed",
						v: _.installed
					}),
					/* @__PURE__ */ d(ea, {
						k: "Registered",
						v: _.registered
					}),
					/* @__PURE__ */ d(ea, {
						k: "In service",
						v: _.inService
					}),
					/* @__PURE__ */ f("div", {
						className: "sb-seg sb-queue",
						title: Object.keys(_.byTransport || {}).length ? "by transport: " + Object.entries(_.byTransport).map(([e, t]) => `${e} ${t}`).join(" · ") : "in-flight generation across web · /v1 · discord · cli",
						children: [/* @__PURE__ */ d("span", {
							className: "sb-k",
							children: "Queue"
						}), _.active || _.pending ? /* @__PURE__ */ f("span", {
							className: "sb-v",
							children: [
								/* @__PURE__ */ f("span", {
									className: _.active ? "sb-active" : "sb-zero",
									children: [_.active, " active"]
								}),
								/* @__PURE__ */ d("span", {
									className: "sb-mid",
									children: "·"
								}),
								/* @__PURE__ */ f("span", {
									className: _.pending ? "sb-pending" : "sb-zero",
									children: [_.pending, " pending"]
								}),
								Object.entries(_.byTransport || {}).map(([e, t]) => /* @__PURE__ */ f("span", {
									className: "sb-transport",
									children: [
										e,
										" ",
										t
									]
								}, e))
							]
						}) : /* @__PURE__ */ d("span", {
							className: "sb-v sb-zero",
							children: "idle"
						})]
					}),
					/* @__PURE__ */ d(tn, {}),
					/* @__PURE__ */ d(ea, {
						k: "Workers",
						v: `${_.wOnline}/${_.wTotal}`,
						sub: "online"
					}),
					/* @__PURE__ */ d(ea, {
						k: "Slots",
						v: _.slotsEnabled === !1 ? "off" : `${_.slotsBusy}/${_.slotsTotal}`,
						sub: _.slotsEnabled === !1 ? void 0 : "busy"
					}),
					/* @__PURE__ */ d(ea, {
						k: "Phones",
						v: `${_.phOnline}/${_.phTotal}`,
						sub: "online"
					}),
					/* @__PURE__ */ d(ta, {
						k: "VRAM",
						used: _.vramUsed,
						total: _.vramTotal,
						show: _.hasVram,
						expanded: h,
						onToggle: () => g((e) => !e)
					}),
					/* @__PURE__ */ d(ta, {
						k: "Host RAM",
						used: _.ramUsed,
						total: _.ramTotal,
						show: !!_.ramTotal,
						sub: "from slots",
						expanded: h,
						onToggle: () => g((e) => !e)
					}),
					/* @__PURE__ */ d("div", {
						className: `sb-alerts${_.alerts.length ? "" : " sb-alerts-ok"}`,
						children: _.alerts.length === 0 ? /* @__PURE__ */ d("span", {
							className: "sb-ok",
							children: "✓ all healthy"
						}) : _.alerts.map((e) => /* @__PURE__ */ d("span", {
							className: `sb-alert sb-alert-${e.cls}`,
							children: e.text
						}, e.k))
					})
				]
			}),
			h && /* @__PURE__ */ d(Yi, {
				workers: t,
				queue: i
			}),
			/* @__PURE__ */ d(Qn, {})
		]
	});
}
function ea({ k: e, v: t, sub: n }) {
	return /* @__PURE__ */ f("div", {
		className: "sb-seg",
		children: [/* @__PURE__ */ d("span", {
			className: "sb-k",
			children: e
		}), /* @__PURE__ */ f("span", {
			className: "sb-v",
			children: [t, n && /* @__PURE__ */ f("span", {
				className: "sb-sub",
				children: [" ", n]
			})]
		})]
	});
}
function ta({ k: e, used: t, total: n, show: r, sub: i, expanded: a, onToggle: o }) {
	return /* @__PURE__ */ f("div", {
		className: `sb-seg sb-meter${o ? " sb-meter-x" : ""}`,
		onClick: o,
		role: o ? "button" : void 0,
		tabIndex: o ? 0 : void 0,
		onKeyDown: o ? ((e) => {
			(e.key === "Enter" || e.key === " ") && (e.preventDefault(), o());
		}) : void 0,
		title: o ? a ? "hide fleet model residency" : "show fleet model residency" : void 0,
		children: [/* @__PURE__ */ f("span", {
			className: "sb-k",
			children: [e, o && /* @__PURE__ */ d("span", {
				className: "sb-caret",
				children: a ? "▾" : "▸"
			})]
		}), r ? /* @__PURE__ */ f(u, { children: [
			/* @__PURE__ */ f("span", {
				className: "sb-v",
				children: [
					Qi(t),
					" ",
					/* @__PURE__ */ f("span", {
						className: "sb-sub",
						children: ["/ ", Qi(n)]
					})
				]
			}),
			/* @__PURE__ */ d("div", {
				className: "sb-bar-track",
				children: /* @__PURE__ */ d("div", {
					className: "sb-bar-fill",
					style: { width: `${Math.min(100, Math.round(t / n * 100))}%` }
				})
			}),
			i && /* @__PURE__ */ d("span", {
				className: "sb-note",
				children: i
			})
		] }) : /* @__PURE__ */ d("span", {
			className: "sb-v",
			children: "—"
		})]
	});
}
//#endregion
//#region src/components/DiscordPanel/DiscordPanel.jsx
var na = (e) => e.model_key ?? e.key, ra = (e) => {
	let t = [];
	return e.channel_id && t.push(`#${e.channel_id}`), e.user_id && t.push(`@${e.user_id}`), t.join(" + ") || "—";
};
function ia({ binding: e, onRemove: t, onPing: n }) {
	let [r, i] = l(""), [a, o] = l(!1), s = async () => {
		if (r.trim()) {
			o(!0);
			try {
				await n(e, r), i("");
			} finally {
				o(!1);
			}
		}
	};
	return /* @__PURE__ */ f("div", {
		className: "dc-binding",
		children: [
			/* @__PURE__ */ f("span", {
				className: "dc-model",
				title: "model this Discord target talks to",
				children: ["🧠 ", e.model_key]
			}),
			/* @__PURE__ */ d("span", {
				className: "dc-arrow",
				children: "→"
			}),
			/* @__PURE__ */ d("span", {
				className: "dc-target",
				title: "Discord channel and/or user",
				children: ra(e)
			}),
			e.label && /* @__PURE__ */ d("span", {
				className: "dc-label",
				children: e.label
			}),
			/* @__PURE__ */ d("input", {
				className: "dc-ping-input",
				placeholder: "push a message to this target…",
				value: r,
				onChange: (e) => i(e.target.value),
				onKeyDown: (e) => {
					e.key === "Enter" && s();
				}
			}),
			/* @__PURE__ */ d("button", {
				className: "dc-ping-btn",
				onClick: s,
				disabled: a || !r.trim(),
				title: "Queue an outbound message; the bot delivers it into Discord",
				children: a ? "…" : "send"
			}),
			/* @__PURE__ */ d("button", {
				className: "dc-remove",
				title: "Remove binding",
				onClick: () => t(e),
				children: "✕"
			})
		]
	});
}
function aa({ embedded: e = !1, models: t = [] }) {
	let [n, r] = l([]), [a, s] = l(null), [c, u] = l(!1), [p, m] = l(""), [h, g] = l(""), [_, v] = l([]), [y, b] = l(!1), [x, S] = l(""), [C, w] = l([]), [T, E] = l(!1), [D, O] = l(""), [k, A] = l(!1), j = i(() => {
		K("/api/discord/bindings").then((e) => {
			r(Array.isArray(e?.bindings) ? e.bindings : []), s(null);
		}).catch((e) => s(e.message)), K("/api/discord/channels").then((e) => v(Array.isArray(e?.channels) ? e.channels : [])).catch(() => {}), K("/api/discord/users").then((e) => w(Array.isArray(e?.users) ? e.users : [])).catch(() => {});
	}, []);
	o(() => {
		j();
		let e = setInterval(j, 1e4);
		return () => clearInterval(e);
	}, [j]);
	let M = i(async () => {
		if (!p) {
			alert("Pick a model to bind.");
			return;
		}
		if (!h.trim() && !x.trim()) {
			alert("Enter a channel ID and/or a user ID.");
			return;
		}
		A(!0);
		try {
			await K("/api/discord/bindings", {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({
					model_key: p,
					channel_id: h.trim() || null,
					user_id: x.trim() || null,
					label: D.trim() || null
				})
			}), g(""), b(!1), S(""), E(!1), O(""), j();
		} catch (e) {
			alert(`Bind failed: ${e.message}`);
		} finally {
			A(!1);
		}
	}, [
		p,
		h,
		x,
		D,
		j
	]), N = i(async (e) => {
		if (confirm(`Remove the binding ${e.model_key} → ${ra(e)}?`)) try {
			await K(`/api/discord/bindings/${encodeURIComponent(e.id)}`, { method: "DELETE" }), j();
		} catch (e) {
			alert(`Remove failed: ${e.message}`);
		}
	}, [j]), P = i(async (e, t) => {
		try {
			await K("/api/discord/outbox", {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({
					binding_id: e.id,
					content: t
				})
			});
		} catch (e) {
			alert(`Send failed: ${e.message}`);
		}
	}, []), F = t.map(na).filter(Boolean);
	return /* @__PURE__ */ f("div", {
		className: "discord-panel",
		children: [/* @__PURE__ */ f("div", {
			className: `dc-bar${e ? " dc-bar-static" : ""}`,
			onClick: e ? void 0 : () => u((e) => !e),
			children: [
				/* @__PURE__ */ d("span", {
					className: "dc-title",
					children: "💬 Discord — model ↔ channel/user bindings"
				}),
				/* @__PURE__ */ f("span", {
					className: "dc-count",
					children: [
						n.length,
						" binding",
						n.length === 1 ? "" : "s"
					]
				}),
				a && /* @__PURE__ */ f("span", {
					className: "dc-err",
					title: a,
					children: ["registry error", /* @__PURE__ */ d(ar, { doc: "registry-error" })]
				}),
				!e && /* @__PURE__ */ d("span", {
					className: "dc-toggle",
					children: c ? "▾" : "▸"
				})
			]
		}), (e || c) && /* @__PURE__ */ f("div", {
			className: "dc-body",
			children: [
				/* @__PURE__ */ f("div", {
					className: "dc-howto",
					children: [
						"Bind a model to a Discord ",
						/* @__PURE__ */ d("b", { children: "channel" }),
						" and/or ",
						/* @__PURE__ */ d("b", { children: "user" }),
						". The hugpy bot (",
						/* @__PURE__ */ d("code", { children: "hugpy bot" }),
						") routes ",
						/* @__PURE__ */ d("code", { children: "@mentions" }),
						" in that channel / from that user to the bound model, and delivers anything you push here into Discord. Copy IDs from Discord with Developer Mode on (right-click → Copy ID)."
					]
				}),
				/* @__PURE__ */ f("div", {
					className: "dc-add",
					children: [
						/* @__PURE__ */ f("select", {
							className: "dc-sel",
							value: p,
							onChange: (e) => m(e.target.value),
							children: [/* @__PURE__ */ d("option", {
								value: "",
								children: "model…"
							}), F.map((e) => /* @__PURE__ */ d("option", {
								value: e,
								children: e
							}, e))]
						}),
						/* @__PURE__ */ f("select", {
							className: "dc-sel",
							value: y ? "__manual__" : h,
							title: "Channels the hugpy bot can see (or enter an ID manually)",
							onChange: (e) => {
								let t = e.target.value;
								t === "__manual__" ? (b(!0), g("")) : (b(!1), g(t));
							},
							children: [
								/* @__PURE__ */ d("option", {
									value: "",
									children: "channel…"
								}),
								_.map((e) => /* @__PURE__ */ f("option", {
									value: e.id,
									children: [
										"#",
										e.name,
										e.guild ? ` (${e.guild})` : ""
									]
								}, e.id)),
								/* @__PURE__ */ d("option", {
									value: "__manual__",
									children: "✏️ enter ID manually…"
								})
							]
						}),
						y && /* @__PURE__ */ d("input", {
							className: "dc-in",
							placeholder: "channel ID",
							value: h,
							autoFocus: !0,
							onChange: (e) => g(e.target.value)
						}),
						/* @__PURE__ */ f("select", {
							className: "dc-sel",
							value: T ? "__manual__" : x,
							title: "Members the hugpy bot can see (or enter a user ID manually)",
							onChange: (e) => {
								let t = e.target.value;
								t === "__manual__" ? (E(!0), S("")) : (E(!1), S(t));
							},
							children: [
								/* @__PURE__ */ d("option", {
									value: "",
									children: "user (optional)…"
								}),
								C.map((e) => /* @__PURE__ */ f("option", {
									value: e.id,
									children: [
										"@",
										e.name,
										e.guild ? ` (${e.guild})` : ""
									]
								}, e.id)),
								/* @__PURE__ */ d("option", {
									value: "__manual__",
									children: "✏️ enter ID manually…"
								})
							]
						}),
						T && /* @__PURE__ */ d("input", {
							className: "dc-in",
							placeholder: "user ID",
							value: x,
							autoFocus: !0,
							onChange: (e) => S(e.target.value)
						}),
						/* @__PURE__ */ d("input", {
							className: "dc-in dc-in-label",
							placeholder: "label (optional)",
							value: D,
							onChange: (e) => O(e.target.value)
						}),
						/* @__PURE__ */ d("button", {
							className: "dc-add-btn",
							onClick: M,
							disabled: k,
							children: k ? "…" : "+ Bind"
						})
					]
				}),
				n.length === 0 && /* @__PURE__ */ d("div", {
					className: "dc-empty",
					children: "No models are bound to Discord yet."
				}),
				n.map((e) => /* @__PURE__ */ d(ia, {
					binding: e,
					onRemove: N,
					onPing: P
				}, e.id))
			]
		})]
	});
}
//#endregion
//#region src/components/BridgePanel/BridgePanel.jsx
var oa = (e) => e.model_key ?? e.key, sa = [
	["auto", "auto — send replies immediately"],
	["defer", "defer — hold every reply for my approval"],
	["directive", "directive — let the model decide per message"]
], ca = [
	["defer", "user-strict — approve every keeper reply"],
	["directive", "keeper-choice — keeper decides (DEFER: escalates)"],
	["auto", "keeper auto — send every reply"]
];
function la({ m: e, onApprove: t, onReject: n }) {
	let r = e.direction === "in", i = e.status === "pending", a = e.status === "rejected", o = r ? `← ${e.author || "discord"}` : e.source === "model" ? "→ 🤖 model" : e.source === "keeper" ? "→ ⛨ keeper" : "→ 🧑 you";
	return /* @__PURE__ */ f("div", {
		className: `br-msg ${r ? "br-in" : "br-out"}${i ? " br-pending" : ""}${a ? " br-rejected" : ""}`,
		children: [
			/* @__PURE__ */ d("span", {
				className: "br-who",
				children: o
			}),
			/* @__PURE__ */ d("span", {
				className: "br-text",
				children: e.content
			}),
			i && /* @__PURE__ */ f("span", {
				className: "br-acts",
				children: [/* @__PURE__ */ d("button", {
					className: "br-ok",
					title: "Approve & send the model’s draft as-is",
					onClick: () => t(e.id),
					children: "✓ send"
				}), /* @__PURE__ */ d("button", {
					className: "br-no",
					title: "Reject",
					onClick: () => n(e.id),
					children: "✕"
				})]
			})
		]
	});
}
function ua({ embedded: e = !1, models: t = [] }) {
	let [n, r] = l(!1), [a, s] = l([]), [u, p] = l([]), [m, h] = l(null), [g, _] = l(""), [v, y] = l(""), [b, x] = l(!1), [S, C] = l(""), [w, T] = l("defer"), [E, D] = l("model"), [O, k] = l(""), [A, j] = l("open"), [M, N] = l(!1), [P, F] = l(null), [I, L] = l([]), R = c(null), z = i(() => {
		K("/api/discord/bridges").then((e) => {
			s(Array.isArray(e?.bridges) ? e.bridges : []), h(null);
		}).catch((e) => h(e.message)), K("/api/discord/channels").then((e) => p(Array.isArray(e?.channels) ? e.channels : [])).catch(() => {});
	}, []), B = i((e) => {
		e && K(`/api/discord/bridges/${encodeURIComponent(e)}/messages`).then((t) => {
			R.current === e && L(Array.isArray(t?.messages) ? t.messages : []);
		}).catch(() => {});
	}, []);
	o(() => {
		z();
		let e = setInterval(z, 5e3);
		return () => clearInterval(e);
	}, [z]), o(() => {
		if (R.current = P, !P) {
			L([]);
			return;
		}
		B(P);
		let e = setInterval(() => B(P), 3e3);
		return () => clearInterval(e);
	}, [P, B]);
	let V = (e) => {
		let t = u.find((t) => t.id === e);
		return t ? `#${t.name}${t.guild ? ` (${t.guild})` : ""}` : `#${e}`;
	}, ee = i(async () => {
		if (E === "model" && !g) {
			alert("Pick a model.");
			return;
		}
		if (!v.trim()) {
			alert("Pick or enter a channel.");
			return;
		}
		N(!0);
		try {
			let e = E === "model" && w === "auto" ? A : "open";
			await K("/api/discord/bridges", {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({
					channel_id: v.trim(),
					model_key: E === "model" ? g : g || null,
					directive: S.trim() || null,
					defer_mode: w,
					brain: E,
					keeper_target: O.trim() || null,
					log_mode: e
				})
			}), C(""), y(""), x(!1), k(""), j("open"), z();
		} catch (e) {
			alert(`Bridge failed: ${e.message}`);
		} finally {
			N(!1);
		}
	}, [
		E,
		g,
		v,
		S,
		w,
		O,
		A,
		z
	]), te = i(async (e) => {
		if (confirm(`Remove the bridge ${e.model_key} ↔ ${V(e.channel_id)}?`)) try {
			await K(`/api/discord/bridges/${encodeURIComponent(e.id)}`, { method: "DELETE" }), P === e.id && F(null), z();
		} catch (e) {
			alert(`Remove failed: ${e.message}`);
		}
	}, [
		P,
		z,
		u
	]), H = i(async (e, t) => {
		try {
			await K(`/api/discord/bridges/${encodeURIComponent(P)}/approve`, {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({
					message_id: e,
					content: t ?? null
				})
			}), B(P);
		} catch (e) {
			alert(`Approve failed: ${e.message}`);
		}
	}, [P, B]), ne = i(async (e) => {
		try {
			await K(`/api/discord/bridges/${encodeURIComponent(P)}/reject`, {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({ message_id: e })
			}), B(P);
		} catch (e) {
			alert(`Reject failed: ${e.message}`);
		}
	}, [P, B]), re = i(async () => {
		if (P && confirm("Clear this bridge’s transcript here?\n\nThis wipes the console-side history only — it does NOT delete anything from the Discord channel.")) try {
			await K(`/api/discord/bridges/${encodeURIComponent(P)}/messages`, { method: "DELETE" }), L([]), B(P);
		} catch (e) {
			alert(`Clear failed: ${e.message}`);
		}
	}, [P, B]), U = t.map(oa).filter(Boolean), W = (e) => e.id === P ? I.filter((e) => e.status === "pending").length : 0, G = E === "keeper" ? ca : sa;
	return /* @__PURE__ */ f("div", {
		className: "bridge-panel",
		children: [/* @__PURE__ */ f("div", {
			className: `br-bar${e ? " br-bar-static" : ""}`,
			onClick: e ? void 0 : () => r((e) => !e),
			children: [
				/* @__PURE__ */ d("span", {
					className: "br-title",
					children: "🔗 Console ↔ Discord bridges"
				}),
				/* @__PURE__ */ d("span", {
					className: "br-count",
					children: a.length
				}),
				m && /* @__PURE__ */ d("span", {
					className: "br-err",
					title: m,
					children: "error"
				}),
				!e && /* @__PURE__ */ d("span", {
					className: "br-toggle",
					children: n ? "▾" : "▸"
				})
			]
		}), (e || n) && /* @__PURE__ */ f("div", {
			className: "br-body",
			children: [
				/* @__PURE__ */ f("div", {
					className: "br-howto",
					children: [
						"Allocate a ",
						/* @__PURE__ */ d("b", { children: "model" }),
						" or a ",
						/* @__PURE__ */ d("b", { children: "keeper" }),
						" to a Discord channel and supervise it from here: inbound messages generate a reply per the ",
						/* @__PURE__ */ d("b", { children: "directive" }),
						"; the mode decides whether it sends automatically, waits for your approval, or lets the brain choose. A",
						/* @__PURE__ */ d("b", { children: " keeper" }),
						" bridge is driven by an attached keeper process (",
						/* @__PURE__ */ d("code", { children: "hugpy keeper --bridge <id>" }),
						") — ",
						/* @__PURE__ */ d("b", { children: "user-strict" }),
						" holds every keeper reply for your approval below; ",
						/* @__PURE__ */ d("b", { children: "keeper-choice" }),
						" lets it decide (",
						/* @__PURE__ */ d("code", { children: "DEFER:" }),
						" escalates)."
					]
				}),
				/* @__PURE__ */ f("div", {
					className: "br-add",
					children: [
						/* @__PURE__ */ f("select", {
							className: "br-sel",
							value: E,
							onChange: (e) => {
								D(e.target.value), T("defer");
							},
							title: "brain",
							children: [/* @__PURE__ */ d("option", {
								value: "model",
								children: "🧠 model"
							}), /* @__PURE__ */ d("option", {
								value: "keeper",
								children: "⛨ keeper"
							})]
						}),
						/* @__PURE__ */ f("select", {
							className: "br-sel",
							value: g,
							onChange: (e) => _(e.target.value),
							children: [/* @__PURE__ */ d("option", {
								value: "",
								children: E === "keeper" ? "model (optional)…" : "model…"
							}), U.map((e) => /* @__PURE__ */ d("option", {
								value: e,
								children: e
							}, e))]
						}),
						/* @__PURE__ */ f("select", {
							className: "br-sel",
							value: b ? "__manual__" : v,
							onChange: (e) => {
								let t = e.target.value;
								t === "__manual__" ? (x(!0), y("")) : (x(!1), y(t));
							},
							children: [
								/* @__PURE__ */ d("option", {
									value: "",
									children: "channel…"
								}),
								u.map((e) => /* @__PURE__ */ f("option", {
									value: e.id,
									children: [
										"#",
										e.name,
										e.guild ? ` (${e.guild})` : ""
									]
								}, e.id)),
								/* @__PURE__ */ d("option", {
									value: "__manual__",
									children: "✏️ enter ID…"
								})
							]
						}),
						b && /* @__PURE__ */ d("input", {
							className: "br-in",
							placeholder: "channel ID",
							value: v,
							onChange: (e) => y(e.target.value)
						}),
						E === "keeper" && /* @__PURE__ */ d("input", {
							className: "br-in",
							placeholder: "keeper target (e.g. host or lxc name)",
							value: O,
							onChange: (e) => k(e.target.value)
						}),
						/* @__PURE__ */ d("select", {
							className: "br-sel",
							value: w,
							onChange: (e) => T(e.target.value),
							children: G.map(([e, t]) => /* @__PURE__ */ d("option", {
								value: e,
								children: t
							}, e))
						}),
						/* @__PURE__ */ f("select", {
							className: "br-sel",
							value: E === "model" && w === "auto" ? A : "open",
							onChange: (e) => j(e.target.value),
							disabled: !(E === "model" && w === "auto"),
							title: "Transcript retention. 'no logs' keeps nothing (ephemeral) — allowed only for an auto model bridge; keeper / defer / session bridges must retain their transcript to work.",
							children: [/* @__PURE__ */ d("option", {
								value: "open",
								children: "📝 open logs"
							}), /* @__PURE__ */ d("option", {
								value: "none",
								children: "🚫 no logs"
							})]
						}),
						/* @__PURE__ */ d("button", {
							className: "br-add-btn",
							onClick: ee,
							disabled: M,
							children: M ? "…" : "+ Bridge"
						})
					]
				}),
				/* @__PURE__ */ d("textarea", {
					className: "br-directive",
					placeholder: "directive — what to focus on, and when to defer to you…",
					value: S,
					onChange: (e) => C(e.target.value),
					rows: 2
				}),
				a.length === 0 && /* @__PURE__ */ d("div", {
					className: "br-empty",
					children: "No bridges yet."
				}),
				a.map((e) => /* @__PURE__ */ f("div", {
					className: "br-row-wrap",
					children: [
						/* @__PURE__ */ f("div", {
							className: "br-row",
							children: [
								e.brain === "keeper" ? /* @__PURE__ */ f("span", {
									className: "br-model",
									title: `keeper${e.keeper_target ? `: ${e.keeper_target}` : ""}`,
									children: ["⛨ ", e.keeper_target || "keeper"]
								}) : /* @__PURE__ */ f("span", {
									className: "br-model",
									title: "model",
									children: ["🧠 ", e.model_key || "—"]
								}),
								/* @__PURE__ */ d("span", {
									className: "br-arrow",
									children: "↔"
								}),
								/* @__PURE__ */ d("span", {
									className: "br-chan",
									title: "Discord channel",
									children: V(e.channel_id)
								}),
								/* @__PURE__ */ d("span", {
									className: `br-mode br-mode-${e.defer_mode}`,
									children: e.brain === "keeper" ? e.defer_mode === "defer" ? "user-strict" : e.defer_mode === "directive" ? "keeper-choice" : "keeper-auto" : e.defer_mode
								}),
								e.log_mode === "none" && /* @__PURE__ */ d("span", {
									className: "br-nolog",
									title: "Ephemeral — no transcript retained",
									children: "no-logs"
								}),
								W(e) > 0 && /* @__PURE__ */ f("span", {
									className: "br-pending-badge",
									children: [W(e), " pending"]
								}),
								/* @__PURE__ */ d("button", {
									className: "br-open",
									onClick: () => F(P === e.id ? null : e.id),
									children: P === e.id ? "hide" : "open"
								}),
								/* @__PURE__ */ d("button", {
									className: "br-remove",
									title: "Remove bridge",
									onClick: () => te(e),
									children: "✕"
								})
							]
						}),
						e.directive && /* @__PURE__ */ f("div", {
							className: "br-dir-show",
							title: "directive",
							children: ["▸ ", e.directive]
						}),
						P === e.id && /* @__PURE__ */ f("div", {
							className: "br-transcript",
							children: [
								I.length === 0 && /* @__PURE__ */ d("div", {
									className: "br-empty",
									children: "No messages yet."
								}),
								I.map((e) => /* @__PURE__ */ d(la, {
									m: e,
									onApprove: H,
									onReject: ne
								}, e.id)),
								/* @__PURE__ */ f("div", {
									className: "br-transcript-foot",
									children: [/* @__PURE__ */ d("span", {
										className: "br-readonly",
										title: "This console cannot message the channel — read-only transcript with approve/reject on model candidates.",
										children: "read-only"
									}), /* @__PURE__ */ d("button", {
										className: "br-clear",
										title: "Clear this transcript (console-side only; does not touch Discord)",
										onClick: re,
										disabled: I.length === 0,
										children: "🗑 clear"
									})]
								})
							]
						})
					]
				}, e.id))
			]
		})]
	});
}
//#endregion
//#region src/components/SessionsPanel/SessionsPanel.jsx
var da = (e) => `${window.location.origin}/api/discord/session/${e}`;
function fa(e, t, n) {
	let r = [
		`You can communicate with Discord channel #${e} via:`,
		`  ${t}`,
		"POST <endpoint>/send with JSON {\"content\":\"...\"} to post (≤1900 chars, delivered ≤8s).",
		"GET  <endpoint>/messages?since=<ts of last seen message> to read replies (poll ~30s).",
		"GET  <endpoint> for a usage refresher."
	];
	return n.trim() && r.push("", `When/how to use it: ${n.trim()}`), r.join("\n");
}
function pa(e) {
	if (navigator.clipboard?.writeText) return navigator.clipboard.writeText(e);
	let t = document.createElement("textarea");
	return t.value = e, document.body.appendChild(t), t.select(), document.execCommand("copy"), document.body.removeChild(t), Promise.resolve();
}
var ma = (e) => e ? (/* @__PURE__ */ new Date(e * 1e3)).toLocaleString() : "—", ha = (e) => e.revoked ? "revoked" : e.expires_at && e.expires_at * 1e3 < Date.now() ? "expired" : "live";
function ga({ embedded: e = !1 }) {
	let [t, n] = l([]), [r, a] = l([]), [s, c] = l(null), [u, p] = l(!1), [m, h] = l(""), [g, _] = l(""), [v, y] = l(""), [b, x] = l(""), [S, C] = l(!1), [w, T] = l(null), [E, D] = l(""), O = i(() => {
		K("/api/discord/sessions").then((e) => {
			n(Array.isArray(e?.sessions) ? e.sessions : []), c(null);
		}).catch((e) => c(e.message)), K("/api/discord/channels").then((e) => a(Array.isArray(e?.channels) ? e.channels : [])).catch(() => {});
	}, []);
	o(() => {
		O();
		let e = setInterval(O, 15e3);
		return () => clearInterval(e);
	}, [O]);
	let k = i((e) => r.find((t) => String(t.id) === String(e))?.name || String(e), [r]), A = i(async () => {
		if (!m) {
			alert("Pick a channel to mint a session for.");
			return;
		}
		C(!0);
		try {
			let e = {
				channel_id: m,
				label: g.trim()
			};
			v.trim() && (e.ttl_hours = parseFloat(v));
			let t = await K("/api/discord/sessions", {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify(e)
			}), n = k(m), r = da(t.token);
			T({
				token: t.token,
				id: t.session.id,
				endpoint: r,
				channelName: n,
				block: fa(n, r, b)
			}), h(""), _(""), y(""), x(""), O();
		} catch (e) {
			alert(`Mint failed: ${e.message}`);
		} finally {
			C(!1);
		}
	}, [
		m,
		g,
		v,
		b,
		k,
		O
	]), j = i(async (e) => {
		if (confirm(`Revoke session ${e.id}? Any agent holding its token loses access immediately.`)) try {
			await K(`/api/discord/sessions/${encodeURIComponent(e.id)}`, { method: "DELETE" }), O();
		} catch (e) {
			alert(`Revoke failed: ${e.message}`);
		}
	}, [O]), M = i(async (e, t) => {
		let n = t === "live" ? `Remove LIVE session ${e.id}? Its token stops working immediately.` : `Remove session ${e.id} from the list?`;
		if (confirm(n)) try {
			await K(`/api/discord/sessions/${encodeURIComponent(e.id)}?purge=1`, { method: "DELETE" }), O();
		} catch (e) {
			alert(`Remove failed: ${e.message}`);
		}
	}, [O]), N = i(async () => {
		if (confirm("Remove ALL revoked/expired sessions from the list? Live sessions are untouched.")) try {
			let e = await K("/api/discord/sessions/prune", { method: "POST" });
			O(), e?.pruned != null && alert(`${e.pruned} dead session${e.pruned === 1 ? "" : "s"} removed.`);
		} catch (e) {
			alert(`Prune failed: ${e.message}`);
		}
	}, [O]), P = i(async (e, t) => {
		await pa(t), D(e), setTimeout(() => D(""), 1500);
	}, []), F = {};
	for (let e of r) (F[e.guild || "—"] ||= []).push(e);
	let I = t.filter((e) => ha(e) === "live").length;
	return /* @__PURE__ */ f("div", {
		className: "sessions-panel",
		children: [/* @__PURE__ */ f("div", {
			className: `sc-bar${e ? " sc-bar-static" : ""}`,
			onClick: e ? void 0 : () => p((e) => !e),
			children: [
				/* @__PURE__ */ d("span", {
					className: "sc-title",
					children: "🎟️ Comms sessions — scoped channel tokens for agents"
				}),
				/* @__PURE__ */ f("span", {
					className: "sc-count",
					children: [I, " live"]
				}),
				s && /* @__PURE__ */ f("span", {
					className: "sc-err",
					title: s,
					children: ["registry error", /* @__PURE__ */ d(ar, { doc: "registry-error" })]
				}),
				!e && /* @__PURE__ */ d("span", {
					className: "sc-toggle",
					children: u ? "▾" : "▸"
				})
			]
		}), (e || u) && /* @__PURE__ */ f("div", {
			className: "sc-body",
			children: [
				/* @__PURE__ */ f("div", {
					className: "sc-howto",
					children: [
						"Mint a ",
						/* @__PURE__ */ d("b", { children: "scoped bearer token" }),
						" bound to one channel and hand it to a terminal or agent session — it can read that channel and post to it, nothing else on the API. Revocable and optionally time-limited; the server stores only the token's hash."
					]
				}),
				/* @__PURE__ */ f("div", {
					className: "sc-add",
					children: [
						/* @__PURE__ */ f("select", {
							className: "sc-sel",
							value: m,
							onChange: (e) => h(e.target.value),
							children: [/* @__PURE__ */ d("option", {
								value: "",
								children: "channel…"
							}), Object.entries(F).map(([e, t]) => /* @__PURE__ */ d("optgroup", {
								label: e,
								children: t.map((e) => /* @__PURE__ */ f("option", {
									value: e.id,
									children: ["#", e.name]
								}, e.id))
							}, e))]
						}),
						/* @__PURE__ */ d("input", {
							className: "sc-in",
							placeholder: "label (optional)",
							value: g,
							onChange: (e) => _(e.target.value)
						}),
						/* @__PURE__ */ d("input", {
							className: "sc-in sc-in-ttl",
							type: "number",
							placeholder: "ttl hours (blank = none)",
							value: v,
							onChange: (e) => y(e.target.value)
						}),
						/* @__PURE__ */ d("button", {
							className: "sc-add-btn",
							onClick: A,
							disabled: S,
							children: S ? "…" : "+ Mint"
						})
					]
				}),
				/* @__PURE__ */ d("input", {
					className: "sc-in sc-in-instr",
					placeholder: "when/how it should be used — folded into the paste-block (e.g. \"post a summary after each step; check replies before destructive actions\")",
					value: b,
					onChange: (e) => x(e.target.value)
				}),
				w && /* @__PURE__ */ f("div", {
					className: "sc-minted",
					children: [
						/* @__PURE__ */ f("div", {
							className: "sc-minted-head",
							children: [
								"session for #",
								w.channelName,
								" — token shown ",
								/* @__PURE__ */ d("b", { children: "once" }),
								" (id ",
								w.id,
								")"
							]
						}),
						/* @__PURE__ */ d("textarea", {
							className: "sc-block",
							readOnly: !0,
							rows: w.block.split("\n").length + 1,
							value: w.block,
							onFocus: (e) => e.target.select()
						}),
						/* @__PURE__ */ f("div", {
							className: "sc-minted-actions",
							children: [
								/* @__PURE__ */ d("button", {
									className: "sc-add-btn",
									onClick: () => P("block", w.block),
									children: E === "block" ? "copied ✓" : "copy paste-block"
								}),
								/* @__PURE__ */ d("button", {
									className: "sc-add-btn",
									onClick: () => P("ep", w.endpoint),
									children: E === "ep" ? "copied ✓" : "copy endpoint"
								}),
								/* @__PURE__ */ d("button", {
									className: "sc-remove",
									onClick: () => T(null),
									children: "dismiss"
								})
							]
						})
					]
				}),
				t.length === 0 && /* @__PURE__ */ d("div", {
					className: "sc-empty",
					children: "No sessions yet."
				}),
				t.length > 0 && /* @__PURE__ */ f("table", {
					className: "sc-table",
					children: [/* @__PURE__ */ d("thead", { children: /* @__PURE__ */ f("tr", { children: [
						/* @__PURE__ */ d("th", { children: "state" }),
						/* @__PURE__ */ d("th", { children: "channel" }),
						/* @__PURE__ */ d("th", { children: "label" }),
						/* @__PURE__ */ d("th", { children: "author" }),
						/* @__PURE__ */ d("th", { children: "created" }),
						/* @__PURE__ */ d("th", { children: "last used" }),
						/* @__PURE__ */ d("th", { children: "expires" }),
						/* @__PURE__ */ d("th", {})
					] }) }), /* @__PURE__ */ d("tbody", { children: t.map((e) => {
						let t = ha(e);
						return /* @__PURE__ */ f("tr", {
							className: `sc-row sc-${t}`,
							children: [
								/* @__PURE__ */ d("td", { children: /* @__PURE__ */ d("span", {
									className: `sc-pill sc-pill-${t}`,
									children: t
								}) }),
								/* @__PURE__ */ f("td", { children: ["#", k(e.channel_id)] }),
								/* @__PURE__ */ d("td", { children: e.label || "—" }),
								/* @__PURE__ */ d("td", { children: e.author || "—" }),
								/* @__PURE__ */ d("td", { children: ma(e.created_at) }),
								/* @__PURE__ */ d("td", { children: ma(e.last_used) }),
								/* @__PURE__ */ d("td", { children: e.expires_at ? ma(e.expires_at) : "never" }),
								/* @__PURE__ */ f("td", { children: [t === "live" && /* @__PURE__ */ d("button", {
									className: "sc-remove",
									onClick: () => j(e),
									children: "revoke"
								}), /* @__PURE__ */ d("button", {
									className: "sc-remove",
									title: "Remove this row from the list",
									onClick: () => M(e, t),
									children: "remove"
								})] })
							]
						}, e.id);
					}) })]
				}),
				t.some((e) => ha(e) !== "live") && /* @__PURE__ */ d("div", {
					className: "sc-minted-actions",
					children: /* @__PURE__ */ f("button", {
						className: "sc-remove",
						onClick: N,
						children: [
							"clear ",
							t.filter((e) => ha(e) !== "live").length,
							" dead session",
							t.filter((e) => ha(e) !== "live").length === 1 ? "" : "s"
						]
					})
				})
			]
		})]
	});
}
//#endregion
//#region src/components/SettingsPanel/ModelDossier.jsx
var _a = 1024 ** 3, va = (e) => e == null ? "—" : `${(e / _a).toFixed(1)} GiB`, ya = (e) => e == null ? "—" : e >= 1e9 ? `${(e / 1e9).toFixed(1)}B` : e >= 1e6 ? `${(e / 1e6).toFixed(0)}M` : String(e), ba = (e) => e == null ? "—" : `${Number(e).toFixed(1)}`;
function xa({ tone: e = "", title: t, children: n }) {
	return /* @__PURE__ */ d("span", {
		className: `sp-dos-badge ${e}`,
		title: t,
		children: n
	});
}
function Sa({ title: e, note: t, children: n }) {
	return /* @__PURE__ */ f("div", {
		className: "sp-dos-sec",
		children: [/* @__PURE__ */ f("div", {
			className: "sp-dos-sec-head",
			children: [/* @__PURE__ */ d("span", {
				className: "sp-dos-sec-title",
				children: e
			}), t && /* @__PURE__ */ d("span", {
				className: "sp-md-dim",
				children: t
			})]
		}), n]
	});
}
function Ca({ row: e }) {
	let t = e.candidate_quality, n = e.incumbent_quality, r = (e) => `${Math.max(0, Math.min(100, Number(e ?? 0)))}%`;
	return e.beats_incumbent === "untested" ? /* @__PURE__ */ f("div", {
		className: "sp-dos-cmp",
		children: [/* @__PURE__ */ d("div", {
			className: "sp-dos-cmp-op",
			children: e.operation
		}), /* @__PURE__ */ f("div", {
			className: "sp-md-dim",
			children: ["untested — ", e.basis]
		})]
	}) : /* @__PURE__ */ f("div", {
		className: "sp-dos-cmp",
		children: [
			/* @__PURE__ */ f("div", {
				className: "sp-dos-cmp-op",
				children: [e.operation, /* @__PURE__ */ f(xa, {
					tone: e.beats_incumbent === "yes" ? "sp-dos-good" : "sp-dos-bad",
					children: [e.beats_incumbent === "yes" ? "beats incumbent" : "below incumbent", e.margin != null && ` ${e.margin > 0 ? "+" : ""}${e.margin}`]
				})]
			}),
			/* @__PURE__ */ f("div", {
				className: "sp-dos-bar-row",
				children: [
					/* @__PURE__ */ d("span", {
						className: "sp-dos-bar-lbl",
						children: "candidate"
					}),
					/* @__PURE__ */ d("div", {
						className: "sp-dos-bar",
						children: /* @__PURE__ */ d("i", { style: { width: r(t) } })
					}),
					/* @__PURE__ */ d("span", {
						className: "sp-dos-bar-num",
						children: ba(t)
					})
				]
			}),
			/* @__PURE__ */ f("div", {
				className: "sp-dos-bar-row",
				children: [
					/* @__PURE__ */ d("span", {
						className: "sp-dos-bar-lbl",
						children: e.incumbent || "incumbent"
					}),
					/* @__PURE__ */ d("div", {
						className: "sp-dos-bar sp-dos-bar-inc",
						children: /* @__PURE__ */ d("i", { style: { width: r(n) } })
					}),
					/* @__PURE__ */ d("span", {
						className: "sp-dos-bar-num",
						children: ba(n)
					})
				]
			}),
			/* @__PURE__ */ d("div", {
				className: "sp-md-dim sp-dos-basis",
				children: e.basis
			})
		]
	});
}
function wa({ samples: e }) {
	return e?.length ? /* @__PURE__ */ d("div", {
		className: "sp-dos-samples",
		children: e.map((e, t) => /* @__PURE__ */ f("div", {
			className: "sp-dos-sample",
			children: [
				/* @__PURE__ */ f("div", {
					className: "sp-dos-sample-head",
					children: [
						/* @__PURE__ */ d("strong", { children: e.operation }),
						/* @__PURE__ */ d("span", {
							className: "sp-md-dim",
							children: e.kind
						}),
						e.seconds != null && /* @__PURE__ */ f("span", {
							className: "sp-md-dim",
							children: [e.seconds, "s"]
						}),
						!e.ok && /* @__PURE__ */ d(xa, {
							tone: "sp-dos-bad",
							children: e.failure || "did not validate"
						})
					]
				}),
				e.snippet && /* @__PURE__ */ d("pre", {
					className: "sp-dos-snippet",
					children: e.snippet
				}),
				e.artifact_ref && /* @__PURE__ */ f("div", {
					className: "sp-md-dim sp-dos-artifact",
					children: ["artifact: ", /* @__PURE__ */ d("code", { children: e.artifact_ref })]
				})
			]
		}, t))
	}) : null;
}
function Ta({ criteria: e, hubId: t }) {
	let [n, r] = l(null), [a, s] = l(""), c = i(() => {
		s(""), K(`/api/llm/review/dossier?criteria=${encodeURIComponent(e)}&hub_id=${encodeURIComponent(t)}`).then(r).catch((e) => s(e.message));
	}, [e, t]);
	if (o(() => {
		c();
	}, [c]), a) return /* @__PURE__ */ f("div", {
		className: "sp-dos sp-empty",
		children: [
			"No dossier for this model yet (",
			a,
			")."
		]
	});
	if (!n) return /* @__PURE__ */ d("div", {
		className: "sp-dos sp-loading",
		children: "loading dossier…"
	});
	let { identity: p, specialization: m, weights: h, trust: g, research: _, community: v, trial: y, verdict: b, unavailable: x } = n;
	return /* @__PURE__ */ f("div", {
		className: "sp-dos",
		children: [
			b && /* @__PURE__ */ f(Sa, {
				title: `Verdict: ${b.verdict}`,
				note: b.judged_by ? `judged by ${b.judged_by}` : "filed from the measured numbers",
				children: [
					/* @__PURE__ */ d(xa, {
						tone: b.confidence === "evidence-backed" ? "sp-dos-good" : "sp-dos-warn",
						children: b.confidence
					}),
					/* @__PURE__ */ d("ul", {
						className: "sp-dos-reasons",
						children: (b.reasons || []).map((e, t) => /* @__PURE__ */ f("li", { children: [e, b.evidence_refs?.[t] && /* @__PURE__ */ d("code", {
							className: "sp-dos-ref",
							children: b.evidence_refs[t]
						})] }, t))
					}),
					b.blocked && /* @__PURE__ */ d("div", {
						className: "sp-note",
						children: b.blocked
					})
				]
			}),
			m && /* @__PURE__ */ f(Sa, {
				title: "Specialization",
				children: [
					/* @__PURE__ */ d("div", {
						className: "sp-dos-line",
						children: m.headline || "—"
					}),
					!!m.emphasis?.length && /* @__PURE__ */ d("div", {
						className: "sp-dos-weights",
						children: m.emphasis.slice(0, 6).map((e, t) => /* @__PURE__ */ f("span", {
							className: "sp-dos-weight",
							title: (e.evidence || []).join(" · "),
							children: [
								e.domain,
								/* @__PURE__ */ d("i", { style: { width: `${Math.round(e.weight * 100)}%` } }),
								/* @__PURE__ */ d("b", { children: e.weight })
							]
						}, t))
					}),
					!!m.finetune_focus?.length && /* @__PURE__ */ d("ul", {
						className: "sp-dos-quotes",
						children: m.finetune_focus.map((e, t) => /* @__PURE__ */ f("li", { children: [
							"“",
							e,
							"”"
						] }, t))
					}),
					p?.base_model && /* @__PURE__ */ f("div", {
						className: "sp-md-dim",
						children: [
							"lineage: ",
							p.relation || "derived",
							" of",
							" ",
							/* @__PURE__ */ d("a", {
								href: `https://huggingface.co/${p.base_model}`,
								target: "_blank",
								rel: "noreferrer",
								children: p.base_model
							})
						]
					})
				]
			}),
			h && /* @__PURE__ */ f(Sa, {
				title: "Weights",
				note: h.params_source ? `params from ${h.params_source}` : "",
				children: [
					/* @__PURE__ */ f("div", {
						className: "sp-dos-line",
						children: [
							/* @__PURE__ */ f(xa, { children: [ya(h.params), " params"] }),
							/* @__PURE__ */ d(xa, { children: h.architecture_family || h.architecture || "arch ?" }),
							/* @__PURE__ */ f(xa, { children: ["ctx ", h.context_length ?? "—"] }),
							g?.license && /* @__PURE__ */ d(xa, {
								tone: "sp-dos-warn",
								title: "licence as declared on the hub",
								children: g.license
							}),
							g?.gated ? /* @__PURE__ */ d(xa, {
								tone: "sp-dos-bad",
								children: "gated"
							}) : null
						]
					}),
					!!h.quants?.length && /* @__PURE__ */ f("table", {
						className: "sp-md-table",
						children: [/* @__PURE__ */ d("thead", { children: /* @__PURE__ */ f("tr", { children: [
							/* @__PURE__ */ d("th", { children: "quant" }),
							/* @__PURE__ */ d("th", { children: "bpw" }),
							/* @__PURE__ */ d("th", { children: "size" }),
							/* @__PURE__ */ d("th", { children: "est VRAM" }),
							/* @__PURE__ */ d("th", { children: "fits" })
						] }) }), /* @__PURE__ */ d("tbody", { children: h.quants.map((e, t) => /* @__PURE__ */ f("tr", {
							className: e.quant === h.best_quant ? "sp-dos-best" : "",
							children: [
								/* @__PURE__ */ d("td", { children: e.quant }),
								/* @__PURE__ */ d("td", {
									className: "sp-md-dim",
									children: e.bits_per_weight ?? "—"
								}),
								/* @__PURE__ */ d("td", { children: va(e.bytes) }),
								/* @__PURE__ */ d("td", { children: va(e.est_vram_bytes) }),
								/* @__PURE__ */ d("td", { children: e.fits_vram == null ? "—" : e.fits_vram ? "yes" : "no" })
							]
						}, t)) })]
					}),
					!!h.notes?.length && /* @__PURE__ */ d("div", {
						className: "sp-md-dim",
						children: h.notes.join(" · ")
					})
				]
			}),
			_ && /* @__PURE__ */ f(Sa, {
				title: "Research (outside the download source)",
				children: [
					_.research_notes && /* @__PURE__ */ f(u, { children: [/* @__PURE__ */ f(xa, {
						tone: "sp-dos-gen",
						children: ["model-generated · ", _.research_notes_model]
					}), /* @__PURE__ */ d("p", {
						className: "sp-dos-notes",
						children: _.research_notes
					})] }),
					!!_.card?.benchmark_claims?.length && /* @__PURE__ */ f("div", {
						className: "sp-dos-claims",
						children: [/* @__PURE__ */ d("span", {
							className: "sp-md-dim",
							children: "claimed by the model card (not measured here):"
						}), _.card.benchmark_claims.slice(0, 10).map((e, t) => /* @__PURE__ */ f(xa, { children: [
							e.benchmark,
							" ",
							e.value
						] }, t))]
					}),
					_.card?.limitations && /* @__PURE__ */ f("details", {
						className: "sp-dos-det",
						children: [/* @__PURE__ */ d("summary", { children: "limitations (from the card)" }), /* @__PURE__ */ d("pre", {
							className: "sp-dos-snippet",
							children: _.card.limitations
						})]
					}),
					_.card?.training_data && /* @__PURE__ */ f("details", {
						className: "sp-dos-det",
						children: [/* @__PURE__ */ d("summary", { children: "training data (from the card)" }), /* @__PURE__ */ d("pre", {
							className: "sp-dos-snippet",
							children: _.card.training_data
						})]
					}),
					!!_.papers?.length && /* @__PURE__ */ d("ul", {
						className: "sp-dos-links",
						children: _.papers.map((e, t) => /* @__PURE__ */ f("li", { children: [/* @__PURE__ */ d("a", {
							href: e.url,
							target: "_blank",
							rel: "noreferrer",
							children: e.title || `arXiv:${e.arxiv_id}`
						}), e.unavailable && /* @__PURE__ */ f("span", {
							className: "sp-md-dim",
							children: [" — ", e.unavailable]
						})] }, t))
					}),
					!!_.cited?.length && /* @__PURE__ */ f("div", {
						className: "sp-md-dim",
						children: ["sources: ", _.cited.join(" · ")]
					})
				]
			}),
			v && /* @__PURE__ */ f(Sa, {
				title: "Community",
				note: `heat ${v.heat}`,
				children: [
					v.model_generated && /* @__PURE__ */ f(xa, {
						tone: "sp-dos-gen",
						children: ["claims extracted by ", v.generated_by]
					}),
					!!v.claims?.length && /* @__PURE__ */ d("ul", {
						className: "sp-dos-quotes",
						children: v.claims.map((e, t) => /* @__PURE__ */ f("li", { children: [
							/* @__PURE__ */ d("b", { children: e.kind }),
							": ",
							e.text,
							e.quote && /* @__PURE__ */ f("span", {
								className: "sp-md-dim",
								children: [
									" — “",
									e.quote,
									"”"
								]
							}),
							e.url && /* @__PURE__ */ f(u, { children: [" ", /* @__PURE__ */ d("a", {
								href: e.url,
								target: "_blank",
								rel: "noreferrer",
								children: "link"
							})] })
						] }, t))
					}),
					!!v.mentions?.length && /* @__PURE__ */ d("ul", {
						className: "sp-dos-links",
						children: v.mentions.slice(0, 8).map((e, t) => /* @__PURE__ */ f("li", { children: [
							/* @__PURE__ */ d("span", {
								className: "sp-md-dim",
								children: e.source
							}),
							" ",
							/* @__PURE__ */ d("a", {
								href: e.url,
								target: "_blank",
								rel: "noreferrer",
								children: e.title || e.url
							})
						] }, t))
					}),
					!v.mentions?.length && /* @__PURE__ */ d("div", {
						className: "sp-empty",
						children: "Nobody has posted about this model on the sources we read."
					})
				]
			}),
			y && /* @__PURE__ */ f(Sa, {
				title: "Trial",
				note: `${y.depth}${y.backend ? ` · ${y.backend}` : ""}`,
				children: [
					y.blocked && /* @__PURE__ */ f("div", {
						className: "sp-note",
						children: ["trial blocked: ", y.blocked]
					}),
					y.scenario_version && /* @__PURE__ */ f("div", {
						className: "sp-md-dim",
						children: [
							"stationary brief ",
							y.scenario_version,
							" (",
							(y.scenario_digest || "").slice(0, 12),
							") — every model is asked the same thing, which is what makes these numbers comparable."
						]
					}),
					(y.comparisons || []).map((e, t) => /* @__PURE__ */ d(Ca, { row: e }, t)),
					/* @__PURE__ */ d(wa, { samples: y.samples })
				]
			}),
			!!x?.length && /* @__PURE__ */ d(Sa, {
				title: "Not available",
				children: /* @__PURE__ */ d("ul", {
					className: "sp-dos-quotes",
					children: x.map((e, t) => /* @__PURE__ */ d("li", {
						className: "sp-md-dim",
						children: e
					}, t))
				})
			})
		]
	});
}
//#endregion
//#region src/components/SettingsPanel/ModelRadar.jsx
function Ea({ name: e }) {
	let [t, n] = l(null), [r, i] = l("");
	if (o(() => {
		K(`/api/llm/review/radar?criteria=${encodeURIComponent(e)}`).then(n).catch((e) => i(e.message));
	}, [e]), r) return /* @__PURE__ */ f("div", {
		className: "sp-empty",
		children: ["radar unavailable: ", r]
	});
	if (!t) return /* @__PURE__ */ d("div", {
		className: "sp-loading",
		children: "loading radar…"
	});
	let a = t.hits || [];
	return /* @__PURE__ */ f("div", {
		className: "sp-dos-radar",
		children: [
			/* @__PURE__ */ d("div", {
				className: "sp-md-dim",
				children: t.detail || ""
			}),
			a.length === 0 && /* @__PURE__ */ f("div", {
				className: "sp-empty",
				children: [
					"Nothing on the radar. (It scans the cached mention pulls for models no card tracks; enable ",
					/* @__PURE__ */ d("code", { children: "radar" }),
					" on this card to run it.)"
				]
			}),
			a.length > 0 && /* @__PURE__ */ f("table", {
				className: "sp-md-table",
				children: [/* @__PURE__ */ d("thead", { children: /* @__PURE__ */ f("tr", { children: [
					/* @__PURE__ */ d("th", { children: "model" }),
					/* @__PURE__ */ d("th", { children: "heat" }),
					/* @__PURE__ */ d("th", { children: "why" })
				] }) }), /* @__PURE__ */ d("tbody", { children: a.map((e, t) => /* @__PURE__ */ f("tr", { children: [
					/* @__PURE__ */ d("td", {
						className: "sp-md-model",
						children: e.hub_id ? /* @__PURE__ */ d("a", {
							href: `https://huggingface.co/${e.hub_id}`,
							target: "_blank",
							rel: "noreferrer",
							children: e.hub_id
						}) : /* @__PURE__ */ d("span", {
							title: "named in posts but no hub repo matched",
							children: e.name
						})
					}),
					/* @__PURE__ */ d("td", { children: e.heat }),
					/* @__PURE__ */ f("td", {
						className: "sp-md-dim",
						children: [e.why, !!e.mentions?.length && /* @__PURE__ */ f(u, { children: [
							" ·",
							" ",
							e.mentions.slice(0, 3).map((e, t) => /* @__PURE__ */ d("a", {
								href: e.url,
								target: "_blank",
								rel: "noreferrer",
								className: "sp-dos-ref-link",
								children: e.source
							}, t))
						] })]
					})
				] }, t)) })]
			})
		]
	});
}
//#endregion
//#region src/components/SettingsPanel/ModelDiscovery.jsx
var Da = (e) => e ? (/* @__PURE__ */ new Date(e * 1e3)).toLocaleString() : "—";
function Oa({ name: e }) {
	let [t, r] = l(null), [i, a] = l(""), [s, c] = l(null);
	return o(() => {
		K(`/api/llm/review/results?criteria=${encodeURIComponent(e)}&best=1&limit=8`).then((e) => r(Array.isArray(e) ? e : [])).catch((e) => a(e.message));
	}, [e]), i ? /* @__PURE__ */ f("div", {
		className: "sp-empty",
		children: ["leaderboard unavailable: ", i]
	}) : t == null ? /* @__PURE__ */ d("div", {
		className: "sp-loading",
		children: "loading finds…"
	}) : t.length === 0 ? /* @__PURE__ */ d("div", {
		className: "sp-empty",
		children: "Nothing recorded for this criteria yet."
	}) : /* @__PURE__ */ f("table", {
		className: "sp-md-table",
		children: [/* @__PURE__ */ d("thead", { children: /* @__PURE__ */ f("tr", { children: [
			/* @__PURE__ */ d("th", { children: "model" }),
			/* @__PURE__ */ d("th", { children: "specialization" }),
			/* @__PURE__ */ d("th", { children: "fit" }),
			/* @__PURE__ */ d("th", { children: "vs incumbent" }),
			/* @__PURE__ */ d("th", { children: "verdict" }),
			/* @__PURE__ */ d("th", { children: "reviewed" })
		] }) }), /* @__PURE__ */ d("tbody", { children: t.map((t, r) => {
			let i = t.payload && t.payload.dossier || null, a = s === t.hub_id;
			return /* @__PURE__ */ f(n, { children: [/* @__PURE__ */ f("tr", {
				className: "sp-md-rowlink",
				onClick: () => c(a ? null : t.hub_id),
				children: [
					/* @__PURE__ */ f("td", {
						className: "sp-md-model",
						children: [/* @__PURE__ */ d("span", {
							className: "sp-dos-caret",
							children: a ? "▾" : "▸"
						}), /* @__PURE__ */ d("a", {
							href: `https://huggingface.co/${t.hub_id}`,
							target: "_blank",
							rel: "noreferrer",
							onClick: (e) => e.stopPropagation(),
							children: t.hub_id
						})]
					}),
					/* @__PURE__ */ d("td", {
						className: "sp-md-dim",
						children: i?.specialization || t.stage
					}),
					/* @__PURE__ */ f("td", { children: [i?.best_quant ? `${i.best_quant}` : "—", i?.est_vram_bytes ? /* @__PURE__ */ f("span", {
						className: "sp-md-dim",
						children: [
							" · ",
							(i.est_vram_bytes / 1024 ** 3).toFixed(1),
							" GiB"
						]
					}) : null] }),
					/* @__PURE__ */ d("td", { children: i ? /* @__PURE__ */ f("span", {
						className: i.beats_incumbent === "yes" ? "sp-dos-good-t" : i.beats_incumbent === "no" ? "sp-dos-bad-t" : "sp-md-dim",
						children: [i.beats_incumbent, i.margin != null && ` ${i.margin > 0 ? "+" : ""}${i.margin}`]
					}) : /* @__PURE__ */ d("span", {
						className: "sp-md-dim",
						children: "—"
					}) }),
					/* @__PURE__ */ d("td", { children: i?.verdict || t.verdict || "—" }),
					/* @__PURE__ */ d("td", {
						className: "sp-md-dim",
						children: Da(t.reviewed_at)
					})
				]
			}), a && /* @__PURE__ */ d("tr", { children: /* @__PURE__ */ d("td", {
				colSpan: 6,
				children: /* @__PURE__ */ d(Ta, {
					criteria: e,
					hubId: t.hub_id
				})
			}) })] }, `${t.hub_id}-${r}`);
		}) })]
	});
}
function ka({ crit: e, onSaved: t, onError: n }) {
	let [r, a] = l(e.query ?? ""), [o, s] = l(e.task ?? ""), [c, u] = l(e.max_downloads_per_run ?? 2), [p, m] = l(""), [h, g] = l(!1), [_, v] = l(!1), [y, b] = l(""), [x, S] = l(e.trial_depth ?? "load-test"), [C, w] = l(e.sample_count ?? 2), [T, E] = l((e.compare_against ?? []).join(", ")), [D, O] = l((e.required_specializations ?? []).join(", ")), [k, A] = l((e.licenses_allowed ?? []).join(", ")), [j, M] = l(e.external_research !== !1), [N, P] = l(e.community !== !1), [F, I] = l(!!e.radar), L = (e) => e.split(",").map((e) => e.trim()).filter(Boolean), R = e.enabled !== !1, z = r !== (e.query ?? "") || o !== (e.task ?? "") || Number(c) !== (e.max_downloads_per_run ?? 2) || x !== (e.trial_depth ?? "load-test") || Number(C) !== (e.sample_count ?? 2) || T !== (e.compare_against ?? []).join(", ") || D !== (e.required_specializations ?? []).join(", ") || k !== (e.licenses_allowed ?? []).join(", ") || j !== (e.external_research !== !1) || N !== (e.community !== !1) || F !== !!e.radar, B = i((i) => {
		let { running: a, running_since: s, error: l, ...u } = e, d = {
			...u,
			query: r,
			task: o || null,
			max_downloads_per_run: Number(c) || 0,
			trial_depth: x,
			sample_count: Number(C) || 0,
			compare_against: L(T),
			required_specializations: L(D),
			licenses_allowed: L(k),
			external_research: j,
			community: N,
			radar: F,
			...i
		};
		m("save"), K(`/api/llm/review/criteria/${encodeURIComponent(e.name)}`, {
			method: "PUT",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify(d)
		}).then(() => t()).catch((t) => n(`Saving "${e.name}" failed: ${t.message}`)).finally(() => m(""));
	}, [
		e,
		r,
		o,
		c,
		x,
		C,
		T,
		D,
		k,
		j,
		N,
		F,
		t,
		n
	]), V = i(() => {
		m("run"), b(""), K("/api/llm/review/run", {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({
				criteria: e.name,
				force: !0
			})
		}).then((e) => b(e.status === "already_running" ? "A run for this criteria is already in flight." : "Run started — results land in the tables below as stages finish.")).catch((t) => n(`Run "${e.name}" failed: ${t.message}`)).finally(() => {
			m(""), t();
		});
	}, [
		e.name,
		t,
		n
	]);
	return /* @__PURE__ */ f("div", {
		className: `sp-md-card${R ? "" : " sp-md-card-off"}`,
		children: [
			/* @__PURE__ */ f("div", {
				className: "sp-row",
				children: [/* @__PURE__ */ f("div", {
					className: "sp-val",
					children: [
						/* @__PURE__ */ d("span", {
							className: "sp-val-main",
							children: e.name
						}),
						e.running && /* @__PURE__ */ d("span", {
							className: "sp-scope sp-scope-fleet",
							children: "running…"
						}),
						!R && /* @__PURE__ */ d("span", {
							className: "sp-scope sp-scope-worker",
							children: "off"
						})
					]
				}), /* @__PURE__ */ f("div", {
					className: "sp-actions",
					children: [
						/* @__PURE__ */ d("button", {
							className: `sp-btn ${R ? "sp-btn-on" : ""}`,
							disabled: p !== "" || R,
							onClick: () => B({ enabled: !0 }),
							children: "On"
						}),
						/* @__PURE__ */ d("button", {
							className: `sp-btn ${R ? "" : "sp-btn-on"}`,
							disabled: p !== "" || !R,
							onClick: () => B({ enabled: !1 }),
							children: "Off"
						}),
						/* @__PURE__ */ d("button", {
							className: "sp-btn",
							disabled: p !== "" || e.running,
							title: "Start a full run right now (works even when the nightly switch is off)",
							onClick: V,
							children: p === "run" ? "…" : "Run now"
						}),
						/* @__PURE__ */ d("button", {
							className: "sp-btn sp-btn-quiet",
							onClick: () => v((e) => !e),
							children: _ ? "hide depth" : "how deep"
						}),
						/* @__PURE__ */ d("button", {
							className: "sp-btn sp-btn-quiet",
							onClick: () => g((e) => !e),
							children: h ? "hide finds" : "what it found"
						})
					]
				})]
			}),
			/* @__PURE__ */ f("div", {
				className: "sp-row sp-md-edit",
				children: [
					/* @__PURE__ */ f("label", {
						className: "sp-md-lbl",
						children: ["query", /* @__PURE__ */ d("input", {
							className: "sp-in",
							value: r,
							placeholder: "HF search text…",
							onChange: (e) => a(e.target.value)
						})]
					}),
					/* @__PURE__ */ f("label", {
						className: "sp-md-lbl",
						children: ["task", /* @__PURE__ */ d("input", {
							className: "sp-in sp-md-in-task",
							value: o ?? "",
							placeholder: "e.g. text-generation",
							onChange: (e) => s(e.target.value)
						})]
					}),
					/* @__PURE__ */ f("label", {
						className: "sp-md-lbl",
						children: ["downloads/run", /* @__PURE__ */ d("input", {
							className: "sp-in sp-md-in-cap",
							type: "number",
							min: "0",
							max: "10",
							value: c,
							onChange: (e) => u(e.target.value)
						})]
					}),
					z && /* @__PURE__ */ d("button", {
						className: "sp-btn sp-btn-on",
						disabled: p !== "",
						onClick: () => B({}),
						children: p === "save" ? "…" : "Save"
					})
				]
			}),
			_ && /* @__PURE__ */ f("div", {
				className: "sp-row sp-md-edit sp-dos-knobs",
				children: [
					/* @__PURE__ */ f("label", {
						className: "sp-md-lbl",
						children: ["trial depth", /* @__PURE__ */ f("select", {
							className: "sp-in sp-md-in-task",
							value: x,
							onChange: (e) => S(e.target.value),
							children: [
								/* @__PURE__ */ d("option", {
									value: "screen-only",
									children: "screen-only — metadata, no download"
								}),
								/* @__PURE__ */ d("option", {
									value: "load-test",
									children: "load-test — download + load on the GPU"
								}),
								/* @__PURE__ */ d("option", {
									value: "full-samples",
									children: "full-samples — + stationary battery"
								})
							]
						})]
					}),
					/* @__PURE__ */ f("label", {
						className: "sp-md-lbl",
						children: ["samples", /* @__PURE__ */ d("input", {
							className: "sp-in sp-md-in-cap",
							type: "number",
							min: "0",
							max: "8",
							value: C,
							onChange: (e) => w(e.target.value)
						})]
					}),
					/* @__PURE__ */ f("label", {
						className: "sp-md-lbl",
						children: ["compare against", /* @__PURE__ */ d("input", {
							className: "sp-in",
							value: T,
							placeholder: "plot.construct, screenplay.complete",
							title: "routing-matrix operation names — the incumbent is whatever the matrix routes there",
							onChange: (e) => E(e.target.value)
						})]
					}),
					/* @__PURE__ */ f("label", {
						className: "sp-md-lbl",
						children: ["required specializations", /* @__PURE__ */ d("input", {
							className: "sp-in",
							value: D,
							placeholder: "code, reasoning",
							title: "screened from tags and the repo name only — no extra fetch",
							onChange: (e) => O(e.target.value)
						})]
					}),
					/* @__PURE__ */ f("label", {
						className: "sp-md-lbl",
						children: ["licences allowed", /* @__PURE__ */ d("input", {
							className: "sp-in",
							value: k,
							placeholder: "apache, mit",
							title: "empty means any licence, which is the default",
							onChange: (e) => A(e.target.value)
						})]
					}),
					/* @__PURE__ */ f("label", {
						className: "sp-md-lbl sp-dos-check",
						children: [/* @__PURE__ */ d("input", {
							type: "checkbox",
							checked: j,
							onChange: (e) => M(e.target.checked)
						}), "external research (card + papers)"]
					}),
					/* @__PURE__ */ f("label", {
						className: "sp-md-lbl sp-dos-check",
						children: [/* @__PURE__ */ d("input", {
							type: "checkbox",
							checked: N,
							onChange: (e) => P(e.target.checked)
						}), "community scan (reddit / HN / discussions)"]
					}),
					/* @__PURE__ */ f("label", {
						className: "sp-md-lbl sp-dos-check",
						children: [/* @__PURE__ */ d("input", {
							type: "checkbox",
							checked: F,
							onChange: (e) => I(e.target.checked)
						}), "gem radar"]
					})
				]
			}),
			y && /* @__PURE__ */ d("div", {
				className: "sp-note",
				children: y
			}),
			h && /* @__PURE__ */ d(Oa, { name: e.name }),
			h && e.radar && /* @__PURE__ */ d(Ea, { name: e.name })
		]
	});
}
function Aa() {
	let [e, t] = l(null), [n, r] = l([]), [a, s] = l(null), c = i(() => {
		K("/api/llm/review/status").then((e) => {
			t(Array.isArray(e?.criteria) ? e.criteria : []), s(null);
		}).catch((e) => s(e.message)), K("/api/llm/review/runs?limit=8").then((e) => r(Array.isArray(e) ? e : [])).catch(() => {});
	}, []);
	return o(() => {
		c();
		let e = setInterval(c, 2e4);
		return () => clearInterval(e);
	}, [c]), /* @__PURE__ */ f("section", {
		className: "sp-sec",
		children: [
			/* @__PURE__ */ f("div", {
				className: "sp-sec-head",
				children: [/* @__PURE__ */ d("h3", {
					className: "sp-sec-title",
					children: "Model discovery (automated search)"
				}), /* @__PURE__ */ d("span", {
					className: "sp-scope sp-scope-fleet",
					children: "nightly ~03:20"
				})]
			}),
			/* @__PURE__ */ f("p", {
				className: "sp-desc",
				children: [
					"Each card is a saved ",
					/* @__PURE__ */ d("strong", { children: "search question" }),
					" the fleet asks Hugging Face every night: candidates are screened on metadata (VRAM fit, quants, context, trust, downloads), the best few are downloaded and ",
					/* @__PURE__ */ d("strong", { children: "load-tested on the GPU" }),
					", and an LLM judge files an adopt/trial/reject verdict. Off pauses a question without touching its settings; the nightly timer keeps ticking and simply skips it.",
					" ",
					/* @__PURE__ */ d("strong", { children: "How deep" }),
					" sets what each card does beyond the screen — the sample battery, the outside research, the community scan and the gem radar. Click a model in “what it found” for its full dossier: specialization, every quant’s VRAM, licence, what the card and the forums claim, the sample outputs, and the verdict with the evidence it cites."
				]
			}),
			a && /* @__PURE__ */ d("div", {
				className: "sp-err",
				children: a
			}),
			e == null && /* @__PURE__ */ d("div", {
				className: "sp-loading",
				children: "loading…"
			}),
			e != null && e.length === 0 && /* @__PURE__ */ d("div", {
				className: "sp-empty",
				children: "No criteria saved on this central. (They live in ~/.config/hugpy/review/ next to the nightly timer.)"
			}),
			(e ?? []).map((e) => e.error ? /* @__PURE__ */ f("div", {
				className: "sp-err",
				children: [
					"criteria ",
					e.name,
					": ",
					e.error
				]
			}, e.name) : /* @__PURE__ */ d(ka, {
				crit: e,
				onSaved: c,
				onError: s
			}, e.name)),
			n.length > 0 && /* @__PURE__ */ f(u, { children: [/* @__PURE__ */ d("p", {
				className: "sp-desc sp-desc-why",
				children: "Recent runs (all criteria, newest first):"
			}), /* @__PURE__ */ f("table", {
				className: "sp-md-table",
				children: [/* @__PURE__ */ d("thead", { children: /* @__PURE__ */ f("tr", { children: [
					/* @__PURE__ */ d("th", { children: "criteria" }),
					/* @__PURE__ */ d("th", { children: "started" }),
					/* @__PURE__ */ d("th", { children: "screened" }),
					/* @__PURE__ */ d("th", { children: "passed" }),
					/* @__PURE__ */ d("th", { children: "downloaded" }),
					/* @__PURE__ */ d("th", { children: "smoked" }),
					/* @__PURE__ */ d("th", { children: "error" })
				] }) }), /* @__PURE__ */ d("tbody", { children: n.map((e, t) => /* @__PURE__ */ f("tr", { children: [
					/* @__PURE__ */ d("td", { children: e.criteria }),
					/* @__PURE__ */ d("td", {
						className: "sp-md-dim",
						children: Da(e.started_at)
					}),
					/* @__PURE__ */ d("td", { children: e.screened ?? "—" }),
					/* @__PURE__ */ d("td", { children: e.passed ?? "—" }),
					/* @__PURE__ */ d("td", { children: e.downloaded ?? "—" }),
					/* @__PURE__ */ d("td", { children: e.smoked ?? "—" }),
					/* @__PURE__ */ d("td", {
						className: "sp-md-err",
						children: e.error || ""
					})
				] }, e.run_id ?? t)) })]
			})] })
		]
	});
}
//#endregion
//#region src/components/SettingsPanel/SettingsPanel.jsx
var ja = {
	settings: "set here",
	fleet: "set here",
	env: "unit drop-in",
	default: "default"
};
function Ma({ source: e }) {
	return /* @__PURE__ */ d("span", {
		className: `sp-src sp-src-${e || "default"}`,
		children: ja[e] || e || "default"
	});
}
function Na({ workers: e = [] }) {
	let [t, n] = l(null), [r, a] = l(null), [s, c] = l(""), [u, p] = l(""), m = i(() => {
		K("/api/llm/evict-policy").then((e) => {
			n(e), a(null);
		}).catch((e) => a(e.message));
	}, []);
	o(() => {
		m();
	}, [m]);
	let h = (e) => {
		c("policy"), p(""), K("/api/llm/evict-policy", {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({ least_reaping: e })
		}).then((t) => {
			n(t), p(e === null ? "Fleet ruling cleared — workers revert to their own drop-in on the next beat." : "Fleet policy set. Every worker adopts it on its next heartbeat.");
		}).catch((e) => a(e.message)).finally(() => c(""));
	};
	return /* @__PURE__ */ f("div", {
		className: "sp-panel",
		children: [
			r && /* @__PURE__ */ d("div", {
				className: "sp-err",
				children: r
			}),
			u && /* @__PURE__ */ d("div", {
				className: "sp-note",
				children: u
			}),
			/* @__PURE__ */ f("section", {
				className: "sp-sec",
				children: [
					/* @__PURE__ */ f("div", {
						className: "sp-sec-head",
						children: [/* @__PURE__ */ d("h3", {
							className: "sp-sec-title",
							children: "Eviction: least reaping"
						}), /* @__PURE__ */ d("span", {
							className: "sp-scope sp-scope-fleet",
							children: "fleet-wide"
						})]
					}),
					/* @__PURE__ */ f("p", {
						className: "sp-desc",
						children: [
							"When an admission needs room, the planner walks candidates in order until it has freed enough, then ",
							/* @__PURE__ */ d("strong", { children: "drops any victim the rest already cover" }),
							". That can satisfy a 15\xA0GiB need with one 35\xA0GiB unload instead of two smaller ones — ",
							/* @__PURE__ */ d("strong", { children: "fewer models disturbed, but less free headroom left behind" }),
							". Turning it off restores the older greedy walk: more models unloaded, more headroom free. On a tight disk that may be what you want."
						]
					}),
					/* @__PURE__ */ d("p", {
						className: "sp-desc sp-desc-why",
						children: "This one is fleet-wide on purpose. Central's eviction preview runs the same drop pass the workers do; if they disagreed, the console would show you one victim list while the fleet unloaded another."
					}),
					t == null ? /* @__PURE__ */ d("div", {
						className: "sp-loading",
						children: "loading…"
					}) : /* @__PURE__ */ f("div", {
						className: "sp-row",
						children: [/* @__PURE__ */ f("div", {
							className: "sp-val",
							children: [/* @__PURE__ */ d("span", {
								className: "sp-val-main",
								children: t.least_reaping ? "On" : "Off (greedy walk)"
							}), /* @__PURE__ */ d(Ma, { source: t.least_reaping_source })]
						}), /* @__PURE__ */ f("div", {
							className: "sp-actions",
							children: [
								/* @__PURE__ */ d("button", {
									className: `sp-btn ${t.least_reaping ? "sp-btn-on" : ""}`,
									disabled: s === "policy" || t.least_reaping,
									onClick: () => h(!0),
									children: "On"
								}),
								/* @__PURE__ */ d("button", {
									className: `sp-btn ${t.least_reaping ? "" : "sp-btn-on"}`,
									disabled: s === "policy" || !t.least_reaping,
									onClick: () => h(!1),
									children: "Off"
								}),
								t.least_reaping_source === "fleet" && /* @__PURE__ */ d("button", {
									className: "sp-btn sp-btn-quiet",
									disabled: s === "policy",
									onClick: () => h(null),
									children: "Clear ruling"
								})
							]
						})]
					})
				]
			}),
			/* @__PURE__ */ d(Aa, {})
		]
	});
}
//#endregion
//#region src/components/EvictionsPanel/evictionFormat.js
function Pa(e) {
	if (e == null) return "?";
	let t = [
		"B",
		"KiB",
		"MiB",
		"GiB",
		"TiB"
	], n = Number(e), r = 0;
	for (; n >= 1024 && r < t.length - 1;) n /= 1024, r++;
	return `${n.toFixed(1)} ${t[r]}`;
}
function Fa(e) {
	if (e == null) return null;
	let t = Number(e);
	return t < 1e3 ? `${Math.round(t)} ms` : t < 6e4 ? `${(t / 1e3).toFixed(1)}s` : `${Math.floor(t / 6e4)}m ${String(Math.round(t % 6e4 / 1e3)).padStart(2, "0")}s`;
}
function Ia(e) {
	if (!e) return "";
	try {
		return (/* @__PURE__ */ new Date(Number(e) * 1e3)).toLocaleTimeString();
	} catch {
		return "";
	}
}
function Q(e) {
	return e == null ? "" : typeof e == "string" ? e : typeof e == "number" || typeof e == "boolean" ? String(e) : Array.isArray(e) ? e.map(Q).filter(Boolean).join(", ") : typeof e == "object" ? Object.entries(e).map(([e, t]) => {
		if (t == null) return null;
		let n = typeof t == "object" ? Q(t) : String(t);
		return n === "" ? null : `${e}: ${n}`;
	}).filter(Boolean).join(" · ") : String(e);
}
function $(e) {
	return e && e._id ? `id:${e._id}` : `sq:${e?.worker_id ?? "?"}:${e?.seq ?? Math.random()}`;
}
var La = {
	fit: "ev-out-fit",
	partial: "ev-out-partial",
	refused: "ev-out-refused",
	"proceeded-unfit": "ev-out-unfit"
};
function Ra(e) {
	return `ev-tier ev-tier-${String(e || "unknown").replace(/[^a-z0-9]+/gi, "-")}`;
}
function za(e, t) {
	for (let n = e.length - 1; n >= 0; n--) {
		let r = e[n];
		if (r.kind === "victim" && r.model_key === t && r.state === "running") return n;
	}
	return -1;
}
function Ba(e, t, n) {
	for (let r = e.length - 1; r >= 0; r--) {
		let i = e[r];
		if (i.kind === t && i.state === "running" && n(i)) return r;
	}
	return -1;
}
function Va(e) {
	return e ? e.human ? e.human : [e.errno_name, e.detail || e.error_class].filter(Boolean).join(": ") || "provisioning failed" : "";
}
function Ha(e) {
	return e && (e.errno_name || e.error_class) || null;
}
function Ua(e) {
	return e ? e.kind === "provision" ? [
		"provision failed",
		e.source,
		Ha(e)
	].filter(Boolean).join(" · ") : e.kind === "resolve" ? ["resolve failed", Q(e.reason) || "no path"].filter(Boolean).join(" · ") : e.kind === "load" ? [
		"load failed",
		e.engine,
		Q(e.error)
	].filter(Boolean).join(" · ") : e.kind === "victim" ? [
		"eviction failed",
		e.model_key,
		Q(e.error)
	].filter(Boolean).join(" · ") : null : null;
}
function Wa(e) {
	if (!e) return null;
	if (e.kind === "provision") return Va(e);
	if (e.kind === "resolve") {
		let t = Q(e.reason) || "could not resolve a path";
		return e.resolved_path ? `${t} (${e.resolved_path})` : t;
	}
	return e.kind === "load" ? Q(e.error) || "load failed" : e.kind === "victim" ? Q(e.error) || "eviction failed" : null;
}
function Ga(e) {
	return e.state === "failed" && (e.kind === "provision" || e.kind === "resolve" || e.kind === "load" || e.kind === "victim");
}
function Ka(e) {
	let t = e.events, n = [], r = null, i = null, a = null, o = null, s = null, c = null;
	for (let e of t) switch (e.incoming_model && !r && (r = e.incoming_model), e.need_bytes != null && s == null && (s = e.need_bytes), e.stage) {
		case "member.select":
			n.push({
				kind: "member",
				key: $(e),
				state: "done",
				group_key: e.group_key,
				model_key: e.model_key,
				as: e.as,
				reason: e.reason,
				verdict: e.verdict,
				need_bytes: e.need_bytes,
				demanded_by: e.demanded_by
			});
			break;
		case "member.skip":
			n.push({
				kind: "memberskip",
				key: $(e),
				group_key: e.group_key,
				model_key: e.model_key,
				reason: e.reason
			});
			break;
		case "headroom.start":
			i = e.trigger || i, a = e.tier || a, e.group && (c = e.group);
			break;
		case "provision.start":
			n.push({
				kind: "provision",
				key: $(e),
				state: "running",
				model_key: e.model_key,
				source: e.source,
				dest_path: e.dest_path
			});
			break;
		case "provision.done": {
			let t = Ba(n, "provision", (t) => t.model_key === e.model_key && t.source === e.source), r = {
				state: "done",
				bytes: e.bytes,
				duration_ms: e.duration_ms
			};
			t >= 0 ? n[t] = {
				...n[t],
				...r
			} : n.push({
				kind: "provision",
				key: $(e),
				model_key: e.model_key,
				source: e.source,
				...r
			});
			break;
		}
		case "provision.fail": {
			let t = Ba(n, "provision", (t) => t.model_key === e.model_key && t.source === e.source), r = {
				state: "failed",
				error_class: e.error_class,
				errno_name: e.errno_name,
				detail: e.detail,
				human: e.human,
				disk_free_bytes: e.disk_free_bytes,
				disk_total_bytes: e.disk_total_bytes,
				disk_mount: e.disk_mount
			};
			t >= 0 ? n[t] = {
				...n[t],
				...r
			} : n.push({
				kind: "provision",
				key: $(e),
				model_key: e.model_key,
				source: e.source,
				...r
			});
			break;
		}
		case "resolve.fail":
			n.push({
				kind: "resolve",
				key: $(e),
				state: "failed",
				model_key: e.model_key,
				resolved_path: e.resolved_path,
				reason: e.reason
			});
			break;
		case "load.start":
			n.push({
				kind: "load",
				key: $(e),
				state: "running",
				model_key: e.model_key,
				engine: e.engine
			});
			break;
		case "load.done": {
			let t = Ba(n, "load", (t) => t.model_key === e.model_key), r = {
				state: "done",
				duration_ms: e.duration_ms,
				engine: e.engine || (t >= 0 ? n[t].engine : null)
			};
			t >= 0 ? n[t] = {
				...n[t],
				...r
			} : n.push({
				kind: "load",
				key: $(e),
				model_key: e.model_key,
				...r
			});
			break;
		}
		case "load.fail": {
			let t = Ba(n, "load", (t) => t.model_key === e.model_key), r = {
				state: "failed",
				error: e.error,
				engine: e.engine || (t >= 0 ? n[t].engine : null)
			};
			t >= 0 ? n[t] = {
				...n[t],
				...r
			} : n.push({
				kind: "load",
				key: $(e),
				model_key: e.model_key,
				...r
			});
			break;
		}
		case "fit.fail":
			n.push({
				kind: "fitfail",
				key: $(e),
				need: e.need_bytes,
				free: e.free_bytes
			});
			break;
		case "candidate.skip":
			n.push({
				kind: "skip",
				key: $(e),
				model_key: e.model_key,
				tier: e.tier,
				reason: e.reason,
				vram_bytes: e.vram_bytes,
				idle_s: e.idle_s
			});
			break;
		case "evict.start":
			n.push({
				kind: "victim",
				key: $(e),
				model_key: e.model_key,
				tier: e.tier,
				state: "running"
			});
			break;
		case "evict.done": {
			let t = za(n, e.model_key), r = {
				state: "done",
				freed_bytes: e.freed_bytes,
				duration_ms: e.duration_ms,
				tier: e.tier || (t >= 0 ? n[t].tier : null)
			};
			t >= 0 ? n[t] = {
				...n[t],
				...r
			} : n.push({
				kind: "victim",
				key: $(e),
				model_key: e.model_key,
				...r
			});
			break;
		}
		case "evict.fail": {
			let t = za(n, e.model_key), r = {
				state: "failed",
				error: e.error,
				tier: e.tier || (t >= 0 ? n[t].tier : null)
			};
			t >= 0 ? n[t] = {
				...n[t],
				...r
			} : n.push({
				kind: "victim",
				key: $(e),
				model_key: e.model_key,
				...r
			});
			break;
		}
		case "reclaim.done":
			n.push({
				kind: "reclaim",
				key: $(e)
			});
			break;
		case "makeroom.verdict":
			n.push({
				kind: "verdict",
				key: $(e),
				action: e.action,
				reason: e.reason,
				evicted: e.evicted,
				freed_bytes: e.freed_bytes
			});
			break;
		case "headroom.done":
			o = e;
			break;
		default: break;
	}
	let l = /* @__PURE__ */ new Set();
	r && l.add(r);
	for (let e of n) e.model_key && l.add(e.model_key);
	if (o && Array.isArray(o.evicted)) for (let e of o.evicted) l.add(e);
	let u = n.find(Ga) || null;
	return {
		id: e.id,
		worker_id: e.worker_id,
		startTs: e.startTs,
		endTs: o ? o.ts : null,
		incoming: r || o && o.incoming_model || null,
		trigger: i,
		tier: a,
		needBytes: s,
		rows: n,
		done: o,
		touched: l,
		group: c,
		failure: Ua(u),
		failureDetail: Wa(u),
		failureRow: u,
		haystack: [
			e.worker_id,
			r,
			...n.map((e) => e.model_key),
			...n.map((e) => e.source),
			...n.map((e) => e.engine),
			...n.map((e) => e.errno_name),
			...n.map((e) => e.error_class),
			...n.map((e) => e.group_key),
			...n.map((e) => e.as),
			...n.map((e) => e.reason),
			c && c.group_key,
			c && c.tick
		].filter(Boolean).join(" ").toLowerCase()
	};
}
function qa(e) {
	return e.rows.filter((e) => e.kind === "victim" && e.state === "done");
}
function Ja(e) {
	let t = 0, n = !1;
	for (let r of qa(e)) r.freed_bytes != null && (t += Number(r.freed_bytes), n = !0);
	return n ? t : null;
}
//#endregion
//#region src/components/EvictionsPanel/evictionStream.js
var Ya = 200, Xa = 300, Za = 5e3, Qa = [], $a = "idle", eo = null, to = 0, no = /* @__PURE__ */ new Set(), ro = /* @__PURE__ */ new Set(), io = null, ao = 0, oo = null, so = !1;
function co() {
	to++;
	for (let e of ro) try {
		e();
	} catch {}
}
function lo(e) {
	let t = [];
	for (let n of e) {
		if (!n || typeof n != "object" || n.stage === "stream.ready") continue;
		let e = $(n);
		no.has(e) || (no.add(e), t.push(n));
	}
	if (!t.length) return;
	let n = Qa.slice(), r = new Map(n.map((e, t) => [e.id, t]));
	for (let e of t) {
		let t = e.run_id || `solo:${$(e)}`, i = r.get(t);
		if (i == null) r.set(t, n.length), n.push({
			id: t,
			worker_id: e.worker_id,
			startTs: Number(e.ts) || Date.now() / 1e3,
			events: [e]
		});
		else {
			let t = n[i];
			n[i] = {
				...t,
				worker_id: t.worker_id || e.worker_id,
				events: [...t.events, e]
			};
		}
	}
	Qa = n.length > Ya ? n.slice(n.length - Ya) : n, co();
}
function uo(e) {
	$a !== e && ($a = e, co());
}
function fo() {
	if (io) return;
	uo("connecting");
	let e;
	try {
		e = new EventSource(H("/api/llm/evictions/stream"), { withCredentials: B().credentials === "include" });
	} catch {
		uo("disconnected");
		return;
	}
	io = e, e.onopen = () => uo("live"), e.onmessage = (e) => {
		let t;
		try {
			t = JSON.parse(e.data);
		} catch {
			return;
		}
		lo([t]);
	}, e.onerror = () => {
		e.close(), io === e && (io = null), uo("disconnected"), ao > 0 && (clearTimeout(oo), oo = setTimeout(() => {
			ao > 0 && fo();
		}, Za));
	};
}
function po() {
	clearTimeout(oo), oo = null, io &&= (io.close(), null), uo("idle");
}
function mo() {
	so || (so = !0, K(`/api/llm/evictions?limit=${Xa}`).then((e) => {
		eo = null, lo(Array.isArray(e?.events) ? e.events : []), co();
	}).catch((e) => {
		eo = e?.message || String(e), co();
	}));
}
function ho() {
	ao++, ao === 1 && (mo(), fo());
}
function go() {
	ao = Math.max(0, ao - 1), ao === 0 && po();
}
function _o() {
	po(), ao > 0 && fo();
}
function vo({ paused: e = !1 } = {}) {
	let [t, n] = l(() => ({
		runs: Qa,
		conn: $a,
		error: eo,
		version: to
	})), [r, a] = l(0), s = c(e);
	return s.current = e, o(() => {
		ho();
		let e = () => {
			s.current ? a((e) => e + 1) : n({
				runs: Qa,
				conn: $a,
				error: eo,
				version: to
			});
		};
		return e(), ro.add(e), () => {
			ro.delete(e), go();
		};
	}, []), o(() => {
		e || (a(0), n({
			runs: Qa,
			conn: $a,
			error: eo,
			version: to
		}));
	}, [e]), {
		runs: t.runs,
		conn: t.conn,
		error: t.error,
		held: r,
		reconnect: i(() => _o(), [])
	};
}
//#endregion
//#region src/components/EvictionsPanel/EvictionsPanel.jsx
function yo({ tier: e }) {
	return e ? /* @__PURE__ */ d("span", {
		className: Ra(e),
		children: e
	}) : null;
}
function bo({ run: e, now: t }) {
	let n = e.done, r = n ? n.outcome || "fit" : null, i = ((e.endTs || t / 1e3) - e.startTs) * 1e3;
	return /* @__PURE__ */ f("div", {
		className: `ev-card ${n ? "" : "ev-card-running"}`,
		children: [
			/* @__PURE__ */ f("div", {
				className: "ev-head",
				children: [
					/* @__PURE__ */ d("span", {
						className: `ev-model ${e.incoming ? "" : "ev-model-sweep"}`,
						children: e.incoming || "headroom sweep"
					}),
					e.trigger && /* @__PURE__ */ d("span", {
						className: `ev-chip ev-trigger ev-trigger-${e.trigger}`,
						children: e.trigger
					}),
					e.worker_id && /* @__PURE__ */ d("span", {
						className: "ev-chip ev-worker",
						children: e.worker_id
					}),
					e.tier && /* @__PURE__ */ d(yo, { tier: e.tier }),
					/* @__PURE__ */ d("span", {
						className: "ev-time",
						children: Ia(e.startTs)
					}),
					/* @__PURE__ */ d("span", {
						className: "ev-elapsed",
						children: Fa(Math.max(0, i))
					}),
					/* @__PURE__ */ d("span", { className: "ev-spacer" }),
					n ? /* @__PURE__ */ d("span", {
						className: `ev-outcome ${La[r] || "ev-out-unfit"}`,
						children: r
					}) : e.failure ? /* @__PURE__ */ d("span", {
						className: "ev-outcome ev-out-failed",
						title: e.failureDetail || e.failure,
						children: "failed"
					}) : /* @__PURE__ */ d("span", {
						className: "ev-outcome ev-out-running",
						children: "running…"
					})
				]
			}),
			e.needBytes != null && /* @__PURE__ */ f("div", {
				className: "ev-need",
				children: ["needs ", Pa(e.needBytes)]
			}),
			e.rows.length === 0 && /* @__PURE__ */ d("div", {
				className: "ev-row ev-row-quiet",
				children: "no candidates walked"
			}),
			e.rows.map((e) => e.kind === "member" ? /* @__PURE__ */ f("div", {
				className: "ev-row ev-row-member",
				children: [
					/* @__PURE__ */ d("span", {
						className: "ev-mark",
						children: "member"
					}),
					/* @__PURE__ */ d("span", {
						className: "ev-chip ev-source",
						children: e.group_key
					}),
					/* @__PURE__ */ d("span", {
						className: "ev-key",
						children: e.model_key
					}),
					e.as && /* @__PURE__ */ f("span", {
						className: "ev-chip",
						children: ["as ", e.as]
					}),
					/* @__PURE__ */ d("span", {
						className: "ev-why",
						children: e.reason || "selected"
					}),
					/* @__PURE__ */ d("span", { className: "ev-spacer" }),
					e.need_bytes != null && e.verdict === "need" && /* @__PURE__ */ f("span", {
						className: "ev-num",
						children: ["declares ", Pa(e.need_bytes)]
					})
				]
			}, e.key) : e.kind === "memberskip" ? /* @__PURE__ */ f("div", {
				className: "ev-row ev-row-skip",
				children: [
					/* @__PURE__ */ d("span", {
						className: "ev-mark",
						children: "member skip"
					}),
					/* @__PURE__ */ d("span", {
						className: "ev-key",
						children: e.model_key
					}),
					/* @__PURE__ */ d("span", {
						className: "ev-why",
						children: e.reason || "no reason recorded"
					})
				]
			}, e.key) : e.kind === "provision" ? /* @__PURE__ */ f("div", {
				className: `ev-row ${e.state === "failed" ? "ev-row-fail" : e.state === "done" ? "ev-row-evict" : "ev-row-inflight"}`,
				title: e.dest_path || void 0,
				children: [
					/* @__PURE__ */ d("span", {
						className: "ev-mark",
						children: "provision"
					}),
					/* @__PURE__ */ d("span", {
						className: "ev-chip ev-source",
						children: e.source || "unknown"
					}),
					/* @__PURE__ */ d("span", {
						className: "ev-key",
						children: e.model_key
					}),
					e.state === "failed" && /* @__PURE__ */ f("span", {
						className: "ev-why",
						children: [Ha(e) ? `${Ha(e)}: ` : "", Va(e)]
					}),
					e.state === "running" && /* @__PURE__ */ d("span", {
						className: "ev-why",
						children: "fetching…"
					}),
					/* @__PURE__ */ d("span", { className: "ev-spacer" }),
					e.state === "done" && e.bytes != null && /* @__PURE__ */ f("span", {
						className: "ev-num ev-num-strong",
						children: [Pa(e.bytes), e.duration_ms == null ? "" : ` in ${Fa(e.duration_ms)}`]
					}),
					e.state === "done" && e.bytes == null && e.duration_ms != null && /* @__PURE__ */ d("span", {
						className: "ev-num",
						children: Fa(e.duration_ms)
					})
				]
			}, e.key) : e.kind === "resolve" ? /* @__PURE__ */ f("div", {
				className: "ev-row ev-row-fail",
				title: e.resolved_path || void 0,
				children: [
					/* @__PURE__ */ d("span", {
						className: "ev-mark",
						children: "resolve"
					}),
					/* @__PURE__ */ d("span", {
						className: "ev-key",
						children: e.model_key
					}),
					/* @__PURE__ */ d("span", {
						className: "ev-why",
						children: Q(e.reason) || "could not resolve a path"
					})
				]
			}, e.key) : e.kind === "load" ? /* @__PURE__ */ f("div", {
				className: `ev-row ${e.state === "failed" ? "ev-row-fail" : e.state === "done" ? "ev-row-evict" : "ev-row-inflight"}`,
				children: [
					/* @__PURE__ */ d("span", {
						className: "ev-mark",
						children: "load"
					}),
					e.engine && /* @__PURE__ */ d("span", {
						className: "ev-chip ev-engine",
						children: e.engine
					}),
					/* @__PURE__ */ d("span", {
						className: "ev-key",
						children: e.model_key
					}),
					e.state === "failed" && /* @__PURE__ */ d("span", {
						className: "ev-why",
						children: Q(e.error) || "load failed"
					}),
					e.state === "running" && /* @__PURE__ */ d("span", {
						className: "ev-why",
						children: "loading…"
					}),
					/* @__PURE__ */ d("span", { className: "ev-spacer" }),
					e.duration_ms != null && /* @__PURE__ */ d("span", {
						className: "ev-num",
						children: Fa(e.duration_ms)
					})
				]
			}, e.key) : e.kind === "skip" ? /* @__PURE__ */ f("div", {
				className: "ev-row ev-row-skip",
				children: [
					/* @__PURE__ */ d("span", {
						className: "ev-mark",
						children: "skip"
					}),
					/* @__PURE__ */ d("span", {
						className: "ev-key",
						children: e.model_key
					}),
					/* @__PURE__ */ d(yo, { tier: e.tier }),
					/* @__PURE__ */ d("span", {
						className: "ev-why",
						children: Q(e.reason) || "no reason recorded"
					}),
					/* @__PURE__ */ d("span", { className: "ev-spacer" }),
					e.vram_bytes != null && /* @__PURE__ */ d("span", {
						className: "ev-num",
						children: Pa(e.vram_bytes)
					}),
					e.idle_s != null && /* @__PURE__ */ f("span", {
						className: "ev-num",
						children: ["idle ", Fa(Number(e.idle_s) * 1e3)]
					})
				]
			}, e.key) : e.kind === "victim" ? /* @__PURE__ */ f("div", {
				className: `ev-row ${e.state === "failed" ? "ev-row-fail" : e.state === "done" ? "ev-row-evict" : "ev-row-inflight"}`,
				children: [
					/* @__PURE__ */ d("span", {
						className: "ev-mark",
						children: e.state === "failed" ? "fail" : e.state === "done" ? "evict" : "…"
					}),
					/* @__PURE__ */ d("span", {
						className: "ev-key",
						children: e.model_key
					}),
					/* @__PURE__ */ d(yo, { tier: e.tier }),
					e.state === "failed" && /* @__PURE__ */ d("span", {
						className: "ev-why",
						children: Q(e.error) || "eviction failed"
					}),
					e.state === "running" && /* @__PURE__ */ d("span", {
						className: "ev-why",
						children: "unloading…"
					}),
					/* @__PURE__ */ d("span", { className: "ev-spacer" }),
					e.freed_bytes != null && /* @__PURE__ */ f("span", {
						className: "ev-num ev-num-strong",
						children: ["freed ", Pa(e.freed_bytes)]
					}),
					e.duration_ms != null && /* @__PURE__ */ d("span", {
						className: "ev-num",
						children: Fa(e.duration_ms)
					})
				]
			}, e.key) : e.kind === "fitfail" ? /* @__PURE__ */ f("div", {
				className: "ev-row ev-row-fitfail",
				children: [/* @__PURE__ */ d("span", {
					className: "ev-mark",
					children: "unfit"
				}), /* @__PURE__ */ f("span", {
					className: "ev-why",
					children: [
						"needs ",
						Pa(e.need),
						" · ",
						Pa(e.free),
						" free"
					]
				})]
			}, e.key) : e.kind === "reclaim" ? /* @__PURE__ */ f("div", {
				className: "ev-row ev-row-quiet",
				children: [/* @__PURE__ */ d("span", {
					className: "ev-mark",
					children: "reclaim"
				}), /* @__PURE__ */ d("span", {
					className: "ev-why",
					children: "allocator reclaim complete"
				})]
			}, e.key) : /* @__PURE__ */ f("div", {
				className: `ev-row ev-row-verdict ev-verdict-${e.action || "unknown"}`,
				children: [
					/* @__PURE__ */ d("span", {
						className: "ev-mark",
						children: "verdict"
					}),
					/* @__PURE__ */ d("span", {
						className: "ev-key",
						children: e.action
					}),
					e.reason != null && /* @__PURE__ */ d("span", {
						className: "ev-why",
						children: Q(e.reason)
					}),
					/* @__PURE__ */ d("span", { className: "ev-spacer" }),
					Array.isArray(e.evicted) && e.evicted.length > 0 && /* @__PURE__ */ f("span", {
						className: "ev-num",
						children: [e.evicted.length, " evicted"]
					}),
					e.freed_bytes != null && /* @__PURE__ */ f("span", {
						className: "ev-num",
						children: ["freed ", Pa(e.freed_bytes)]
					})
				]
			}, e.key)),
			n && (Q(n.reason) || n.note || Array.isArray(n.evicted) && n.evicted.length > 0) && /* @__PURE__ */ f("div", {
				className: "ev-foot",
				children: [
					Array.isArray(n.evicted) && n.evicted.length > 0 && /* @__PURE__ */ f("span", {
						className: "ev-foot-evicted",
						children: ["unloaded: ", n.evicted.join(", ")]
					}),
					n.reason != null && /* @__PURE__ */ d("span", {
						className: "ev-foot-reason",
						children: Q(n.reason)
					}),
					n.note && /* @__PURE__ */ d("span", {
						className: "ev-foot-note",
						children: n.note
					})
				]
			})
		]
	});
}
function xo() {
	let [e, t] = l(!1), [n, r] = l(""), [i, a] = l(() => Date.now()), { runs: c, conn: p, error: m, held: h, reconnect: g } = vo({ paused: e });
	o(() => {
		let e = setInterval(() => a(Date.now()), 1e3);
		return () => clearInterval(e);
	}, []);
	let _ = s(() => {
		let e = n.trim().toLowerCase(), t = [];
		for (let n = c.length - 1; n >= 0; n--) {
			let r = Ka(c[n]);
			(!e || r.haystack.includes(e)) && t.push(r);
		}
		return t;
	}, [c, n]);
	return /* @__PURE__ */ f("div", {
		className: "ev-panel",
		children: [
			/* @__PURE__ */ f("div", {
				className: "ev-toolbar",
				children: [
					/* @__PURE__ */ d("span", {
						className: `ev-dot ev-dot-${p}`,
						title: `stream ${p}`
					}),
					/* @__PURE__ */ d("span", {
						className: "ev-conn",
						children: p
					}),
					/* @__PURE__ */ d("button", {
						className: `ev-btn ${e ? "ev-btn-paused" : ""}`,
						onClick: () => t((e) => !e),
						title: e ? "Resume — queued events land in order; the stream never dropped" : "Pause the view. The stream stays open and buffers.",
						children: e ? `Paused${h ? ` (${h})` : ""}` : "Live"
					}),
					/* @__PURE__ */ d("input", {
						className: "ev-filter",
						value: n,
						placeholder: "filter by model or worker…",
						onChange: (e) => r(e.target.value)
					}),
					p === "disconnected" && /* @__PURE__ */ d("button", {
						className: "ev-btn ev-btn-quiet",
						onClick: g,
						children: "Reconnect"
					}),
					/* @__PURE__ */ d("span", { className: "ev-spacer" }),
					/* @__PURE__ */ f("span", {
						className: "ev-count",
						children: [
							_.length,
							" run",
							_.length === 1 ? "" : "s"
						]
					})
				]
			}),
			m && /* @__PURE__ */ d("div", {
				className: "ev-err",
				children: m
			}),
			_.length === 0 && /* @__PURE__ */ d("div", {
				className: "ev-empty",
				children: c.length === 0 ? /* @__PURE__ */ f(u, { children: [/* @__PURE__ */ d("p", {
					className: "ev-empty-line",
					children: "No eviction activity recorded yet."
				}), /* @__PURE__ */ f("p", {
					className: "ev-empty-sub",
					children: [
						"Eviction events are emitted by the box that does the unloading, so they only appear from workers running a release that carries the worker-side emitters. Central does no local model serving, but it does clear the video card for a render — those runs show up here tagged ",
						/* @__PURE__ */ d("code", { children: "reservation" }),
						"."
					]
				})] }) : /* @__PURE__ */ f("p", {
					className: "ev-empty-line",
					children: [
						"No runs match “",
						n,
						"”."
					]
				})
			}),
			/* @__PURE__ */ d("div", {
				className: "ev-list",
				children: _.map((e) => /* @__PURE__ */ d(bo, {
					run: e,
					now: i
				}, e.id))
			})
		]
	});
}
//#endregion
//#region src/Auth/AuthProvider.tsx
var So = "hugpy", Co = {
	me: "/me",
	login: "/login",
	logout: "/logout",
	register: "/register",
	changePassword: "/change-password"
}, wo = r(null);
function To({ children: e, base: t, mode: n, endpoints: r, fetch: a, credentials: u = "include" }) {
	let [f, p] = l({ status: "checking" }), [m, h] = l("open"), [g, _] = l(!0), v = s(() => a ?? ((...e) => globalThis.fetch(...e)), [a]), y = s(() => ({
		...Co,
		...r
	}), [r]), b = c(null), x = i(async () => (b.current || (b.current = (async () => {
		if (n === "open") return {
			mode: "open",
			base: t ?? "",
			reachable: !0
		};
		if (n === "external" && t) return {
			mode: "external",
			base: t,
			reachable: !0
		};
		let e = await me();
		return {
			mode: e.mode,
			base: t ?? e.base ?? "",
			reachable: e.reachable
		};
	})(), b.current.then((e) => {
		h(e.mode), _(e.reachable);
	})), b.current), [n, t]), S = i(async (e, t) => {
		let n = await x();
		return v(`${n.base}${e}`, {
			credentials: u,
			...t
		});
	}, [
		x,
		v,
		u
	]), C = i(async () => {
		if ((await x()).mode === "open") {
			p({
				status: "authed",
				user: {
					username: "local",
					local: !0
				}
			});
			return;
		}
		try {
			let e = await S(y.me, {
				method: "GET",
				headers: { Accept: "application/json" }
			});
			if (e.ok) {
				p({
					status: "authed",
					user: await e.json()
				});
				return;
			}
			(e.status === 401 || e.status === 403) && p({ status: "guest" });
		} catch {
			p((e) => e.status === "checking" ? { status: "guest" } : e);
		}
	}, [
		x,
		S,
		y.me
	]), w = i(async (e, t) => (await x()).mode === "open" || (await S(y.login, {
		method: "POST",
		headers: {
			"Content-Type": "application/json",
			Accept: "application/json"
		},
		body: JSON.stringify({
			username: e,
			password: t,
			site: So
		})
	})).ok ? (await C(), !0) : (p({ status: "guest" }), !1), [
		x,
		S,
		y.login,
		C
	]), T = i(async () => {
		if ((await x()).mode === "open") {
			p({ status: "guest" });
			return;
		}
		try {
			await S(y.logout, {
				method: "POST",
				headers: { Accept: "application/json" }
			});
		} finally {
			p({ status: "guest" });
		}
	}, [
		x,
		S,
		y.logout
	]), E = i(async (e, t) => {
		try {
			let t = await e.json();
			if (t && typeof t.error == "string" && t.error.trim()) return t.error;
		} catch {}
		return t;
	}, []), D = i(async ({ username: e, email: t, password: n }) => {
		let r = await S(y.register, {
			method: "POST",
			headers: {
				"Content-Type": "application/json",
				Accept: "application/json"
			},
			body: JSON.stringify({
				username: e,
				email: t,
				password: n
			})
		});
		return r.ok ? { ok: !0 } : {
			ok: !1,
			error: await E(r, "There was an issue registering your account.")
		};
	}, [
		S,
		y.register,
		E
	]), O = i(async ({ currentPassword: e, newPassword: t }) => {
		let n = await S(y.changePassword, {
			method: "POST",
			headers: {
				"Content-Type": "application/json",
				Accept: "application/json"
			},
			body: JSON.stringify({
				current_password: e,
				new_password: t
			})
		});
		return n.ok ? (await C(), { ok: !0 }) : {
			ok: !1,
			error: await E(n, "There was an issue changing your password.")
		};
	}, [
		S,
		y.changePassword,
		E,
		C
	]);
	o(() => {
		C();
	}, [C]);
	let k = s(() => ({
		state: f,
		mode: m,
		reachable: g,
		refresh: C,
		signIn: w,
		signOut: T,
		signUp: D,
		changePassword: O
	}), [
		f,
		m,
		g,
		C,
		w,
		T,
		D,
		O
	]);
	return /* @__PURE__ */ d(wo.Provider, {
		value: k,
		children: e
	});
}
function Eo() {
	let e = a(wo);
	if (!e) throw Error("useAuth must be used inside <AuthProvider>");
	return e;
}
//#endregion
//#region src/Auth/Login/Login.tsx
var Do = () => {
	let e = _(), t = g(), { signIn: n } = Eo(), r = t.state, i = `${r?.from?.pathname || "/"}${r?.from?.search || ""}${r?.from?.hash || ""}`, [a, s] = l(r?.justRegistered ? "Account created. You can sign in once an administrator approves it." : null), [c, u] = l(!1), [m, h] = l("Credentials incorrect!"), [D, O] = l(!1), [k] = l(r?.username || "");
	return o(() => {
		r?.justRegistered && window.history.replaceState({}, "");
	}, [r?.justRegistered]), /* @__PURE__ */ d(w, {
		style: { marginTop: "1rem" },
		children: /* @__PURE__ */ d(T, { children: /* @__PURE__ */ f(x, {
			sx: {
				marginTop: 8,
				marginBottom: 8,
				display: "flex",
				flexDirection: "column",
				alignItems: "center"
			},
			children: [
				/* @__PURE__ */ d(S, {
					component: "h1",
					variant: "h5",
					children: "Sign in"
				}),
				/* @__PURE__ */ d(E, {
					in: !!a,
					children: /* @__PURE__ */ d(C, {
						severity: "info",
						sx: { mt: 2 },
						children: a
					})
				}),
				/* @__PURE__ */ d(E, {
					in: c,
					children: /* @__PURE__ */ d(C, {
						severity: "error",
						sx: { mt: 2 },
						children: m
					})
				}),
				/* @__PURE__ */ f(x, {
					component: "form",
					onSubmit: async (t) => {
						t.preventDefault(), s(null);
						let r = new FormData(t.currentTarget), a = String(r.get("username") || "").trim(), o = String(r.get("password") || "");
						if (!a || !o) {
							h("Username and password are required."), u(!0);
							return;
						}
						O(!0), u(!1);
						try {
							if (!await n(a, o)) {
								h("Credentials incorrect or sign-in not permitted."), u(!0);
								return;
							}
							e(i, { replace: !0 });
						} catch (e) {
							console.error("Login error:", e), h("Login request failed."), u(!0);
						} finally {
							O(!1);
						}
					},
					noValidate: !0,
					sx: { mt: 1 },
					children: [
						/* @__PURE__ */ d(y, {
							margin: "normal",
							required: !0,
							fullWidth: !0,
							id: "username",
							label: "User Name",
							name: "username",
							defaultValue: k,
							error: c,
							autoComplete: "username",
							autoFocus: !0,
							disabled: D
						}),
						/* @__PURE__ */ d(y, {
							margin: "normal",
							required: !0,
							fullWidth: !0,
							name: "password",
							label: "Password",
							type: "password",
							id: "password",
							error: c,
							autoComplete: "current-password",
							disabled: D
						}),
						/* @__PURE__ */ d(v, {
							type: "submit",
							fullWidth: !0,
							variant: "contained",
							disabled: D,
							sx: {
								mt: 3,
								mb: 2
							},
							children: D ? "Signing in..." : "Sign In"
						}),
						/* @__PURE__ */ f(b, {
							direction: "row",
							spacing: 2,
							sx: {
								justifyContent: "flex-end",
								mt: 2
							},
							children: [
								/* @__PURE__ */ d(p, {
									to: "/register",
									children: "Register"
								}),
								/* @__PURE__ */ d(p, {
									to: "/change-password",
									children: "Change Password"
								}),
								/* @__PURE__ */ d(p, {
									to: "/",
									children: "Back"
								})
							]
						})
					]
				})
			]
		}) })
	});
};
//#endregion
//#region src/Auth/Logout/Logout.tsx
function Oo() {
	let e = _(), { signOut: t } = Eo();
	return o(() => {
		let n = !1;
		return (async () => {
			await t(), n || e("/", { replace: !0 });
		})(), () => {
			n = !0;
		};
	}, [t, e]), null;
}
//#endregion
//#region src/Auth/Register/Register.tsx
function ko() {
	let e = _(), { signUp: t } = Eo(), [n, r] = l(!1), [i, a] = l("There was an issue registering your account."), [o, s] = l(!1);
	return /* @__PURE__ */ d(w, {
		style: { marginTop: "1rem" },
		children: /* @__PURE__ */ d(T, { children: /* @__PURE__ */ f(x, {
			sx: {
				marginTop: 8,
				marginBottom: 8,
				display: "flex",
				flexDirection: "column",
				alignItems: "center"
			},
			children: [
				/* @__PURE__ */ d(S, {
					component: "h1",
					variant: "h5",
					children: "Sign up"
				}),
				/* @__PURE__ */ d(E, {
					in: n,
					children: /* @__PURE__ */ d(C, {
						severity: "error",
						children: i
					})
				}),
				/* @__PURE__ */ f(x, {
					component: "form",
					noValidate: !0,
					onSubmit: async (n) => {
						n.preventDefault();
						let i = new FormData(n.currentTarget), o = String(i.get("username") || "").trim(), c = String(i.get("email") || "").trim(), l = String(i.get("password") || "");
						if (!o || !c || !l) {
							a("Username, email, and password are required."), r(!0);
							return;
						}
						s(!0), r(!1);
						try {
							let n = await t({
								username: o,
								email: c,
								password: l
							});
							if (!n.ok) {
								a(n.error || "There was an issue registering your account."), r(!0);
								return;
							}
							e("/login", {
								replace: !0,
								state: {
									justRegistered: !0,
									username: o
								}
							});
						} catch (e) {
							console.error("Register error:", e), a("Registration request failed."), r(!0);
						} finally {
							s(!1);
						}
					},
					sx: { mt: 1 },
					children: [
						/* @__PURE__ */ d(y, {
							margin: "normal",
							required: !0,
							fullWidth: !0,
							id: "username",
							label: "Username",
							name: "username",
							autoComplete: "username",
							autoFocus: !0,
							error: n,
							disabled: o
						}),
						/* @__PURE__ */ d(y, {
							margin: "normal",
							required: !0,
							fullWidth: !0,
							id: "email",
							label: "Email Address",
							name: "email",
							type: "email",
							autoComplete: "email",
							error: n,
							disabled: o
						}),
						/* @__PURE__ */ d(y, {
							margin: "normal",
							required: !0,
							fullWidth: !0,
							name: "password",
							label: "Password",
							type: "password",
							id: "password",
							autoComplete: "new-password",
							error: n,
							disabled: o
						}),
						/* @__PURE__ */ d(v, {
							type: "submit",
							fullWidth: !0,
							variant: "contained",
							disabled: o,
							sx: {
								mt: 3,
								mb: 2
							},
							children: o ? "Creating account..." : "Sign Up"
						}),
						/* @__PURE__ */ f(b, {
							direction: "row",
							spacing: 2,
							sx: {
								justifyContent: "flex-end",
								mt: 2
							},
							children: [/* @__PURE__ */ d(p, {
								to: "/login",
								children: "Login"
							}), /* @__PURE__ */ d(p, {
								to: "/",
								children: "Back"
							})]
						})
					]
				})
			]
		}) })
	});
}
//#endregion
//#region src/Auth/ChangePassword/ChangePassword.tsx
function Ao() {
	let e = _(), { changePassword: t } = Eo(), [n, r] = l(!1), [i, a] = l(!1), [o, s] = l(""), [c, u] = l(!1);
	return /* @__PURE__ */ d(w, {
		style: { marginTop: "1rem" },
		children: /* @__PURE__ */ d(T, { children: /* @__PURE__ */ f(x, {
			sx: {
				marginTop: 8,
				marginBottom: 8,
				display: "flex",
				flexDirection: "column",
				alignItems: "center"
			},
			children: [
				/* @__PURE__ */ d(S, {
					component: "h1",
					variant: "h5",
					children: "Change Password"
				}),
				/* @__PURE__ */ d(E, {
					in: n,
					children: /* @__PURE__ */ d(C, {
						severity: "error",
						children: o
					})
				}),
				/* @__PURE__ */ d(E, {
					in: i,
					children: /* @__PURE__ */ d(C, {
						severity: "success",
						children: o
					})
				}),
				/* @__PURE__ */ f(x, {
					component: "form",
					noValidate: !0,
					onSubmit: async (n) => {
						n.preventDefault();
						let i = new FormData(n.currentTarget), o = String(i.get("current_password") || ""), c = String(i.get("new_password") || ""), l = String(i.get("verify_password") || "");
						if (r(!1), a(!1), s(""), !o || !c || !l) {
							r(!0), s("All password fields are required.");
							return;
						}
						if (c !== l) {
							r(!0), s("New password and verify password do not match.");
							return;
						}
						if (c.length < 10) {
							r(!0), s("New password must be at least 10 characters.");
							return;
						}
						u(!0);
						try {
							let n = await t({
								currentPassword: o,
								newPassword: c
							});
							if (!n.ok) {
								r(!0), s(n.error || "There was an issue changing your password.");
								return;
							}
							a(!0), s("Password updated successfully."), setTimeout(() => {
								e("/", { replace: !0 });
							}, 800);
						} catch (e) {
							console.error("Change password error:", e), r(!0), s("Change password request failed.");
						} finally {
							u(!1);
						}
					},
					sx: { mt: 1 },
					children: [
						/* @__PURE__ */ f(D, {
							container: !0,
							spacing: 2,
							children: [
								/* @__PURE__ */ d(D, {
									size: { xs: 12 },
									children: /* @__PURE__ */ d(y, {
										required: !0,
										fullWidth: !0,
										name: "current_password",
										label: "Current Password",
										type: "password",
										id: "CurrentPassword",
										autoComplete: "current-password",
										error: n,
										disabled: c
									})
								}),
								/* @__PURE__ */ d(D, {
									size: { xs: 12 },
									children: /* @__PURE__ */ d(y, {
										required: !0,
										fullWidth: !0,
										name: "new_password",
										label: "New Password",
										type: "password",
										id: "NewPassword",
										autoComplete: "new-password",
										error: n,
										disabled: c
									})
								}),
								/* @__PURE__ */ d(D, {
									size: { xs: 12 },
									children: /* @__PURE__ */ d(y, {
										required: !0,
										fullWidth: !0,
										name: "verify_password",
										label: "Verify Password",
										type: "password",
										id: "VerifyPassword",
										autoComplete: "new-password",
										error: n,
										disabled: c
									})
								})
							]
						}),
						/* @__PURE__ */ d(v, {
							type: "submit",
							fullWidth: !0,
							variant: "contained",
							disabled: c,
							sx: {
								mt: 3,
								mb: 2
							},
							children: c ? "Updating..." : "Change Password"
						}),
						/* @__PURE__ */ f(b, {
							direction: "row",
							spacing: 2,
							sx: {
								justifyContent: "flex-end",
								mt: 2
							},
							children: [/* @__PURE__ */ d(p, {
								to: "/login",
								children: "Login"
							}), /* @__PURE__ */ d(p, {
								to: "/",
								children: "Back"
							})]
						})
					]
				})
			]
		}) })
	});
}
//#endregion
//#region src/Auth/LoginForm.jsx
function jo() {
	let { signIn: e } = Eo(), t = _(), n = g(), r = n.state?.from?.pathname || "/console", i = new URLSearchParams(n.search).get("next"), a = i && i.startsWith("/") && !i.startsWith("//") ? i : null, [o, s] = l(null), [c, p] = l(!1);
	return /* @__PURE__ */ f(u, { children: [/* @__PURE__ */ d(be, {}), /* @__PURE__ */ d("div", {
		className: "login-page",
		children: /* @__PURE__ */ f("form", {
			onSubmit: async (n) => {
				n.preventDefault(), s(null), p(!0);
				let i = new FormData(n.currentTarget);
				try {
					await e(String(i.get("username") || ""), String(i.get("password") || "")) ? a ? window.location.assign(a) : t(r, { replace: !0 }) : s("Credentials incorrect or sign-in not permitted.");
				} catch {
					s("Login request failed.");
				} finally {
					p(!1);
				}
			},
			className: "login-card",
			children: [
				/* @__PURE__ */ d("img", {
					className: "login-lockup",
					src: Ft,
					alt: "hugpy — inference you own"
				}),
				o && /* @__PURE__ */ d("div", {
					className: "login-error",
					children: o
				}),
				/* @__PURE__ */ d("input", {
					name: "username",
					placeholder: "Username",
					autoComplete: "username",
					autoFocus: !0,
					disabled: c
				}),
				/* @__PURE__ */ d("input", {
					name: "password",
					type: "password",
					placeholder: "Password",
					autoComplete: "current-password",
					disabled: c
				}),
				/* @__PURE__ */ d("button", {
					type: "submit",
					className: "btn-primary",
					disabled: c,
					children: c ? "Signing in…" : "Sign in"
				})
			]
		})
	})] });
}
//#endregion
//#region src/Auth/PrivateRoute.jsx
function Mo() {
	let { state: e } = Eo(), t = g();
	return e.status === "checking" ? /* @__PURE__ */ d("div", {
		style: {
			padding: 24,
			color: "#888"
		},
		children: "Checking session…"
	}) : e.status === "authed" ? /* @__PURE__ */ d(h, {}) : /* @__PURE__ */ d(m, {
		to: "/",
		replace: !0,
		state: { from: t }
	});
}
//#endregion
//#region src/components/EvictionsPanel/EvictionFeed.jsx
var No = 50;
function Po({ run: e, now: t, selected: n, expanded: r, onToggle: i }) {
	let a = e.done, o = a ? a.outcome || "fit" : null, s = qa(e), c = Ja(e), l = e.rows.filter((e) => e.kind === "skip"), u = ((e.endTs || t / 1e3) - e.startTs) * 1e3;
	return /* @__PURE__ */ f("div", {
		className: `evf-run${n && e.touched.has(n) ? " evf-hit" : ""}${a ? "" : " evf-running"}`,
		children: [/* @__PURE__ */ f("button", {
			className: "evf-line",
			onClick: i,
			title: e.failureDetail || (r ? "Collapse" : "Show the candidates this pass walked"),
			children: [
				/* @__PURE__ */ d("span", {
					className: `evf-caret${r ? " evf-caret-open" : ""}`,
					children: "›"
				}),
				/* @__PURE__ */ d("span", {
					className: `evf-model${e.incoming ? "" : " evf-model-sweep"}${n && e.incoming === n ? " evf-model-sel" : ""}`,
					children: e.incoming || "headroom sweep"
				}),
				e.failure ? /* @__PURE__ */ d("span", {
					className: "evf-victims evf-fail",
					title: e.failureDetail || e.failure,
					children: e.failure
				}) : s.length > 0 ? /* @__PURE__ */ f("span", {
					className: "evf-victims",
					children: [/* @__PURE__ */ d("span", {
						className: "evf-arrow",
						children: "unloaded"
					}), s.map((e) => /* @__PURE__ */ d("span", {
						className: `evf-victim${n && e.model_key === n ? " evf-victim-sel" : ""}`,
						children: e.model_key
					}, e.key))]
				}) : /* @__PURE__ */ d("span", {
					className: "evf-victims evf-none",
					children: l.length > 0 ? `${l.length} protected, none unloaded` : "no eviction"
				}),
				/* @__PURE__ */ d("span", { className: "evf-spacer" }),
				c != null && /* @__PURE__ */ d("span", {
					className: "evf-freed",
					children: Pa(c)
				}),
				e.worker_id && /* @__PURE__ */ d("span", {
					className: "evf-chip",
					children: e.worker_id
				}),
				/* @__PURE__ */ d("span", {
					className: "evf-time",
					children: Ia(e.startTs)
				}),
				a ? /* @__PURE__ */ d("span", {
					className: `evf-outcome ${La[o] || "ev-out-unfit"}`,
					children: o
				}) : e.failure ? /* @__PURE__ */ d("span", {
					className: "evf-outcome ev-out-failed",
					children: "failed"
				}) : /* @__PURE__ */ d("span", {
					className: "evf-outcome ev-out-running",
					children: Fa(Math.max(0, u))
				})
			]
		}), r && /* @__PURE__ */ f("div", {
			className: "evf-detail",
			children: [
				e.trigger && /* @__PURE__ */ f("span", {
					className: "evf-dchip",
					children: ["trigger ", e.trigger]
				}),
				e.needBytes != null && /* @__PURE__ */ f("span", {
					className: "evf-dchip",
					children: ["needs ", Pa(e.needBytes)]
				}),
				e.rows.length === 0 && /* @__PURE__ */ d("div", {
					className: "evf-drow evf-dquiet",
					children: "no candidates walked"
				}),
				e.rows.map((e) => e.kind === "member" ? /* @__PURE__ */ f("div", {
					className: `evf-drow evf-dgroup${n && e.model_key === n ? " evf-drow-sel" : ""}`,
					title: e.reason || void 0,
					children: [
						/* @__PURE__ */ d("span", {
							className: "evf-dmark",
							children: "member"
						}),
						/* @__PURE__ */ d("span", {
							className: "evf-dchip-inline",
							children: e.group_key
						}),
						/* @__PURE__ */ d("span", {
							className: "evf-dkey",
							children: e.model_key
						}),
						e.as && /* @__PURE__ */ f("span", {
							className: "evf-dchip-inline",
							children: ["as ", e.as]
						}),
						e.reason && /* @__PURE__ */ d("span", {
							className: "evf-dwhy",
							children: e.reason
						})
					]
				}, e.key) : e.kind === "memberskip" ? /* @__PURE__ */ f("div", {
					className: `evf-drow evf-dquiet${n && e.model_key === n ? " evf-drow-sel" : ""}`,
					title: e.reason || void 0,
					children: [
						/* @__PURE__ */ d("span", {
							className: "evf-dmark",
							children: "skip"
						}),
						/* @__PURE__ */ d("span", {
							className: "evf-dkey",
							children: e.model_key
						}),
						/* @__PURE__ */ d("span", {
							className: "evf-dwhy",
							children: e.reason
						})
					]
				}, e.key) : e.kind === "provision" ? /* @__PURE__ */ f("div", {
					className: `evf-drow ${e.state === "failed" ? "evf-dfail" : e.state === "done" ? "evf-devict" : "evf-dinflight"}${n && e.model_key === n ? " evf-drow-sel" : ""}`,
					title: e.state === "failed" ? Va(e) : e.dest_path || void 0,
					children: [
						/* @__PURE__ */ d("span", {
							className: "evf-dmark",
							children: "prov"
						}),
						/* @__PURE__ */ d("span", {
							className: "evf-dchip-inline",
							children: e.source || "unknown"
						}),
						/* @__PURE__ */ d("span", {
							className: "evf-dkey",
							children: e.model_key
						}),
						e.state === "failed" && /* @__PURE__ */ f("span", {
							className: "evf-dwhy",
							children: [Ha(e) ? `${Ha(e)}: ` : "", Va(e)]
						}),
						e.state === "running" && /* @__PURE__ */ d("span", {
							className: "evf-dwhy",
							children: "fetching…"
						}),
						e.state === "done" && e.bytes != null && /* @__PURE__ */ d("span", {
							className: "evf-dwhy",
							children: Pa(e.bytes)
						}),
						e.state === "done" && e.duration_ms != null && /* @__PURE__ */ d("span", {
							className: "evf-dwhy",
							children: Fa(e.duration_ms)
						})
					]
				}, e.key) : e.kind === "resolve" ? /* @__PURE__ */ f("div", {
					className: `evf-drow evf-dfail${n && e.model_key === n ? " evf-drow-sel" : ""}`,
					title: e.resolved_path || void 0,
					children: [
						/* @__PURE__ */ d("span", {
							className: "evf-dmark",
							children: "resolve"
						}),
						/* @__PURE__ */ d("span", {
							className: "evf-dkey",
							children: e.model_key
						}),
						/* @__PURE__ */ d("span", {
							className: "evf-dwhy",
							children: Q(e.reason) || "could not resolve a path"
						})
					]
				}, e.key) : e.kind === "load" ? /* @__PURE__ */ f("div", {
					className: `evf-drow ${e.state === "failed" ? "evf-dfail" : e.state === "done" ? "evf-devict" : "evf-dinflight"}${n && e.model_key === n ? " evf-drow-sel" : ""}`,
					children: [
						/* @__PURE__ */ d("span", {
							className: "evf-dmark",
							children: "load"
						}),
						e.engine && /* @__PURE__ */ d("span", {
							className: "evf-dchip-inline",
							children: e.engine
						}),
						/* @__PURE__ */ d("span", {
							className: "evf-dkey",
							children: e.model_key
						}),
						e.state === "failed" && /* @__PURE__ */ d("span", {
							className: "evf-dwhy",
							children: Q(e.error) || "load failed"
						}),
						e.state === "running" && /* @__PURE__ */ d("span", {
							className: "evf-dwhy",
							children: "loading…"
						}),
						e.duration_ms != null && /* @__PURE__ */ d("span", {
							className: "evf-dwhy",
							children: Fa(e.duration_ms)
						})
					]
				}, e.key) : e.kind === "skip" ? /* @__PURE__ */ f("div", {
					className: `evf-drow evf-dskip${n && e.model_key === n ? " evf-drow-sel" : ""}`,
					children: [
						/* @__PURE__ */ d("span", {
							className: "evf-dmark",
							children: "skip"
						}),
						/* @__PURE__ */ d("span", {
							className: "evf-dkey",
							children: e.model_key
						}),
						/* @__PURE__ */ d("span", {
							className: Ra(e.tier),
							children: e.tier
						}),
						/* @__PURE__ */ d("span", {
							className: "evf-dwhy",
							children: Q(e.reason) || "no reason recorded"
						})
					]
				}, e.key) : e.kind === "victim" ? /* @__PURE__ */ f("div", {
					className: `evf-drow ${e.state === "failed" ? "evf-dfail" : e.state === "done" ? "evf-devict" : "evf-dinflight"}${n && e.model_key === n ? " evf-drow-sel" : ""}`,
					children: [
						/* @__PURE__ */ d("span", {
							className: "evf-dmark",
							children: e.state === "failed" ? "fail" : e.state === "done" ? "evict" : "…"
						}),
						/* @__PURE__ */ d("span", {
							className: "evf-dkey",
							children: e.model_key
						}),
						/* @__PURE__ */ d("span", {
							className: Ra(e.tier),
							children: e.tier
						}),
						e.state === "failed" && /* @__PURE__ */ d("span", {
							className: "evf-dwhy",
							children: Q(e.error)
						}),
						e.freed_bytes != null && /* @__PURE__ */ f("span", {
							className: "evf-dwhy",
							children: ["freed ", Pa(e.freed_bytes)]
						}),
						e.duration_ms != null && /* @__PURE__ */ d("span", {
							className: "evf-dwhy",
							children: Fa(e.duration_ms)
						})
					]
				}, e.key) : e.kind === "fitfail" ? /* @__PURE__ */ f("div", {
					className: "evf-drow evf-dquiet",
					children: [/* @__PURE__ */ d("span", {
						className: "evf-dmark",
						children: "unfit"
					}), /* @__PURE__ */ f("span", {
						className: "evf-dwhy",
						children: [
							"needs ",
							Pa(e.need),
							" · ",
							Pa(e.free),
							" free"
						]
					})]
				}, e.key) : e.kind === "reclaim" ? null : /* @__PURE__ */ f("div", {
					className: "evf-drow evf-dverdict",
					children: [
						/* @__PURE__ */ d("span", {
							className: "evf-dmark",
							children: "verdict"
						}),
						/* @__PURE__ */ d("span", {
							className: "evf-dkey",
							children: e.action
						}),
						e.reason != null && /* @__PURE__ */ d("span", {
							className: "evf-dwhy",
							children: Q(e.reason)
						})
					]
				}, e.key)),
				a && Q(a.reason) && /* @__PURE__ */ f("div", {
					className: "evf-drow evf-dquiet",
					children: [/* @__PURE__ */ d("span", {
						className: "evf-dmark",
						children: "why"
					}), /* @__PURE__ */ d("span", {
						className: "evf-dwhy",
						children: Q(a.reason)
					})]
				})
			]
		})]
	});
}
function Fo({ selectedModel: e = null }) {
	let [t, n] = l(!1), [r, i] = l(!1), [a, c] = l(() => /* @__PURE__ */ new Set()), [p, m] = l(() => Date.now()), { runs: h, conn: g, error: _, held: v, reconnect: y } = vo({ paused: t }), b = s(() => h.some((e) => !e.events.some((e) => e.stage === "headroom.done")), [h]);
	o(() => {
		if (!b) return;
		let e = setInterval(() => m(Date.now()), 1e3);
		return () => clearInterval(e);
	}, [b]);
	let x = s(() => {
		let t = [];
		for (let n = h.length - 1; n >= 0 && t.length < No; n--) {
			let i = Ka(h[n]);
			r && e && !i.touched.has(e) || t.push(i);
		}
		return t;
	}, [
		h,
		r,
		e
	]), S = s(() => e ? x.filter((t) => t.touched.has(e)).length : 0, [x, e]), C = (e) => c((t) => {
		let n = new Set(t);
		return n.has(e) ? n.delete(e) : n.add(e), n;
	});
	return /* @__PURE__ */ f("div", {
		className: "evf",
		children: [
			/* @__PURE__ */ f("div", {
				className: "evf-bar",
				children: [
					/* @__PURE__ */ d("span", {
						className: `ev-dot ev-dot-${g}`,
						title: `stream ${g}`
					}),
					/* @__PURE__ */ d("span", {
						className: "evf-title",
						children: "Evictions"
					}),
					e && /* @__PURE__ */ f("span", {
						className: "evf-sel",
						title: `Highlighting passes involving ${e}`,
						children: [e, /* @__PURE__ */ d("span", {
							className: "evf-sel-n",
							children: S
						})]
					}),
					/* @__PURE__ */ d("span", { className: "evf-spacer" }),
					e && /* @__PURE__ */ d("button", {
						className: `evf-btn${r ? " evf-btn-on" : ""}`,
						onClick: () => i((e) => !e),
						title: r ? "Showing only this model — click to see its neighbours again" : "Show only passes involving the selected model",
						children: "only this"
					}),
					/* @__PURE__ */ d("button", {
						className: `evf-btn${t ? " evf-btn-on" : ""}`,
						onClick: () => n((e) => !e),
						title: t ? "Resume — held events land in order; the stream never dropped" : "Freeze this view. The stream stays open and keeps collecting.",
						children: t ? `paused${v ? ` (${v})` : ""}` : "live"
					}),
					g === "disconnected" && /* @__PURE__ */ d("button", {
						className: "evf-btn",
						onClick: y,
						children: "reconnect"
					})
				]
			}),
			_ && /* @__PURE__ */ d("div", {
				className: "evf-err",
				children: _
			}),
			/* @__PURE__ */ d("div", {
				className: "evf-list",
				children: x.length === 0 ? /* @__PURE__ */ d("div", {
					className: "evf-empty",
					children: r && e ? /* @__PURE__ */ f(u, { children: [
						"No eviction passes have involved ",
						/* @__PURE__ */ d("code", { children: e }),
						" yet."
					] }) : /* @__PURE__ */ d(u, { children: "No eviction activity yet. Passes appear here the moment a worker makes room — call a model from chat and watch it land." })
				}) : x.map((t) => /* @__PURE__ */ d(Po, {
					run: t,
					now: p,
					selected: e,
					expanded: a.has(t.id),
					onToggle: () => C(t.id)
				}, t.id))
			})
		]
	});
}
//#endregion
//#region src/components/ChatPanel/useChats.js
var Io = "hugpy.chats.v1", Lo = [];
function Ro() {
	try {
		return JSON.parse(localStorage.getItem(Io)) || {};
	} catch {
		return {};
	}
}
function zo(e) {
	let t = {};
	for (let [n, r] of Object.entries(e)) !Array.isArray(r) || r.length === 0 || (t[n] = r.map((e) => {
		let { status: t, ...n } = e;
		return n.attachment?.dataUrl && (n.attachment = {
			...n.attachment,
			dataUrl: void 0
		}), n;
	}));
	return t;
}
function Bo() {
	let [e, t] = l(Ro);
	return o(() => {
		try {
			localStorage.setItem(Io, JSON.stringify(zo(e)));
		} catch {}
	}, [e]), {
		chats: e,
		getMessages: i((t) => e[t] || Lo, [e]),
		setMessages: i((e, n) => {
			t((t) => {
				let r = t[e] || Lo, i = typeof n == "function" ? n(r) : n;
				return {
					...t,
					[e]: i
				};
			});
		}, []),
		clearChat: i((e) => {
			t((t) => {
				if (!(e in t)) return t;
				let n = { ...t };
				return delete n[e], n;
			});
		}, [])
	};
}
function Vo(e = {}) {
	return !1;
}
//#endregion
//#region src/App/App.jsx
function Ho({ banner: e = null }) {
	let [t, n] = l([]), [r, a] = l(!0), [s, u] = l(null), [p, m] = l(() => {
		try {
			return localStorage.getItem("hugpy.activeChat") || null;
		} catch {
			return null;
		}
	}), [h, g] = l({}), [_, v] = l([]), y = typeof window < "u" && window.matchMedia("(max-width: 640px)").matches, [b, x] = l(y), [S, C] = l(() => {
		let e = y ? "overview" : "status";
		try {
			let t = localStorage.getItem("hugpy.activeTab");
			return [
				"overview",
				"status",
				"compute",
				"models",
				"add",
				"api",
				"nodes",
				"evictions",
				"settings"
			].includes(t) ? t : e;
		} catch {
			return e;
		}
	});
	o(() => {
		try {
			localStorage.setItem("hugpy.activeTab", S);
		} catch {}
	}, [S]), o(() => {
		let e = window.matchMedia("(max-width: 640px)"), t = (e) => x(e.matches);
		return e.addEventListener("change", t), () => e.removeEventListener("change", t);
	}, []), o(() => {
		!b && S === "overview" && C("status");
	}, [b, S]);
	let { signOut: w } = Eo(), { chats: T, getMessages: E, setMessages: D } = Bo(), O = i((e) => {
		p && D(p, e);
	}, [p, D]);
	o(() => {
		try {
			p ? localStorage.setItem("hugpy.activeChat", p) : localStorage.removeItem("hugpy.activeChat");
		} catch {}
	}, [p]);
	let k = i(() => {
		K("/api/models").then((e) => {
			n(e), a(!1), u(null);
		}).catch((e) => {
			u(e.message), a(!1);
		});
	}, []), A = i(() => {
		K("/api/llm/workers").then((e) => v(Array.isArray(e) ? e : [])).catch(() => {});
	}, []);
	o(() => {
		k();
	}, [k]), o(() => {
		A();
		let e = setInterval(A, 1e4);
		return () => clearInterval(e);
	}, [A]);
	let j = i((e, t) => {
		K(`/api/llm/workers/${encodeURIComponent(e.id)}/assign`, {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({ model_key: t })
		}).then(A).catch((e) => alert(`Assign failed: ${e.message}`));
	}, [A]), M = i(async (e, t) => {
		try {
			return await K(`/api/llm/workers/${encodeURIComponent(e.id)}/probe`, {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({ model_key: t })
			});
		} catch (e) {
			return {
				ok: !1,
				fit: !1,
				error: e.message
			};
		}
	}, []);
	o(() => {
		let e = Object.values(h).filter((e) => e.status === "queued" || e.status === "running");
		if (!e.length) return;
		let t = setInterval(() => {
			e.forEach((e) => {
				K(`/api/jobs/${e.id}`).then((e) => {
					g((t) => ({
						...t,
						[e.id]: {
							...t[e.id],
							...e
						}
					})), e.status === "completed" && k();
				}).catch(() => {});
			});
		}, 1e3);
		return () => clearInterval(t);
	}, [h, k]);
	let N = i((e) => {
		K(`/api/models/${encodeURIComponent(e)}/download`, { method: "POST" }).then((t) => g((n) => ({
			...n,
			[t.id]: {
				...t,
				model_key: t.model_key ?? e
			}
		}))).catch((e) => alert(`Download failed: ${e.message}`));
	}, []), P = i((e) => {
		m(e);
	}, []), [F, I] = Y("hugpy.models.feedPct", 32), L = Number.isFinite(Number(F)) ? Math.max(0, Math.min(85, Number(F))) : 32, R = c(null), z = i((e) => {
		e.preventDefault();
		let t = R.current;
		if (!t) return;
		let n = (e) => {
			let n = t.getBoundingClientRect();
			if (!n.height) return;
			let r = (n.bottom - e.clientY) / n.height * 100;
			I(r < 6 ? 0 : Math.min(85, r));
		}, r = () => {
			window.removeEventListener("mousemove", n), window.removeEventListener("mouseup", r), document.body.style.cursor = "", document.body.style.userSelect = "";
		};
		window.addEventListener("mousemove", n), window.addEventListener("mouseup", r), document.body.style.cursor = "row-resize", document.body.style.userSelect = "none";
	}, [I]), B = i((e) => {
		confirm(`Delete downloaded files for ${e}?`) && K(`/api/models/${encodeURIComponent(e)}`, { method: "DELETE" }).then(() => k()).catch((e) => alert(`Delete failed: ${e.message}`));
	}, [k]), V = i((e) => {
		confirm(`Prune "${e}" from the model registry?\n\nThis removes the catalog entry (no files are on disk). Re-discovery or re-adding the model brings it back.`) && K(`/api/models/${encodeURIComponent(e)}/prune`, { method: "POST" }).then(() => k()).catch((e) => alert(`Prune failed: ${e.message}`));
	}, [k]), ee = i((e, t) => {
		n((n) => n.map((n) => (n.model_key ?? n.key) === e ? {
			...n,
			media: t
		} : n)), K(`/api/models/${encodeURIComponent(e)}/media`, {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({ enabled: t })
		}).catch((r) => {
			alert(`Media toggle failed: ${r.message}`), n((n) => n.map((n) => (n.model_key ?? n.key) === e ? {
				...n,
				media: !t
			} : n));
		});
	}, []), te = i((e, t) => {
		n((n) => n.map((n) => (n.model_key ?? n.key) === e ? {
			...n,
			media_default: t
		} : t && n.media_default ? {
			...n,
			media_default: !1
		} : n)), K(`/api/models/${encodeURIComponent(e)}/media-default`, {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({ default: t })
		}).catch((e) => {
			alert(`Default media model failed: ${e.message}`), k();
		});
	}, [k]), H = i((e) => {
		K(`/api/jobs/${e}/cancel`, { method: "POST" }).then(() => {
			g((t) => t[e] ? {
				...t,
				[e]: {
					...t[e],
					status: "cancelled"
				}
			} : t);
		}).catch((e) => alert(`Cancel failed: ${e.message}`));
	}, []), ne = i((e) => {
		K(`/api/jobs/${e}/retry`, { method: "POST" }).then((t) => {
			if (t && t.retried === !1) {
				alert(`Retry failed: ${t.reason || "unknown"}`);
				return;
			}
			g((t) => t[e] ? {
				...t,
				[e]: {
					...t[e],
					status: "running",
					error: null,
					stalled: !1
				}
			} : t);
		}).catch((e) => alert(`Retry failed: ${e.message}`));
	}, []), re = i((e) => {
		let t = e.id ?? e.job_id;
		t && (g((n) => ({
			...n,
			[t]: {
				...e,
				id: t,
				model_key: e.model_key,
				hub_id: e.hub_id,
				status: e.status ?? "queued"
			}
		})), k());
	}, [k]), U = {};
	Object.values(h).forEach((e) => {
		e.model_key && (U[e.model_key] = e);
	});
	let W = {};
	Object.values(h).forEach((e) => {
		e.hub_id && (W[e.hub_id] = e);
	});
	let G = {};
	return Object.values(h).forEach((e) => {
		e.hub_id && (G[e.hub_id] = !0);
	}), t.forEach((e) => {
		e.hub_id && e.status === "installed" && (G[e.hub_id] = !0);
	}), /* @__PURE__ */ f("div", {
		className: "layout",
		children: [
			/* @__PURE__ */ f(be, {
				banner: e,
				children: [/* @__PURE__ */ d("button", {
					onClick: () => Vo(),
					title: "Open help",
					className: "hugpy-navbar-action",
					children: "Help"
				}), /* @__PURE__ */ d("button", {
					onClick: w,
					title: "Sign out of the console",
					className: "hugpy-navbar-action",
					children: "Sign out"
				})]
			}),
			!b && /* @__PURE__ */ d($i, {
				models: t,
				workers: _
			}),
			/* @__PURE__ */ d("nav", {
				className: "tabbar",
				role: "tablist",
				children: [
					...b ? [{
						id: "overview",
						label: "Overview",
						badge: null
					}] : [],
					{
						id: "status",
						label: "Status",
						badge: null
					},
					{
						id: "compute",
						label: "Compute",
						badge: _.filter((e) => e.status === "online").length || null
					},
					{
						id: "models",
						label: p ? "Models 💬" : "Models",
						badge: t.filter((e) => e.status === "installed").length || null
					},
					{
						id: "add",
						label: "Add models 🤗",
						badge: null
					},
					{
						id: "api",
						label: "API",
						badge: null
					},
					{
						id: "nodes",
						label: "Agent Nodes",
						badge: null
					},
					{
						id: "evictions",
						label: "Evictions",
						badge: null
					},
					{
						id: "settings",
						label: "Settings",
						badge: null
					}
				].map((e) => /* @__PURE__ */ f("button", {
					role: "tab",
					"aria-selected": S === e.id,
					className: `tab ${S === e.id ? "tab-active" : ""}`,
					onClick: () => C(e.id),
					children: [e.label, e.badge != null && /* @__PURE__ */ d("span", {
						className: "tab-badge",
						children: e.badge
					})]
				}, e.id))
			}),
			/* @__PURE__ */ f("div", {
				className: "tab-body",
				children: [
					S === "overview" && b && /* @__PURE__ */ d("div", {
						className: "tab-pane",
						children: /* @__PURE__ */ d($i, {
							models: t,
							workers: _
						})
					}),
					S === "models" && /* @__PURE__ */ f("div", {
						className: "body",
						children: [/* @__PURE__ */ f("section", {
							className: "models-pane",
							"data-chat-open": !!p,
							children: [
								/* @__PURE__ */ d(fi, { models: t }),
								/* @__PURE__ */ d(mi, { workers: _ }),
								/* @__PURE__ */ f("div", {
									className: "models-pane-split-host",
									ref: R,
									children: [
										/* @__PURE__ */ f("div", {
											className: "models-pane-list",
											style: { height: `${100 - L}%` },
											children: [
												r && /* @__PURE__ */ d("div", {
													className: "placeholder",
													children: "Loading models…"
												}),
												s && /* @__PURE__ */ f("div", {
													className: "placeholder error",
													children: ["API error: ", s]
												}),
												!r && !s && /* @__PURE__ */ d(Xn, {
													models: t,
													jobsByModel: U,
													activeChat: p,
													onDownload: N,
													onChat: P,
													onDelete: B,
													onPrune: V,
													onSetMedia: ee,
													onSetMediaDefault: te,
													onCancel: H,
													onRetry: ne,
													workers: _,
													onAssignWorker: j,
													onProbeWorker: M,
													onRefresh: k
												})
											]
										}),
										/* @__PURE__ */ d("div", {
											className: "models-split",
											role: "separator",
											"aria-orientation": "horizontal",
											"aria-label": "Resize the eviction feed",
											title: "Drag to resize · double-click to collapse",
											onMouseDown: z,
											onDoubleClick: () => I((e) => e <= 4 ? 32 : 0),
											children: /* @__PURE__ */ d("span", { className: "models-split-grip" })
										}),
										/* @__PURE__ */ d("div", {
											className: "models-pane-feed",
											style: { height: `${L}%` },
											children: /* @__PURE__ */ d(Fo, { selectedModel: p })
										})
									]
								})
							]
						}), p && /* @__PURE__ */ d(Xt, {
							modelKey: p,
							model: t.find((e) => (e.model_key ?? e.key) === p),
							messages: E(p),
							setMessages: O,
							onClose: () => m(null),
							chats: T,
							models: t,
							onSwitchChat: P
						}, p)]
					}),
					S === "add" && /* @__PURE__ */ d("div", {
						className: "tab-pane",
						children: /* @__PURE__ */ d(Sn, {
							embedded: !0,
							models: t,
							onJobStarted: re,
							onCancelJob: H,
							onRetryJob: ne,
							pendingByHub: G,
							jobsByHub: W
						})
					}),
					S === "compute" && /* @__PURE__ */ f("div", {
						className: "tab-pane",
						children: [/* @__PURE__ */ d(si, { models: t }), /* @__PURE__ */ d(bi, {})]
					}),
					S === "status" && /* @__PURE__ */ d("div", {
						className: "tab-pane",
						children: /* @__PURE__ */ d(zi, {
							models: t,
							workers: _
						})
					}),
					S === "evictions" && /* @__PURE__ */ d("div", {
						className: "tab-pane",
						children: /* @__PURE__ */ d(xo, {})
					}),
					S === "settings" && /* @__PURE__ */ d("div", {
						className: "tab-pane",
						children: /* @__PURE__ */ d(Na, { workers: _ })
					}),
					S === "api" && /* @__PURE__ */ f("div", {
						className: "tab-pane",
						children: [
							/* @__PURE__ */ d(Me, { models: t }),
							/* @__PURE__ */ d(aa, { models: t }),
							/* @__PURE__ */ d(ua, { models: t }),
							/* @__PURE__ */ d(ga, {})
						]
					}),
					S === "nodes" && /* @__PURE__ */ d("div", {
						className: "tab-pane",
						children: /* @__PURE__ */ d(Mi, {})
					})
				]
			})
		]
	});
}
//#endregion
//#region src/HugpyConsole.tsx
function Uo(e) {
	return /* @__PURE__ */ d(G, {
		...e,
		children: /* @__PURE__ */ d(Ho, {})
	});
}
//#endregion
export { Mi as AgentNodesPanel, Me as ApiAccess, To as AuthProvider, ua as BridgePanel, Ao as ChangePassword, Xt as ChatPanel, aa as DiscordPanel, Sn as HFSearch, Uo as HugpyConsole, G as HugpyProvider, Kt as Landing, Do as Login, jo as LoginForm, Oo as Logout, Xn as ModelTable, Qn as PeersBar, bi as PhoneBrickPanel, Mo as PrivateRoute, ko as Register, si as WorkersPanel, z as configureHugpy, K as fetchJson, he as getAuthBase, me as getAuthConfig, B as getHugpyConfig, U as hugpyFetch, V as resetHugpyConfig, ne as resolveApiOrigin, H as resolveApiUrl, se as uploadFile, Eo as useAuth, ie as useHugpyConfig };

//# sourceMappingURL=hugpy-ui.mjs.map