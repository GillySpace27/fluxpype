# Plumber: own address and Dock icon (scope)

Status: scope draft 1 (2026-11-10). Builds on Fluxpype Plumber (PR #47).

## 1. Goal

1. Open Plumber at a name of its own, not `localhost:<port>`.
2. Open it from a Dock icon. Per Gilly, the icon is a thin installed link to the running web server, not a second app to maintain.

## 2. Recommendation

Three small pieces. Together they need no new codebase:

- **Addresses:** `http://plumber.localhost:8765` for real runs and `http://demo.plumber.localhost:8775` for the demo.
- **Always-on server:** a macOS LaunchAgent (launchd) keeps the real Plumber running, so the address always answers.
- **Dock icon:** Plumber serves a web-app manifest and icons. Then Safari's **File > Add to Dock** (macOS 14 and later) or Chrome's **Install app** makes a Dock icon that opens Plumber in its own window. The browser maintains that icon, not us.

One command sets it all up: `flux-plumb --install`. `flux-plumb --uninstall` reverses it.

## 3. Problems found while scoping (fix these first, whatever we choose)

These come from reading the current code.

1. **The token changes on every restart.** `cli.py` creates a new `secrets.token_urlsafe(24)` at each start, and the browser cookie stores the old one. A fixed Dock link would therefore hit "unauthorized" after every restart. Fix: reuse the existing `<home>/token` when it is present and has mode 0600, and add `flux-plumb --rotate-token` for when you want a new one.
2. **Demo and real Plumber overwrite each other's login.** Cookies are scoped to the host name, not the port. On `localhost:8765` and `localhost:8766` both instances set the same `flux_plumber` cookie, so logging into one logs you out of the other. That is happening on your Mac right now. Distinct host names (`plumber.localhost` and `demo.plumber.localhost`) fix it. Also give the cookie name a per-home suffix as a backstop.
3. **The Host check would refuse the new names.** `create_app` only allows `localhost`, `127.0.0.1` and `::1`. Add `plumber.localhost` and `*.plumber.localhost`. These still resolve only to loopback, so this doesn't loosen the DNS-rebinding guard.
4. **A bare link has no token.** The first visit to a new address has no cookie. Handle it with a small login page that explains where to find the token, plus a one-time `flux-plumb --open` that opens the token URL so the cookie is set. Keep the cookie lifetime at 30 days and refresh it on each visit.

## 4. Part 1: a unique address

| Option | Setup | Verdict |
|---|---|---|
| `plumber.localhost:<port>` | None. RFC 6761 reserves `.localhost` for loopback, and Chrome and Firefox resolve any `*.localhost` to 127.0.0.1 without touching DNS. | **Recommended.** I have not checked whether Safari resolves `*.localhost` subdomains; verify on your Mac before relying on it in Safari. |
| `plumber.localhost` with no port (port 80) | Recent macOS lets non-root processes bind ports below 1024 (to verify). Two instances on port 80 would need a router, and port 80 collides with anything else you run. | Optional later step. Not worth it for v1. |
| `/etc/hosts` name such as `plumber.local` | Needs sudo. `.local` belongs to Bonjour (mDNS), and lookups can stall for seconds. | Avoid. |
| HTTPS with a local CA (mkcert) | Installs a root certificate. | Not needed: browsers already treat `localhost` and its subdomains as a secure context, which is what installable web apps require. |

Ports: real 8765 and demo 8775, both fixed and recorded in `server.json`. When a port is busy, Plumber should exit with a clear error rather than move silently to another port, because a moved port breaks the Dock link.

## 5. Part 2: the Dock icon

| Option | What we maintain | Not-running behavior | Verdict |
|---|---|---|---|
| **Web-app manifest + browser install** (Safari Add to Dock, Chrome Install app) | `manifest.webmanifest`, three PNG icons and a theme color, all shipped in `web/dist` | The browser shows a connection error. The LaunchAgent prevents that. | **Recommended**, together with the LaunchAgent. |
| Generated launcher `.app` (`flux-plumb --install-app` writes a bundle in `~/Applications` holding a shell script, `Info.plist` and an `.icns`) | One template script | It can start Plumber first, then open the URL. | Good fallback if you'd rather not run Plumber all the time. It opens a browser tab, or a chromeless window via Chrome's `--app=` flag (to verify). |
| Native wrapper (Electron, Tauri, pywebview) | A real second app: builds, signing, updates | Can manage the server | Rejected: this is the separate software you said you don't want. |

**LaunchAgent details:**
- `~/Library/LaunchAgents/com.gillyspace27.fluxpype.plumber.plist`, started with `RunAtLoad` and `KeepAlive` (on crash only).
- It runs `.venv-plumber/bin/python -m fluxpype.plumber.cli --headless --port 8765 --python ~/vscode/fluxons/.venv/bin/python`.
- Environment: launchd does not read your shell profile, so `--install` copies the PATH and PERL5LIB that FLUX needs into the plist. That puts the fluxenv Perl ahead of Homebrew's. It records them in `<home>/install.json` so `--install` can be rerun after an environment change.
- Logs go to `<home>/server.log`. Detached jobs are unaffected by restarts, as now.
- The demo is not installed as an agent. It stays something you start by hand.

## 6. Work

| Step | Content | Size (estimate) |
|---|---|---|
| A0 | Stable token, per-home cookie name, Host allowlist for `*.plumber.localhost`, login page, `--open`, `--rotate-token`, with tests | ~200 LOC |
| A1 | Manifest, icons (rendered from the existing sun logo at 192, 512 and maskable 512), `apple-touch-icon` and theme color | ~60 LOC plus assets |
| A2 | `flux-plumb --install` / `--uninstall` (LaunchAgent plist, captured environment, `launchctl bootstrap` / `bootout`) and `--status`; tests run on plist generation only | ~200 LOC |
| A3 | Optional `--install-app` launcher bundle | ~120 LOC |
| A4 | README "Run Plumber like an app" section, with the `cd` shown | docs |

A0 is worth doing even if nothing else is: problems 1 and 2 already affect the two Plumbers running on your Mac.

## 7. Open questions (defaults in brackets)

- Should Plumber start at login? [Yes, via the LaunchAgent. The cost is one idle Python process.]
- Safari or Chrome for the Dock icon? [Whichever you use. The manifest serves both.]
- Name in the Dock and window title? ["Plumber". "Fluxpype Plumber" gets cut off under the icon.]
- Should the demo get its own Dock icon? [No.]

## 8. Risks

- The Safari behavior with `*.localhost` and the low-port rule are unverified (marked above). A1 and A2 do not depend on either.
- An always-on scheduler will start queued jobs whenever the Mac is awake. That's intended, but the queue's pause switch is the off button.
- A captured PATH goes stale if the FLUX environment moves. `--status` reports when the interpreter or Perl in `install.json` is missing.
