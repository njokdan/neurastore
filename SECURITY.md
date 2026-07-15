# Security Policy

## Reporting a vulnerability

Please **do not** open a public GitHub issue for a security
vulnerability. Instead, report it privately via GitHub's
[private vulnerability reporting](https://github.com/njokdan/neurastore/security/advisories/new)
(Security tab → Report a vulnerability).

Include, if you can:
- A description of the vulnerability and its potential impact
- Steps to reproduce it, or a proof-of-concept
- The version/commit you tested against

You should get an acknowledgment within a few days. This is currently a
small project maintained without a dedicated security team, so please
be patient — but every report will be taken seriously and credited
(unless you'd prefer otherwise) once a fix ships.

## Current security posture — read this before deploying

NeuraStore's security model is opt-in, not on-by-default, and it's
important to understand what that means before running it anywhere
beyond local development:

- **No authentication by default.** `NEURASTORE_API_KEYS` must be
  explicitly set, or anyone who can reach the server has full
  read/write/delete access. The server logs a clear warning either way
  at startup — check your logs.
- **No TLS by default.** Traffic is plain HTTP unless you put a
  TLS-terminating reverse proxy in front (see `deploy/Caddyfile` and
  `README.md`'s TLS section). Don't expose an un-proxied instance to
  the public internet.
- **No rate limiting by default.** Set `NEURASTORE_RATE_LIMIT_RPS` if
  you need protection against runaway or abusive clients.
- **Collection names are validated** against path traversal (letters,
  digits, underscore, hyphen only) — this one *is* always on, since a
  collection name becomes a directory name on disk.
- **Anomaly detection is advisory only.** Even enabled, it flags
  unusual request patterns to a log for human review — it never blocks
  a request on its own.

If you're running this anywhere other than local development, the
minimum recommended configuration is: `NEURASTORE_API_KEYS` set, a
reverse proxy providing TLS in front, and `NEURASTORE_RATE_LIMIT_RPS`
set to something appropriate for your expected traffic.

## Scope

This policy covers the NeuraStore server (`src/`), the Python client
and CLI (`client/python/`), and the provided deployment configuration
(`Dockerfile`, `docker-compose*.yml`, `deploy/`). Vulnerabilities in
upstream dependencies (Rust crates, Python packages) should generally
be reported to those projects directly, unless NeuraStore's specific
usage of them creates the vulnerability.
