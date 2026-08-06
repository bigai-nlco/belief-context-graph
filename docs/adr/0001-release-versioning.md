# ADR-0001: Release versioning and Dashboard packaging

- Date: 2026-08-06
- Status: Accepted
- Decision makers: project owner (jun)

## Context

BCG ships three components: the Python SDK/runtime (`bcg`), the terminal
Agent (`bcg-agent`, npm package), and the graph Dashboard (Vite app). The
refactor plan (step 14) requires a release versioning policy and a decision
on whether the Dashboard is a release artifact.

## Decision 1: Version strategy — lockstep majors + release manifest

- The three components share the **same major/minor release line**
  (currently `1.0.x`); the published version numbers stay in lockstep.
- **Patch versions may advance independently** for hotfixes.
- Every release generates a `release-manifest.json` recording the exact
  component combination: Python version, Agent version, contract
  `schema_version`, Dashboard version, and lockfile state. A release is
  traceable to exactly one combination.
- Rationale: the components are developed and shipped together, have no
  independent release history, and the cross-language contract
  (`contracts/`) already ties their interfaces together. Independent
  semver + compatibility matrix would add bookkeeping without users.

## Decision 2: Dashboard — release artifact, deployed independently

- The Dashboard **is a release artifact**: its version enters the release
  manifest and its static bundle is produced by the release pipeline.
- It is **not installed by `install.sh` or `make install`**: it is deployed
  independently (static hosting or `npm run preview` / `vite preview`).
- `scripts/release-manifest.py` records the Dashboard version and the
  bundle build state; the dashboard remains a dev tool inside the repo
  (`make dev-dashboard` / `npm run dev`) and a previewable bundle
  (`npm run build`).

## Consequences

- `make release` (or the release pipeline) must run
  `scripts/release-manifest.py` and commit/attach the manifest.
- Lockfile drift (uncommitted `uv.lock` / `package-lock.json` changes)
  fails the manifest check.
- install.sh stays Python + Agent only; Dashboard deployment is documented
  separately (dashboard/README.md).
