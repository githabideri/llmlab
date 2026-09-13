# Failure corpus

The regression suite for `classifier/` (replayed by `qualify.py`, Q14, on every
prepare). One file per failure *shape*; the paired `.expect` names the class the
classifier must return.

## Substitution table (sanitization)

Real logs are excerpted, never pasted wholesale. Replacements applied:

| real (private) | corpus form |
|---|---|
| any internal build path | `/opt/<build>/` |
| any model store path | `/mnt/<models>/` |
| internal hostnames / LXC ids / IPs | never present (excerpt to the error line only) |

## Provenance per fixture

| fixture | source |
|---|---|
| meta-assert-then-oom.log | real sanitized excerpt, 2026-09-12 B1 (meta-backend assert preceding the OOM) |
| oom-25gb.log | real sanitized excerpt shape, 2026-09-10 documented wall (25.4 GB single-device allocation, no higher-precedence marker) |
| oom-8gib.log | reconstructed (MiB-unit OOM below the 24 GB wall threshold; the 09-12 matcher was unit-blind for exactly this) |
| cgroup-oom.log | reconstructed from the 2026-09-10 container oom-kill (kernel format stable) |
| unsupported-arch.log | real sanitized excerpt, 2026-09-12 B0 control cell |
| xid.log | reconstructed (NVRM Xid kernel format is stable) |
| model-load-fail.log | real sanitized excerpt shape, 2026-09-12 B0 tail |
| unknown-garbage.log | synthetic (the UNKNOWN path must exist and page the owner) |
| health-empty200.resp | shape fixture: vLLM >=0.2 /health is an empty 200 (09-02/09-10 body-grepping probes) |
| sse-malformed.resp | shape fixture (09-02 stage-1 SSE parsing bugs) |
| usage-absent.resp | shape fixture (09-10: no usage object -> token closure unavailable) |
