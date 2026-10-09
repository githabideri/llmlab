# Strata — reading what the engine reports

**Status:** Active (added 2026-10-09)

Strata is the sparse-expert engine serving the flash-next class models. Launch flags are
serving facts and belong on the model card; this doc owns the *semantics* of what the engine
reports, so a card can point here instead of carrying an explanation.

## `spec` is the effective window, not the flag

The engine mutates the parsed `--spec` value at startup when the prompt/suffix drafter is
active (on by default, `--suffix-draft 3`). With `mtp_max_t == 0` it takes `mtp_max_t = spec`
and widens the verify window to `min(spec + 2, 8)` (`src/program/generate.cpp` L2192-2199,
v0.1.41). So a launch line of `--spec 4` reports `spec: 6` in `/metrics`.

Read the reported number as the effective window, the flag as the request. A reported `spec`
larger than the flag is the default widening, not config drift — confirm by checking that the
reported `mtp_max` equals the flag value and `lookup` equals the drafter default.

`--lookup-chain K` widens further (`max(spec, min(mtp_max_t + K, 8))`). Note the field naming
trap: the metrics `lookup` field is `suffix_draft`, not `lookup_chain`.

## The other fields

| Field | What it is | What it is evidence of |
|---|---|---|
| `arena_mib` | arena memory in use | whether the resident budget is being consumed as intended |
| `expert_slots` | resident expert slots | ladder fill — compare with the model's expert count, not with a target |
| `vram_free_mib` | free VRAM | headroom for the next tier; near zero means the ladder is at its ceiling |
| `pcie_frac` | PCIe transfer fraction | how much work is spilling off-card — the number that separates a resident run from a spilling one |

These are engine-reported state, not measured performance. Measured performance belongs on the
card, with the data file behind it.
