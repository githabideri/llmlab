# Strata — reading what the engine reports

**Status:** Active (added 2026-10-09)

Strata is the sparse-expert engine serving the flash-next class models. Launch flags are
serving facts and belong on the model card; this doc owns the *semantics* of what the engine
reports, and the project's supply-chain rules for running it, so a card can point here
instead of carrying an explanation.

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

## Security & supply chain

The official distribution path (`bash <(curl … update.sh)`, then `python3 setup.py
--install`) has three supply-chain properties worth knowing before using it; the October 2026
critique of the project ([the "maybe wait on this one" video](https://www.youtube.com/watch?v=i18vpg0T-mk),
Unbiased Bob, and the follow-ups it triggered) made them explicit:

1. **`update.sh` is a moving target by design.** It fetches the newest release binary at run
   time, the default is *auto-update on* (`STRATA_NO_AUTO_UPDATE=0`), and the model weights
   come from whatever repo string is in the installer at that moment
   (`huggingface.co/Strata-ML/…`). No hash pinning, no TOFU check.
2. **`setup.py` executes on your host during install.** It grew from ~1,700 to ~7,000 lines in
   a few weeks — the whole codebase effectively ships through this one file. The maintainer
   acknowledged the criticism: "setup.py has become a remote code execution channel. That is
   … an oversight. It should be broken into smaller pieces and the install step should be
   pinned to a hash."
3. **The git history was rewritten 2026-10-06** ([issue #1276](https://github.com/Niko1221/Strata/issues/1276):
   "Git history was rewritten. All PRs/commits are gone") — the same day the v0.1.41 model
   line was announced. A clone on the old history cannot fetch-upgrade; a version move is a
   deliberate re-branch.

Two further points from the video, verified against our builds: the MCP client ships in the
binary (never configured here; its egress code is inert), and an image-URL fetch path exists
(wired but unused — we serve local models only).

### How this project runs Strata (both 3060 deployments)

None of the three properties above is active:

- **Pinned source, compiled locally.** Both boxes build the engine from a source clone at an
  exact commit (the v0.1.39/v0.1.41 tags, and for the 35B box the branch merge that carries
  the v0.1.41 main) with in-tree CMake builds; the build logs are kept. No release binary is
  ever downloaded.
- **Local modifications are declared and minimal.** The only changes are a couple of
  documented `setup.py` lines (skipping the ~36 GB expert-weight staging where the box can't
  fit it); the unmodified original is kept alongside. Nothing else in the tree is touched.
- **No auto-update, ever** (`STRATA_NO_AUTO_UPDATE=1`). A version move is a deliberate,
  logged action (new clone, rebuild, re-test), never a background fetch.
- **No network exposure.** The engine API binds to loopback only; where a service needs a
  front door it is a local proxy with its own random API key. The weights are local files;
  the runtime makes no outbound calls in this configuration.

### If you are running the official install

You inherit the trust model above: the installer's word is the supply chain. Mitigations in
order of effect: build from a pinned tag commit instead of `update.sh`; disable auto-update;
keep your own auth in front of the API; and treat every update as a re-decision, because the
project's history has already been rewritten once. The video's author maintains a fork
(`UnbiasedStrata`) as an alternative distribution.

**Status as of 2026-10-10:** the maintainer has acknowledged the issues publicly and promised
the pinned-hash install; no fixing commits had landed yet (upstream's newest commit still
predates the video). Check the upstream repo before assuming the official path is improved.

**The llama.cpp side, checked (2026-10-10):** mainline has merged the one community PR aimed at this class of engine — #29887, "a GPU cache for MoE experts kept in host memory" (2026-10-07): an LRU cache for host-resident experts, only misses uploaded, small batches only, off by default (`--moe-cache-mib N`). It cannot be deployed on a 12 GB card at 128K context (no VRAM headroom — see the
[10-10 Re-A/B report](../reports/2026-10-10-llamacpp-mainline-rebuild-35b-a3b-single-3060.md)), which is part of why the Strata card on such a box still earns its keep.
