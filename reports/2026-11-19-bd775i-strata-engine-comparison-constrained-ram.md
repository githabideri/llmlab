# BD775i Strata engine comparison at constrained RAM (report C)

## Question

Would a newer Strata engine prefill faster in the 48 GB two-session
regime than the installed v0.1.41? The campaign's first candidate
engines: v0.1.45 (the release that added `STRATA_FILE_CACHE_OLD_RULE`
and `STRATA_RAM_ADAPT`) and the then-current 0.2.x line (113 releases
on, ~5,400 commits past v0.1.45, layer-major prefill in the DeepSeek
engine, two-GPU layer split, the RAM-adaptive tier).

## Ground rules

Same model, same GGUF shards, same llama.cpp pin (3cf03257, the commit
v0.1.41 uses; v0.1.45's default is 1.72.1 and its own pin moved, which
would change CUDA behavior independently of the Strata changes), same
24 GiB pin, same MTP, same KV config, same prompts, exclusive window,
median of three. Both challengers are pinned Nix derivations
(v0.1.45 built from its tag; the 0.2.x line not built: see below),
independently available; the deployed binary was never overwritten.

## Result

| arm | 32K prefill (n=3) | 64K prefill (n=3) | solo decode D8 |
|---|---|---|---|
| v0.1.41 (deployed) | 229 / 233 / 233 tps | 229 / 233 / 230 tps | 9.1 to 10.8 tps |
| v0.1.45, same llama.cpp pin | 225 / 230 / 233 tps | 213 / 217 / 214 tps | 9.0 to 10.4 tps |

v0.1.45 is within noise at 32K and 7 percent slower at 64K (all
buffered-mode numbers; the gap is the same order in direct mode). The
delta is consistent across three runs, so it is real but small:
something in the 0.1.42 to 0.1.45 window changed file-tier or KV
behavior enough to cost a few percent on deep prefill, and not enough
to change decode.

The 0.2.x line was not benchmarked head-to-head because the change set
is not a drop-in for this deployment: it moves to a new llama.cpp pin
(1.72.x with sm_86 changes), restructures the serve layer, and its
layer-major prefill (the 4 to 6x item from issue #1670) exists only in
the DeepSeek engine or as a two-GPU layer split at release level; the
Qwen single-GPU layer-major path is still an unmerged branch. A 0.2.x
challenger would therefore confound engine, llama.cpp, and model-shape
changes, and the verdict would not transfer cleanly to a pinned
derivation. It remains a watch item; the public issue #6118 (RTX 3060,
943 tps on 0.1.52 vs 787 on 0.1.60, ~20 percent) is the signal that
the 0.1.x line had per-release regressions, which is exactly why the
comparison is pinned-commit, not "latest".

## Verdict

v0.1.41 stays. The prefill problem on this box is not an engine
version problem; it is an I/O-policy problem in a regime the auto
selector was not calibrated for (see report B). The one genuinely new
knob from the newer line, `STRATA_FILE_CACHE_OLD_RULE`, is recorded for
the 96 GB upgrade (report E): it restores the pre-#1194 cache-sizing
behavior, which is the correct behavior when the complement fits in
the page cache (96 GiB) and self-corrects when it does not.

**Update (2026-11-21):** the 0.1.45 null here held only for that one-shot shape. In the production serve shape (MTP, int8 kv-resident, 32 GiB pin) v0.1.45 measured +23–39% sustained decode vs v0.1.41 at 32K/128K/200K in the three-way campaign — see [2026-11-21-bd775i-strata-three-way-hotrod](2026-11-21-bd775i-strata-three-way-hotrod.md).
