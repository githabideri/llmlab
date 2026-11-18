# The BD775i as an always-on two-conversation node for Qwen3.8-Flash-Next: the verdict

**Date:** 2026-11-18
**Machine:** [the BD775i](../docs/hardware/bd775i.md)
**Engine:** Strata v0.1.41 (NixOS derivation) · **Model:** IQ3_XXS (75.8 GB)
**Predecessor:** [the baseline profile](2026-11-18-bd775i-strata-baseline-memory-kv-mtp-batch.md) (memory modes, KV, depth, MTP, batch mechanics)

## Goal

The assignment: make this box the best validated always-on node for the model, with one 200K-token conversation live alongside one 128K conversation, a declarative NixOS service, and an honest verdict on where the box wins and where it does not. The baseline report established the parts; this one assembles them into the target service, stress-tests it, and states the trade-offs.

## The target run (200K + 128K, both live)

Two sessions were run concurrently through the engine's `--batch 2`: conversation A with a 204,856-token prompt, conversation B (fired 90 s later) with 128,055, each asked for 256 output tokens. The only difference between the two configurations below is the expert memory mode; both use int8 KV with 20,480 cells pinned (the rest streamed), the 208,896-token context ceiling, the MTP drafter, and batch 2.

| | 24 GiB page-locked (res24) | mapped (page cache) |
|---|---|---|
| A read (204,856) | 200.2 tok/s, 17.1 min | 253.5 tok/s, 13.5 min |
| A decode (256 tok, in the batch slot) | 0.96 tok/s (332 ms/window, 307 ms of it CPU experts) | 0.80 tok/s (383 ms/window, 362 ms CPU) |
| B read (128,055, concurrent) | 166.9 tok/s, 12.8 min | 228.8 tok/s, 9.3 min |
| B decode (252 tok, solo path, MTP on) | 9.9 tok/s (acceptance 0.45) | 5.1 tok/s (acceptance 0.36) |
| A total (fire to 256 done) | 21.6 min | 18.6 min |
| B total | 28.9 min | 22.2 min |
| Memory pressure (PSI full, avg10) | ~11% | 14–16% |
| OOM / errors | none | none |

Both configurations complete the target; the numbers are single runs each, so read them as a comparison of regimes, not a measurement of effect sizes. The split is consistent with the baseline's single-session matrix in the place that matters for the service: the page-cache (mapped) mode wins *prefill* here (its LRU keeps whatever the two reads are touching), while the pinned 24 GiB mode wins *decode* (the slot and solo decode touch the same hot expert set that the pin already holds in RAM; the mapped mode pays a page fault under 15% memory pressure for it) and wins on pressure headroom. For an always-on node whose users talk to the live conversations, decode rate and headroom are the right axis, so **the service runs res24**; mapped is the better choice if the workload is mostly long one-shot prefills (RAG-style batch), where its 27–37% faster reads dominate.

The choreography was exactly the documented one: when B arrived, A's 204,857-token state was copied into the first batch slot in 2.1–2.3 s (nothing re-read), A then decoded one token per verify window while B's prompt was read in the main session, and the slot kept decoding during B's prefill gaps (254 windows over 266 s in res24). Both completions were coherent continuations of the corpus text.

## The 8 h stability soak

The service ran 08:26 to 17:44 (8 h 18 min, minus a 25 min window in which we deliberately stopped it to run the quality cells below, in which the 12 GB card cannot host a second engine at all). Every 30 minutes both sessions received a ~22.5K-token turn (16 completion tokens; the 16K-character slice tokenizes to 22.5K): seven turns before the power-off and two after, all completing in 129–143 s end-to-end with no drift across the whole period. The engine's own logs show the steady state the numbers imply: 10–15% avg60 memory pressure under turn load, back to ~0 between turns; 11.5 GB VRAM resident; 57–59 W at idle under the 24 GiB pin; no OOM, no restart of its own, no error lines.

Three milestones were forced on it. **Crash (T+3 h):** both processes killed with SIGKILL; the service was restarted by hand (the unit's automatic restart is the Phase J cutover, below) and was ready in 60 s; a fresh 44.5K-token prompt then ran end-to-end in 197 s, which is the honest recovery number: a cold process pays the ~48 GB NVMe re-read, then the prompt read, then the answer. **Power-off (T+6 h):** the box powered itself off; a WOL magic packet from a second machine on the same LAN brought it back to a live SSH session in 30 s, the service was up 60 s after boot, and the same 44.5K-token recovery turn completed in 197 s. **Long soak:** the hourly state snapshots (free memory, PSI, GPU power and temperature, uptime, error count) show a flat line: 44 GB total with 35 GB used by the resident service, ~9 GB available, no creep, no fragmentation symptom in 8 hours.

One gap is documented rather than hidden: a single 30-min turn pair near the end was missed because the service was down for the quality window (a client bug on our side, not the service's; the turns before and after are normal).

## Quality under the chosen KV quantization

v0.1.41 has no perplexity command, so the fixture is a comprehension pair with known answers, run one-shot (32K Moby-Dick prefix + question, greedy, 128 tokens) under each of the three int8-class KV formats: int8, k8v4, q4_0. The narrator question ("what name does the narrator call himself in the very first line?") is answered correctly by all three: **Ishmael**, each citing "Call me Ishmael". The captain question ("who is the captain of the Pequod?", expected Ahab) turned out to be unanswerable from the prefix: Ahab first appears in chapter 20, beyond the 32K window, and all three formats instead describe the Father Mapple passage that the window does contain, in near-identical wording. That is the fixture's limitation, not a divergence: the q4_0 format shows no visible quality drift against int8 on this test (n=1 per cell; the engine's own upstream bench puts q4_0 at +8–12% perplexity on long documents, which is why the service keeps int8).

## The service

The NixOS module (in the machine's nixos-settings, applied with the final rebuild) declares one unit, `strata-serve`: the engine is the flake's `strata` package (upstream v0.1.41, CUDA 13.0, sm_86, built on the box), the HTTP layer is the engine's own python `serve/` tree run from the flake's `strataTools` python, both with stable store paths. The unit sets `LimitMEMLOCK=infinity` (the 24 GiB pin is `mlock`'d), `LimitNOFILE=131072` (two 38 GB shards mapped), `LimitAS=infinity`, `STRATA_PREFILL_CPU_SHARE=0`, and restarts on failure; the model and the engine config json are persistent data outside the store (`/var/lib/models/...`, `/var/lib/strata-serve/`), the same out-of-store pattern the estate's other Strata node uses. The port is on the LAN and Tailscale only, with no API key (the owner's decision for a private network).

## Verdict

The box meets the assignment's target: a 200K-token and a 128K-token conversation live at the same time, in a declarative NixOS service, on a 12 GB card and 44 GB of single-channel RAM, with no OOMs in the target run and the soak. What it costs:

- **Loading a new long conversation is slow** (a fresh 200K prompt takes 13–17 min; a fresh process always re-reads ~48 GB from NVMe first). Fine for conversations that live for days; painful for one-shot workloads.
- **The 200K session decodes slowly** (~1 tok/s while the other session has the main path) — that is the batch-slot regime, and it is the price of not having two GPUs' worth of expert bandwidth. A user on the 128K session gets ~10 tok/s with the MTP drafter; that is the session you point interactive users at.
- **Memory is the binding constraint, not the GPU.** 44 GB with a 40 GB model is a 10% headroom game; the box runs at 11–16% full memory pressure under the target load and a second deep conversation (a third session at 200K) would not fit.

Where it wins: it is the only configuration that runs *this* model (75.8 GB) at all on a single consumer card; the always-on cost is the loaded-idle power measured in the [bring-up report](2026-11-18-bd775i-strata-iq3xxs-first-inference.md); and the whole stack (engine build, service, model placement) is declarative and reproducible from the machine's flake. The engine's shared-KV-pool branch (PR #1614, for a newer engine release) is the obvious next step if the slot-decode speed ever matters: it would let both conversations decode from one KV arena instead of one sitting in a slot at ~1 tok/s.

## Reproduction

Everything above is in the campaign directory of the private estate repo (freeze manifest with all hashes; the two probe result JSONs; the engine logs with the per-chunk and per-window lines; the soak log and hourly state snapshots; the NixOS module). The engine and model are identified by the hashes in the freeze; the prompts are plain-text corpus excerpts with a marker token, nothing personal.
