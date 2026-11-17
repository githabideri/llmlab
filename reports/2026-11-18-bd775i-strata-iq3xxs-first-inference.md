# 2026-11-18 — Strata v0.1.41 first inference: IQ3_XXS Flash-Next on a NixOS-built 3060 node

**Goal:** bring up a new single-3060 node (Minisforum BD775i SE, RTX 3060 12 GB,
48 GB DDR5, NixOS) with the Strata engine built declaratively from the upstream
v0.1.41 tag, and produce the model's first tokens on it with the ISTA
IQ3_XXS quant of Qwen3.8-Flash-Next.

This is a **first-contact record, not a campaign profile**: one short prompt,
one-shot runs, the engine's shipped (generic) expert profile, and no energy
metering. The sustained/depth/cold-I/O numbers come from the follow-up campaign.

## Setup

- **Box:** [Minisforum BD775i SE](../docs/hardware/bd775i.md) — Ryzen 7 7745HX
  (8C/16T), 48 GB DDR5-5200, Samsung 512 GB NVMe, RTX 3060 12 GB (Gen4 x16;
  26.8 GB/s host→device measured by the engine's own PCIe probe). NixOS 26.11
  (flakes), driver 595.104.02 (open module).
- **Engine:** Strata v0.1.41, **built by a NixOS flake package on the box
  itself** (the project's Dockerfile says "build it on the PC it runs on";
  the native-CPU form is therefore the faithful one, and the cost is that the
  binary is not reproducible on other builders). CUDA 13.0 comes from nixpkgs'
  `cudaPackages_13_0` (merged toolkit prefix + components); sm_86 only; the
  pinned llama.cpp dependency is the exact commit the project's own tooling
  fetches. No `setup.py`, no container image.
- **Model:** `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`, the **IQ3_XXS**
  variant, 75.8 GB in two shards on the NVMe (HF CLI download).
- **Pack:** the project's `tools/iq_pack.py` against shard 1 — a *native* IQ
  pack: 1.5 GB `dense.bin` (the float tensors), `native_experts.txt` (the
  quantized experts stay in GGUF form and are read from the shards at runtime),
  `index.txt` (1,079 tensors; 302 served natively), and the exported tokenizer.
  The layout matches the one already proven on the GPU server's flash-next lane.

## Commands

Token ids for the prompt came from the project's reference tokenizer
(`tools/strata_tokenizer.py`, verified by its 16-string round-trip) run from a
separate flake Python package; the engine call is flat flags (there is no
`generate` subcommand in v0.1.41):

```bash
strata \
  --pack <packdir> \
  --native <modeldir>/…-00001-of-00002.gguf \
  --expert-profile <packdir>/expert-profile.bin \
  --expert-cache auto --prefill auto --spec 4 \
  --tokens-file <ids> --max-new 64 --greedy
```

The flag set is more constrained than it looks, and the engine states each
requirement as you hit it:

- A **native (IQ) pack requires** `--native <shard1>` (the shards it reads),
  `--spec T` with `T ≥ 2` (a verify window), and `--prefill <chunk>` (chunked
  prompt; `auto` picks the largest chunk the expert cache can lend).
- `--spec` in turn **needs the device residency table**, which is only built
  when an `--expert-profile` is loaded (a profile-filled cache makes the
  residency static, so the hit decision moves onto the device and the token
  graph carries it). The profile **shipped in the Strata tree** (all 24,576
  layer/expert pairs ranked, version 1) passes the loader's checks for this
  model — it is generic, not workload-tuned — and `--expert-cache auto` fills
  whatever VRAM the card can afford in rank order, so the profile's declared
  slot count does not bind a smaller card.

## Observations

- Load: 39.97 GiB expert arena read in ~33 s (**1.22 GiB/s**, the NVMe's
  sustained number), 1.47 GB of dense weights in 0.2 s, PLE table (320,001,536
  rows) mapped from shard 2, ~4.2 GB VRAM at rest. With the shipped profile and
  `--expert-cache auto` the engine placed **4,132 expert slots** in the VRAM
  tier and decided the residency on-device.
- The 5-token prompt ("The capital of France is") produced 64 coherent,
  factually correct tokens: *" Paris. The capital of Germany is Berlin. The
  capital of Italy is Rome. The capital of Spain is Madrid. The capital of
  Portugal is Lisbon. The capital of"* — the quant's output quality holds up on
  this card.
- The engine's prompt path ran the CPU share (idle cores take some of the
  streamed experts on sub-1K chunks); speculating ran 29 verify rounds with
  the default suffix drafter (no MTP — that drafter is a separate ~5 GB
  fetch, deferred).

## Metrics

One-shot `generate`, 5-token prompt, greedy, 64 new tokens, on the box above:

| Run | Decode | Prefill (4 tok) | TTFT |
|-----|--------|-----------------|------|
| cold (empty page cache) | **14.45 t/s** (4,429 ms) | 8.30 t/s (482 ms) | 1.53 s |
| warm (cache filled by a preceding run) | **14.33 t/s** (4,467 ms) | 10.77 t/s (371 ms) | 1.08 s |

A shorter 32-token run measured 12.09 t/s. Cold and warm decode come out
statistically the same here: at a 5-token context the per-token expert set is
small (the engine reports ~6 distinct experts per layer per round) and most of
it is served from the VRAM tier, so the disk only matters at load time and at
prompt depth — which this test does not exercise. Take the prefill row with the
same caution (N=4 tokens, one-time setup dominates).

No energy figures: no wall meter on the node yet.

## Conclusion

The box works: a NixOS-built, nixpkgs-CUDA Strata produces correct tokens from
the IQ3_XXS Flash-Next on a single 12 GB 3060, with the model's 76 GB living on
the local NVMe and ~4 GB VRAM in use — comfortably inside the engine's stated
single-card target. Two follow-ups are queued: the campaign itself (sustained
decode at 8K/32K/64K depth, true-cold NVMe cells, nonces, and the Q2 variant
the CPU-side cells need), and a serve deployment (learned expert profile via
`--expert-profile-save`, the low-RAM/mmap mode the 48 GB box needs, and the
MTP drafter) if the node becomes an always-on endpoint.
