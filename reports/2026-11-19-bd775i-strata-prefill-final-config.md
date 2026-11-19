# Final configuration for the two-session 48 GB service (report D)

## What changed and what did not

The deployed service keeps its engine (v0.1.41, same llama.cpp commit
3cf03257), its model (the published GGUF IQ3_XXS + BF16 MTP), its
routing and KV settings (int8 KV, kv-resident 20,480, batch 2, max
context 208,896, 24 GiB expert pin), and its unit structure. One
addition:

1. `STRATA_UNBUFFERED_LOAD=1` in the unit environment: the file tier
   reads the 16 GiB complement with O_DIRECT instead of through a page
   cache that cannot hold it. This is the entire prefill fix: 2.2 to
   2.8x on every context length measured, decode unchanged, physical
   NVMe volume down ~2x, major faults down ~40x.
   (`STRATA_PREFILL_CPU_SHARE=0` was already in the unit; it stays,
   for consistency with the benchmarked state.)

Two mechanisms were evaluated and left out of this config, deliberately:

- The in-RAM conversation cache (`--conversation-cache-mib`, off by
  default) parks a finished conversation's KV so its next turn skips
  the re-prefill. Tested at 8 GiB with the 2.5 GiB admission floor: it
  parks one deep conversation (128K, 2.8 GB snapshot, ~1 s) but on 44
  GiB the ledger cannot hold two parked conversations plus the pin
  plus KV (admission skips the second, and parking one destroys the
  other's prefix reuse). It is a single-deep-session option (a variant
  config exists on the node); the two-session config ships without it.
- Slot save/restore (llama-server's `/slots/N?action=save|restore`) is
  disabled whenever batch mode is on (upstream #465: parallel requests
  share the engine's control FIFO). Session persistence across an
  engine restart is therefore in-RAM only; a restart costs one
  re-prefill per lost session (423 s at 200K, direct).

## What it does

Two consumers: one 200K-token conversation, one 128K-token
conversation, both live, batch 2, MTP for the main-session decode:

| cell | measured (final config) |
|---|---|
| fresh 32K prefill | ~72 s (was 200 s) |
| fresh 64K prefill | ~142 s (was 358 s) |
| fresh 128K prefill | ~283 s (was 684 s) |
| fresh 200K prefill | ~212 s prefix-warm / ~423 s cold (was 499 to 660 s) |
| follow-up on a 200K session whose KV is gone | ~471 s re-prefill (direct); 1,029 s measured on the previous buffered config. The opt-in in-RAM conversation cache (8 GiB) fits one parked conversation but not two on 44 GiB, so the two-session config ships without it |
| two-consumer 512-token decode | ~1 to 5.6 tps per session (unchanged: slot-decode bound) |

The 32K/64K/128K fresh numbers are the candidate battery (three
consecutive days of runs behind the service); the two-consumer decode
is the structural limit of this box at this pin size and does not move
with I/O (report B, "what this box cannot reach").

## Rollback

The previous config is a single environment-variable difference.
Rollback is: remove `STRATA_UNBUFFERED_LOAD=1` from the unit, restart
the unit, ~2.5 min to serving. The known-good binary, config, and MTP
pack are untouched; the v0.1.41 derivation is still the active one.
The challenger v0.1.45 derivation (report C) remains in the Nix store
as an independently available pinned build.

## Exposure and access

The endpoint is LAN + Tailscale only, no public route (unchanged from
the deployment it replaces). For wider client rollout an API key is
prepared in the varlock-managed config (the serve layer checks a
Bearer token or `x-api-key` header when `api_key` is set in the
config); flipping it on is a config edit plus a unit restart, and it
does not change the network exposure. It is not flipped on during
this campaign.

## What was deliberately not changed

- The pin size stays at 24 GiB: it is the largest that fits two deep
  sessions plus MTP plus process in 44 GiB, and it is the fastest
  (report B budget table).
- Chunk size stays at the engine's auto value (8,192): 4K chunks are
  1.7 to 1.8x slower, 16K/32K no better (report B).
- The GGUF stager stays on its auto (SSD) profile: forcing the RAM
  profile was not distinguishable from noise (report B).
- Decode stays int8 KV with kv-resident 20,480: the only regime that
  fits; FP16 KV OOMs in two-session.
- No 0.2.x engine: not a drop-in (new llama.cpp pin, restructured
  serve, layer-major not in the Qwen single-GPU path at release level);
  watch item (report C).
