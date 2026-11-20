# LLM hub: the picker card, structured model rows, and the BD775i onboarding

**Date:** 2026-11-20
**Area:** the homelab's llm-hub (the fleet's inference registry, `llmlab/hub`), and the
[BD775i SE](../docs/hardware/bd775i.md) single-GPU node
**Status:** done — deployed on the hub host and the node, verified end to end

The fleet's hub and its newest card landed together in one window, because the card
*needed* the hub change to be readable: the BD775i is the first hub card whose two
rows are the **same model in two placements** (one 12 GB card, one engine at a
time, two serving profiles), and a card like that is unreadable in a flat
table of identical-looking rows. The change is two halves:

1. a **box-owned mux front** on the node (the fleet's established pattern: the
   Omen laptop and both 3060 boxes each run a small stdlib-python mux that
   owns the card's public port, synthesizes `/v1/models` with live per-model
   state, 202-acks `load`/`unload` and runs the switch in the background);
2. a **picker card** in the hub UI plus a few structured config fields, so a
   card shows its *active* profile in full and everything else as one
   compact line each, gated by what was actually measured.

## The mux (node side)

The node's two profiles are systemd units (the declarative NixOS
[deployment](../docs/hardware/bd775i.md)): the **dual** profile (24 GiB
pinned experts, batch 2 — the always-on two-consumer 200K+128K service) and
the **solo** profile (32 GiB pin, batch 1 — the single-user long-context
profile measured in the [solo campaign](2026-11-20-bd775i-solo-long-context-decode.md)).
They cannot run together (one 12 GB card), so they were previously switched
with a hand-run script.

The new `bd775i-mux` (stdlib python, declared in the node's NixOS flake like
everything else on that box) sits on the port consumers already used:

- **Port map.** The mux owns the one exposed inference port (8080,
  LAN/tailnet, keyless as the node's other endpoint). Both engine units are
  local-only now (8082 dual, 8081 solo); a GPU sidecar (the hub's nvidia-smi
  poller) is a declared unit on 9421.
- **Row ids.** The mux advertises the model as two rows,
  `qwen3.8-flash-next-iq3xxs-dual` and `…-solo`. It rewrites the OpenAI
  `model` field to the engine's canonical id on the way in and back on the
  way out — on *streaming* responses too, with a carry buffer so a needle
  split across TCP chunk boundaries is still rewritten (unit-tested at
  chunk sizes 1, 7 and 50 bytes).
- **Backward compatibility.** Requests carrying the *canonical* id (every
  pre-mux consumer: the fleet's pi-web agents, two bots) are routed to the
  **active** profile with the bytes untouched — the cutover changed nothing
  client-side, which is the point of taking over the existing port instead
  of introducing a new one.
- **Switching.** `POST /v1/models/load {row-id}` returns **202**
  ("switching") inside the hub's 2 s verdict budget; the mux then stops the
  other unit, waits for the GPU to actually free (< 500 MiB, not an
  assumption), starts the target and polls its `/v1/models` until it reports
  loaded (measured ~46 s with a warm NVMe, ~2.5–4 min cold — 75 GB of GGUF
  shards through the disk plus the expert pin). `unload` stops a unit.
  Deliberately there is **no blocking auto-switch on chat**: a switch
  exceeds any client's request budget, so a chat request naming a
  non-active profile gets a 503 telling the caller to load it first. The
  out-of-band switch script still works for humans (the hub's polling
  converges on whatever state results).

## The picker card (hub side)

The hub's card layout changed to what the skill's pre-test protocol had
sketched as its "UI doctrine" for a while and never built:

- **Card head** carries identity chips — an optional server-level `site`
  string plus the GPU name from the sidecar — and, when `site` is present,
  the long free-form server `description` is withheld from the UI. It stays
  in the config and on `GET /api/models` (the agent surface), which is where
  prose belongs; the UI renders data instead.
- **The active model renders in full**: description, live rates (t/s, pp/s,
  MTP acceptance, cache), the 30-min sparkline, diagnostics, load/unload.
- **Inactive models are picker lines**: one compact row each —
  `quant · ctx · measured rate · matrix-relation-to-active` — expanded to
  the full row on click. For the BD775i both rows show the same identity
  and quant; what differs is the pin size, the batch mode, the measured
  rate and the relation — exactly the decision a user actually makes.
- **`switching…`**: the hub now surfaces the upstream status value verbatim
  as `status_raw` on every model row (router and mux kinds). A mux in the
  middle of a profile change reports `switching`, which the UI renders as a
  pulsing state with the button disabled; the existing 202-ack + state
  polling contract is untouched.
- **Structured hints.** A server's `models.<id>` may now carry `identity`,
  `quant`, `measured` (the last measured facts, UI-facing) and `matrix`
  (row-id → tested relation to that row, the tested-only doctrine made
  visible). Where present the UI renders them instead of the prose desc;
  where absent (all the fleet's pre-existing cards, migrated only to `site`
  in this change) the old layout renders unchanged — the hub code ships
  independently of any config migration, which is what made the one-window
  rollout possible.

The per-model `desc` scheme is unchanged (fixed field order, banned state
words/dates/narrative, 180-character cap, mechanical check at deploy time);
the two new BD775i descs pass it at 174/177 characters.

## Verification

- Node side, on the box: canonical-id chat through the mux byte-transparent
  (non-streaming and SSE); row-id chat with the response `model` rewritten
  in every streamed chunk; `/metrics` proxy (the hub's Strata JSON-dashboard
  adapter parses it); the switch in both directions (dual→solo 46 s warm,
  solo→dual multi-minute cold) with the catalog showing the right
  loaded/unloaded pair each time; both the LAN and the tailnet face of the
  mux reachable, engine ports not.
- Hub side: the existing test suite (engine-JSON adapter, vLLM metric
  semantics, the UI state matrix) passes with the change; the new fields
  were exercised against the **live** mux by a throwaway unauthenticated hub
  instance before the real deploy — `GET /api/servers` and `GET /api/models`
  return `site`, `status_raw`, `quant`, `measured`, `matrix` for both rows;
  the production hub host then picked the card up via the normal
  config-deploy path (the desc checker ran in that path and passed).
- Consumer lockstep: the fleet's pi-web registries gained the two row ids
  as a new provider (LAN for the same-site machine, the tailnet address for
  the cross-site one); a drift note covers the laptop copy that only its
  owner can change.

## What this is not

- Not a new engine and not a new model: same Strata pin, same IQ3_XXS
  weights, same NixOS units as before. The node's inference quality and
  performance numbers are the ones in the two BD775i reports of 2026-11-18
  and 2026-11-20; this change is purely about *reachability and
  readability*.
- The hub still never sits in the inference path (consumers talk to the
  node's port directly); the mux is the node's own, in the fleet's
  established pattern, so a hub outage degrades to monitoring, not service.


## Addendum (same day, after the owner looked at the live card)

Two corrections, both about identity labels, both mine:

1. **The site chip is gone.** The card head briefly showed a `site` label chip
   plus the GPU name; the owner's reaction ("Freistadt? wtf?") was the correct
   one, for two reasons. The label was noise on a card whose identity is the
   name plus the models, and the value was *wrong*: the sites had been
   derived from the LAN address prefix, but the estate's two sites run
   separate physical LANs that share the same 0.x space, so the prefix
   carries no site information. The corrected assignment came from the
   estate's machine curation (this node is on the site the rest of its
   fleet neighbours live on). The chip is removed from the UI entirely
   (the field stays in the config and on the API, where it is data, not
   display), and with it the last free-form line of the card head: the
   server description is now never rendered in the UI, only on
   `GET /api/models`.
2. **State is what a card should show, not geography.** The one thing the
   owner actually needed to know — *which profile is live right now, and
   what the other one costs* — is exactly the loaded/idle column plus the
   picker line; that part of the revamp stood up to the look and stays.

## Addendum 2 (same day, correcting addendum 1)

The explanation in addendum 1 was wrong and is corrected: the estate's two
sites do **not** share an address space. Vienna's LAN is the 192.168.0/24
and Freistadt's is the 192.168.178/24 — separate /24s, so the address prefix is a
perfectly good site signal. The actual error was an inverted map (0.x was
attributed to Freistadt instead of Vienna); the site values in the config
and on the card were corrected against the estate's inventory and are right,
but the "shared 0.x space" sentence above is false and should be read as
retracted.

## Addendum 3 (picker v2: active-only default, the panel, the bar)

After seeing the first version, the owner pointed at the design this picker
had been waiting for since the doctrine was written: the card's default view
is the **active profile only**; a **"switch model"** button opens the
**panel** with every model loadable on the card, and each row carries its
tested-matrix relation to what is running — `coexists` loads alongside,
`exclusive` replaces (with the measured reason), `untested` is explicitly
not selectable. The doctrine's **measured-load bar** landed too: a per-server
`capacity` config block (measured VRAM totals against the measured pool,
from the pre-test windows) renders above the roster; it is hidden on cards
without measured totals rather than estimated. The switch-progress line
carries the mux's phase and elapsed seconds where the mux reports them,
otherwise a client-side elapsed counter. One interaction subtlety was fixed
along the way: the grid deliberately skips re-rendering while the cursor
sits on it, which silently swallowed the first panel toggles — the toggle
now forces the next render explicitly.
