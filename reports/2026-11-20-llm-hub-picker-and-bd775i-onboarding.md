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

## Addendum 4 (fleet fixes found while reviewing the v2 build)

Three fixes landed in the same deploy:

- **The tick period was the fleet sum, and dead machines set it.** The
  poller ran one server per iteration, so each unreachable endpoint
  (two 4 s fetch timeouts per attempt) added its full cost to every
  tick; with three machines down the "2 s" tick ran at ~29 s for
  *every* card. The vLLM card's 1-min prompt-compute window — the data
  behind its pp/s lane — self-clears after a 15 s gap, so it could never
  accumulate and the lane read flat while the card was actively
  generating. The tick is now **one poll thread per server in parallel**
  (a re-entry guard makes a wedged fetch harmless; a 15 s join cap keeps
  one server from ever stalling the fleet), offline servers are probed
  at most once every 15 s on their own clock, and the offline state is
  time-based (~30 s without a success) instead of a fixed poll count.
  The tick is back to 2 s and the vLLM pp lane fills again (small
  numbers while the conversation runs on ~96 % prefix-cache hits; a
  fresh prefill spikes it like the other cards).
- **Card order: online first.** `/api/servers` stable-sorts online
  machines to the top (config order inside each group), so the three
  down machines no longer sit between the live ones. The config order
  was set in the same change (the box the 27B runs on, then the
  48 GiB box, then the dual-3060, then the GPU-only visibility card).
- **The 48 GiB box's boot default is now the solo profile** (owner
  decision: it serves one long conversation, not two). Only the
  `wantedBy` symlinks flipped — the unit bodies are byte-identical, so
  a rebuild never restarts a running profile; the two-consumer dual
  profile stays one switch away.

## Addendum 5 (measured-load tooltips; the vLLM pp lane shows the phase speed;
fpx2 reports both of its 3060s again)

- **The measured-load bar now explains itself.** On cards that have the bar,
  the label's hover tooltip spells out what it is (measured per-profile VRAM
  totals against the measured pool, shown only where a pre-test window
  measured them, never estimated; *partial* = a loaded profile without a
  total) and why vLLM cards don't have one (the engine sizes its pool to
  the cards at startup, so the bar would be a permanent red carrying no
  information — the live pressure there is the KV% in the model rows). On
  cards without the bar, a quiet hoverable line takes the bar's place with
  the card-specific reason ("measured in a pre-test window" for the
  llama/strata cards; the pool-sizing explanation for vLLM).
- **The vLLM pp lane now shows the same number as the row cell.** It was
  fed by a 1-min work-rate window, which dilutes: a 9 s prefill at 700 t/s
  reads ~100/s over 60 s — the owner saw "100 in the chart, 700 in the
  cell" and was right to call it off. The lane now carries the
  phase-clock speed of the last *completed* prefill, held for 2 min
  (during a chunked prefill the computed-token counter is flat, so there
  is no honest in-flight rate; the live-prefill timer covers that
  interval). Verified live: a prefill completing at 447.5 t/s shows
  447.5 across the lane for the hold, where the work-rate window read 9.1.
- **The dual-3060 machine's sidecar reported one card again.** Its
  per-host GPU index filter was stale (written before the second 3060
  existed), so the board showed a single 12 GB row and read as "one
  3060". The filter now lists both indices; the board shows both cards
  again (both holding the 27B's VRAM). Both links are currently at the
  degraded Gen1 (x16 and x1) per the 2026-11-18 v4 note — a reboot
  restores the negotiated generation.

## Addendum 6 (row parity: the kv cell, a live ctx cell for strata cards,
one hover affordance rule, the glossary out of the way)

- **The vLLM kv cell is back in the row and now carries its context**:
  shown at any occupancy (a near-empty long-context pool is a fact, not
  noise), with the tokens, the pool size in concurrent contexts, and the
  10m/30m/1h/24h peaks in the hover.
- **Card parity.** Each card kind now shows what its engine honestly
  measures: vLLM cards keep kv/req/tpot/ttft/phase; the strata cards
  (the 48 GiB box, the 35B on the laptop) gain a live **ctx** cell —
  the deepest prompt in flight or among the last few completed requests,
  straight from the engine's dashboard — which is the llama-side
  counterpart of the KV cell (for kv-resident profiles, depth past the
  resident window means RAM streaming); plain llama.cpp builds expose no
  KV-pool meter and no per-request dashboard, and the glossary says so
  instead of inventing one.
- **One affordance rule for hover explanations**: anything carrying a
  tooltip shows the question-mark help cursor — the measured-load label
  included, where it was missing while the row labels had it.
- **The footer's glossary is collapsed by default** (a details/summary
  one-liner; click expands). The always-visible line keeps the
  operational facts: the hub is observability only, never in the
  inference path; poll cadence; the API as source of truth.
