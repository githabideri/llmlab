#!/usr/bin/env python3
"""Unit tests for vision_routing.route_vision — the Phase 1 acceptance matrix.

Run: python3 hub/test_vision_routing.py   (stdlib unittest, no network)

Test 4 of the acceptance matrix (hub unavailable → consumer static fallback)
is a consumer-side contract and is not testable here; it is documented in
hub/README.md and the design report.
"""
import unittest

import vision_routing as vr

# Three-node fleet mirroring the production shape (names are synthetic).
A, B, C = "node-a", "node-b", "node-c"
M35, M27 = "Qwen3.6-35B", "Qwen3.8-27B"
URL = {"node-a": "http://a:8081", "node-b": "http://b:8080", "node-c": "http://c:8080"}


def cand(server, model=M35, **kw):
    base = dict(
        server=server, model=model, url=URL[server], kind="llama.cpp",
        online=True, state_age_s=2.0, loaded=True,
        input_modalities=["text", "image"], enabled=True,
        tier=10, parallelism=2, busy_slots=0, n_proc=0, n_deferred=0,
        gpu_util_max=10.0, gpus_stale=False, tpp=None, tgen=None,
    )
    base.update(kw)
    return vr.normalize_candidate(base)


# The measured-fleet default tiers (see the 2026-09-07 design report §2).
TIERS = {f"{A}/{M35}": 0, f"{B}/{M35}": 5, f"{C}/{M35}": 7, f"{A}/{M27}": 20}


def fleet():
    """All three 35Bs loaded and idle; A additionally holds an idle 27B.
    Parallelism mirrors the real fleet: A's 35B runs parallel=2, the
    single-GPU backup nodes and the 27B run parallel=1."""
    fl = [cand(A), cand(B, parallelism=1), cand(C, parallelism=1)]
    fl[0]["tier"] = 0
    fl[1]["tier"] = 5
    fl[2]["tier"] = 7
    fl.append(cand(A, M27, tier=20, parallelism=1))
    return fl


class AcceptanceMatrix(unittest.TestCase):

    def test_1_all_idle_prefers_node_a_35b(self):
        r = vr.route_vision(fleet())
        self.assertTrue(r["ok"], r)
        self.assertEqual((r["choice"]["server"], r["choice"]["model"]), (A, M35))
        self.assertIn("preferred_tier", r["reason_codes"])
        self.assertIn("immediate_capacity", r["reason_codes"])
        self.assertEqual(r["choice"]["capacity"], "immediate")

    def test_2_offline_node_never_chosen_or_alternate(self):
        fl = fleet()
        fl[1]["online"] = False
        r = vr.route_vision(fl)
        self.assertTrue(r["ok"])
        self.assertNotEqual(r["choice"]["server"], B)
        self.assertNotIn(B, {a["server"] for a in r["alternates"]})
        self.assertIn("server_offline",
                      {x["code"] for x in r["rejected"] if x["name"].startswith(B)})

    def test_3a_evicted_35b_best_loaded_backup_wins_no_mutation(self):
        # The incident shape, minus the slot pin: A/35B evicted, A/27B loaded
        # and idle, B and C loaded and idle. B (tier 5) must beat C (7) and
        # A/27B (20); the advisor must only recommend, never load.
        fl = fleet()
        fl[0]["loaded"] = False                      # 35B evicted
        r = vr.route_vision(fl)
        self.assertTrue(r["ok"])
        self.assertEqual((r["choice"]["server"], r["choice"]["model"]), (B, M35))
        codes = {x["code"] for x in r["rejected"]}
        self.assertIn("not_loaded", codes)

    def test_3b_evicted_and_backups_down_honest_degraded(self):
        fl = fleet()
        for c in fl:
            if c["model"] == M35:
                c["loaded"] = False                  # only A/27B remains
        r = vr.route_vision(fl)
        self.assertTrue(r["ok"])
        self.assertEqual(r["choice"]["model"], M27)
        # and when *nothing* is loaded anywhere:
        for c in fl:
            c["loaded"] = False
        r2 = vr.route_vision(fl)
        self.assertFalse(r2["ok"])
        self.assertIn("not_loaded", r2["reason_codes"])
        self.assertIsNone(r2["choice"])

    def test_5_shared_gpu_contention_demotes(self):
        fl = fleet()
        fl[2]["gpu_util_max"] = 95.0                 # C: doc-AI on the card
        r = vr.route_vision(fl)
        self.assertEqual(r["choice"]["server"], A)   # A still wins outright
        # A out of the picture: B (idle) must beat C (contended).
        fl[0]["online"] = False
        r2 = vr.route_vision(fl)
        self.assertEqual(r2["choice"]["server"], B)
        self.assertIn(B, {a["server"] for a in r2["alternates"]} | {r2["choice"]["server"]})
        # C remains usable as last resort, with the contention code attached.
        fl[1]["online"] = False
        r3 = vr.route_vision(fl)
        self.assertEqual(r3["choice"]["server"], C)
        self.assertIn("gpu_contention", r3["reason_codes"])

    def test_6_saturated_single_slot_loses_to_idle(self):
        # B and C are parallel=1 with their only slot pinned; A/35B
        # (parallel=2) still has one free slot -> it must win.
        fl = fleet()
        fl[1]["n_proc"] = 1
        fl[2]["n_proc"] = 1
        r = vr.route_vision(fl)
        self.assertEqual((r["choice"]["server"], r["choice"]["model"]), (A, M35))
        self.assertEqual(r["choice"]["capacity"], "immediate")

    def test_6b_the_incident_parallel1_prefill(self):
        # A/27B (parallel=1) pinned by a long prefill; A/35B evicted; B idle.
        fl = fleet()
        fl[0]["loaded"] = False
        fl[3]["n_proc"] = 1                          # 27B: slot occupied
        r = vr.route_vision(fl)
        self.assertEqual((r["choice"]["server"], r["choice"]["model"]), (B, M35))
        self.assertNotIn("only_queued_available", r["reason_codes"])
        # ...and when the saturated 27B is the ONLY loaded vision model, it
        # is returned with its queuing stated honestly.
        for c in fl:
            if c["server"] != A:
                c["online"] = False
        r2 = vr.route_vision(fl)
        self.assertTrue(r2["ok"])
        self.assertEqual(r2["choice"]["model"], M27)
        self.assertEqual(r2["choice"]["capacity"], "queued")
        self.assertIn("only_queued_available", r2["reason_codes"])
        self.assertIn("queued", r2["reason_codes"])

    def test_7_stale_state_rejected(self):
        fl = fleet()
        fl[0]["state_age_s"] = 120.0                 # > default 60 s
        r = vr.route_vision(fl)
        self.assertTrue(r["ok"])
        self.assertNotEqual(r["choice"]["server"], A)
        stale = [x for x in r["rejected"]
                 if x["name"] == f"{A}/{M35}"]
        self.assertEqual(stale[0]["code"], "stale_state")

    def test_request_shape_accepted_but_ignored(self):
        r = vr.route_vision(fleet(), request={"images": 3,
                                              "max_width": 1920,
                                              "max_height": 1080})
        self.assertTrue(r["ok"])
        self.assertEqual(r["choice"]["server"], A)


class NormalizationAndPolicy(unittest.TestCase):

    def test_modality_none_is_unknown_not_no(self):
        # llama.cpp: absent metadata stays capable (older-build quirk)
        c = cand(A, input_modalities=None)
        self.assertTrue(vr.image_capable(c))
        self.assertTrue(vr.route_vision([c])["ok"])
        # vLLM: /v1/models exposes no modalities — absent means text-only
        # until explicitly opted in.
        v = cand(B, input_modalities=None, kind="vllm")
        self.assertFalse(vr.image_capable(v))
        r = vr.route_vision([v])
        self.assertFalse(r["ok"])
        self.assertIn("no_image_modality", r["reason_codes"])
        v2 = cand(B, input_modalities=None, kind="vllm", vision_override=True)
        self.assertTrue(vr.route_vision([v2])["ok"])

    def test_modality_text_only_rejected(self):
        c = cand(A, input_modalities=["text"])
        r = vr.route_vision([c])
        self.assertFalse(r["ok"])
        self.assertIn("no_image_modality", r["reason_codes"])

    def test_config_override_restores_blind_metadata(self):
        # Older llama.cpp builds reported ["text"] despite a loaded mmproj.
        c = cand(A, input_modalities=["text"], vision_override=True)
        r = vr.route_vision([c])
        self.assertTrue(r["ok"])
        c2 = cand(A, input_modalities=["text", "image"], vision_override=False)
        r2 = vr.route_vision([c2])
        self.assertIn("no_image_modality", r2["reason_codes"])

    def test_policy_disabled(self):
        r = vr.route_vision(fleet(), policy={"enabled": False})
        self.assertFalse(r["ok"])
        self.assertIn("disabled", r["reason_codes"])

    def test_policy_validation_bounds(self):
        with self.assertRaises(ValueError):
            vr.route_vision(fleet(), policy={"stale_after_s": -5})
        with self.assertRaises(ValueError):
            vr.route_vision(fleet(), policy={"gpu_busy_threshold": "high"})
        with self.assertRaises(ValueError):
            vr.route_vision("not a list")

    def test_unknown_capacity_is_immediate_with_penalty(self):
        c = cand(A, parallelism=None, busy_slots=None, n_proc=None)
        self.assertEqual(vr.capacity_class(c), "unknown")
        self.assertEqual(vr.route_vision([c])["choice"]["capacity"], "immediate")
        # A loaded model with busy metrics absent still loses to an
        # affirmative idle one when both are eligible:
        r = vr.route_vision([cand(A, n_proc=0), c])
        self.assertEqual(r["choice"]["server"], A)

    def test_deferred_request_means_queued(self):
        c = cand(A, n_deferred=1)
        self.assertEqual(vr.capacity_class(c), "queued")

    def test_alternates_bounded_and_ordered(self):
        r = vr.route_vision(fleet())
        self.assertEqual(len(r["alternates"]), 2)
        self.assertEqual([a["server"] for a in r["alternates"]], [B, C])
        for a in r["alternates"] + [r["choice"]]:
            self.assertIn("score", a)
            self.assertIsInstance(a["score"], float)

    def test_deterministic(self):
        a = vr.route_vision(fleet())
        b = vr.route_vision(fleet())
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main(verbosity=2)
