#!/usr/bin/env python3
"""Unit tests for parse_engine_json() — the JSON-dashboard /metrics adapter
added 2026-10-08 so engines that serve a JSON metrics dashboard (Strata
serve) feed the hub's llama-router poll path instead of tripping the
'metrics degraded' badge, plus an end-to-end _poll_router check with the
HTTP layer monkey-patched.

Run: python3 hub/test_engine_json.py   (stdlib unittest)
"""
import importlib.util
import json
import os
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("llm_hub", os.path.join(HERE, "llm-hub.py"))
hub = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hub)


def _listing(loaded=True):
    return json.dumps({"data": [{
        "id": "m1",
        "status": {"value": "loaded" if loaded else "unloaded"},
    }]})


class TestAdapter(unittest.TestCase):

    def test_maps_dashboard_counters(self):
        p = hub.parse_engine_json(json.dumps({
            "totals": {"prompt_tokens": 1000, "reused": 400, "output_tokens": 50,
                       "prompt_ms": 120.0, "decode_ms": 30.0,
                       "drafts_offered": 8, "drafts_accepted": 5},
            "live": {"queued": 2},
        }))
        self.assertEqual(p["llamacpp:tokens_predicted_total"], 50.0)
        self.assertEqual(p["llamacpp:tokens_predicted_seconds_total"], 0.03)
        # the engine's prompt counter includes reused; the hub wants the split
        self.assertEqual(p["llamacpp:prompt_tokens_total"], 600.0)
        self.assertEqual(p["llamacpp:prompt_tokens_cached_total"], 400.0)
        self.assertEqual(p["llamacpp:prompt_tokens_seconds_total"], 0.12)
        self.assertEqual(
            p["llamacpp:spec_decode_num_draft_tokens_total"], 8.0)
        self.assertEqual(
            p["llamacpp:spec_decode_num_accepted_tokens_total"], 5.0)
        self.assertEqual(p["llamacpp:requests_processing"], 2.0)
        # get_metric() must find every key (prefix probing)
        for k in p:
            self.assertIsNotNone(hub.get_metric(p, k[len("llamacpp:"):]))

    def test_rejects_prom_text(self):
        prom = ("llamacpp:tokens_predicted_total{model=\"M\"} 42\n"
                "# a comment line\n")
        self.assertIsNone(hub.parse_engine_json(prom))

    def test_rejects_unknown_json(self):
        self.assertIsNone(hub.parse_engine_json(json.dumps({"hello": 1})))
        self.assertIsNone(hub.parse_engine_json(json.dumps(
            {"totals": {"prompt_tokens": 3}})))   # no output_tokens
        self.assertIsNone(hub.parse_engine_json(json.dumps([1, 2, 3])))
        self.assertIsNone(hub.parse_engine_json(None))
        self.assertIsNone(hub.parse_engine_json(""))


class TestRouterPath(unittest.TestCase):
    """End-to-end: _poll_router with a JSON-dashboard engine behind it."""

    def _srv(self):
        return hub.Server({"name": "t", "kind": "llama-router",
                           "url": "http://x",
                           "models": {"m1": {"desc": "d"}}})

    def _poll(self, s, metrics_body, t, loaded=True):
        calls = []

        def fake(url, timeout=None):
            calls.append(url)
            if url.endswith("/v1/models"):
                return 200, _listing(loaded)
            return 200, metrics_body
        orig = hub.http_json
        hub.http_json = fake
        try:
            s._poll_router(t)
        finally:
            hub.http_json = orig
        return calls

    def _body(self, **kw):
        return json.dumps({
            "totals": {
                "prompt_tokens": kw.get("prompt", 1000),
                "reused": kw.get("reused", 400),
                "output_tokens": kw.get("out", 50),
                "prompt_ms": kw.get("pms", 120.0),
                "decode_ms": kw.get("dms", 30.0),
                "drafts_offered": kw.get("doffer", 8),
                "drafts_accepted": kw.get("dacc", 5),
            },
            "live": {"queued": kw.get("queued", 0)},
        })

    def test_metrics_ok_and_rates(self):
        s = self._srv()
        self._poll(s, self._body(), 1000.0)
        st = s.models["m1"]
        self.assertTrue(st["metrics_ok"],
                        "adapter must clear the 'metrics degraded' flag")
        self.assertIsNone(st["tgen"])              # first sample: no rate yet
        # second poll 2 s later: decode advanced 10 tokens over 20 ms of
        # the engine's own clock -> 500 t/s, bounded, not wall-artifacted
        self._poll(s, self._body(prompt=1400, reused=700, out=60,
                                 pms=150.0, dms=50.0,
                                 doffer=12, dacc=10), 1002.0)
        st = s.models["m1"]
        self.assertAlmostEqual(st["tgen"], 500.0, delta=1)
        # tpp on the prompt clock: non-cached prompt 600->700 over 30 ms;
        # the cached share is the reused delta over the total prompt delta
        # (300 of 400)
        self.assertAlmostEqual(st["tpp"], 100 / 0.03, delta=10)
        self.assertAlmostEqual(st["cache_hit"], 300 / 400, delta=0.01)
        # spec acceptance: (10-5) accepted over (12-8) offered
        self.assertAlmostEqual(st["spec_accept"], 5 / 4, delta=0.01)

    def test_counter_restart_rebaselines(self):
        s = self._srv()
        self._poll(s, self._body(out=100, dms=10.0), 1000.0)
        self._poll(s, self._body(out=110, dms=12.0), 1002.0)
        st = s.models["m1"]
        self.assertIsNotNone(st["tgen"])
        # engine restart: totals drop below the baseline at t=1040 — no
        # spike; the stale pre-restart rate has decayed by RATE_DECAY
        self._poll(s, self._body(out=5, dms=1.0), 1040.0)
        st = s.models["m1"]
        self.assertIsNone(st["tgen"])              # decayed, not spiking

    def test_loaded_false_no_metrics_call(self):
        s = self._srv()
        calls = self._poll(s, None, 1000.0, loaded=False)
        self.assertFalse(any("metrics" in c for c in calls))
        self.assertFalse(s.models["m1"]["metrics_ok"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
