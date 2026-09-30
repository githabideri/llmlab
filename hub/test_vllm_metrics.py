#!/usr/bin/env python3
"""Unit tests for the vLLM metrics path of llm-hub.py (2026-09-30 semantics
fix) plus the UI state derivation (run via node against the real index.html
function — no re-implementation is tested).

Run: python3 hub/test_vllm_metrics.py   (stdlib unittest; node for the UI)

Covers the spec's matrix:
  A  2 s burst artifact must not surface as prefill performance
  B  phase-clock prefill speed (computed KV / prefill-phase seconds)
  C  cached prompt: computed share is what gets computed, not the total
  D  counter reset (engine restart) -> clean re-baseline, no spike
  E  mux model/engine switch -> all short-window state re-baselined
  F  initial prefill: live timer, no invented rate, no degradation inputs
  G  length-limited completions never mark anything degraded
  H  high KV with no queue/preemption is busy pressure, not degraded
  I  preemption alone degrades
  J  missing phase histograms (older build): no crash, None, no invented values
"""
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("llm_hub", os.path.join(HERE, "llm-hub.py"))
hub = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hub)


# ---------------------------------------------------------------------------
# synthetic vLLM /metrics body
# ---------------------------------------------------------------------------

def _hist(name, n, sumv):
    """A histogram with a fixed bucket set (base/cur must match)."""
    lines = []
    for le in ("1.0", "5.0", "10.0", "100.0"):
        lines.append(f'vllm:{name}_bucket{{engine="0",model_name="{M}",le="{le}"}} 0.0')
    lines.append(f'vllm:{name}_bucket{{engine="0",model_name="{M}",le="+Inf"}} {n}')
    lines.append(f'vllm:{name}_sum{{engine="0",model_name="{M}"}} {sumv}')
    lines.append(f'vllm:{name}_count{{engine="0",model_name="{M}"}} {n}')
    return lines


def _pkhist(name, n, sumv):
    lines = []
    for le in ("100.0", "1000.0", "10000.0", "100000.0"):
        lines.append(f'vllm:{name}_bucket{{engine="0",model_name="{M}",le="{le}"}} 0.0')
    lines.append(f'vllm:{name}_bucket{{engine="0",model_name="{M}",le="+Inf"}} {n}')
    lines.append(f'vllm:{name}_sum{{engine="0",model_name="{M}"}} {sumv}')
    lines.append(f'vllm:{name}_count{{engine="0",model_name="{M}"}} {n}')
    return lines


M = "M"

def vbody(lc=0.0, lh=0.0, pt=None, gen=0.0, kv=0.0, run=0, wait=0, pre=0,
          stop=0, length=0, abort=0,
          ttft_n=0, ttft_sum=0.0,
          pf_n=0, pf_sum=0.0, dd_n=0, dd_sum=0.0, pk_n=0, pk_sum=0.0,
          by_source=True, phase=True, prefix_counters=True):
    """One synthetic vLLM /metrics body. pt defaults to lc+lh (the engine
    counts full prompt length; by_source splits it)."""
    if pt is None:
        pt = lc + lh
    L = []
    if by_source:
        for src, v in (("local_compute", lc), ("local_cache_hit", lh),
                       ("external_kv_transfer", 0.0)):
            L.append(f'vllm:prompt_tokens_by_source_total{{engine="0",'
                     f'model_name="{M}",source="{src}"}} {v}')
    L.append(f'vllm:prompt_tokens_total{{engine="0",model_name="{M}"}} {pt}')
    L.append(f'vllm:prompt_tokens_cached_total{{engine="0",model_name="{M}"}} {lh}')
    if prefix_counters:
        L.append(f'vllm:prefix_cache_queries_total{{engine="0",model_name="{M}"}} {pt}')
        L.append(f'vllm:prefix_cache_hits_total{{engine="0",model_name="{M}"}} {lh}')
    L.append(f'vllm:generation_tokens_total{{engine="0",model_name="{M}"}} {gen}')
    L.append(f'vllm:kv_cache_usage_perc{{engine="0",model_name="{M}"}} {kv}')
    L.append(f'vllm:num_requests_running{{engine="0",model_name="{M}"}} {run}')
    L.append(f'vllm:num_requests_waiting{{engine="0",model_name="{M}"}} {wait}')
    L.append(f'vllm:num_preemptions_total{{engine="0",model_name="{M}"}} {pre}')
    for r, n in (("stop", stop), ("length", length), ("abort", abort)):
        if n:
            L.append(f'vllm:request_success_total{{engine="0",model_name="{M}",'
                     f'finished_reason="{r}"}} {n}')
    L.append('vllm:engine_sleep_state{engine="0",model_name="M",sleep_state="awake"} 1.0')
    L += _hist("time_to_first_token_seconds", ttft_n, ttft_sum)
    if phase:
        L += _hist("request_prefill_time_seconds", pf_n, pf_sum)
        L += _hist("request_decode_time_seconds", dd_n, dd_sum)
        L += _pkhist("request_prefill_kv_computed_tokens", pk_n, pk_sum)
    return "\n".join(L) + "\n"


def srv(kind="vllm"):
    return hub.Server({"name": "t", "kind": kind, "url": "http://x",
                       "models": {"M": {"desc": "d"}}})


# ---------------------------------------------------------------------------

class TestVllmMetrics(unittest.TestCase):

    def test_a_2s_burst_not_performance(self):
        """A 10K computed prefill landing in one 2 s poll: the old wall rate
        read 5,000 t/s; the headline must be the phase-clock speed."""
        s = srv()
        t = 1000.0
        s._apply_vllm_metrics("(vllm)", vbody(lc=10000), t)
        t += 2
        # the prefill completed: 10K computed KV in 8.5 s of prefill phase
        s._apply_vllm_metrics(
            "(vllm)", vbody(lc=20000, pf_n=1, pf_sum=8.5, pk_n=1, pk_sum=10000), t)
        st = s.models["(vllm)"]
        # the artifact value must be nowhere to be found
        self.assertNotEqual(st["tpp"], 5000)
        self.assertAlmostEqual(st["tpp"], 10000 / 8.5, delta=1)
        self.assertIsNone(st["pp_tput_1m"])          # window not yet full
        self.assertEqual(st["prompt_tokens_computed_raw"], 20000)

    def test_b_prefill_speed_from_phase_clocks(self):
        """spec §16B: computed delta 6030 / prefill time 4.7 s -> ~1283 t/s."""
        s = srv()
        t = 1000.0
        s._apply_vllm_metrics("(vllm)", vbody(lc=1000), t)
        s._apply_vllm_metrics(
            "(vllm)", vbody(lc=7030, pf_n=1, pf_sum=4.7, pk_n=1, pk_sum=6030), t + 30)
        st = s.models["(vllm)"]
        self.assertAlmostEqual(st["tpp"], 6030 / 4.7, delta=1)
        self.assertEqual(st["phase"]["prefill"]["n"], 1)
        self.assertEqual(st["phase"]["window_s"], 30)
        self.assertLessEqual(st["prefill_speed_age"], 30)

    def test_c_cached_prompt(self):
        """spec §16C: 30K requested, 1500 computed, 28.5K cache-served."""
        s = srv()
        t = 1000.0
        s._apply_vllm_metrics("(vllm)", vbody(lc=0, lh=0), t)
        s._apply_vllm_metrics(
            "(vllm)",
            vbody(lc=1500, lh=28500, pt=30000,
                  pf_n=1, pf_sum=1.2, pk_n=1, pk_sum=1500), t + 30)
        st = s.models["(vllm)"]
        pw = st["prompt_work"]
        self.assertEqual(pw["total"], 30000)
        self.assertEqual(pw["computed"], 1500)      # not 30000
        self.assertEqual(pw["cached"], 28500)
        self.assertAlmostEqual(pw["hit_ratio"], 0.95, places=3)
        self.assertAlmostEqual(st["tpp"], 1500 / 1.2, delta=1)

    def test_d_counter_reset(self):
        """spec §16D: engine restart mid-window -> no spike, no negative rate."""
        s = srv()
        t = 1000.0
        s._apply_vllm_metrics("(vllm)", vbody(lc=2_000_000), t)
        s._apply_vllm_metrics("(vllm)", vbody(lc=10_000, pf_n=1, pf_sum=2.0,
                                              pk_n=1, pk_sum=500), t + 2)
        st = s.models["(vllm)"]
        # 1-min ring must have been cleared (single sample again)
        self.assertIsNone(st["pp_tput_1m"])
        # the performance figure comes from the phase histograms, which
        # survived the restart
        self.assertAlmostEqual(st["tpp"], 500 / 2.0, delta=1)
        self.assertEqual(st["prompt_tokens_computed_raw"], 10_000)

    def test_e_mux_engine_switch(self):
        """spec §16E: switching the active model re-baselines every window."""
        s = srv("vllm-mux")
        t = 1000.0
        s._apply_vllm_metrics("A", vbody(lc=5000), t)
        s._apply_vllm_metrics("A", vbody(lc=9000), t + 2)
        s._apply_vllm_metrics("B", vbody(lc=100), t + 4)   # engine switch
        self.assertEqual(len(s._winbuf), 1)
        self.assertEqual(len(s._lc_hist["B"]), 1)
        self.assertEqual(s._metrics_mid, "B")
        # B's first sample cannot produce a straddled rate
        st = s.models["B"]
        self.assertIsNone(st["pp_tput_1m"])
        self.assertIsNone(st["tpp"])

    def test_f_initial_prefill(self):
        """spec §16F: running request, no completed stats -> live timer,
        no invented rate, nothing that could read as degradation."""
        s = srv()
        t = 1000.0
        s._apply_vllm_metrics("(vllm)", vbody(run=1, kv=0.05), t)
        st = s.models["(vllm)"]
        self.assertIsNone(st["tpp"])
        self.assertTrue(st["live_prefill"]["active"])
        self.assertEqual(st["live_prefill"]["elapsed_s"], 0)
        s._apply_vllm_metrics("(vllm)", vbody(run=1, kv=0.09), t + 12)
        st = s.models["(vllm)"]
        self.assertEqual(st["live_prefill"]["elapsed_s"], 12)
        # first token: gen + ttft move -> prefill over, timer cleared
        s._apply_vllm_metrics(
            "(vllm)", vbody(run=1, kv=0.09, gen=1, ttft_n=1, ttft_sum=12.0,
                            stop=1, pf_n=1, pf_sum=11.5, pk_n=1, pk_sum=18000),
            t + 14)
        st = s.models["(vllm)"]
        self.assertIsNone(st["live_prefill"])
        self.assertAlmostEqual(st["tpp"], 18000 / 11.5, delta=2)

    def test_g_length_never_degrades(self):
        """spec §16G: even 100% length-limited completions must not change
        health. The hub's data carries the count; the UI rule (tested below
        against the real modelState) must not map it to degraded."""
        s = srv()
        t = 1000.0
        s._apply_vllm_metrics("(vllm)", vbody(length=1, run=1), t)
        s._apply_vllm_metrics(
            "(vllm)", vbody(length=50, run=1, gen=100), t + 30)
        st = s.models["(vllm)"]
        self.assertEqual(st["finish"]["length"], 49)
        # and the preemption-based trigger must be untouched by this
        self.assertIn("preempt_win", st)
        self.assertEqual(st["preempt_win"], 0)

    def test_h_high_kv_no_consequences(self):
        """spec §16H: 95% KV, no queue, no preemption -> no degraded inputs."""
        s = srv()
        t = 1000.0
        s._apply_vllm_metrics("(vllm)", vbody(kv=0.95, run=1, gen=50), t)
        s._apply_vllm_metrics("(vllm)", vbody(kv=0.95, run=1, gen=90), t + 30)
        st = s.models["(vllm)"]
        self.assertEqual(st["preempt_win"], 0)
        self.assertIsNone(st["preempt_total"] if st["preempt_total"] else None)

    def test_i_preemption_degrades(self):
        """spec §16I: one preemption in the window is the degraded signal,
        independent of KV level."""
        s = srv()
        t = 1000.0
        s._apply_vllm_metrics("(vllm)", vbody(pre=0, kv=0.2, run=1), t)
        s._apply_vllm_metrics("(vllm)", vbody(pre=1, kv=0.2, run=1), t + 30)
        st = s.models["(vllm)"]
        self.assertEqual(st["preempt_win"], 1)
        self.assertEqual(st["preempt_total"], 1)

    def test_j_missing_phase_histograms(self):
        """spec §16J: older build without the phase histograms -> no crash,
        None, nothing invented; the by_source-free fallback still works."""
        s = srv()
        t = 1000.0
        s._apply_vllm_metrics(
            "(vllm)", vbody(phase=False, by_source=False, prefix_counters=True), t)
        s._apply_vllm_metrics(
            "(vllm)",
            vbody(phase=False, by_source=False, prefix_counters=True,
                  lc=4000, lh=2000, pt=6000, run=1), t + 2)
        st = s.models["(vllm)"]
        self.assertIsNone(st["tpp"])
        self.assertIsNone(st["phase"])
        self.assertTrue(st["metrics_ok"])
        # fallback computed share via queries - hits
        self.assertEqual(st["prompt_tokens_computed_raw"], 4000)
        self.assertIsNotNone(st["prompt_work"])
        self.assertEqual(st["prompt_work"]["total"], 6000)
        self.assertEqual(st["prompt_work"]["computed"], 4000)

    def test_prometheus_export(self):
        """new series present, old ambiguous gauge gone for vLLM, counter
        monotone across scrapes, reset visible as a drop (a Prometheus
        counter reset)."""
        s = srv()
        hub.SERVERS[:] = [s]
        t = 1000.0
        s._apply_vllm_metrics("(vllm)", vbody(lc=1000, pf_n=1, pf_sum=2.0,
                                             pk_n=1, pk_sum=2000), t)
        txt1 = hub.prom_text()
        # 60 s of steady work (100 computed tokens per 2 s poll) so the
        # 1-min throughput gauge has a full window; completed prefill
        # samples keep arriving so the phase figures stay alive
        lc, pf_n, pf_sum, pk_sum = 1000.0, 1, 2.0, 2000.0
        for _ in range(30):
            t += 2
            lc += 100
            pf_n += 1
            pf_sum += 0.5
            pk_sum += 500
            s._apply_vllm_metrics("(vllm)", vbody(lc=lc, pf_n=pf_n,
                                                 pf_sum=pf_sum, pk_n=pf_n,
                                                 pk_sum=pk_sum), t)
        txt2 = hub.prom_text()
        for needle in ("hub_model_prompt_tokens_computed_total",
                       "hub_model_prompt_compute_throughput_tokens_per_second",
                       "hub_model_prefill_in_flight",
                       "hub_model_prefill_speed_age_seconds",
                       "hub_model_prefill_speed_tokens_per_second",
                       "hub_model_prefill_p50_seconds",
                       "hub_model_prefill_p95_seconds"):
            self.assertIn(needle, txt2)
        # the 1-min gauge: 3000 computed tokens over 60 s == 50 t/s
        tput = float(re.search(
            r"hub_model_prompt_compute_throughput_tokens_per_second\{[^}]*\} ([0-9.]+)",
            txt2).group(1))
        self.assertAlmostEqual(tput, 50.0, delta=1)
        # the retired gauge must not come back for vLLM models
        self.assertNotIn("hub_model_prompt_tokens_per_second", txt2)
        self.assertNotIn("hub_model_prefill_avg_seconds", txt2)
        v1 = float(re.search(r"hub_model_prompt_tokens_computed_total\{[^}]*\} ([0-9.]+)", txt1).group(1))
        v2 = float(re.search(r"hub_model_prompt_tokens_computed_total\{[^}]*\} ([0-9.]+)", txt2).group(1))
        self.assertGreater(v2, v1)
        # and a reset reads as a drop = a Prometheus reset
        s._apply_vllm_metrics("(vllm)", vbody(lc=5, pf_n=1, pf_sum=1.0,
                                             pk_n=1, pk_sum=10), t + 2)
        txt3 = hub.prom_text()
        v3 = float(re.search(r"hub_model_prompt_tokens_computed_total\{[^}]*\} ([0-9.]+)", txt3).group(1))
        self.assertLess(v3, v2)

    def test_llama_branch_unchanged(self):
        """the llama.cpp model keeps the old gauge name (its rate is a
        child-clock rate — the semantics were already correct)."""
        s = srv("llama-router")
        hub.SERVERS[:] = [s]
        st = s.models.setdefault("M", {"id": "M", "kind": "llama.cpp",
                                       "loaded": True})
        st.update({"tgen": 10.0, "tpp": 500.0})
        txt = hub.prom_text()
        self.assertIn("hub_model_prompt_tokens_per_second", txt)
        self.assertNotIn("hub_model_prompt_tokens_computed_total", txt)


# ---------------------------------------------------------------------------
# the UI state derivation — run the REAL index.html function in node
# ---------------------------------------------------------------------------

def _extract_modelstate():
    src = open(os.path.join(HERE, "ui", "index.html"), encoding="utf-8").read()
    m = re.search(r"function modelState\(m\) \{", src)
    assert m, "modelState not found in index.html"
    i = m.start()
    depth, j = 0, src.index("{", i)
    for k in range(j, len(src)):
        if src[k] == "{":
            depth += 1
        elif src[k] == "}":
            depth -= 1
            if depth == 0:
                break
    return src[i:k + 1]


def test_ui_state_matrix():
    fn = _extract_modelstate()
    cases = {
        # (name, model object, expected state)
        "G1 one length, idle":      ({"stale": False, "loaded": True, "kv_used": 0.1,
                                      "finish": {"length": 1}, "preempt_win": 0}, "idle"),
        "G2 100% length, running":  ({"stale": False, "loaded": True, "kv_used": 0.2,
                                      "finish": {"total": 50, "length": 50},
                                      "req_running": 1, "preempt_win": 0}, "active"),
        "H  95% kv, no consequences": ({"stale": False, "loaded": True, "kv_used": 0.95,
                                        "req_running": 1, "req_waiting": 0,
                                        "preempt_win": 0}, "busy"),
        "I  one preemption":        ({"stale": False, "loaded": True, "kv_used": 0.3,
                                      "req_running": 1, "preempt_win": 1}, "degraded"),
        "I2 preemption, high kv":   ({"stale": False, "loaded": True, "kv_used": 0.85,
                                      "req_running": 1, "preempt_win": 2}, "degraded"),
        "Q  queue + 90% kv":        ({"stale": False, "loaded": True, "kv_used": 0.9,
                                      "req_running": 1, "req_waiting": 3,
                                      "preempt_win": 0}, "degraded"),
        "S  stalled engine":        ({"stale": False, "loaded": True, "kv_used": 0.4,
                                      "req_running": 1, "stalled": True,
                                      "preempt_win": 0}, "degraded"),
        "P  metrics parse failure": ({"stale": False, "loaded": True, "kv_used": 0.1,
                                      "metrics_ok": False, "preempt_win": 0}, "degraded"),
        "L  live prefill is active": ({"stale": False, "loaded": True, "kv_used": 0.05,
                                       "req_running": 1, "preempt_win": 0,
                                       "live_prefill": {"active": True, "elapsed_s": 9}},
                                      "active"),
    }
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as f:
        f.write(fn + "\n")
        for name, (model, want) in cases.items():
            f.write("console.log(JSON.stringify("
                    + "{n: " + json.dumps(name)
                    + ", s: modelState(" + json.dumps(model)
                    + "), w: " + json.dumps(want) + "}))\n")
        path = f.name
    out = subprocess.run(["node", path], capture_output=True, text=True)
    os.unlink(path)
    assert out.returncode == 0, out.stderr
    for line in out.stdout.strip().splitlines():
        r = json.loads(line)
        assert r["s"] == r["w"], f'{r["n"]}: got {r["s"]}, want {r["w"]}'


if __name__ == "__main__":
    unittest.main(verbosity=2, exit=False)
    test_ui_state_matrix()
    print("ui state matrix: all cases passed")
