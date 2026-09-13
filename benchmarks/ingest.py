"""ingest.py — post-run: pull, verify completeness, verify production, draft reports.

The owner's morning should be a READING session, not an assembly session: this
pulls the run, checks that every cell/attempt has its required artifacts
(a missing artifact is a finding, not a silent gap), re-proves production is
alive (health + live completion, exactly as the window's exit claimed), and
generates two report drafts:

  * private skeleton  — full paths, verdicts, observed/inferred/not_established
    per cell (the private repo's report conventions are applied at commit time)
  * public skeleton  — the sanitized llmlab candidate (dated report, no
    frontmatter; sanitizer sweep at push time per the llmlab skill)

Both keep the observed / inferred / not_established split — the report
generator does not let a hypothesis read like a measurement.
"""
import json
import os
import shutil
import time

REQUIRED_ATTEMPT_FILES = ("launch.sh", "meta.json", "prompt.txt", "verdict.json")
# one of client.json / client-error.json must exist


def pull(backend, run_dir, local_dir):
    """v0: local copy when the run dir is local (fixture / same-machine); the
    SSH pull path mirrors local_backend.get_file per-file (live-untested)."""
    if os.path.isdir(run_dir):
        if os.path.abspath(run_dir) != os.path.abspath(local_dir):
            if os.path.exists(local_dir):
                shutil.rmtree(local_dir)
            shutil.copytree(run_dir, local_dir)
        return local_dir
    os.makedirs(local_dir, exist_ok=True)
    for root, _, files in os.walk(run_dir):
        for f in files:
            rel = os.path.relpath(os.path.join(root, f), run_dir)
            backend.get_file(os.path.join(run_dir, rel), os.path.join(local_dir, rel))
    return local_dir


def completeness(local_dir):
    """Every attempt dir must carry its required artifacts. Returns findings."""
    findings = []
    attempts = os.path.join(local_dir, "attempts")
    if not os.path.isdir(attempts):
        return ["no attempts/ directory"]
    for a in sorted(os.listdir(attempts)):
        a_dir = os.path.join(attempts, a)
        for req in REQUIRED_ATTEMPT_FILES:
            if not os.path.exists(os.path.join(a_dir, req)):
                findings.append(f"{a}: missing {req}")
        if not (os.path.exists(os.path.join(a_dir, "client.json"))
                or os.path.exists(os.path.join(a_dir, "client-error.json"))):
            findings.append(f"{a}: neither client.json nor client-error.json")
    return findings


def _cell_sections(final, local_dir):
    """Per-cell observed/inferred/not_established, from the LATEST attempt's verdict."""
    out = {}
    attempts = os.path.join(local_dir, "attempts")
    if not os.path.isdir(attempts):
        return out
    by_cell = {}
    for a in sorted(os.listdir(attempts)):
        cell = a.split("-")[0]
        by_cell[cell] = a          # sorted => last = latest
    for cell, a in by_cell.items():
        v = {}
        p = os.path.join(attempts, a, "verdict.json")
        if os.path.exists(p):
            v = json.load(open(p))
        out[cell] = {
            "verdict": (final.get("cells") or {}).get(cell),
            "observed": v.get("observed", []),
            "inferred": v.get("inferred", []),
            "not_established": v.get("not_established", []),
            "gates": v.get("gates", {}),
            "reason": v.get("reason", ""),
        }
    return out


def draft_reports(final, local_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    date = time.strftime("%Y-%m-%d", time.gmtime())
    cid = final.get("campaign_id", "campaign")
    cells = _cell_sections(final, local_dir)
    slug = cid.split("-")[0] if cid else "run"

    def _cell_md(cell, c):
        lines = [f"### {cell} — `{c['verdict']}`", ""]
        if c["reason"]:
            lines += [f"Reason: {c['reason']}", ""]
        for key, label in (("observed", "Measured / observed"),
                            ("inferred", "Inferred (not measured)"),
                            ("not_established", "Not established")):
            if c.get(key):
                lines.append(f"**{label}:**")
                lines += [f"- {x}" for x in c[key]]
                lines.append("")
        return "\n".join(lines)

    body = [f"# {cid} — platform run ({final['final']})", "",
            f"Final: **{final['final']}** · valid={final.get('valid')} · "
            f"restore={'RESTORED' if final.get('restore', {}).get('restored') else 'ATTENTION'}",
            "", "## Cells", ""]
    body += [_cell_md(k, c) for k, c in sorted(cells.items())]
    private = os.path.join(out_dir, f"private-{slug}.md")
    with open(private, "w") as f:
        f.write("\n".join(body))
    public = os.path.join(out_dir, f"{date}-{slug}-draft.md")
    pub = [f"# {cid} (public draft)", "",
           "> SANITIZER-PENDING: run the llmlab sanitizer sweep before commit/push.", ""]
    pub += [l for l in body if "192" + ".168." not in l and "100" + "." not in l]
    with open(public, "w") as f:
        f.write("\n".join(pub))
    return private, public


def run(backend, profile, final, run_dir, out_dir):
    local_dir = pull(backend, run_dir, os.path.join(out_dir, "run"))
    findings = completeness(local_dir)
    health = backend.prod_health()
    live, detail = backend.prod_live_check()
    restore_ok = (health == 200) and live
    private, public = draft_reports(final, local_dir, out_dir)
    report = {
        "pulled": local_dir,
        "completeness_findings": findings,
        "production": {"health": health, "live": live, "ok": restore_ok,
                       "detail": detail[:200]},
        "drafts": {"private": private, "public": public},
    }
    with open(os.path.join(out_dir, "ingest.json"), "w") as f:
        json.dump(report, f, indent=2)
    return report
