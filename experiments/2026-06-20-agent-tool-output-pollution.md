# Agent Tool Output Pollution — Context Window Corruption

**Date:** 2026-06-20
**Status:** Mitigated (client-side validation), root cause open (server-side)
**Samples:** `garbled-output-sample.json`, `lynx-output-sample.txt`, `agent-browser-output-sample.txt`

## Problem

When an LLM agent calls web extraction tools (`agentread`/`agentsearch -x`), the returned content is injected directly into the agent's context window. If the content contains binary or garbled data, the model produces corrupted output for multiple subsequent turns and must be reset.

### Confirmed Trigger

URL: `https://toolhalla.ai/blog/firecrawl-vs-crawl4ai-vs-jina-reader-2026`

- **Expected:** ~27KB clean text (blog post comparing web scraping tools)
- **Got:** 15,000 chars, 59% non-printable bytes (Brotli-compressed data decoded as UTF-8)
- **Root cause:** Cloudflare bypass adapter sends `Accept-Encoding: gzip, deflate, br`, server responds with Brotli-compressed data, httpx fails to decompress, raw compressed bytes decoded as UTF-8 → replacement characters (U+FFFD = `ef bf bd`)
- **Frequency:** ~5+ incidents observed

### Garbled Output Sample

`garbled-output-sample.json` — 38KB JSON with 15,000-char content field, 59% non-printable. Captured from AgentSearch API response.

## Mitigation: Client-Side Validation

Added `validate-content.py` to the websearch toolchain. Validates all `content` fields before they enter the agent context:

| Check | Threshold | Action |
|-------|-----------|--------|
| Non-printable ratio | >15% | Reject → clean error message |
| Single-line length | >1000 chars + HTML tags | Reject (minified HTML) |
| Content length | <200 chars | Reject (too short) |

Replaces bad content with: `[CONTENT ERROR: binary/compressed (59% non-printable). Strategy 'X' returned unusable data for URL. Try a different URL or use fallback tools.]`

**Result:** Agent sees clean error message instead of binary garbage. Context stays intact.

## Fallback: Browser-Based Extraction

When the API kill chain fails, two browser-based fallbacks provide clean output:

### Lynx (Text-Mode Browser)

```bash
lynx -dump -nolist -width 120 "https://example.com"
```

| Property | Value |
|----------|-------|
| Install | `apt install lynx` |
| Size | ~2MB |
| Speed | ~1-2s |
| JS rendering | ❌ No |
| Sample output | `lynx-output-sample.txt` (30,992 bytes, ~1% non-printable) |

### Agent-Browser (Full Chrome Automation)

```bash
agent-browser open "https://example.com"
agent-browser get text body
agent-browser close
```

| Property | Value |
|----------|-------|
| Install | `bun install -g agent-browser && agent-browser install` |
| Size | ~230MB (npm + Chrome) |
| Speed | ~2-5s |
| JS rendering | ✅ Full Chrome |
| Sample output | `agent-browser-output-sample.txt` (27,559 bytes, 0.3% non-printable) |

### Comparison

| Method | Bytes | Non-print | JS | Quality |
|--------|-------|-----------|----|----|
| API (kill chain) | 15,000 | 59.0% | — | **BINARY GARBAGE** |
| lynx -dump | 30,992 | ~1.0% | ❌ | Clean text |
| agent-browser | 27,559 | 0.3% | ✅ | Clean text + JS |

Both browsers return clean, readable content. Agent-browser has slight edge in Unicode handling and no navigation artifacts.

## Open Questions

1. **Server-side fix:** AgentSearch Cloudflare bypass adapter needs Brotli decompression fix. Not yet patched upstream.
2. **Model sensitivity:** How much garbled content triggers breakdown in which models? Need systematic testing across model families.
3. **Recovery patterns:** Can models recover from partial pollution? What's the contamination radius (how many subsequent turns are affected)?
4. **Fallback integration:** Should browsers be auto-invoked on validation failure, or kept as manual fallback?

## Future Research

- [ ] Test garbled content across model families (Llama, Qwen, Mistral, etc.)
- [ ] Measure minimum pollution threshold for each model
- [ ] Document recovery patterns and contamination radius
- [ ] Evaluate auto-fallback vs manual fallback tradeoffs
- [ ] Assess whether AgentSearch server-side fix is available upstream

## Related

- [Tool Output Context Pollution Report](../../homelab/reports/2026-06-13_report_tool-output-context-pollution.md) — Detailed incident analysis
- [Local AI Agent Web Access Landscape](../../homelab/reports/2026-06-13_research_local-ai-agent-web-access-landscape.md) — Tooling research
- [Websearch Toolchain](../../homelab/scripts/websearch/README.md) — Setup and usage
