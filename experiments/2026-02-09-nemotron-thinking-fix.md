# Nemotron "Lost in Thought" Fix Experiment

**Date:** 2026-02-09
**Model:** Nemotron-3-Nano-30B-A3B-IQ4_NL
**Hardware:** 2× RTX 3060 12GB (tensor-split 1,1)

## Problem

~1.4% of turns produce thinking tokens but no visible output. Model emits stop token after `</think>` without generating response text.

## Root Cause (from research)

- Known issue: llama.cpp GitHub #12339
- Model has `enable_thinking: True` by default in Jinja template
- Thinking can exhaust output budget before visible text is generated
- Template uses `<think>...</think>` XML tags

## Baseline Config

```bash
/opt/llama.cpp/build/bin/llama-server \
  --model /mnt/models/gguf/nemotron-3-nano-30b-a3b/Nemotron-3-Nano-30B-A3B-IQ4_NL.gguf \
  --ctx-size 131072 -b 128 -ub 64 \
  --n-gpu-layers -1 --tensor-split 1,1 \
  --flash-attn 1 --jinja \
  --host 0.0.0.0 --port 8080
```

**Issues:**
- No `--reasoning-format` → thinking mixed in `content` field
- No `--reasoning-budget` → unlimited but uncontrolled

## Test Config

```bash
/opt/llama.cpp/build/bin/llama-server \
  --model /mnt/models/gguf/nemotron-3-nano-30b-a3b/Nemotron-3-Nano-30B-A3B-IQ4_NL.gguf \
  --ctx-size 131072 -b 128 -ub 64 \
  --n-gpu-layers -1 --tensor-split 1,1 \
  --flash-attn 1 --jinja \
  --reasoning-format deepseek --reasoning-budget -1 \
  --host 0.0.0.0 --port 8080
```

**Changes:**
- `--reasoning-format deepseek` → extracts `<think>` into `reasoning_content` field
- `--reasoning-budget -1` → unlimited thinking (model decides when to stop)

## Expected Behavior

- `reasoning_content`: contains thinking/planning
- `content`: contains visible output only
- If thinking-only failure occurs, `content` will be empty (clear diagnostic)

## Test Plan

1. Restart server with new flags
2. Run failure-prone prompt 5× in llmlab
3. Check for thinking-only responses
4. Examine response structure via sessions_history

## Results

### Test 1: Weather query (simple)
- **Prompt:** "can you check the weather here for tomorrow?"
- **Result:** ✅ Success
- **Thinking:** Properly extracted into `thinking` field
- **Output:** "Sure! Could you let me know which location..."
- **Tool calls:** Read SKILL.md, read memory file — all worked

### Test 2: File creation (multi-step)
- **Prompt:** "Create a file /tmp/nemotron-test.txt with text, cat it back"
- **Delivery:** Via `sessions_send` (Matrix mention routing issue — see below)
- **Result:** ✅ Task completed successfully
- **Thinking:** Model reasoned through steps properly
- **Output:** "File created at `/tmp/nemotron-test.txt` with content..."
- **Issue:** Response didn't route back to Matrix room (sessions_send context issue)

### Reasoning format: WORKING ✅
The `--reasoning-format deepseek` flag properly separates:
- `thinking` field: internal reasoning
- `text` field: visible output

No "lost in thought" failures observed in limited testing.

### Side issue: Matrix mentions
Agent-to-agent mentions via `message` tool don't trigger `requireMention`.
See: `notes/matrix/agent-to-agent-mentions.md`

---

## Test 3: faster-whisper full install (stress test)

**Prompt:** Install faster-whisper with uv, add to PATH, create skill wrapper
**Context:** 131k

### What LocalBot accomplished:
1. ✅ Created wrapper script at `/var/lib/clawdbot/.local/bin/faster-whisper`
2. ❌ `faster-whisper --help` failed: `ModuleNotFoundError: No module named 'av._core'` (FFmpeg dependency issue)
3. ✅ Spawned sub-agent for skill creation
4. 🔴 **Context exhausted** at 131072 tokens mid-thought

### Issues discovered:
1. **av package:** Python package installed but C extensions missing (needs system FFmpeg)
2. **Context burnout:** `--reasoning-budget -1` (unlimited) + verbose thinking = context wall

### Lesson:
- Consider `--reasoning-budget 4096` or similar cap for long tasks
- 131k context may not be enough for complex multi-step tasks with unlimited thinking

### Next steps:
- Try 262k context
- Fix av/FFmpeg dependency
- Retry faster-whisper task

---

## Server Restart Log

**2026-02-09 22:17 UTC** — Restarted with new flags
- PID: 6154
- Health check: `{"status":"ok"}`
- Flags confirmed: `--reasoning-format deepseek --reasoning-budget -1`

