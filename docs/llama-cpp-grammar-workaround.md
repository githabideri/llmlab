# llama.cpp Grammar Repetition Threshold Workaround

## Problem

Recent llama.cpp builds have a hardcoded grammar repetition threshold that's too low for complex tool-call schemas, causing "Failed to parse input at pos..." errors during tool-enabled inference.

## Root Cause

In `src/llama-grammar.cpp`, the constant `MAX_REPETITION_THRESHOLD` defaults to **2000**. Complex JSON schemas with many optional parameters (like tool-call definitions) can exceed this limit, triggering grammar parsing failures.

**Symptoms:**
```
Failed to parse input at pos XXXX: <tool_call>...
error parsing grammar ... repetition exceeds sane defaults
```

**Related issues:**
- [#20867](https://github.com/ggml-org/llama.cpp/issues/20867) - `MAX_REPETITION_THRESHOLD` (2000) breaks tool-calling grammars for tools with many optional parameters
- [#20860](https://github.com/ggml-org/llama.cpp/issues/20860) - Bug: Failed to parse grammar
- [#20260](https://github.com/ggml-org/llama.cpp/issues/20260) - PEG parser fails when model outputs text before `<tool_call>`

## Affected Versions

- Builds with `MAX_REPETITION_THRESHOLD` = 2000 (default)
- Recent regressions starting around commit `990e4d9` (March 2026)
- No runtime flag exists to override this threshold

## Workaround

Increase the threshold in source code before building:

**File:** `src/llama-grammar.cpp`

```diff
-#define MAX_REPETITION_THRESHOLD 2000
+#define MAX_REPETITION_THRESHOLD 100000
```

**Build:**
```bash
cd /path/to/llama.cpp
cmake -B build
cmake --build build --config Release -t llama-server -j$(nproc)
```

**Deploy:**
```bash
# Backup existing binary
cp ./build/bin/llama-server ./build/bin/llama-server.bak

# Replace or use new binary
# Restart service to load new binary
systemctl restart llama-server
```

## Verification

Test with a tool-enabled request. The grammar parsing error should no longer occur.

## What This Fixes

- Grammar compile failure for tools with many optional parameters
- "Failed to parse input at pos..." errors with tool-call XML (`<tool_call>`)

## What This Does NOT Fix

- **Qwen "thinking" text before tool calls:** If the model emits natural language before the `<tool_call>` tag, the PEG parser may still fail. This is a separate issue.
- **Solution for thinking issue:** Disable thinking with `--chat-template-kwargs '{"enable_thinking": false}'`

## Notes

- This is a source-level patch; no runtime flag exists yet to override the threshold
- The value 100000 provides headroom for complex schemas while staying reasonable
- After rebuilding, services must be restarted to load the patched binary
- Consider pinning llama.cpp to a known-good commit until upstream adds runtime control

## References

- Primary issue: https://github.com/ggml-org/llama.cpp/issues/20867
- llama.cpp source: `src/llama-grammar.cpp`
- Build docs: https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md
