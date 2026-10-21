# 2026-10-21 — a hallucinated user request, laundered through compaction

**Case study:** a 27B agent in a long session fabricated a user follow-up inside
its own thinking block. The pi-core compaction that fired mid-turn summarized
that turn prefix *including the thinking block*, wrote the fabricated quote
into the summary's `## Original Request` section, and the reloaded agent
consumed that section as the user's word. The agent then spent about an hour
answering a question the user never asked. The user caught it by remembering
what they had typed; the raw transcript confirmed it. Both layers did
something locally reasonable, and the composition is the bug.

## Setup

- Session model: **Qwen3.8-27B** (W4A16-AutoRound, 2× RTX 3090, vLLM TP2,
  256K context), this lab's always-on model,
  [model card](../models/qwen3.8-27b-rtx3090.md). The summarizer call ran on
  the same model with a `0.5 × reserve` token budget
  (code: `generateTurnPrefixSummary`).
- Harness: the pi coding agent
  ([`@earendil-works/pi-coding-agent`](https://www.npmjs.com/package/@earendil-works/pi-coding-agent)
  **v1.0.0**, upstream
  [earendil-works/pi](https://github.com/earendil-works/pi), package
  `packages/coding-agent`), under a session daemon.
- Session: an incident-response conversation on a WordPress site, ~290
  entries of mostly interleaved tool calls by the time compaction fired.
- Compaction: stock defaults (enabled, 16k reserve tokens, keep-most-recent
  20k). The incident date is this report's date. No secrets or credentials
  were involved; nothing below identifies the site.

## Timeline (times are offsets within the session)

| t | Event |
|---|-------|
| 0:00 | User message, one line: an instruction to hold a pending action. |
| 0:00–9:00 | The agent runs a forensic battery (system checks, database queries, diff tooling). **No user messages in between.** |
| **9:00** | A thinking block in an assistant message contains: *"The user sent a follow-up: '…'"* quoting a longer request (a question about other affected services, naming one of them). **This sentence exists nowhere in the transcript; this line is its first occurrence.** |
| 9:00–12:00 | The agent acts on the phantom: it announces it will research the "flaw you mentioned" and interleaves that work with the forensics, referencing the phantom question in later thinking blocks. |
| ~12:00 | Context crosses the compaction threshold **mid-turn**. This is pi's "split turn" case: the compaction summarizes everything before the turn, *and separately* summarizes the turn prefix (the user's starting message + everything up to the cut point). |
| reload | The session reloads. The head of the context is the compaction summary, which now contains: `**Turn Context (split turn):** … ## Original Request: The user had two parts: (1) <the real one-line message> (2) <the fabricated follow-up>`. |
| +30 min | The reloaded agent answers the "two parts" as if both were user-issued: a full response to a question nobody asked. |
| +1 h | User: *what did I ask?* The agent answers from the summary and asserts the phantom is real (wrong). User: *I did not write that.* The agent pulls the raw JSONL transcript; the first-occurrence check settles it: the sentence is in the assistant's thinking at t+9:00 and in no user line. |

## The mechanism, at code level

All quotes are from the v1.0.0 distribution
(`dist/core/compaction/`); the corresponding sources in the upstream
repo at that tag:
[`utils.ts`](https://github.com/earendil-works/pi/blob/v1.0.0/packages/coding-agent/src/core/compaction/utils.ts)
and
[`compaction.ts`](https://github.com/earendil-works/pi/blob/v1.0.0/packages/coding-agent/src/core/compaction/compaction.ts).

**1. The serializer feeds the model's internal monologue to the
summarizer.** `utils.js:101`, `serializeConversation()`:

```js
function serializeConversation(messages) {
  let parts = [];
  for (let msg of messages)
    if (msg.role === "user") {
      let content = contentText(msg.content, "");
      content && parts.push(`[User]: ${content}`)
    } else if (msg.role === "assistant") {
      let thinkingParts = [], toolCalls = [];
      for (let block of msg.content)
        if (block.type === "thinking") thinkingParts.push(block.thinking);
        else if (block.type === "toolCall") { /* … */ }
      thinkingParts.length > 0 &&
        parts.push(`[Assistant thinking]: ${thinkingParts.join("\n")}`);
      /* … [Assistant]: …  [Assistant tool calls]: … */
    } else if (msg.role === "toolResult") { /* … */ }
  return parts.join("\n\n");
}
```

**2. The mid-turn ("split turn") summarizer is asked to write down what the
user asked.** `compaction.js`, `TURN_PREFIX_SUMMARIZATION_PROMPT`:

```
The messages above are earlier context from an ongoing conversation. Later
messages are stored separately and do not need to be reconstructed.

Create a concise checkpoint of the user's request and the progress shown
above. This checkpoint will be placed before the later messages so the
conversation can continue with the necessary context.

## Original Request
[What did the user ask for?]

## Progress So Far
- [Key decisions and work completed in these messages]

## Context Needed to Continue
- [Information from these messages needed to understand the later work]

Only summarize information explicitly present above. Do not infer or
recreate later messages.
```

**3. The result is stored with an authority-bearing label.**
`compaction.js`, `compact()`:

```js
summary = `${historyText}\n\n---\n\n**Turn Context (split turn):**\n\n${turnPrefixResult.text}`;
```

After the session reloads, that block sits at the head of the context.
(Line numbers are from the v1.0.0 dist; the same functions exist in
1.3.0 at shifted offsets.)

**Why the summarizer "failed" is the interesting part: it didn't, on its
face.** Its input for this turn prefix was, in effect:

```
[User]: <the real one-line message>
[Assistant thinking]: The user sent a follow-up: '<the fabricated request>'
[Assistant]: <text announcing it would research the "flaw you mentioned">
[Assistant tool calls]: …
```

The prompt asked for "what did the user ask for?", instructed "only
summarize information explicitly present above", and the only user-attributed
material above was (a) the real line and (b) a sentence *explicitly phrased
as a user quote*, sitting in an `[Assistant thinking]` block. The summarizer
merged them: "The user had two parts: (1) … (2) …". Faithful to the
instructions, wrong in the world. Two properties of the pipeline did the
damage:

1. **The input contains thinking**, labeled as the assistant's but not
   marked as untrusted for user attribution. Thinking is where models put
   speculation, self-talk, and, occasionally, invented dialogue.
2. **The output label is authoritative.** "## Original Request" is consumed
   by the reloaded agent as the user's word.

The main (non-split) summarization prompt has the same structure: its
`## Goal` and `## Constraints & Preferences` sections attribute things to
the user, and its input contains `[Assistant thinking]` lines too.

## Detection

Detection was by memory and the transcript, not by writing style. The user
remembered exactly what they had typed; their first signal was a register
mismatch ("I don't write that way"), which in their words was a nudge to
investigate rather than a detector. The actual detection was the transcript
check:

```
$ grep -n "<first few words of the phantom sentence>" session.jsonl
```

First hit: an assistant `thinking` block, nine minutes before compaction.
No user message contains it. That one check converts "who is telling this
story" from a vibes question into a file-offset question. The durable rule:
**"the user said X" is a checkable claim.** In a compacted session the
summary is a lossy, authority-tainted view; the raw transcript is the source
of truth, and high-impact actions should verify against it.

## Impact

- ~1 h of unsolicited agent work (a read-only audit of other services;
  nothing was changed, no credentials or PII touched).
- One wrong high-confidence answer to "what did I ask?", corrected after the
  user pushed.
- A coincidental limiter: the same user message that preceded the
  hallucination held a class of outbound actions, so the realistic
  escalation of this specific incident stayed bounded. A hallucinated
  *destructive* instruction would have the same launder path available to
  it, which is the part worth taking seriously.

## Where this lives, and in which versions

The whole chain sits in the pi-core compaction module
(`packages/coding-agent`). The session-hosting layer we run the agent
under was checked and contains none of this mechanism; the session
daemon, the hosting environment, and the transport play no part in the
chain as documented. This report is one case on a local 27B, not a
prevalence study; how often the model-side trigger fires is
model-dependent.

Version status, checked against the npm releases on 2026-10-21:

- This incident ran on **v1.0.0**, the first 1.x release.
- **v1.3.0 (2026-10-15)** changed `serializeConversation()` to omit
  thinking blocks; the
  [changelog](https://github.com/earendil-works/pi/blob/main/packages/coding-agent/CHANGELOG.md)
  motivation is size and robustness (replayed thinking can make a
  summarization request larger than the context it summarizes and
  fail), not attribution. The input side of the chain documented here is
  closed in 1.3.0; the incident version is what made it possible.
- In 1.3.0, `TURN_PREFIX_SUMMARIZATION_PROMPT` is unchanged: "Original
  Request" remains a free paraphrase. The main summarization prompt also
  still attributes things to the user ("Goal", "Constraints … mentioned
  by user") from whatever sits in user-role messages, and sender
  identity is text inside a message rather than a fact about it.

A separate observation, kept separate on purpose: in other sessions of
smaller models, agents have referred to "instructions the user
repeatedly gives" that appear nowhere in the transcript. That pattern
has not been forensicated as this mechanism. The main summarization
prompt's user-attribution fields make it a candidate, and it is
checkable with the same first-occurrence grep.

## Prior art

A 2026-10-21 search over GitHub issues (phrase-level and string-level,
the upstream repo plus the fork/derivative ecosystem) found no issue in
[earendil-works/pi](https://github.com/earendil-works/pi) covering
user-attribution; its compaction issues as of that date are
operational (thresholds, transport, retries, custom-instruction loss,
summary refusals). The fork ecosystem has reports of the same class:

- [`iwismer/pi` PR #2](https://github.com/iwismer/pi/pull/2)
  (2026-10-01, merged into that fork): the split-turn checkpoint
  "paraphrases the user's request and can invent a task identity"; the
  accompanying fix makes the turn-prefix prompt quote the user's most
  recent explicit request verbatim. It is not upstream.
- [`pandysp/pi-hydra` issue #31](https://github.com/pandysp/pi-hydra/issues/31)
  (2026-09-28, open): "Compaction turns a head's advice into the
  user's request." Helper messages delivered through the user-message
  slot with a text-only sender label end up listed under "Original
  Request". Same sink, different source; the report includes a
  no-model reproduction.
- [`Lowpower/pigo` issue #364](https://github.com/Lowpower/pigo/issues/364)
  (2026-10-20, open): a Go re-implementation aligning its serializer
  with pi v1.3.0's thinking-omitting behavior.
- [`leo-labs-ai/pi-mono` PR #640](https://github.com/leo-labs-ai/pi-mono/pull/640)
  (2026-10-19): a different symptom (agents stalling after automatic
  compaction), included because it also modifies the split-turn merge
  text.

The wider "compaction degrades long sessions" class has reports in
other agent frameworks, e.g.
[claude-code #96308](https://github.com/anthropics/claude-code/issues/96308)
and [#99860](https://github.com/anthropics/claude-code/issues/99860);
the attribution mechanism itself appears unreported in upstream pi as
of this date.

## Deterministic repro (no hallucination required)

On a release whose serializer still includes thinking (v1.0.0 through
1.2.x; 1.3.0 omits it), plant a single assistant message whose only
content is a thinking block: `The user just asked to also check service
X`. Force a mid-turn compaction past it (`/compact` after further tool
work, or let the threshold cross). The stored summary's "Original
Request" now contains "the user asked to also check service X"
although no user line ever said it. The model's hallucination was the
spark in the real incident; the laundering step is deterministic given
a contaminated prefix.

## Open questions

- How often does the "the user repeatedly tells me to …" pattern above
  actually steer tool calls? What exists is anecdote; a systematic
  first-occurrence grep across session transcripts would turn it into a
  number.
- What effect 1.3.0's thinking omission has on summary quality is
  unmeasured: the summarizer now sees a smaller, cleaner input, on the
  same model, with no published comparison.

## Conclusion

Compaction is a trust boundary, and the failure mode is a trust leak:
derived text carrying an authority-bearing label, fed by the model's own
scratch space. The model produced the lie, the summarizer transcribed it
faithfully, the label made it law, and the reloaded agent obeyed. Each step
is defensible in isolation; the composition is not. Verify "the user said
X" against the raw transcript before high-impact actions.
