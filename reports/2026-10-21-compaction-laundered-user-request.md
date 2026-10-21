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

- The estate's always-on 27B (Qwen3.8, dual RTX 3090, vLLM) as the session
  model, running the pi coding agent
  (`@earendil-works/pi-coding-agent` **v1.0.0**, upstream
  `github.com/earendil-works/pi`, package `packages/coding-agent`), under a
  session daemon.
- Session: an incident-response conversation on a WordPress site, ~290
  entries of mostly interleaved tool calls by the time compaction fired.
- Compaction: stock defaults (enabled, 16k reserve tokens, keep-most-recent
  20k). The summarizer call is made with the session's model and a
  `0.5 × reserve` token budget (code: `generateTurnPrefixSummary`).
- The incident date is this report's date. No secrets or credentials were
  involved; nothing below identifies the site.

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

All quotes are from the pi-core v1.0.0 distribution
(`dist/core/compaction/`).

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
instructions, wrong in the world. Two design properties did the damage:

1. **The input contains thinking**, labeled as the assistant's but not
   marked as untrusted for user attribution. Thinking is where models put
   speculation, self-talk, and, occasionally, invented dialogue. It is the
   least user-like content in a transcript and the most plausible-looking
   source of "the user said X".
2. **The output label is authoritative.** "## Original Request" is consumed
   by the reloaded agent as the user's word. Had the section been labeled
   "what the summarized context *appears to* be asking", the reloaded agent
   would have had a reason to check before acting.

The **main** (non-split) summarization prompt has the same vector: it asks
for `## Goal [What is the user trying to accomplish?]` and `## Constraints &
Preferences [… mentioned by user]` while its input again contains
`[Assistant thinking]`. That is the likely source of the recurring pattern,
seen across sessions of smaller models, where the agent writes "the user
repeatedly tells me to …" about instructions that were never given: model-side
claims about the user, promoted into durable summary sections by compaction.

## Detection

Not by style. The user remembered exactly what they had typed; their first
signal was a register mismatch ("I don't write that way"), which in their
words was a nudge to investigate rather than a detector. The actual
detection was the transcript check:

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
  hallucination held a class of outbound actions, so the worst realistic
  escalation of this specific incident was bounded. The general worst case
  is not: a hallucinated *destructive* instruction laundered into "Original
  Request" is a deletion, an exfiltration, or a mass action, not an audit.

Scope note: this is not a prompt-injection attack (no external input), and
it implicates no session-hosting layer; the pi-web distro we run contains
zero occurrences of the "Turn Context / Original Request" mechanism. It is
pi-core plus the model, so **every pi session with compaction enabled is
exposed**, on any host, with any model that ever quotes imagined user
utterances in thinking (small local models do this more often; the "the
user repeatedly tells me to …" pattern above is the chronic form).

## Proposed fixes (for the upstream issue)

1. **`serializeConversation()`:** exclude `thinking` blocks from
   summarization input, or label them explicitly ("model internal monologue,
   untrusted for user attribution").
2. **Both summarization prompts:** "User attribution must come **only**
   from `[User]` lines. Content in `[Assistant thinking]` is the model's
   internal monologue and must never be attributed to the user."
3. **Mechanical anchor before persisting:** the turn's actual user message
   is known (it is in the prefix). Check the produced "Original Request"
   against it (containment/match); on mismatch, tag the section
   "unverified, model-claimed" or drop it.
4. **Relabel:** "## Original Request" → something that signals derived data
   ("## Requested task (summarized; verify against transcript)").

1 and 2 are prompt/serializer changes only; 3 is a small code check; 4 costs
a rename. Fixes 1+2 would have broken the chain at the input; 3 is the guard
that makes the failure detectable rather than silent; 4 changes how the next
agent *reads* the summary even when 1–3 miss an edge.

**Deterministic repro (no hallucination required):** in any pi session, plant
a single assistant message whose only content is a thinking block:
`The user just asked to also check service X`. Force a mid-turn compaction
past it (`/compact` after further tool work, or let the threshold cross).
The stored summary's "Original Request" now contains "the user asked to also
check service X" although no user line ever said it. The model's
hallucination was the spark in the real incident; the laundering step is
deterministic given a contaminated prefix.

## Open questions

- The summarizer here ran on the session's own 27B with a smaller budget. A
  stronger summarizer might resist attributing `[Assistant thinking]`
  content to the user; that is an empirical question, not a design one, and
  the prompt fix should not depend on the summarizer being good.
- Whether pi should treat compaction summaries as a distinct message type
  with a "derived, verify against transcript" envelope on every re-injection
  (today the summary text and the label do all the work).
- The "user repeatedly tells me to …" chronic form: how often does it
  actually steer tool calls? We have anecdotes; a systematic count over
  session transcripts would turn it into a number.

## Conclusion

Compaction is a trust boundary, and the failure mode is a trust leak:
derived text carrying an authority-bearing label, fed by the model's own
scratch space. The model produced the lie, the summarizer transcribed it
faithfully, the label made it law, and the reloaded agent obeyed. Each step
is defensible in isolation; the composition is not. Verify "the user said
X" against the raw transcript before high-impact actions, and keep compaction
summaries out of the position of issuing orders in the user's name.
