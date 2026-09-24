# speaker-viewer

A single self-contained HTML file (no dependencies, no build) that visualizes
**speaker diarization**: per-speaker color lanes with a synchronized playhead,
click-to-seek, and a transcript colored per word-midpoint.

It renders any diarization payload in the JSON shape used by the
[audio-assist console](../../reports/2026-09-24-audio-assist-transcription-diarization-console.md)
(and by `nvidia/Nemotron-3-Diarization` directly, after the model card's
`/diar` → JSON conversion): 8-speaker ceiling, 80 ms frames for Nemotron, or any
coarser segmentation.

## Try it

```sh
python3 -m http.server 8090
# open  http://localhost:8090/diar-viewer.html?dataUrl=sample.json
```

(`sample.json` is a small synthetic 3-speaker conversation — it renders lanes
and transcript without any audio; add an `audioUrl` in the JSON or pass one via
the embed API for a playhead.)

## Three ways to feed it

1. **Query string**: `?dataUrl=<url>` (or `?file=<name>` when served by a host
   that puts JSON under `/viewer/data/<name>` — the audio-assist service does).
2. **Embed (iframe)**: the page posts
   `{__diarViewer: 1, diar: <obj>, transcript: <obj>, audioUrl?: <url>, title?: <str>, model?: <str>}` —
   exactly how the console embeds it.
3. **Direct URL to JSON** of the shape:
   ```json
   {
     "diar": {
       "model": "nvidia/Nemotron-3-Diarization",
       "n_speakers": 5, "duration_s": 120.0, "rtfx": 35,
       "segments": [{"speaker": 0, "start": 0.0, "end": 3.4}, ...]
     },
     "transcript": {
       "segments": [
         {"start": 0.0, "end": 3.4,
          "words": [{"start": 0.0, "end": 0.3, "word": "The"}, ...]},
         ...
       ]
     }
   }
   ```
   `diar.segments[].speaker` is 0-based; `transcript` also accepts the raw
   WhisperX shape (`{"transcription": {"segments": [...]}}`).

## Rendering rules

- **Speaker colors**: a fixed 8-color palette, speaker *N* gets color N (stable
  across pages and sessions).
- **Word assignment**: a transcript word is shown in the color of the speaker
  active at the word's **midpoint**; two speakers at the midpoint ⇒ overlap
  (rendered `+`); none ⇒ unassigned (`–`). Overlaps are intentionally rare in
  practice (the Nemotron model's 80 ms frame granularity keeps most words
  single-speaker).
- **Lanes**: one block per diar segment; click a lane to seek the media player.
- **Playhead**: synchronized to the media element; without media it is inert.

## Provenance

This file ships with the homelab's audio-assist service (one file, two
deployments: standalone on the diarization box with sample clips, and embedded
in the web console). MIT-like internal tooling; the file has zero external
dependencies and works offline.
