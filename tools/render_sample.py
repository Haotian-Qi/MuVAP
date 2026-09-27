"""Render one packed sample as a self-contained HTML page, to check alignment.

Everything on the page is read through `PackedConversationDataset`, so what it
shows is what a training step is handed - the same 25 Hz sampling, the same
loudness normalization, the same label projection. A picture built any other way
could agree with the model and still be wrong about the data.

The point is to catch a desync you cannot see in a tensor shape. Play the audio
and watch: a speaker's mouth should move while their voice-activity row is lit,
the per-speaker future bins should light up *before* they start, and for an
event the marked decision frame should sit at the very end, with the answer
still in the future.

```bash
python tools/render_sample.py --pack /data/AVCC/packed/events.test --index 0 \\
    --output event.html          # self-contained page: scrub, step, inspect
python tools/render_sample.py --pack /data/AVCC/packed/segments.val --index 0 \\
    --output segment.mp4         # a video with the audio muxed in
```

The format follows the extension. The page carries its own audio and face crops
as data URIs, so it opens anywhere, including over SSH with nothing installed at
the other end; the video needs ffmpeg to mux, and plays anywhere at all.
"""

import argparse
import base64
import io
import json
import math
import sys
import wave
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.dataloaders.conversation import PackedConversationDataset  # noqa: E402
from preprocess.muvap.schema import MODEL_FPS, SAMPLE_RATE  # noqa: E402
from projection_window import ProjectionWindow  # noqa: E402

FACE = 112
GRID_COLUMNS = 25


def wav_uri(waveform: np.ndarray) -> str:
    """The exact audio the model reads, as a playable data URI."""
    pcm = np.clip(waveform, -1.0, 1.0)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes((pcm * 32767.0).astype("<i2").tobytes())
    return "data:audio/wav;base64," + base64.b64encode(buffer.getvalue()).decode()


def sprite_uri(faces: np.ndarray, quality: int = 80) -> str:
    """One speaker's face track as a grid, addressed per frame by the page."""
    frames = len(faces)
    rows = math.ceil(frames / GRID_COLUMNS)
    sheet = np.full((rows * FACE, GRID_COLUMNS * FACE), 128, dtype=np.uint8)
    for index, face in enumerate(faces):
        row, column = divmod(index, GRID_COLUMNS)
        sheet[row * FACE : (row + 1) * FACE, column * FACE : (column + 1) * FACE] = face
    ok, encoded = cv2.imencode(".jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("could not encode the face sheet")
    return "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode()


def envelope(waveform: np.ndarray, frames: int) -> list[float]:
    """Per-frame audio level, so the waveform lines up with the label rows."""
    per_frame = round(SAMPLE_RATE / MODEL_FPS)
    usable = frames * per_frame
    block = np.abs(waveform[:usable]).reshape(frames, per_frame).max(axis=1)
    peak = float(block.max()) or 1.0
    return [round(float(value) / peak, 4) for value in block]


def build(dataset, index: int, gvap, svap, start: int = 0, length: int | None = None) -> dict:
    item = dataset[index]
    metadata = item["metadata"]
    speakers = list(metadata["speaker_ids"])

    if "audio" not in item:
        raise SystemExit(
            "this is an embedding pack, which holds no audio or faces to show; "
            "point --pack at the media pack it was extracted from"
        )

    # Trim before anything is measured or encoded, so every stream on the page
    # is cut at the same frame and the audio cannot drift from the labels.
    total = int(item["mask"].shape[0])
    start = max(0, min(start, total - 1))
    frames = total - start if length is None else max(1, min(length, total - start))
    if (start, frames) != (0, total):
        span = slice(start, start + frames)
        per_frame = round(SAMPLE_RATE / MODEL_FPS)
        item = dict(item)
        item["audio"] = item["audio"][:, start * per_frame : (start + frames) * per_frame]
        for key, axis in (("visual", 1), ("vad", 1), ("visual_mask", 1), ("bbox", 1),
                          ("mask", 0), ("gvap_gt", 0), ("svap_gt", 1)):
            if key in item:
                item[key] = item[key][span] if axis == 0 else item[key][:, span]
        metadata = dict(metadata)
        metadata["start_sec"] = metadata.get("start_sec", 0.0) + start / MODEL_FPS

    waveform = item["audio"][0].numpy()

    payload = {
        "sample_id": metadata["sample_id"],
        "kind": dataset.kind,
        "frames": frames,
        "fps": MODEL_FPS,
        "speakers": speakers,
        "tracked": metadata.get("tracked", [True] * len(speakers)),
        "source_fps": metadata["source_fps"],
        "source_frames": metadata["source_frames"],
        "start_sec": metadata.get("start_sec", 0.0),
        "audio": wav_uri(waveform),
        "envelope": envelope(waveform, frames),
        "faces": [sprite_uri(item["visual"][row].numpy().astype(np.uint8)) for row in range(len(speakers))],
        "vad": item["vad"].numpy().astype(int).tolist(),
        "visible": item["visual_mask"].numpy().astype(int).tolist(),
        "bbox": np.round(item["bbox"].numpy(), 4).tolist(),
        "scored": item["mask"].numpy().astype(int).tolist(),
        "columns": GRID_COLUMNS,
        "face": FACE,
        "event": metadata.get("event"),
        "gvap_bins": gvap.bin_sec,
        # The bins one codebook *row* encodes, which is the future half only.
        "gvap_row_sec": gvap.row_sec,
        "svap_bins": svap.bin_sec,
    }
    # An event pack stores no projected labels - its ground truth is the
    # annotation, and its window ends where the future it would project begins.
    # They are still worth seeing, so they are derived here from the stored
    # activity and the trailing frames whose window runs off the end are marked:
    # those are reading silence that is an artefact of the cut, not the corpus.
    import torch

    svap_gt, gvap_gt = item.get("svap_gt"), item.get("gvap_gt")
    if svap_gt is None or gvap_gt is None:
        with torch.no_grad():
            activity = item["vad"].float()
            svap_gt = svap.get_labels(activity.unsqueeze(1))
            gvap_gt = gvap.get_labels(activity.unsqueeze(0))[0]
        payload["derived"] = True
        payload["horizon"] = max(gvap.fut_frames, svap.fut_frames)

    payload["svap"] = svap_gt.numpy().astype(int).tolist()
    payload["gvap"] = gvap_gt.numpy().astype(int).tolist()
    # The ranked pair on the 1.4 / 0.6 / 0.6 / 1.4 window - two history bins and
    # two future ones, the shape the role-based VAP is read in. Under
    # `role_relative` this is the class itself; under `role_future` it is the
    # window the ranking used before the class re-encoded the future finer.
    window = gvap.role_window(item["vad"].float().unsqueeze(0))[0]
    payload["gvap_window"] = window.numpy().astype(int).tolist()
    payload["gvap_window_sec"] = gvap.coarse_sec
    payload["gvap_hist"] = gvap.num_hist_bins
    # What the class actually claims, decoded back to the two ranked rows, so
    # the page can show the target rather than a class index.
    payload["gvap_rows"] = gvap.decode(gvap_gt.long()).numpy().astype(int).tolist()
    return payload


TEMPLATE = """<title>__TITLE__</title>
<style>
  :root {
    --bg:#fbfaf8; --ink:#1b1a18; --muted:#6b6762; --line:#ddd8d1;
    --panel:#fff; --speak:#2f7d5d; --idle:#e8e4de; --hide:#c8563c;
    --future:#3f6fb5; --mark:#c8563c;
  }
  @media (prefers-color-scheme: dark) { :root:not([data-theme=light]) {
    --bg:#16161a; --ink:#ecebe8; --muted:#9a958e; --line:#33322f;
    --panel:#1f1f24; --idle:#2c2b30;
  }}
  :root[data-theme=dark] {
    --bg:#16161a; --ink:#ecebe8; --muted:#9a958e; --line:#33322f;
    --panel:#1f1f24; --idle:#2c2b30;
  }
  body { background:var(--bg); color:var(--ink); font:14px/1.5 system-ui,sans-serif;
         padding-block:24px; padding-left:20px; padding-right:20px; max-width:1100px; margin:0 auto; }
  h1 { font-size:18px; margin:0 0 2px; }
  .sub { color:var(--muted); margin:0 0 18px; font-size:13px; }
  .panel { background:var(--panel); border:1px solid var(--line); border-radius:10px;
           padding:14px; margin-bottom:14px; }
  .faces { display:flex; gap:14px; flex-wrap:wrap; }
  .face { text-align:center; }
  .face canvas { width:132px; height:132px; border-radius:8px; border:2px solid var(--line);
                 image-rendering:pixelated; background:#888; display:block; }
  .face.speaking canvas { border-color:var(--speak); }
  .face.hidden canvas { border-color:var(--hide); opacity:.45; }
  .face .name { font-weight:600; margin-top:6px; }
  .face .state { color:var(--muted); font-size:12px; font-variant-numeric:tabular-nums; }
  .bins { display:flex; gap:3px; justify-content:center; margin-top:5px; }
  .bin { width:30px; padding:2px 0; border-radius:3px; background:var(--idle);
         color:var(--muted); font:11px ui-monospace,monospace; text-align:center; }
  .bin.on { background:var(--future); color:#fff; }
  .now { display:flex; gap:26px; align-items:center; flex-wrap:wrap; }
  .now .role { display:flex; gap:8px; align-items:center; }
  .now .role b { font-weight:600; font-size:12px; min-width:38px; }
  .now h2 { font-size:13px; margin:0 0 8px; font-weight:600; }
  audio { width:100%; margin-bottom:10px; }
  canvas#timeline { width:100%; height:auto; display:block; cursor:crosshair; }
  .legend { color:var(--muted); font-size:12px; margin-top:8px; }
  .legend b { color:var(--ink); font-weight:600; }
  kbd { background:var(--idle); border:1px solid var(--line); border-radius:4px;
        padding:1px 5px; font:12px ui-monospace,monospace; }
  table { border-collapse:collapse; font-size:13px; }
  td { padding:2px 14px 2px 0; }
  td.k { color:var(--muted); }
  .mono { font:12px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace; }
</style>

<h1 id="title"></h1>
<p class="sub" id="subtitle"></p>

<div class="panel">
  <audio id="audio" controls></audio>
  <div class="faces" id="faces"></div>
</div>

<div class="panel" id="nowpanel">
  <h2 id="nowtitle">GVAP at this frame &mdash; the conversation ranked to a pair</h2>
  <div class="now" id="now"></div>
</div>

<div class="panel">
  <canvas id="timeline"></canvas>
  <div class="legend" id="legend"></div>
</div>

<div class="panel"><table id="meta"></table></div>

<script id="payload" type="application/json">__PAYLOAD__</script>
<script>
const D = JSON.parse(document.getElementById("payload").textContent);
const FPS = D.fps, N = D.frames, S = D.speakers.length;
const audio = document.getElementById("audio");
audio.src = D.audio;

document.getElementById("title").textContent =
  (D.kind === "events" ? "Event " : "Segment ") + D.sample_id;
document.getElementById("subtitle").textContent =
  `${N} model frames at ${FPS} Hz (${(N/FPS).toFixed(2)} s) · ${S} speakers · `
  + `source ${D.source_frames} frames at ${D.source_fps} fps · starts ${D.start_sec.toFixed(2)} s into the segment`;

/* faces: one sprite grid per speaker, addressed per frame */
const sheets = [], cells = [];
D.speakers.forEach((name, i) => {
  const wrap = document.createElement("div");
  wrap.className = "face";
  const c = document.createElement("canvas");
  c.width = D.face; c.height = D.face;
  wrap.appendChild(c);
  const n = document.createElement("div");
  n.className = "name";
  n.textContent = "speaker " + name + (D.tracked[i] ? "" : " (audio only)");
  const st = document.createElement("div");
  st.className = "state";
  const bins = document.createElement("div");
  bins.className = "bins";
  const boxes = (D.svap_bins || []).map(sec => {
    const b = document.createElement("div");
    b.className = "bin"; b.textContent = sec;
    bins.appendChild(b); return b;
  });
  wrap.appendChild(n); wrap.appendChild(st); wrap.appendChild(bins);
  document.getElementById("faces").appendChild(wrap);
  cells.push({ wrap, ctx: c.getContext("2d"), state: st, boxes });
  const img = new Image(); img.src = D.faces[i]; sheets.push(img);
});

/* GVAP for the current frame, decoded back into its bins */
const gvapBoxes = [];
if (D.gvap_rows) {
  const secs = D.gvap_bins.slice(D.gvap_bins.length - D.gvap_rows[0][0].length);
  const host = document.getElementById("now");
  [["Scurr", 0], ["Snext", 1]].forEach(([title, slot]) => {
    const role = document.createElement("div");
    role.className = "role";
    const label = document.createElement("b"); label.textContent = title;
    role.appendChild(label);
    const bins = document.createElement("div"); bins.className = "bins";
    const boxes = secs.map(sec => {
      const b = document.createElement("div");
      b.className = "bin"; b.textContent = sec;
      bins.appendChild(b); return b;
    });
    role.appendChild(bins); host.appendChild(role);
    gvapBoxes.push({ slot, boxes });
  });
  const klass = document.createElement("div");
  klass.className = "role"; klass.id = "gvapclass";
  host.appendChild(klass);
} else {
  // An event window stops where the silence starts, so there is no future left
  // inside it to project a target over. Saying so beats an empty panel.
  document.getElementById("nowtitle").textContent = "No projection target here";
  document.getElementById("now").innerHTML =
    "<span style='color:var(--muted)'>An event window ends where the mutual silence "
    + "begins, so the frames a GVAP or SVAP target would need are deliberately cut "
    + "off. The event's own hold/shift annotation is the ground truth instead. "
    + "Open a segment sample to see the projected bins.</span>";
}

/* timeline */
const tl = document.getElementById("timeline");
const ROW = 17, GAP = 5, PAD = 104, RIGHT = 16;

/* The same rows the video lays out: a VAD band spans the window, while the
   GVAP and SVAP projections exist only ahead of the current frame. */
function timelineRows(D) {
  const rows = [["audio", "audio", null], ["GVAP Scurr", "gvap", 0], ["GVAP Snext", "gvap", 1]];
  D.speakers.forEach((name, i) => {
    rows.push(["SVAP spk " + name, "svap", i]);
    rows.push(["VAD spk " + name, "vad", i]);
  });
  return rows;
}
const ROWS = timelineRows(D);
const TOPS = ROWS.map((_, i) => 6 + i * (ROW + GAP));
const RULER = 6 + ROWS.length * (ROW + GAP);
const H = RULER + 26;
const W = Math.max(1040, N * 3 + PAD + RIGHT);
const PLOT = W - PAD - RIGHT, PPF = PLOT / N;
tl.width = W; tl.height = H;
const g = tl.getContext("2d");
const css = getComputedStyle(document.documentElement);
const col = k => css.getPropertyValue(k).trim();

/* Bins laid end to end at their real durations, so the block a projection
   covers is exactly the span of time it predicts over. */
function binSpans(seconds, history) {
  let offset = -seconds.slice(0, history || 0).reduce((a, b) => a + b, 0) * FPS;
  return seconds.map(v => { const s = [offset, v * FPS]; offset += v * FPS; return s; });
}

const base = document.createElement("canvas");
base.width = W; base.height = H;
const b = base.getContext("2d");

function drawBase() {
  b.fillStyle = col("--panel"); b.fillRect(0, 0, W, H);
  ROWS.forEach(([title, kind, index], i) => {
    const y = TOPS[i];
    b.font = "11px system-ui,sans-serif"; b.textAlign = "right";
    b.fillStyle = kind === "vad" ? col("--ink") : col("--muted");
    b.fillText(title, PAD - 10, y + ROW - 4);
    if (kind === "audio") {
      b.fillStyle = col("--ink");
      for (let t = 0; t < N; t++) {
        const h = Math.max(1, D.envelope[t] * ROW);
        b.fillRect(PAD + t * PPF, y + ROW - h, Math.ceil(PPF) + 0.5, h);
      }
    } else if (kind === "vad") {
      for (let t = 0; t < N; t++) {
        b.fillStyle = D.vad[index][t] ? col("--speak") : col("--idle");
        b.fillRect(PAD + t * PPF, y, Math.ceil(PPF) + 0.5, ROW);
        if (!D.visible[index][t]) {
          b.fillStyle = col("--hide");
          b.fillRect(PAD + t * PPF, y + ROW - 3, Math.ceil(PPF) + 0.5, 3);
        }
      }
    }
  });
  b.fillStyle = col("--muted"); b.textAlign = "center"; b.font = "11px system-ui,sans-serif";
  for (let s = 0; s <= N / FPS; s++) {
    const x = PAD + s * FPS * PPF;
    b.fillRect(x, RULER, 1, 5);
    b.fillText(s + "s", x, RULER + 18);
  }
}

function drawFrame(t) {
  g.clearRect(0, 0, W, H);
  g.drawImage(base, 0, 0);
  const now = PAD + t * PPF, limit = PAD + PLOT;

  ROWS.forEach(([title, kind, index], i) => {
    if (kind !== "gvap" && kind !== "svap") return;
    const y = TOPS[i];
    const values = kind === "gvap" ? D.gvap_window[t][index] : D.svap[index][t];
    const spans = kind === "gvap"
      ? binSpans(D.gvap_window_sec, D.gvap_hist)
      : binSpans(D.svap_bins, 0);
    let left = null, right = now;
    spans.forEach(([start, span], k) => {
      const a = Math.max(PAD, now + start * PPF);
      const z = Math.min(now + (start + span) * PPF, limit);
      if (z <= a) return;
      if (left === null) left = a;
      g.fillStyle = values[k] ? col("--future") : col("--idle");
      g.fillRect(a, y, z - a - 1, ROW);
      right = z;
    });
    if (left !== null && right > left) {
      g.strokeStyle = col("--muted"); g.lineWidth = 1;
      g.strokeRect(left + 0.5, y + 0.5, right - left - 1, ROW - 1);
    }
  });

  if (D.event) {
    let last = N - 1; while (last > 0 && !D.scored[last]) last--;
    const edge = PAD + (last + 1) * PPF;
    g.fillStyle = col("--hide"); g.fillRect(edge, TOPS[0], 2, RULER - TOPS[0]);
  }
  /* the current frame */
  g.fillStyle = col("--mark"); g.fillRect(now - 1, TOPS[0], 2.5, RULER - TOPS[0]);
}

let headX = null;
function playhead() {
  if (headX !== null) { g.putImageData(saved, headX - 1, 0); }
  const t = Math.min(N - 1, Math.floor(audio.currentTime * FPS));
  const w = (W - PAD) / N;
  const x = Math.round(PAD + t * w);
  saved = g.getImageData(x - 1, 0, 3, H);
  headX = x;
  g.fillStyle = col("--mark"); g.fillRect(x, 0, 2, H - 22);
}
let saved = null;

/* per-frame view */
function show(t) {
  D.speakers.forEach((name, i) => {
    const sheet = sheets[i], cell = cells[i];
    if (sheet.complete) {
      const r = Math.floor(t / D.columns), c = t % D.columns;
      cell.ctx.drawImage(sheet, c * D.face, r * D.face, D.face, D.face, 0, 0, D.face, D.face);
    }
    const speaking = D.vad[i][t], visible = D.visible[i][t];
    cell.wrap.className = "face" + (speaking ? " speaking" : "") + (visible ? "" : " hidden");
    cell.state.textContent = (speaking ? "speaking" : "silent") + (visible ? "" : " · no face");
    if (D.svap) cell.boxes.forEach((b, k) => b.classList.toggle("on", !!D.svap[i][t][k]));
  });
  gvapBoxes.forEach(({ slot, boxes }) =>
    boxes.forEach((b, k) => b.classList.toggle("on", !!D.gvap_rows[t][slot][k])));
  if (D.gvap) document.getElementById("gvapclass").textContent = "class " + D.gvap[t];
  document.getElementById("legend").innerHTML =
    `frame <b>${t}</b> / ${N - 1} · <b>${(t / FPS).toFixed(2)} s</b> into this window`
    + (D.event ? ` · <b>${D.event.label}</b>, speaker ${D.event.previous} → ${D.event.following},`
        + ` pause ${D.event.pause.toFixed(3)} s` : "")
    + ` · green = speaking now, blue = an active future bin, red line = this frame.`
    + ` <kbd>←</kbd><kbd>→</kbd> step a frame, <kbd>space</kbd> play.`;
}

let frame = -1;
function tick() {
  const t = Math.max(0, Math.min(N - 1, Math.floor(audio.currentTime * FPS)));
  if (t !== frame) { frame = t; show(t); drawFrame(t); }
  requestAnimationFrame(tick);
}

/* metadata table */
const meta = document.getElementById("meta");
const rowsOut = [
  ["sample", D.sample_id], ["pack kind", D.kind],
  ["model frames", `${N} at ${FPS} Hz = ${(N / FPS).toFixed(2)} s`],
  ["source frames", `${D.source_frames} at ${D.source_fps} fps`],
  ["speakers", D.speakers.map((n, i) => n + (D.tracked[i] ? "" : " (audio only)")).join(", ")],
  ["audio samples", `${Math.round(N * 16000 / FPS)} at 16 kHz, normalized as the model reads it`],
];
if (D.event) rowsOut.push(["event", `${D.event.label}: ${D.event.previous} → ${D.event.following}, pause ${D.event.pause} s`]);
meta.innerHTML = rowsOut.map(([k, v]) => `<tr><td class="k">${k}</td><td class="mono">${v}</td></tr>`).join("");

tl.addEventListener("click", e => {
  const rect = tl.getBoundingClientRect();
  const x = (e.clientX - rect.left) * (W / rect.width);
  audio.currentTime = Math.max(0, (x - PAD) / PPF / FPS);
});
addEventListener("keydown", e => {
  if (e.key === "ArrowLeft") { audio.pause(); audio.currentTime = Math.max(0, (frame - 1) / FPS); e.preventDefault(); }
  if (e.key === "ArrowRight") { audio.pause(); audio.currentTime = Math.min((N - 1) / FPS, (frame + 1) / FPS); e.preventDefault(); }
  if (e.key === " ") { audio.paused ? audio.play() : audio.pause(); e.preventDefault(); }
});

let pending = sheets.length;
sheets.forEach(img => img.onload = () => { if (--pending === 0) show(0); });
drawBase(); drawFrame(0); show(0); tick();
</script>
"""


# Drawn in BGR, because OpenCV is what composes the video.
INK = (28, 26, 27)
PAPER = (248, 250, 251)
MUTED = (130, 130, 134)
SPEAK = (93, 125, 47)
FUTURE = (181, 111, 63)
HIDE = (60, 86, 200)
IDLE = (222, 228, 232)
#: The playhead, and the frame an event is decided on.
MARK = (48, 48, 214)


def _band(canvas, y, height, values, colour, x0, width, faded=False):
    """One label row across the whole timeline."""
    step = width / len(values)
    shade = tuple(int(c + (255 - c) * 0.55) for c in colour) if faded else colour
    for index, value in enumerate(values):
        left = int(x0 + index * step)
        right = max(left + 1, int(x0 + (index + 1) * step))
        canvas[y : y + height, left:right] = shade if value else IDLE


def _label(canvas, text, x, y, colour=MUTED, scale=0.38):
    cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 1, cv2.LINE_AA)


def bin_boxes(canvas, values, seconds, x, y, colour, box=30, height=16):
    """One row of future bins, lit where the label says that bin is active."""
    for index, on in enumerate(values):
        left = x + index * (box + 3)
        canvas[y : y + height, left : left + box] = colour if on else IDLE
        _label(canvas, f"{seconds[index]:g}", left + 5, y + height - 4,
               PAPER if on else MUTED, 0.32)
    return x + len(values) * (box + 3)


def draw_now(canvas, payload, frame, y, width):
    """The labels for *this* frame only, decoded into their bins.

    The timeline underneath says what happens across the whole window; this says
    what the model is being asked to predict at the frame the playhead is on.
    """
    rows = payload.get("gvap_rows")
    if not rows:
        return
    seconds = payload["gvap_bins"][-len(rows[0][0]):]
    _label(canvas, "GVAP  the conversation, ranked to a pair", 16, y + 12, INK, 0.42)
    x = 16
    for slot, title in ((0, "Scurr"), (1, "Snext")):
        _label(canvas, title, x, y + 34, MUTED, 0.38)
        x = bin_boxes(canvas, rows[frame][slot], seconds, x + 42, y + 22, FUTURE)
        x += 26
    klass = payload["gvap"][frame] if "gvap" in payload else None
    if klass is not None:
        _label(canvas, f"class {klass}", x, y + 34, MUTED, 0.38)


ROW, GAP, LABEL_WIDTH, RIGHT = 15, 5, 108, 16


def timeline_rows(payload):
    """The bottom section, top to bottom.

    `kind` says how a row is drawn: `vad` spans the whole window, while `gvap`
    and `svap` are projections and exist only ahead of the current frame.
    """
    rows = [("audio", "audio", None)]
    rows.append(("GVAP Scurr", "gvap", 0))
    rows.append(("GVAP Snext", "gvap", 1))
    for index, name in enumerate(payload["speakers"]):
        rows.append((f"SVAP spk {name}", "svap", index))
        rows.append((f"VAD spk {name}", "vad", index))
    return rows


def timeline_geometry(payload, width):
    """Where every row sits, and how a frame maps to a pixel."""
    rows = timeline_rows(payload)
    x0 = LABEL_WIDTH
    plot = width - LABEL_WIDTH - RIGHT
    per_frame = plot / payload["frames"]
    tops = {}
    y = 6
    for position, (title, kind, index) in enumerate(rows):
        tops[position] = y
        y += ROW + GAP
    return {
        "rows": rows, "tops": tops, "x0": x0, "plot": plot,
        "per_frame": per_frame, "ruler": y, "height": y + 26,
    }


def bin_spans(seconds, fps, history=0):
    """Each bin as (offset, width) in frames, laid end to end around now.

    The widths are the bins' real durations, so a window is drawn exactly as
    wide as the time it covers; equal-width boxes would misrepresent one whose
    bins run 0.2 to 1.4 s. `history` bins are placed *before* the current frame,
    which is what puts the classic window's two past bins left of the playhead
    and its two future bins right of it.
    """
    spans, offset = [], -sum(seconds[:history]) * fps
    for value in seconds:
        spans.append((offset, value * fps))
        offset += value * fps
    return spans


def timeline_static(payload, width):
    """Everything that does not move: the audio, the VAD bands, the ruler."""
    geometry = timeline_geometry(payload, width)
    canvas = np.full((geometry["height"], width, 3), PAPER, dtype=np.uint8)
    x0, plot, per_frame = geometry["x0"], geometry["plot"], geometry["per_frame"]

    for position, (title, kind, index) in enumerate(geometry["rows"]):
        y = geometry["tops"][position]
        bright = kind == "vad"
        _label(canvas, title, 8, y + ROW - 3, INK if bright else MUTED, 0.36)
        if kind == "audio":
            for frame, value in enumerate(payload["envelope"]):
                tall = max(1, int(value * ROW))
                left = int(x0 + frame * per_frame)
                canvas[y + ROW - tall : y + ROW, left : max(left + 1, int(x0 + (frame + 1) * per_frame))] = INK
        elif kind == "vad":
            _band(canvas, y, ROW, payload["vad"][index], SPEAK, x0, plot)
            # A frame with no face is still scored for the global label, so it
            # is marked rather than left looking like ordinary silence.
            for frame, visible in enumerate(payload["visible"][index]):
                if not visible:
                    left = int(x0 + frame * per_frame)
                    canvas[y + ROW - 3 : y + ROW, left : max(left + 1, int(x0 + (frame + 1) * per_frame))] = HIDE
        else:
            canvas[y : y + ROW, x0 : x0 + plot] = PAPER

    y = geometry["ruler"]
    for second in range(int(payload["frames"] / payload["fps"]) + 1):
        x = int(x0 + second * payload["fps"] * per_frame)
        canvas[y : y + 4, x : x + 1] = MUTED
        _label(canvas, f"{second}s", x - 6, y + 16, MUTED, 0.32)
    return canvas, geometry


def draw_projection(canvas, payload, frame, geometry, offset_y=0):
    """The projection bins for this frame, and the playhead they start at.

    Both windows are drawn where they apply: starting at the current frame and
    running forward over the span each bin covers. What the model is being asked
    at this instant is therefore the block immediately right of the red line.
    """
    x0, per_frame = geometry["x0"], geometry["per_frame"]
    now = x0 + frame * per_frame
    limit = x0 + geometry["plot"]

    for position, (title, kind, index) in enumerate(geometry["rows"]):
        if kind not in ("gvap", "svap"):
            continue
        y = geometry["tops"][position] + offset_y
        if kind == "gvap":
            values = payload["gvap_window"][frame][index]
            spans = bin_spans(
                payload["gvap_window_sec"], payload["fps"], payload["gvap_hist"]
            )
        else:
            values = payload["svap"][index][frame]
            spans = bin_spans(payload["svap_bins"], payload["fps"])
        block_left = max(int(x0), int(now + spans[0][0] * per_frame))
        block_right = block_left
        for (start, span), on in zip(spans, values):
            left = max(int(x0), int(now + start * per_frame))
            right = min(int(now + (start + span) * per_frame), int(limit))
            if right <= left:
                continue
            canvas[y : y + ROW, left:right] = FUTURE if on else IDLE
            # A hairline between bins, so four lit bins still read as four.
            canvas[y : y + ROW, right - 1 : right] = PAPER
            block_right = right
        # The window's extent, which an all-quiet row would otherwise lose
        # against the background.
        if block_right > block_left:
            canvas[y : y + 1, block_left:block_right] = MUTED
            canvas[y + ROW - 1 : y + ROW, block_left:block_right] = MUTED

    top = geometry["tops"][0] + offset_y
    bottom = geometry["ruler"] + offset_y
    x = int(now)
    canvas[top:bottom, max(0, x - 1) : x + 2] = MARK


def render_video(payload, output: Path, face_size: int = 168) -> None:
    """Compose the sample frame by frame and mux its own audio back in."""
    import shutil
    import subprocess
    import tempfile

    if shutil.which("ffmpeg") is None:
        raise SystemExit("rendering a video needs ffmpeg on PATH")

    speakers = payload["speakers"]
    frames = payload["frames"]
    faces = [
        cv2.imdecode(
            np.frombuffer(base64.b64decode(uri.split(",", 1)[1]), np.uint8),
            cv2.IMREAD_GRAYSCALE,
        )
        for uri in payload["faces"]
    ]

    header, pad = 34, 16
    block = face_size + 46
    width = max(1040, pad * 2 + len(speakers) * (face_size + pad))
    strip, geometry = timeline_static(payload, width)
    height = header + block + strip.shape[0] + 8

    with tempfile.TemporaryDirectory() as work:
        silent = Path(work) / "silent.mp4"
        writer = cv2.VideoWriter(
            str(silent), cv2.VideoWriter_fourcc(*"mp4v"), payload["fps"], (width, height)
        )
        decision = None
        if payload.get("event"):
            decision = max(i for i, on in enumerate(payload["scored"]) if on)

        for t in range(frames):
            canvas = np.full((height, width, 3), PAPER, dtype=np.uint8)
            title = f"{payload['sample_id']}   frame {t}/{frames - 1}   {t / payload['fps']:6.2f}s"
            _label(canvas, title, pad, 22, INK, 0.46)
            if payload.get("event"):
                event = payload["event"]
                _label(
                    canvas,
                    f"{event['label'].upper()}  speaker {event['previous']} -> "
                    f"{event['following']}   pause {event['pause']:.2f}s",
                    width - 330, 22, HIDE if event["label"] == "shift" else SPEAK, 0.46,
                )

            for index, name in enumerate(speakers):
                left = pad + index * (face_size + pad)
                sheet = faces[index]
                row, column = divmod(t, payload["columns"])
                crop = sheet[
                    row * FACE : (row + 1) * FACE, column * FACE : (column + 1) * FACE
                ]
                tile = cv2.cvtColor(cv2.resize(crop, (face_size, face_size)), cv2.COLOR_GRAY2BGR)
                speaking = payload["vad"][index][t]
                visible = payload["visible"][index][t]
                border = SPEAK if speaking else (HIDE if not visible else MUTED)
                canvas[header : header + face_size, left : left + face_size] = tile
                cv2.rectangle(
                    canvas, (left - 2, header - 2),
                    (left + face_size + 1, header + face_size + 1), border, 3,
                )
                state = "speaking" if speaking else ("no face" if not visible else "silent")
                _label(canvas, f"speaker {name}  -  {state}", left, header + face_size + 18,
                       INK if speaking else MUTED, 0.42)

            top = header + block
            canvas[top : top + strip.shape[0]] = strip
            if decision is not None:
                edge = int(geometry["x0"] + (decision + 1) * geometry["per_frame"])
                canvas[top + geometry["tops"][0] : top + geometry["ruler"], edge : edge + 2] = HIDE
            draw_projection(canvas, payload, t, geometry, offset_y=top)
            writer.write(canvas)
        writer.release()

        wave_path = Path(work) / "audio.wav"
        wave_path.write_bytes(base64.b64decode(payload["audio"].split(",", 1)[1]))
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(silent), "-i", str(wave_path),
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
             "-c:a", "aac", "-shortest", str(output)],
            check=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, required=True, help="a media pack")
    parser.add_argument("--index", type=int, default=0, help="which sample")
    parser.add_argument(
        "--start", type=int, default=0, help="first model frame to render"
    )
    parser.add_argument(
        "--frames",
        type=int,
        help="how many model frames to render; 750 is one 30 s training window",
    )
    parser.add_argument("--output", type=Path, default=Path("sample.html"))
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "config/yaml/muvap.yaml",
        help="config whose projection windows label the sample",
    )
    args = parser.parse_args()

    import yaml

    with args.config.open() as handle:
        cfg = yaml.safe_load(handle)["muvap"]
    gvap = ProjectionWindow(**cfg["gvap_projection_window"])
    svap = ProjectionWindow(**cfg["svap_projection_window"])

    with (args.pack / "dataset.json").open() as handle:
        kind = json.load(handle)["kind"]
    dataset = PackedConversationDataset(
        args.pack, gvap_projection=gvap, svap_projection=svap, kind=kind
    )
    if not 0 <= args.index < len(dataset):
        raise SystemExit(f"--index must be in 0..{len(dataset) - 1}")

    payload = build(dataset, args.index, gvap, svap, args.start, args.frames)
    if args.output.suffix.lower() in {".mp4", ".mov", ".mkv"}:
        render_video(payload, args.output)
    else:
        html = TEMPLATE.replace("__TITLE__", payload["sample_id"]).replace(
            "__PAYLOAD__", json.dumps(payload)
        )
        args.output.write_text(html)
    size = args.output.stat().st_size / 1e6
    print(f"{payload['sample_id']}: {payload['frames']} frames, "
          f"{len(payload['speakers'])} speakers -> {args.output} ({size:.1f} MB)")


if __name__ == "__main__":
    main()
