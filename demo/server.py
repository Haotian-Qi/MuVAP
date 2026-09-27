"""Live MuVAP: camera and microphone in, turn-taking predictions out.

The browser captures video and audio and streams them here; this holds a rolling
window of exactly what the training loader would have built, runs the frozen VAP
and ASD modules and the fusion over it, and sends back the newest frame's
predictions. Nothing is annotated and nothing is cached from a corpus - every
number on the page came out of the model a moment earlier.

**The frame clock is the audio.** A webcam's frame rate wanders, and the model's
25 Hz timeline is not negotiable: one model frame is exactly 640 audio samples.
So the server advances its clock on audio, not on video, and each model frame
takes whichever crop was most recent when that audio arrived. Video jitter then
shows up as a repeated or dropped crop instead of as drift between the two
streams, which is the failure that would quietly invalidate every prediction.

**Why the whole window is re-run.** Measured on this machine, one forward pass
over a 10 s window with three speakers costs 48 ms - about 21 Hz, against the
10 Hz this needs. A KV cache would save part of that, but only part: the cost is
dominated by the visual frontend, which is convolutional and has to see each new
crop regardless. Re-running the window is also exactly what training did, so
there is no second inference path to keep honest.
"""

import asyncio
import json
import math
import os
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.media import ASD_MAX_GAIN_DB, ASD_PEAK_LIMIT, ASD_TARGET_RMS_DBFS  # noqa: E402
from demo.streaming import StreamingEncoders  # noqa: E402
from demo.tracker import SPEAKER_CAPACITY, LiveTracker, crop_face  # noqa: E402
from models.muvap import frozen_pair, load_fusion  # noqa: E402
from projection_window import ProjectionWindow  # noqa: E402

FPS = 25
SAMPLE_RATE = 16_000
SAMPLES_PER_FRAME = SAMPLE_RATE // FPS
FACE = 112

CONTEXT_SEC = 10.0
CONTEXT_FRAMES = int(CONTEXT_SEC * FPS)
#: Range the page may move `shift_scale` over. It multiplies the shift row
#: before the rows are renormalised, so 1.0 is the plain readout and the
#: bounds are only there to keep a stray control message from pinning
#: p(shift) flat at 0 or 1.
SHIFT_SCALE_MIN, SHIFT_SCALE_MAX = 0.25, 8.0
#: The model's own rate. Re-running the whole window costs 44 ms and would cap
#: this at ~20 Hz; caching the per-frame visual features (demo/streaming.py)
#: takes it to about 20 ms, which leaves room at 25.
INFER_HZ = 25
#: Boxes follow faces at the video rate. On the GPU a detection is 6.5 ms, so
#: this is affordable and a box is never more than one frame old.
DETECT_HZ = 25
#: Only used if the detector falls back to the CPU. onnxruntime sizes its pool
#: from the core count and would take all 24, stalling the model's own CPU-side
#: work - launching kernels, normalising the window, copying logits back - for
#: as long as a detection lasts. Measured: unbounded gives 44 ms p95 on the
#: forward pass, four threads 39 ms, and detection does not care.
DETECT_THREADS = 4
#: cuDNN 9, which onnxruntime needs and this torch does not have. Install it
#: beside the environment rather than in it: torch is pinned to cuDNN 8 and
#: loads it by absolute path, and the two sonames differ, so both live in the
#: process at once without either seeing the other's. Point `MUVAP_ORT_CUDNN` at
#: that directory; unset, face detection runs on the CPU instead.
_ORT_CUDNN_ENV = os.environ.get("MUVAP_ORT_CUDNN")
ORT_CUDNN = Path(_ORT_CUDNN_ENV) if _ORT_CUDNN_ENV else None
#: The slowest rate at which identity is re-anchored when ArcFace is too
#: expensive to run every frame - which is only true on the CPU fallback, where
#: a full roster costs 165 ms. On the GPU identity is 13.4 ms and runs on every
#: detection, as the offline tracker does.
IDENTITY_SEC = 4.0

HERE = Path(__file__).resolve().parent
app = FastAPI(title="MuVAP live")
STATE = {}


def normalise(block: np.ndarray) -> np.ndarray:
    """The training gain rule, applied to the window rather than a recording.

    Offline this is computed over a whole recording so a slice cannot drift with
    its own content. Live there is no whole recording, so it is computed over the
    window in hand - the closest honest equivalent, and the reason a very quiet
    or very loud moment can move the level a little.
    """
    centred = block - block.mean()
    rms = float(np.sqrt(np.mean(centred**2)))
    peak = float(np.abs(centred).max())
    if rms <= 1e-8 or peak <= 1e-8:
        return centred
    gain = min(
        10 ** (ASD_TARGET_RMS_DBFS / 20) / rms,
        10 ** (ASD_MAX_GAIN_DB / 20),
        ASD_PEAK_LIMIT / peak,
    )
    return centred * gain


class Session:
    """One browser's rolling window and the predictions drawn from it."""

    def __init__(self, models):
        self.models = models
        # Ring buffers, not deques. A deque of Python floats has to be rebuilt
        # into an array on every forward pass, and at 160k samples that cost
        # more than the model did. These are written in place and read as a
        # contiguous slice, so the window is free to take.
        self.samples = CONTEXT_FRAMES * SAMPLES_PER_FRAME
        self.audio_ring = np.zeros(2 * self.samples, dtype=np.float32)
        self.crop_ring = np.full(
            (2 * CONTEXT_FRAMES, SPEAKER_CAPACITY, FACE, FACE), 128, dtype=np.uint8
        )
        # Presence rides in a ring too. It used to be a deque that the reader
        # appended to while the inference thread walked it, which raised
        # "deque mutated during iteration" the moment the two ran concurrently.
        self.present_ring = np.zeros((2 * CONTEXT_FRAMES, SPEAKER_CAPACITY), dtype=bool)
        self.head = 0                                  # model frames written
        self.encoded = 0                               # frames the cache has seen
        self.stream = StreamingEncoders(
            models["encoders"], SPEAKER_CAPACITY, CONTEXT_FRAMES, models["device"]
        )
        self.pending = np.zeros(0, dtype=np.float32)
        self.frame = None                              # newest decoded camera frame
        self.tracks = []
        self.frames_seen = 0
        self.last_detect = 0.0
        self.lock = asyncio.Lock()
        # The readout knob, not a model weight: it reweights the shift row of
        # p_future before the 0.5 threshold, so the page can be moved off the
        # window's own default without restarting the demo.
        self.shift_scale = models["window"].default_shift_scale

    def fold_encoded(self):
        """Keep the feature cache aligned when the crop ring folds back."""
        self.encoded = max(0, self.encoded - CONTEXT_FRAMES)

    def push_audio(self, samples: np.ndarray) -> int:
        """Advance the clock: every 640 samples completes one model frame."""
        # The browser sends exactly one model frame per message, so `pending` is
        # normally empty and the concatenate is skipped. It still exists because
        # nothing guarantees a message is frame-aligned.
        self.pending = samples if not len(self.pending) else np.concatenate(
            [self.pending, samples]
        )
        made = 0
        while len(self.pending) >= SAMPLES_PER_FRAME:
            block, self.pending = (
                self.pending[:SAMPLES_PER_FRAME],
                self.pending[SAMPLES_PER_FRAME:],
            )
            if self.head >= 2 * CONTEXT_FRAMES:        # fold back, keeping the window
                self.audio_ring[: self.samples] = self.audio_ring[-self.samples :]
                self.crop_ring[:CONTEXT_FRAMES] = self.crop_ring[-CONTEXT_FRAMES:]
                self.present_ring[:CONTEXT_FRAMES] = self.present_ring[-CONTEXT_FRAMES:]
                self.head = CONTEXT_FRAMES
                self.fold_encoded()
            self.audio_ring[
                self.head * SAMPLES_PER_FRAME : (self.head + 1) * SAMPLES_PER_FRAME
            ] = block
            self.crop_ring[self.head] = self._current_crops()
            row = np.zeros(SPEAKER_CAPACITY, dtype=bool)
            for track in self.tracks:
                if track["bbox"] is not None and track["speaker"] < SPEAKER_CAPACITY:
                    row[track["speaker"]] = True
            self.present_ring[self.head] = row
            self.head += 1
            self.frames_seen += 1
            made += 1
        return made

    def _current_crops(self):
        blank = np.full((FACE, FACE), 128, dtype=np.uint8)
        if self.frame is None:
            return np.stack([blank] * SPEAKER_CAPACITY)
        out = []
        for index in range(SPEAKER_CAPACITY):
            track = next((t for t in self.tracks if t["speaker"] == index), None)
            out.append(
                crop_face(self.frame, track["bbox"]) if track and track["bbox"] else blank
            )
        return np.stack(out)

    def push_frame(self, frame):
        self.frame = frame

    def detect_due(self):
        now = time.monotonic()
        if now - self.last_detect < 1.0 / DETECT_HZ or self.frame is None:
            return False
        self.last_detect = now
        return True

    def identity_due(self, since: float) -> bool:
        """Whether this detection should also run ArcFace.

        On the GPU, always - full identity is 13.4 ms, and matching on position
        alone between passes was losing a speaker in a fifth of frames because a
        face that moves or is briefly occluded is then matched by where it used
        to be rather than by who it is. The offline tracker runs identity on
        every frame and this now does the same.

        On the CPU fallback it cannot: a full roster is 165 ms there. Identity
        is then asked for when it actually decides something - a track coasting
        or missing, or the roster short - and otherwise on a slow floor.
        """
        if self.models["detector_on_gpu"]:
            return True
        roster = self.models["tracker"].tracks
        if len(roster) < SPEAKER_CAPACITY or any(t.bbox is None or t.missed for t in roster):
            return True
        return since >= IDENTITY_SEC

    def detect(self, identity=False):
        """Runs in a worker thread: detection must never stall the stream."""
        tracker = self.models["tracker"]
        self.tracks = tracker.update(self.frame, identity=identity)
        for slot in tracker.reassigned:
            self.forget_speaker(slot)

    def forget_speaker(self, slot: int) -> None:
        """A different person now holds this row, so drop the old one's history.

        The window is ten seconds long and the model reads all of it. Without
        this, a speaker who takes a vacated row inherits the previous occupant's
        face and voice activity for as long as the overlap lasts.
        """
        self.crop_ring[:, slot] = 128
        self.present_ring[:, slot] = False
        self.stream.forget(slot)

    @property
    def context(self):
        return min(self.head, CONTEXT_FRAMES)

    def ready(self):
        # A prediction needs enough history for the projection window to mean
        # something; before that the page shows the model warming up.
        return self.context >= FPS * 2

    @torch.no_grad()
    def predict(self):
        began = time.perf_counter()
        head = self.head                       # sampled once: the reader keeps writing
        frames = min(head, CONTEXT_FRAMES)
        start = head - frames
        device = self.models["device"]

        # Encode whatever arrived since the last pass. Doing it here rather than
        # in the reader keeps every CUDA call on one thread, and the cache is
        # then exactly as many frames deep as the window being read.
        pushed = head - self.encoded
        for index in range(self.encoded, head):
            self.stream.push(torch.from_numpy(self.crop_ring[index]).to(device))
        self.encoded = head
        t_push = time.perf_counter()

        audio = self.audio_ring[start * SAMPLES_PER_FRAME : head * SAMPLES_PER_FRAME]
        a = torch.from_numpy(normalise(audio)).to(device)[None, None]
        present = torch.from_numpy(
            self.present_ring[start:head].T.copy()
        ).to(device)[None]

        t_prep = time.perf_counter()
        with torch.autocast(device, torch.bfloat16):
            vap, asd = self.stream.run(a)
            global_logits, speaker_logits = self.models["fusion"](vap, asd, present)
        t_fwd = time.perf_counter()

        shift = self.models["window"].get_shift_hold(
            global_logits[0, 0].float().cpu(), shift_scale=self.shift_scale
        )["p_shift"].numpy()
        activity = speaker_logits[0, :, :, 0].sigmoid().float().cpu().numpy()
        done = time.perf_counter()
        self.stages = {
            "pushed": pushed,
            "push_ms": round((t_push - began) * 1000, 1),
            "prep_ms": round((t_prep - t_push) * 1000, 1),
            "fwd_ms": round((t_fwd - t_prep) * 1000, 1),
            "post_ms": round((done - t_fwd) * 1000, 1),
        }
        return shift, activity, (done - began) * 1000


def gpu_detector_providers() -> list[str]:
    """Put face detection on the GPU, or say why it stayed on the CPU.

    Detection is 48 ms on the CPU and 6.5 ms on the GPU, and the CPU version
    spends that time competing with the model's own host work, which showed up
    directly in the tail of the forward pass. The obstacle is that onnxruntime
    links cuDNN 9 while this torch bundles 8.9; the libraries have different
    sonames, so preloading 9 from its own directory satisfies onnxruntime
    without disturbing the copy torch has already opened.
    """
    import ctypes

    if ORT_CUDNN is None:
        print("MUVAP_ORT_CUDNN is not set; face detection stays on the CPU")
        return ["CPUExecutionProvider"]
    if not ORT_CUDNN.is_dir():
        print(f"cuDNN 9 not found at {ORT_CUDNN}; face detection stays on the CPU")
        return ["CPUExecutionProvider"]
    # Dependency order: the rest link against graph and ops.
    order = ("libcudnn_graph.so.9", "libcudnn_ops.so.9",
             "libcudnn_engines_precompiled.so.9", "libcudnn_engines_runtime_compiled.so.9",
             "libcudnn_heuristic.so.9", "libcudnn_adv.so.9", "libcudnn_cnn.so.9",
             "libcudnn.so.9")
    for name in order:
        try:
            ctypes.CDLL(str(ORT_CUDNN / name), mode=ctypes.RTLD_GLOBAL)
        except OSError as error:
            print(f"could not preload {name} ({error}); face detection stays on the CPU")
            return ["CPUExecutionProvider"]
    return ["CUDAExecutionProvider", "CPUExecutionProvider"]


def cap_onnx_threads(threads: int) -> None:
    """Bound the detector's thread pool before any session is built.

    onnxruntime takes its intra-op width from the machine, and insightface
    builds its sessions with no options, so there is no argument to pass. The
    initialiser is wrapped instead - `InferenceSession` itself cannot be
    replaced, because insightface subclasses it at import time.
    """
    import onnxruntime as ort

    if getattr(ort.InferenceSession, "_muvap_capped", False):
        return
    original = ort.InferenceSession.__init__

    def capped(self, path, sess_options=None, **kwargs):
        options = sess_options or ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        original(self, path, sess_options=options, **kwargs)

    capped._muvap_capped = True
    ort.InferenceSession.__init__ = capped
    ort.InferenceSession._muvap_capped = True


def load_models(vap_weights, asd_weights, muvap_weights, config, device):
    encoders = frozen_pair(vap_weights, asd_weights).to(device).eval()
    fusion, cfg = load_fusion(muvap_weights, config)
    providers = gpu_detector_providers()
    on_gpu = providers[0] == "CUDAExecutionProvider"
    if not on_gpu:
        cap_onnx_threads(DETECT_THREADS)
    from insightface.app import FaceAnalysis

    face = FaceAnalysis(
        name="buffalo_l", allowed_modules=["detection", "recognition"], providers=providers
    )
    face.prepare(ctx_id=0 if on_gpu else -1, det_size=(640, 640))
    print("face detection on " + ("the GPU" if on_gpu else "the CPU"))
    # A cold first pass costs the best part of a second while kernels are
    # selected. Paying it here means the first visitor does not.
    with torch.no_grad(), torch.autocast(device, torch.bfloat16):
        warm_a = torch.zeros(1, 1, CONTEXT_FRAMES * SAMPLES_PER_FRAME, device=device)
        warm_v = torch.full(
            (1, SPEAKER_CAPACITY, CONTEXT_FRAMES, FACE, FACE), 128, dtype=torch.uint8, device=device
        )
        vap, asd = encoders(warm_a, warm_v)
        fusion.to(device)(vap, asd)
    return {
        "encoders": encoders,
        "fusion": fusion.to(device).eval(),
        "window": ProjectionWindow(**cfg["gvap_projection_window"]),
        "tracker": LiveTracker(face),
        "detector_on_gpu": on_gpu,
        "device": device,
    }


@app.get("/")
def index():
    return FileResponse(HERE / "static" / "live.html")


@app.get("/health")
def health():
    models = STATE.get("models")
    return {
        "ready": models is not None,
        "device": models["device"] if models else None,
        "context_sec": CONTEXT_SEC,
        "infer_hz": INFER_HZ,
        "capacity": SPEAKER_CAPACITY,
    }


@app.websocket("/ws")
async def stream(socket: WebSocket):
    await socket.accept()
    session = Session(STATE["models"])
    # One tracker per connection, so a new visitor starts with a clean roster.
    session.models = dict(STATE["models"])
    session.models["tracker"] = LiveTracker(STATE["models"]["tracker"].app)
    sent_upto = 0

    async def infer_loop():
        """Own task, so a 60 ms forward pass never stalls the audio clock."""
        nonlocal sent_upto
        deadline = time.monotonic()
        while True:
            # Paced to a deadline, not by sleeping a fixed slice after the work:
            # sleeping 100 ms and then spending 70 ms thinking gives 6 Hz, not 10.
            deadline += 1.0 / INFER_HZ
            delay = deadline - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
            else:
                deadline = time.monotonic()
            if not session.ready():
                continue
            started = time.perf_counter()
            shift, activity, compute_ms = await asyncio.to_thread(session.predict)
            # Every model frame since the last message, not just the newest one.
            # The clock runs at 25 Hz and this loop at 10, so sending one value
            # per message would stretch a 10 s window into 25 s of screen.
            clock = session.frames_seen
            # Exactly the frames the page has not been given. A forward pass can
            # finish twice inside one model frame, and `max(1, ...)` used to send
            # the newest frame again when that happened - which drew 785 frames
            # for 749 elapsed and stretched the 10 s window on screen.
            new = min(clock - sent_upto, len(shift))
            if new <= 0:
                continue
            sent_upto = clock
            # Only the rows a speaker actually holds. The buffers are sized for
            # SPEAKER_CAPACITY, but a row nobody occupies is masked out and the
            # model's output for it is meaningless - measured, a masked row has
            # no effect on the others at all - so it is not sent and the page
            # never draws a speaker who does not exist.
            seated = sorted({t["speaker"] for t in session.tracks})
            await socket.send_text(json.dumps({
                "t": clock,
                "n": new,
                "shift": [round(float(x), 4) for x in shift[-new:]],
                "speakers": seated,
                "activity": [
                    [round(float(x), 4) for x in activity[s][-new:]] for s in seated
                ],
                "capacity": SPEAKER_CAPACITY,
                "shift_scale": round(session.shift_scale, 3),
                "tracks": [
                    {"speaker": t["speaker"], "bbox": t["bbox"]}
                    for t in session.tracks if t["bbox"]
                ],
                "frames": session.context,
                # The model's own cost, and the same measured from the loop -
                # the gap between them is scheduling, not thinking, and is the
                # honest thing to show when a reply arrives late.
                "latency_ms": round(compute_ms, 1),
                "round_trip_ms": round((time.perf_counter() - started) * 1000, 1),
                "stages": session.stages,
            }))

    async def detect_loop():
        last_identity = 0.0
        deadline = time.monotonic()
        while True:
            deadline += 1.0 / DETECT_HZ
            delay = deadline - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
            else:
                deadline = time.monotonic()
            if session.frame is None:
                continue
            now = time.monotonic()
            identity = session.identity_due(now - last_identity)
            if identity:
                last_identity = now
            await asyncio.to_thread(session.detect, identity)

    async def guarded(name, coro):
        """A task that dies quietly takes the demo with it; say so instead."""
        try:
            await coro()
        except asyncio.CancelledError:
            raise
        except Exception:
            print(f"[{name}] stopped:\n{traceback.format_exc()}", flush=True)

    workers = [
        asyncio.create_task(guarded("infer", infer_loop)),
        asyncio.create_task(guarded("detect", detect_loop)),
    ]
    try:
        while True:
            message = await socket.receive()
            if "bytes" in message and message["bytes"] is not None:
                payload = message["bytes"]
                kind = payload[0]
                if kind == 1:                                   # audio, float32 mono
                    session.push_audio(
                        np.frombuffer(payload[1:], dtype=np.float32).copy()
                    )
                elif kind == 2:                                 # video, JPEG
                    frame = cv2.imdecode(
                        np.frombuffer(payload[1:], dtype=np.uint8), cv2.IMREAD_COLOR
                    )
                    if frame is not None:
                        session.push_frame(frame)
            elif message.get("text") is not None:
                if message["text"] == "close":
                    break
                # Control messages are the only text the page sends. A bad one
                # is ignored rather than fatal: the demo keeps predicting at
                # whatever scale it already had.
                try:
                    control = json.loads(message["text"])
                    wanted = float(control["shift_scale"])
                except (ValueError, TypeError, KeyError):
                    continue
                if math.isfinite(wanted):
                    session.shift_scale = min(max(wanted, SHIFT_SCALE_MIN),
                                              SHIFT_SCALE_MAX)
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        for worker in workers:
            worker.cancel()


def main():
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vap-weights", required=True)
    parser.add_argument("--asd-weights", required=True)
    parser.add_argument(
        "--muvap-weights", required=True,
        help="published MuVAP release, or a training .ckpt",
    )
    parser.add_argument("--config", default=HERE.parent / "config/yaml/muvap.yaml")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8443)
    parser.add_argument("--certfile")
    parser.add_argument("--keyfile")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    STATE["models"] = load_models(
        args.vap_weights, args.asd_weights, args.muvap_weights, args.config, args.device
    )
    print(f"models ready on {args.device}; {CONTEXT_SEC:g}s context at {INFER_HZ} Hz")
    uvicorn.run(
        app, host=args.host, port=args.port,
        ssl_certfile=args.certfile, ssl_keyfile=args.keyfile, log_level="warning",
    )


if __name__ == "__main__":
    main()
