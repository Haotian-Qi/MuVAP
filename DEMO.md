# Running the live demo

A browser sends its camera and microphone to this server; the server holds a
rolling ten seconds of exactly what the training loader would have built, runs
the frozen VAP and ASD modules and the fusion over it, and streams the
predictions back. Nothing on the page is annotated - every number came out of
the model a moment earlier.

This file is the runbook. The reasoning behind the tracking rules lives in the
module docstring of `demo/tracker.py`, and the fusion itself is `docs/muvap.md`.

## Before you start

The demo needs the two frozen modules, a trained fusion checkpoint, and a TLS
certificate. Released weights come from the Hugging Face repo named in the
[README](README.md); the checkpoint is whatever `train_muvap.py` wrote.

A camera only opens on a secure origin, so the server has to speak https. A
self-signed certificate is enough - the browser warns once and you accept it:

```bash
mkdir -p cert
openssl req -x509 -newkey rsa:2048 -nodes -days 365 \
  -keyout cert/key.pem -out cert/cert.pem \
  -subj "/CN=$(hostname)" \
  -addext "subjectAltName=DNS:$(hostname),DNS:localhost,IP:127.0.0.1,IP:$(hostname -I | awk '{print $1}')"
```

The demo needs a few packages beyond the training extras - `fastapi`,
`uvicorn`, `websockets`, `insightface`, `onnxruntime-gpu`, `opencv-python`.

## Start it

```bash
python -u -m demo.server \
  --vap-weights weights/vap-role-cpc \
  --asd-weights weights/asd-cpc \
  --checkpoint <run>/best.ckpt \
  --certfile cert/cert.pem \
  --keyfile cert/key.pem \
  --port 8443
```

Startup takes about a minute: the frozen pair and the fusion are loaded, then a
warm-up pass is run so the first visitor does not pay for kernel selection. It
is ready when it prints:

```
face detection on the GPU
models ready on cuda; 10s context at 25 Hz
```

Long runs belong in tmux, since the process must outlive the shell:

```bash
tmux new-session -d -s live -n server
tmux send-keys -t live:server 'python -u -m demo.server \
  --vap-weights weights/vap-role-cpc \
  --asd-weights weights/asd-cpc \
  --checkpoint <run>/best.ckpt \
  --certfile cert/cert.pem --keyfile cert/key.pem \
  --port 8443 2>&1 | tee live_server.log' C-m
tmux attach -t live          # detach with ctrl-b d
```

Waiting for it to come up, rather than guessing at a sleep:

```bash
until curl -sk https://127.0.0.1:8443/health | grep -q '"ready":true'; do sleep 3; done
```

## The link

Open `https://<host>:8443/` in a browser. Plain http, or an address the
certificate does not cover, will not get a camera. If the browser refuses the
self-signed certificate for camera access, tunnel instead and use localhost,
which counts as secure on its own:

```bash
ssh -L 8443:localhost:8443 <host>
# then open https://localhost:8443/
```

`GET /health` answers without a camera and is the quickest check that the
process is up:

```bash
curl -sk https://127.0.0.1:8443/health
# {"ready":true,"device":"cuda","context_sec":10.0,"infer_hz":25,"capacity":6}
```

## When it does not work

**"face detection on the CPU"** in the startup log. Detection still works but is
48 ms instead of 6.5 ms, and it competes with the model for cores. onnxruntime
needs cuDNN 9, which this torch does not ship. Install a copy *beside* the
environment rather than into it - torch is pinned to cuDNN 8.9 and installing 9
over it would break torch - and point `MUVAP_ORT_CUDNN` at the library
directory:

```bash
python -m pip install --target <ortdeps> "nvidia-cudnn-cu12>=9,<10"
export MUVAP_ORT_CUDNN=<ortdeps>/nvidia/cudnn/lib
```

The two sonames differ, so both live in the process at once without either
seeing the other's. Unset, or pointed at a directory that does not exist,
detection falls back to the CPU and the log line says which library failed.

**Port already in use.** An older server is still running: `tmux ls`, then stop
it as below, or `ss -ltnp | grep 8443` to find the process.

**The page loads but nothing moves.** The websocket is not connected or the
camera was refused - the page shows the reason in a banner. Check the server log
for a `[infer] stopped:` or `[detect] stopped:` traceback; a worker that dies is
reported rather than failing silently.

## Stop it

```bash
tmux send-keys -t live:server C-c
ss -ltnp | grep 8443                   # silent when it is down
```

## Knobs

All in `demo/server.py` unless noted. None need changing to run it.

| Constant | Value | What it is |
| --- | --- | --- |
| `FPS` | 25 | The model's frame rate. One frame is 640 audio samples. |
| `CONTEXT_SEC` | 10.0 | Length of the window the model reads. |
| `INFER_HZ` | 25 | How often a forward pass is started. |
| `DETECT_HZ` | 25 | How often faces are detected. |
| `SPEAKER_CAPACITY` | 6 | Rows the buffers hold (`demo/tracker.py`). |
| `NEW_SPEAKER_COST` | 0.78 | Above this, a face is somebody new (`demo/tracker.py`). |

Two things worth knowing before changing anything:

The clock is the **audio**, never the video. A webcam's frame rate wanders and
25 Hz is not negotiable, so the server advances one model frame per 640 samples
and each frame takes whichever crop was newest. Video jitter becomes a repeated
crop rather than drift between the streams.

The **speaker count is dynamic**. Rows are created as people arrive and a
speaker who leaves keeps their identity, so they get their own row back on
sight. The model consumes the speaker axis as a set inside each frame, so it
runs unchanged at any count and costs the same: 22.4 ms at one speaker, 22.7 ms
at eight. `SPEAKER_CAPACITY` is only the size of the buffers. It was trained on
two- and three-speaker conversations, so more than that runs but is
extrapolation.
