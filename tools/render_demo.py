"""Build a self-contained demo page: a minute of conversation, with predictions.

Where `render_sample.py` inspects the *data*, this shows the *model* running on
it. The original recording plays on top; underneath, a ten-second window of the
timeline scrolls past a fixed playhead, carrying what MuVAP predicted frame by
frame against what the corpus says.

Three readouts, all taken from one forward pass:

* **shift probability** from the global head - the same `p_shift` the hold/shift
  decision thresholds at 0.5, drawn as a curve so the decision is a line rather
  than a verdict;
* **per-speaker activity** from the first SVAP bin. That bin covers the 0.2 s
  straddling now, which makes it the model's own answer to "is this face
  talking", and on this clip it matches the annotation 90% of the time;
* **the annotation itself**, drawn behind the prediction in a lighter tone so
  agreement and disagreement are both visible at a glance.
"""

import argparse
import base64
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

FACE = 112
GRID = 25


def sprite(faces: np.ndarray, quality: int = 78) -> str:
    rows = math.ceil(len(faces) / GRID)
    sheet = np.full((rows * FACE, GRID * FACE), 128, dtype=np.uint8)
    for index, face in enumerate(faces):
        r, c = divmod(index, GRID)
        sheet[r * FACE : (r + 1) * FACE, c * FACE : (c + 1) * FACE] = face
    ok, buf = cv2.imencode(".jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("could not encode faces")
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


TEMPLATE = """<title>__TITLE__</title>
<style>
  :root{--bg:#faf9f7;--ink:#1b1a18;--muted:#6f6b66;--line:#e0dbd4;--panel:#fff;
        --gt:#b9d6c4;--pred:#2f7d5d;--idle:#ece8e2;--hot:#c8442c;--cool:#b9b5ae;--mark:#d62828}
  @media(prefers-color-scheme:dark){:root:not([data-theme=light]){
        --bg:#141417;--ink:#eeece9;--muted:#9a958e;--line:#33322f;--panel:#1d1d21;
        --gt:#33574a;--idle:#2a2a2f;--cool:#4a4842}}
  body{background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,sans-serif;
       margin:0 auto;max-width:1000px;padding-block:20px;padding-left:16px;padding-right:16px}
  h1{font-size:17px;margin:0 0 2px}
  .sub{color:var(--muted);font-size:13px;margin:0 0 14px}
  .panel{background:var(--panel);border:1px solid var(--line);border-radius:10px;
         padding:12px;margin-bottom:12px}
  .stage{position:relative;line-height:0}
  video{width:100%;border-radius:6px;display:block;background:#000}
  canvas#ov{position:absolute;left:0;top:0;width:100%;height:100%;pointer-events:none}
  .faces{display:flex;gap:12px;justify-content:center;margin-top:12px}
  .face{text-align:center}
  .face canvas{width:104px;height:104px;border-radius:8px;border:3px solid var(--cool);
               image-rendering:pixelated;display:block;background:#888}
  .face.on canvas{border-color:var(--pred)}
  .face.off canvas{border-color:var(--hot)}
  .face .nm{font-size:12px;margin-top:5px;color:var(--muted)}
  .face .nm b{color:var(--ink)}
  canvas#strip{width:100%;height:auto;display:block}
  .legend{color:var(--muted);font-size:12px;margin-top:8px;display:flex;gap:16px;flex-wrap:wrap}
  .sw{display:inline-block;width:22px;height:9px;border-radius:2px;vertical-align:middle;margin-right:5px}
</style>

<h1 id="ttl"></h1>
<p class="sub" id="sub"></p>

<div class="panel">
  <div class="stage"><video id="v" controls playsinline></video><canvas id="ov"></canvas></div>
  <div class="faces" id="faces"></div>
</div>

<div class="panel">
  <canvas id="strip"></canvas>
  <div class="legend">
    <span><i class="sw" style="background:var(--hot)"></i>shift probability above 0.5</span>
    <span><i class="sw" style="background:var(--cool)"></i>below 0.5</span>
    <span><i class="sw" style="background:var(--gt)"></i>annotated speech</span>
    <span><i class="sw" style="background:var(--pred)"></i>predicted speech (SVAP bin 1)</span>
    <span>boxes and borders follow the <b>prediction</b>, never the annotation</span>
  </div>
</div>

<script id="p" type="application/json">__PAYLOAD__</script>
<script>
const D=JSON.parse(document.getElementById("p").textContent);
const FPS=D.fps,N=D.frames,S=D.speakers.length,WIN=D.window_sec*FPS;
const FACE_S=D.face,GRID_C=D.columns;
const v=document.getElementById("v");v.src=D.video;
document.getElementById("ttl").textContent=D.title;
document.getElementById("sub").textContent=
  `${(N/FPS).toFixed(0)} s around a ${D.event.label} at ${D.event.at.toFixed(1)} s `
  +`(speaker ${D.event.previous} → ${D.event.following}) · `
  +`${D.window_sec} s of timeline scrolls past the playhead`;

const sheets=[],cells=[];
D.speakers.forEach((nm,i)=>{
  const w=document.createElement("div");w.className="face";
  const c=document.createElement("canvas");c.width=FACE_S;c.height=FACE_S;w.appendChild(c);
  const n=document.createElement("div");n.className="nm";w.appendChild(n);
  document.getElementById("faces").appendChild(w);
  cells.push({w,ctx:c.getContext("2d"),n});
  const im=new Image();im.src=D.faces[i];sheets.push(im);
});

const st=document.getElementById("strip");
const ROW=22,GAP=6,PAD=78,RIGHT=12,PROB=76;
const W=980,H=PROB+GAP+S*(ROW+GAP)+22;
st.width=W;st.height=H;const g=st.getContext("2d");
const css=getComputedStyle(document.documentElement),col=k=>css.getPropertyValue(k).trim();
const PLOT=W-PAD-RIGHT, PPF=PLOT/WIN;

function draw(f){
  g.fillStyle=col("--panel");g.fillRect(0,0,W,H);
  const half=WIN/2, from=f-half;                       // playhead fixed at centre
  const x=t=>PAD+(t-from)*PPF;

  /* shift probability, filled to the axis: red above the threshold, grey below */
  const top=4,bot=top+PROB, mid=top+PROB*0.5;
  const lo=Math.max(0,Math.floor(from)), hiT=Math.min(N,from+WIN+1);
  const area=()=>{                       // the whole region under the curve
    g.beginPath();g.moveTo(x(lo),bot);
    for(let t=lo;t<hiT;t++)g.lineTo(x(t),bot-D.shift[t]*PROB);
    g.lineTo(x(hiT-1),bot);g.closePath();
  };
  /* Clipped in two bands rather than drawn as two shapes: everything above the
     0.5 line is red, everything below it grey, and the curve itself decides
     where the boundary falls. */
  g.save();g.beginPath();g.rect(PAD,mid,PLOT,bot-mid);g.clip();
  area();g.fillStyle=col("--cool");g.globalAlpha=.55;g.fill();g.restore();g.globalAlpha=1;
  g.save();g.beginPath();g.rect(PAD,top,PLOT,mid-top);g.clip();
  area();g.fillStyle=col("--hot");g.globalAlpha=.85;g.fill();g.restore();g.globalAlpha=1;
  g.strokeStyle=col("--ink");g.lineWidth=1.5;g.beginPath();
  for(let t=lo;t<hiT;t++){const y=bot-D.shift[t]*PROB;t===lo?g.moveTo(x(t),y):g.lineTo(x(t),y);}
  g.stroke();
  g.setLineDash([5,4]);g.strokeStyle=col("--muted");g.lineWidth=1;
  g.beginPath();g.moveTo(PAD,mid);g.lineTo(PAD+PLOT,mid);g.stroke();g.setLineDash([]);
  g.fillStyle=col("--muted");g.font="11px system-ui,sans-serif";g.textAlign="right";
  g.fillText("p(shift)",PAD-8,top+12);g.fillText("0.5",PAD-8,mid+4);

  /* one row per speaker: annotation behind, prediction in front */
  let y=bot+GAP;
  D.speakers.forEach((nm,i)=>{
    g.fillStyle=col("--idle");g.fillRect(PAD,y,PLOT,ROW);
    for(let t=Math.max(0,Math.floor(from));t<Math.min(N,from+WIN+1);t++){
      const w=Math.ceil(PPF)+.5;
      if(D.vad[i][t]){g.fillStyle=col("--gt");g.fillRect(x(t),y,w,ROW);}
      if(D.act[i][t]>0.5){g.fillStyle=col("--pred");g.fillRect(x(t),y+ROW*0.30,w,ROW*0.70);}
    }
    g.fillStyle=col("--muted");g.textAlign="right";g.fillText("speaker "+nm,PAD-8,y+ROW-6);
    y+=ROW+GAP;
  });

  /* the event, and the playhead */
  const ex=x(D.event.frame);
  if(ex>PAD&&ex<PAD+PLOT){g.strokeStyle=col("--muted");g.setLineDash([3,3]);g.lineWidth=1;
    g.beginPath();g.moveTo(ex,top);g.lineTo(ex,y-GAP);g.stroke();g.setLineDash([]);
    g.fillStyle=col("--muted");g.textAlign="center";g.fillText(D.event.label,ex,y+9);}
  g.fillStyle=col("--mark");g.fillRect(PAD+half*PPF-1,top,2.5,y-GAP-top);

  g.fillStyle=col("--muted");g.textAlign="center";
  for(let s=Math.ceil(from/FPS);s<=(from+WIN)/FPS;s++){
    if(s<0||s>N/FPS)continue;g.fillText(s+"s",x(s*FPS),H-4);
  }
}

/* The video reports no time until it has metadata, so this never indexes with
   NaN and the page draws frame 0 while it loads rather than throwing. */
const ov=document.getElementById("ov"),og=ov.getContext("2d");
function boxes(f){
  if(!ov.width)return;
  og.clearRect(0,0,ov.width,ov.height);
  D.speakers.forEach((nm,i)=>{
    if(!D.vis[i][f])return;                       // no face tracked this frame
    const b=D.bbox[i][f], on=D.act[i][f]>0.5;
    const X=b[0]*ov.width, Y=b[1]*ov.height, X2=b[2]*ov.width, Y2=b[3]*ov.height;
    if(X2<=X||Y2<=Y)return;
    og.strokeStyle=on?col("--pred"):col("--hot");og.lineWidth=Math.max(2,ov.width/380);
    og.strokeRect(X,Y,X2-X,Y2-Y);
    const label=`${nm}  ${D.act[i][f].toFixed(2)}`;
    og.font=`${Math.round(ov.width/55)}px system-ui,sans-serif`;
    const w=og.measureText(label).width+10, h=ov.width/42;
    og.fillStyle=on?col("--pred"):col("--hot");og.fillRect(X,Math.max(0,Y-h),w,h);
    og.fillStyle="#fff";og.textAlign="left";og.fillText(label,X+5,Math.max(0,Y-h)+h*0.76);
  });
}
v.addEventListener("loadedmetadata",()=>{ov.width=v.videoWidth;ov.height=v.videoHeight;last=-1;});

function frameAt(){
  const t=Number.isFinite(v.currentTime)?v.currentTime:0;
  return Math.max(0,Math.min(N-1,Math.floor(t*FPS)));
}
let last=-1;
function tick(){
  const f=frameAt();
  if(f!==last){
    last=f;draw(f);boxes(f);
    D.speakers.forEach((nm,i)=>{
      const c=cells[i],sh=sheets[i];
      if(sh.complete){const r=Math.floor(f/GRID_C),q=f%GRID_C;
        c.ctx.drawImage(sh,q*FACE_S,r*FACE_S,FACE_S,FACE_S,0,0,FACE_S,FACE_S);}
      const on=D.act[i][f]>0.5;
      c.w.className="face "+(on?"on":"off");
      c.n.innerHTML=`<b>speaker ${nm}</b><br>${on?"speaking":"silent"} · p=${D.act[i][f].toFixed(2)}`;
    });
  }
  requestAnimationFrame(tick);
}
let pending=sheets.length;sheets.forEach(im=>im.onload=()=>{if(--pending===0)last=-1;});
draw(0);tick();
</script>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="npz from the model run")
    parser.add_argument("--video", type=Path, required=True, help="the matching clip")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window-sec", type=float, default=10.0)
    parser.add_argument("--title", default="MuVAP on AVCC")
    parser.add_argument("--event-label", default="shift")
    parser.add_argument("--event-at", type=float, default=30.0)
    parser.add_argument("--previous", default="0")
    parser.add_argument("--following", default="2")
    args = parser.parse_args()

    d = np.load(args.data, allow_pickle=False)
    speakers = [str(s) for s in d["speakers"]]
    frames = int(d["shift"].shape[0])
    payload = {
        "title": args.title,
        "fps": 25,
        "frames": frames,
        "window_sec": args.window_sec,
        "speakers": speakers,
        "face": FACE,
        "columns": GRID,
        "video": "data:video/mp4;base64,"
        + base64.b64encode(args.video.read_bytes()).decode(),
        "faces": [sprite(d["faces"][i]) for i in range(len(speakers))],
        "shift": [round(float(x), 4) for x in d["shift"]],
        # The first SVAP bin: the model's own "is this face talking now".
        "act": [[round(float(x), 4) for x in d["svap"][i, :, 0]] for i in range(len(speakers))],
        "vad": [[int(x) for x in d["vad"][i]] for i in range(len(speakers))],
        "vis": [[int(x) for x in d["vis"][i]] for i in range(len(speakers))],
        "bbox": [[[round(float(v), 4) for v in f] for f in d["bbox"][i]] for i in range(len(speakers))],
        "event": {
            "frame": int(d["event_frame"][0]),
            "at": args.event_at,
            "label": args.event_label,
            "previous": args.previous,
            "following": args.following,
        },
    }
    html = TEMPLATE.replace("__TITLE__", args.title).replace(
        "__PAYLOAD__", json.dumps(payload)
    )
    args.output.write_text(html)
    print(f"{args.output}  ({args.output.stat().st_size / 1e6:.1f} MB, "
          f"{frames} frames, {len(speakers)} speakers)")


if __name__ == "__main__":
    main()
