"""Turn one run of `tools/demo_infer.py` into a video that can sit in a slide.

Everything on the canvas is one frame of the model's state, and every frame was
produced from ten seconds of history and nothing after it, so the video plays at
the rate the model actually runs at. Four readouts, arranged so the eye can get
from a face to a prediction without leaving the frame:

* **the recording**, with one box per tracked face. The box carries the speaker
  id the model indexes that face by, and it thickens and lights a dot the moment
  the model says that face is talking - the first SVAP bin, which covers the
  0.2 s straddling now;
* **the per-speaker tracks**, five seconds of that same decision scrolling past
  a fixed now-line at the right edge. This is history: what is drawn has already
  been decided and never moves again;
* **the SVAP head itself**, beside each track - all four future bins for that
  speaker, so the track's on/off is visible as the first column of a profile
  that reaches 0.8 s ahead;
* **the GVAP head**, ranked. The global head is a distribution over 136
  role-relative states, and the five most likely are drawn as the states they
  are - two rows of four bins, `Scurr` over `Snext`, past to the left of the
  now-line - rather than as five numbers.

The rendering is deliberately dumb: no smoothing, no thresholding beyond the
0.5 the model is scored at, nothing interpolated between frames. What is on the
screen at frame `t` is what came out of the forward pass at frame `t`.
"""

import argparse
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

W, H = 1920, 1080
MARGIN = 32
FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")

#: One colour per speaker row, reused by the box, the chip and the track so a
#: face and its prediction never have to be matched by position.
SPEAKER_COLOURS = [
    (240, 162, 60),    # amber
    (91, 155, 245),    # blue
    (70, 196, 155),    # teal
    (214, 122, 196),   # orchid
    (232, 106, 92),    # coral
    (176, 176, 92),    # olive
]

THEMES = {
    "dark": {
        "bg": (16, 17, 20),
        "panel": (26, 28, 33),
        "panel_edge": (46, 49, 56),
        "ink": (237, 235, 232),
        "muted": (142, 138, 133),
        "faint": (58, 61, 68),
        "idle": (37, 39, 45),
        "now": (214, 60, 48),
    },
    "light": {
        "bg": (248, 247, 245),
        "panel": (255, 255, 255),
        "panel_edge": (223, 219, 213),
        "ink": (27, 26, 24),
        "muted": (111, 107, 102),
        "faint": (208, 204, 198),
        "idle": (236, 233, 227),
        "now": (198, 54, 42),
    },
}


def fonts():
    def load(name, size):
        return ImageFont.truetype(str(FONT_DIR / name), size)

    return {
        "h1": load("DejaVuSans-Bold.ttf", 30),
        "sub": load("DejaVuSans.ttf", 19),
        "panel": load("DejaVuSans-Bold.ttf", 18),
        "body": load("DejaVuSans.ttf", 16),
        "small": load("DejaVuSans.ttf", 14),
        "tiny": load("DejaVuSans.ttf", 12),
        "num": load("DejaVuSansMono-Bold.ttf", 19),
        "numsmall": load("DejaVuSansMono.ttf", 14),
        "chip": load("DejaVuSans-Bold.ttf", 24),
        "big": load("DejaVuSans-Bold.ttf", 44),
        "box": load("DejaVuSans-Bold.ttf", 26),
        "next": load("DejaVuSans-Bold.ttf", 40),
    }


def scalar(value):
    """A stored scalar, whether numpy kept it 0-d or as a one-element array."""
    return np.asarray(value).reshape(-1)[0]


def mix(a, b, t):
    """`a` toward `b`; used to tint a panel rather than keep a second palette."""
    return tuple(int(round(x + (y - x) * t)) for x, y in zip(a, b))


def panel(draw, box, theme, title=None, font=None):
    draw.rounded_rectangle(box, 12, fill=theme["panel"], outline=theme["panel_edge"])
    if title:
        draw.text((box[0] + 18, box[1] + 14), title, font=font, fill=theme["muted"])


class Layout:
    """Where everything lives, given the shape of the picture being shown.

    Two panels: the recording, and under it the four tracks. The picture takes
    whatever height is left once the tracks have the room they need, so a 16:9
    clip and a wide cropped strip both fill the frame without the tracks ever
    being squeezed.
    """

    #: What the four tracks need. Below this they stop being readable from the
    #: back of a room, so the picture gives way rather than they do.
    TRACKS_H = 372
    GAP = 22

    def __init__(self, aspect: float):
        video_h = H - 2 * MARGIN - self.GAP - self.TRACKS_H
        video_w = round(video_h * aspect)
        if video_w > W - 2 * MARGIN:                  # wider than the frame allows
            video_w = W - 2 * MARGIN
            video_h = round(video_w / aspect)
        left = (W - video_w) // 2
        top = MARGIN + (H - 2 * MARGIN - self.GAP - self.TRACKS_H - video_h) // 2
        self.video = (left, top, left + video_w, top + video_h)
        self.tracks = (MARGIN, H - MARGIN - self.TRACKS_H, W - MARGIN, H - MARGIN)

        self.chip_x = MARGIN + 22
        self.svap_x = MARGIN + 86
        self.track_x0 = MARGIN + 320
        self.track_x1 = W - MARGIN - 22

        # The shift curve on top, then one row per speaker. The curve is worth
        # half as much again as a speaker row: it is a plot, not a bar.
        pad, gap = 16, 8
        usable = self.TRACKS_H - 2 * pad - 3 * gap
        unit = usable / 4.5
        self.shift_top = self.tracks[1] + pad
        self.shift_h = unit * 1.5
        self.row_gap = gap
        self.row_h = unit
        self.row_top = self.shift_top + self.shift_h + gap

    @property
    def now_x(self):
        """The red line: the midpoint of the lane, with past to its left."""
        return (self.track_x0 + self.track_x1) / 2


def parse_crop(text):
    """`x0,y0,x1,y1` in fractions of the frame, or None for the whole frame."""
    if not text:
        return None
    parts = [float(v) for v in text.split(",")]
    if len(parts) != 4:
        raise ValueError("--crop takes four fractions: x0,y0,x1,y1")
    x0, y0, x1, y1 = parts
    if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
        raise ValueError(f"--crop {text} is not a rectangle inside the frame")
    return (x0, y0, x1, y1)


def cropped(frame, crop):
    """The visible part of a source frame, and the box that maps onto it.

    Cropping is a change to what is *shown*, never to what was run: the model
    read 112x112 crops of the faces, which this cannot touch. It exists so a
    recording with a burned-in overlay - a stream's chat, a broadcaster's
    banner - can be put on a slide without the overlay coming along.
    """
    if crop is None:
        return frame, (0.0, 0.0, 1.0, 1.0)
    height, width = frame.shape[:2]
    x0, y0, x1, y1 = crop
    return frame[round(y0 * height) : round(y1 * height),
                 round(x0 * width) : round(x1 * width)], crop


def place(box, crop):
    """A normalised box in frame coordinates, in cropped-frame coordinates."""
    x0, y0, x1, y1 = crop
    return (
        (box[0] - x0) / (x1 - x0),
        (box[1] - y0) / (y1 - y0),
        (box[2] - x0) / (x1 - x0),
        (box[3] - y0) / (y1 - y0),
    )


def draw_video(canvas, draw, frame, data, index, fonts_, theme, layout, crop):
    """The recording, with a box per tracked face and the speaking indicator."""
    x0, y0, x1, y1 = layout.video
    width, height = x1 - x0, y1 - y0
    visible_frame, window = cropped(frame, crop)
    shot = Image.fromarray(cv2.cvtColor(visible_frame, cv2.COLOR_BGR2RGB)).resize(
        (width, height), Image.LANCZOS
    )
    canvas.paste(shot, (x0, y0))
    draw.rounded_rectangle(layout.video, 10, outline=theme["panel_edge"])

    for speaker in range(len(data["speakers"])):
        if not data["visible"][speaker, index]:
            continue
        box = place(data["bbox"][speaker, index], window)
        bx0, by0 = x0 + box[0] * width, y0 + box[1] * height
        bx1, by1 = x0 + box[2] * width, y0 + box[3] * height
        if bx1 <= bx0 or by1 <= by0 or bx1 < x0 or bx0 > x1 or by1 < y0 or by0 > y1:
            continue                                   # cropped out of the picture
        colour = SPEAKER_COLOURS[speaker % len(SPEAKER_COLOURS)]
        probability = float(data["svap"][speaker, index, 0])
        speaking = probability > 0.5

        # Wider while the model says this face is talking; the dim ring around
        # the box is the same signal again, so it survives being seen small.
        if speaking:
            draw.rounded_rectangle(
                (bx0 - 7, by0 - 7, bx1 + 7, by1 + 7), 10, outline=mix(colour, theme["bg"], 0.6), width=3
            )
        draw.rounded_rectangle(
            (bx0, by0, bx1, by1), 6, outline=colour, width=7 if speaking else 3
        )

        # The tag: the speaker id the model indexes this face by, and a dot
        # that fills while the model says it is talking. No number - the box
        # already carries the decision in its width.
        label = str(data["speakers"][speaker])
        tag_h = 36
        tag_w = 34 + draw.textlength(label, font=fonts_["box"]) + 14
        ty = max(y0 + 2, by0 - tag_h - 6)
        draw.rounded_rectangle((bx0, ty, bx0 + tag_w, ty + tag_h), 8, fill=colour)
        cy = ty + tag_h / 2
        if speaking:
            draw.ellipse((bx0 + 11, cy - 8, bx0 + 27, cy + 8), fill=(255, 255, 255))
        else:
            draw.ellipse(
                (bx0 + 11, cy - 8, bx0 + 27, cy + 8),
                outline=mix(colour, (0, 0, 0), 0.45), width=2,
            )
        draw.text((bx0 + 34, cy), label, font=fonts_["box"], fill=(20, 20, 20), anchor="lm")


def draw_shift(draw, data, index, fonts_, theme, layout, window_sec, fps):
    """p(shift) as a curve, filled against the 0.5 line it is thresholded at.

    The fill is split by the threshold rather than by the curve: everything the
    curve encloses above 0.5 is drawn in the shift colour and everything below
    it in grey, so the decision is a region of the plot rather than a verdict
    printed somewhere. The curve stops at the now-line, because a streaming
    model has not computed the rest of it yet.
    """
    top = layout.shift_top
    bottom = top + layout.shift_h
    span = int(window_sec * fps)
    pixels = (layout.now_x - layout.track_x0) / span
    half = bottom - layout.shift_h * 0.5

    draw.rounded_rectangle(
        (layout.track_x0, top, layout.track_x1, bottom), 6,
        fill=mix(theme["idle"], theme["panel"], 0.62),
    )
    draw.rounded_rectangle(
        (layout.track_x0, top, layout.now_x, bottom), 6, fill=theme["idle"]
    )

    shift = data["p_shift"]
    first = index - span + 1
    hold_fill = mix(theme["muted"], theme["idle"], 0.72)

    # Drawn a pixel column at a time: the split at 0.5 is then exact, and no
    # polygon has to be intersected with the threshold line to find it.
    previous = None
    for x in range(int(layout.track_x0), int(layout.now_x) + 1):
        frame = first + (x - layout.track_x0) / pixels
        if frame < 0 or frame > index:
            continue
        value = float(shift[int(round(frame))])
        y = bottom - layout.shift_h * value
        draw.line((x, max(y, half), x, bottom), fill=hold_fill)
        if y < half:
            draw.line((x, y, x, half), fill=theme["now"])
        if previous is not None:
            draw.line((x - 1, previous, x, y), fill=theme["ink"], width=2)
        previous = y

    x = layout.track_x0                               # the 0.5 line, dashed
    while x < layout.track_x1:
        draw.line((x, half, min(x + 7, layout.track_x1), half), fill=theme["muted"], width=1)
        x += 13

    # The one number on the canvas: what the curve reads right now.
    value = float(shift[index])
    accent = theme["now"] if value > 0.5 else mix(theme["ink"], theme["panel"], 0.30)
    draw.text(
        (layout.track_x0 - 26, (top + bottom) / 2), f"{value:.2f}", font=fonts_["big"],
        fill=accent, anchor="rm",
    )


def draw_tracks(draw, data, index, fonts_, theme, layout, window_sec, fps):
    """Per speaker: the SVAP profile now, what was predicted, and what was said.

    The now-line is the middle of the lane, and the two halves are not the same
    kind of thing. Left of it is the past, carrying both the model's decision
    (the thick bar) and the corpus's (the thin ribbon under it); right of it is
    the future, where a streaming model has nothing to show, so only the
    annotation continues. The clip therefore runs into its own answer.
    """
    speakers = list(data["speakers"])
    span = int(window_sec * fps)
    pixels = (layout.now_x - layout.track_x0) / span
    svap_bins = np.asarray(data["svap_bin_sec"], dtype=np.float64)
    truth_frames = data["vad"].shape[1]

    for row, name in enumerate(speakers):
        top = layout.row_top + row * (layout.row_h + layout.row_gap)
        colour = SPEAKER_COLOURS[row % len(SPEAKER_COLOURS)]
        speaking = float(data["svap"][row, index, 0]) > 0.5

        # Every part of a row is a fraction of its height, so the panel fills
        # whatever the picture above it left over.
        height = layout.row_h
        chip_s = min(52, height * 0.42)
        chip_y = top + height * 0.30
        chip = (layout.chip_x, chip_y, layout.chip_x + chip_s, chip_y + chip_s)
        draw.rounded_rectangle(
            chip, 10, fill=colour if speaking else mix(colour, theme["panel"], 0.65)
        )
        draw.text(
            ((chip[0] + chip[2]) / 2, (chip[1] + chip[3]) / 2 + 1), str(name),
            font=fonts_["chip"], fill=(20, 20, 20) if speaking else mix(colour, (0, 0, 0), 0.3),
            anchor="mm",
        )

        # All four bins of this speaker's head. The first is the one the box and
        # the lane are drawn from; the rest reach 0.8 s ahead of it.
        bar_w, gap = 32, 12
        tall = min(62, height * 0.50)
        base = top + height * 0.74
        for b in range(len(svap_bins)):
            value = float(data["svap"][row, index, b])
            bx = layout.svap_x + b * (bar_w + gap)
            draw.rectangle(
                (bx, base - tall, bx + bar_w, base), fill=mix(theme["idle"], theme["panel"], 0.25)
            )
            if tall * value > 1:
                draw.rectangle(
                    (bx, base - tall * value, bx + bar_w, base),
                    fill=colour if b == 0 and speaking else mix(colour, theme["panel"], 0.45),
                )
        draw.line(
            (layout.svap_x, base - tall / 2, layout.svap_x + 4 * (bar_w + gap) - gap, base - tall / 2),
            fill=theme["faint"], width=1,
        )

        # The lane. The future half is lighter, so the eye is told the two
        # halves differ before it reads anything in them.
        lane_top, lane_bottom = top + height * 0.16, top + height * 0.84
        draw.rounded_rectangle(
            (layout.track_x0, lane_top, layout.track_x1, lane_bottom), 6,
            fill=mix(theme["idle"], theme["panel"], 0.62),
        )
        draw.rounded_rectangle(
            (layout.track_x0, lane_top, layout.now_x, lane_bottom), 6, fill=theme["idle"]
        )

        ribbon = max(10.0, (lane_bottom - lane_top) * 0.24)
        predicted_bottom = lane_bottom - ribbon
        first = index - span + 1
        offset = max(0, -first)

        history = data["svap"][row, max(0, first) : index + 1, 0] > 0.5
        for begin, stop in runs(history):
            left = layout.track_x0 + (offset + begin) * pixels
            right = max(layout.track_x0 + (offset + stop) * pixels, left + 2)
            draw.rounded_rectangle((left, lane_top + 4, right, predicted_bottom), 4, fill=colour)

        truth_to = min(truth_frames, index + span + 1)
        truth = data["vad"][row, max(0, first) : truth_to] > 0
        for begin, stop in runs(truth):
            left = layout.track_x0 + (offset + begin) * pixels
            right = max(layout.track_x0 + (offset + stop) * pixels, left + 2)
            draw.rectangle(
                (left, predicted_bottom + 5, right, lane_bottom - 4),
                fill=mix(colour, theme["panel"], 0.4),
            )

    bottom = layout.row_top + 3 * (layout.row_h + layout.row_gap) - layout.row_gap
    draw.line(
        (layout.now_x, layout.shift_top, layout.now_x, bottom), fill=theme["now"], width=3
    )


def runs(flags):
    """Contiguous True spans of a boolean array, as `(start, stop)` pairs."""
    if not len(flags):
        return []
    padded = np.concatenate([[False], flags, [False]])
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return list(zip(edges[::2], edges[1::2]))


def render(args) -> None:
    raw = np.load(args.data, allow_pickle=False)
    data = {key: raw[key] for key in raw.files}
    data["speakers"] = [str(s) for s in data["speakers"]]
    if args.shift_scale is not None:
        if "p_future" not in data:
            raise SystemExit("this npz predates --shift-scale; re-run demo_infer.py")
        rows = data["p_future"].astype(np.float64).copy()
        rows[:, 1] *= args.shift_scale
        data["p_shift"] = (rows[:, 1] / np.clip(rows.sum(axis=1), 1e-8, None)).astype(np.float32)
        print(f"  p(shift) re-read at shift_scale {args.shift_scale:g}")
    frames = int(scalar(data["frames"]))
    fps = float(scalar(data["fps"]))
    theme = THEMES[args.theme]
    fonts_ = fonts()
    crop = parse_crop(args.crop)

    video = Path(args.video) if args.video else Path(str(data["video"]))
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise SystemExit(f"could not open {video}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(scalar(data["start_frame"])))
    source = (
        int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    span = (1.0, 1.0) if crop is None else (crop[2] - crop[0], crop[3] - crop[1])
    layout = Layout((source[0] * span[0]) / (source[1] * span[1]))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    silent = output.with_suffix(".silent.mp4")
    encoder = subprocess.Popen(
        [
            "ffmpeg", "-y", "-v", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", f"{fps}",
            "-i", "-",
            "-an", "-c:v", "libx264", "-preset", "slow", "-crf", str(args.crf),
            "-pix_fmt", "yuv420p", str(silent),
        ],
        stdin=subprocess.PIPE,
    )

    for index in range(frames):
        ok, frame = capture.read()
        if not ok:
            frames = index
            break
        canvas = Image.new("RGB", (W, H), theme["bg"])
        draw = ImageDraw.Draw(canvas)
        draw_video(canvas, draw, frame, data, index, fonts_, theme, layout, crop)
        panel(draw, layout.tracks, theme)
        draw_shift(draw, data, index, fonts_, theme, layout, args.window_sec, fps)
        draw_tracks(draw, data, index, fonts_, theme, layout, args.window_sec, fps)
        encoder.stdin.write(canvas.tobytes())
        if index % int(fps * 5) == 0:
            print(f"    {index / fps:5.1f}s / {frames / fps:.1f}s", flush=True)
    capture.release()
    encoder.stdin.close()
    if encoder.wait() != 0:
        raise SystemExit("ffmpeg failed while encoding the frames")

    # The excerpt's own audio, muxed on after the fact: the picture is built
    # frame by frame and the sound is simply the recording, unchanged.
    start = float(scalar(data["start_frame"])) / fps
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error",
            "-i", str(silent),
            "-ss", f"{start:.3f}", "-t", f"{frames / fps:.3f}", "-i", str(video),
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", str(output),
        ],
        check=True,
    )
    silent.unlink()
    print(f"  {output}  ({frames / fps:.1f}s, {output.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="npz from tools/demo_infer.py")
    parser.add_argument("--video", default=None, help="override the recording it names")
    parser.add_argument("--output", required=True)
    parser.add_argument("--window-sec", type=float, default=5.0)
    parser.add_argument(
        "--shift-scale", type=float, default=None,
        help="re-read p(shift) at this scale from the stored rows, without "
             "running the model again",
    )
    parser.add_argument("--theme", choices=sorted(THEMES), default="light")
    parser.add_argument(
        "--crop", default=None, metavar="x0,y0,x1,y1",
        help="show only this part of the frame, in fractions - for cutting a "
             "burned-in overlay off a recording. The model's input is unaffected.",
    )
    parser.add_argument("--crf", type=int, default=18)
    render(parser.parse_args())


if __name__ == "__main__":
    main()
