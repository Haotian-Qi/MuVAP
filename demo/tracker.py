"""Live face tracking with a persistent speaker gallery.

The corpus was tracked by `face/face_detection.py`: InsightFace `buffalo_l` for
detection and ArcFace embeddings, then Hungarian matching of detections to
tracks on cosine distance with a small spatial term, a rejection threshold so a
track never snaps onto an unrelated object, and a coast period before a track is
considered lost. Detection, embedding, crop geometry and the matching cost are
kept exactly as they are there, because the ASD module was trained on crops
produced by them.

The tracking rules around that cost are not the offline ones, because the
problem is not the offline problem. Offline the cast is fixed and known: the
video is scanned, the best frame picked, one anchor stored per speaker, and
nobody arrives or leaves. Live, people walk in and out, and a speaker who leaves
has to get their own row back when they return - so identity has to survive
absence, and a face seen at a new angle must not be mistaken for a new person.

Three things follow, each measured rather than assumed. On a three-way
conversation, ArcFace distance between two views of the *same* person reaches
0.70 at worst, while two *different* people never come closer than 0.84:

  same person, all pairs      median 0.22   p99 0.54   max 0.70
  different people, all pairs               p1  0.90   min 0.84

1. **Two thresholds, not one.** `MAX_MATCH_COST` decides whether a detection
   continues a speaker; `NEW_SPEAKER_COST` decides whether it is somebody new.
   With a single threshold at 0.55, a face at a different angle - 0.55 to 0.70
   from its own record - both failed to match and counted as a stranger, which
   is how one person arriving became two speakers. The band between the two is
   deliberately undecided: no match, and no new identity either.

2. **Several exemplars per speaker, not one anchor.** A speaker remembers up to
   `MAX_EXEMPLARS` embeddings covering different angles, and is matched on the
   closest of them. This is what makes a profile view match a frontal record,
   and it removes the offline tracker's dependence on picking one good frame: a
   poor first sighting is one exemplar among several rather than a record that
   can never be recovered from. View quality only breaks ties when the set is
   full - measured, it predicts a good exemplar far too weakly to be trusted
   with more than that.

3. **Speakers are never forgotten.** A speaker who leaves keeps their exemplars
   and is still matched against every frame, with no spatial term, since they
   have no position to be near. If every row is occupied, the longest-absent
   speaker is evicted to the archive and re-admitted on sight.

New identities also serve a short probation (`PENDING_FRAMES`) before taking a
row, so one blurred or half-occluded detection cannot claim a speaker slot.
"""

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

#: How many speaker rows exist at most. Not a property of the model, which
#: consumes the speaker axis as a set inside each frame and runs unchanged at
#: any count - measured, one forward pass costs 22.4 ms at one speaker and
#: 22.7 ms at eight, because the GPU is latency-bound on small operations rather
#: than throughput-bound on this one. It is only the size of the buffers, so it
#: is set generously and the roster grows into it as people arrive.
SPEAKER_CAPACITY = 6

#: Kept identical to the offline tracker, so crops match what ASD was trained on.
DET_SCORE_THRESHOLD = 0.62
MAX_MATCH_COST = 0.55
TRACK_MAX_MISSED = 30
SPATIAL_WEIGHT = 0.2
SPATIAL_SCALE = 200.0
MIN_AREA_FRACTION = 0.10

#: Above this, a detection is someone the gallery has never seen. Placed in the
#: measured gap between the worst same-person pair (0.70) and the closest
#: different-person pair (0.84), so neither side sits near it.
NEW_SPEAKER_COST = 0.78
#: How many views of one face to keep. Enough to span frontal and both profiles.
MAX_EXEMPLARS = 6
#: A view closer than this to one already held is the same angle, so it refreshes
#: that exemplar instead of taking a slot of its own.
EXEMPLAR_DIVERSITY = 0.25
#: Consecutive frames a stranger must be seen for before taking a speaker row.
PENDING_FRAMES = 8
#: A detection overlapping a speaker who is already there by more than this is
#: not a stranger, whatever the embedding says. A small, clipped or compressed
#: face can land 0.88 from its own record - measured, a known person looks like
#: a stranger in about 1% of views, and probation alone does not catch it when
#: the bad views run together. Nobody can walk in on top of someone who is
#: already standing there, so position settles it: the view is discarded rather
#: than being allowed to found an identity.
NEW_SPEAKER_MAX_IOU = 0.3
#: How long a speaker's last position keeps blocking new identities there. Their
#: box is cleared when they stop being tracked, but the seat they were in is not
#: free the instant they are lost - a face reappearing there within a few
#: seconds is them. Past this, the position is stale and someone else may
#: legitimately be standing in it.
POSITION_HOLD_FRAMES = 125


def l2(x, axis=-1):
    return x / np.maximum(np.linalg.norm(x, axis=axis, keepdims=True), 1e-10)


def cosine_distance(a, b):
    return 1.0 - np.dot(l2(np.asarray(a), 1), l2(np.asarray(b), 1).T)


def iou(a, b):
    """Overlap of two boxes, 0 when they do not touch."""
    if a is None or b is None:
        return 0.0
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    overlap = (x1 - x0) * (y1 - y0)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return overlap / max(area_a + area_b - overlap, 1e-6)


def view_quality(det_score, kps, bbox):
    """How much a view is worth remembering.

    Detection confidence, times how square-on the face is - the nose sitting
    between the eyes rather than off to one side - times how much of it there is
    to measure. Only ever used to choose between exemplars; the gallery does not
    depend on it being right.
    """
    width = max(bbox[2] - bbox[0], 1)
    if kps is None:
        return float(det_score) * 0.5 * min(1.0, width / 60.0)
    kps = np.asarray(kps, dtype=np.float32)
    eye_span = max(abs(kps[1, 0] - kps[0, 0]), 1e-3)
    yaw = abs(kps[2, 0] - (kps[0, 0] + kps[1, 0]) / 2) / eye_span
    return float(det_score) * max(0.0, 1.0 - 2.0 * yaw) * min(1.0, width / 60.0)


# eq=False throughout: these records hold numpy arrays, so a generated __eq__
# compares them elementwise and `x in list` raises on the ambiguous truth value.
# Identity is what the lists actually mean here anyway.
@dataclass(eq=False)
class Speaker:
    """One person, and every angle of them the session has seen."""

    speaker_id: int
    exemplars: list = field(default_factory=list)     # unit embeddings
    qualities: list = field(default_factory=list)     # view_quality of each
    bbox: list | None = None
    #: Where this speaker was last actually seen. Unlike `bbox` it is never
    #: cleared, so a speaker who has just been lost still holds their position
    #: against a new identity being founded on top of them.
    last_bbox: list | None = None
    missed: int = 0
    seen: int = 0
    present: bool = False
    absent_for: int = 0

    def occupies(self):
        """The position this speaker still claims, if it is recent enough."""
        if self.bbox is not None:
            return self.bbox
        if self.absent_for <= POSITION_HOLD_FRAMES:
            return self.last_bbox
        return None

    def distance(self, emb) -> float:
        """Distance to the closest angle of this person on record."""
        return float(cosine_distance([emb], self.exemplars).min())

    def remember(self, emb, quality):
        """Keep this view if it adds an angle, or improves one already held."""
        emb = l2(np.asarray(emb, dtype=np.float32))
        if not self.exemplars:
            self.exemplars.append(emb)
            self.qualities.append(quality)
            return
        gaps = cosine_distance([emb], self.exemplars)[0]
        nearest = int(np.argmin(gaps))
        if gaps[nearest] < EXEMPLAR_DIVERSITY:
            # An angle already on record. Keep whichever view of it is better,
            # which is how a poor first sighting gets replaced.
            if quality > self.qualities[nearest]:
                self.exemplars[nearest], self.qualities[nearest] = emb, quality
            return
        if len(self.exemplars) < MAX_EXEMPLARS:
            self.exemplars.append(emb)
            self.qualities.append(quality)
            return
        worst = int(np.argmin(self.qualities))
        if quality > self.qualities[worst]:
            self.exemplars[worst], self.qualities[worst] = emb, quality

    def observe(self, emb, bbox, quality):
        self.bbox, self.last_bbox = bbox, bbox
        self.missed, self.seen = 0, self.seen + 1
        self.present, self.absent_for = True, 0
        self.remember(emb, quality)

    def mark_missed(self):
        self.missed += 1
        self.absent_for += 1
        if self.missed > TRACK_MAX_MISSED:
            self.bbox, self.present = None, False


@dataclass(eq=False)
class Pending:
    """A stranger on probation, not yet given a row."""

    emb: np.ndarray
    bbox: list
    quality: float
    frames: int = 1
    idle: int = 0


@dataclass(eq=False)
class LiveTracker:
    """Detections in, stable speaker ids out."""

    app: object
    max_speakers: int = SPEAKER_CAPACITY
    speakers: list = field(default_factory=list)      # holding a model row
    archive: list = field(default_factory=list)       # evicted, still known
    pending: list = field(default_factory=list)
    #: Rows whose occupant changed in the last update, so the caller can drop
    #: whatever history it has buffered for them.
    reassigned: set = field(default_factory=set)

    # ------------------------------------------------------------------ detect

    def boxes_only(self, frame):
        """Boxes, keypoints and scores, without paying for identity."""
        detector = self.app.models["detection"]
        found, points = detector.detect(frame, max_num=0, metric="default")
        if found is None or len(found) == 0:
            return [], [], []
        height, width = frame.shape[:2]
        boxes, keeps, scores = [], [], []
        for row, kps in zip(found, points if points is not None else [None] * len(found)):
            if row[4] < DET_SCORE_THRESHOLD:
                continue
            x1, y1, x2, y2 = (int(v) for v in row[:4])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)
            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])
                keeps.append(kps)
                scores.append(float(row[4]))
        if not boxes:
            return [], [], []
        areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
        largest = max(areas)
        order = sorted(range(len(boxes)), key=lambda i: -areas[i])
        order = [i for i in order if areas[i] >= MIN_AREA_FRACTION * largest][: self.max_speakers]
        return [boxes[i] for i in order], [keeps[i] for i in order], [scores[i] for i in order]

    def embed(self, frame, boxes, keypoints):
        """ArcFace embeddings for boxes already found, for identity only."""
        from insightface.app.common import Face

        model = self.app.models["recognition"]
        out = []
        for box, kps in zip(boxes, keypoints):
            face = Face(bbox=np.array(box, dtype=np.float32), kps=kps, det_score=1.0)
            model.get(frame, face)
            out.append(face.normed_embedding)
        return out

    # ------------------------------------------------------------------ update

    def update(self, frame, identity=True, now=None):
        """Detect, and assign every face to a speaker id."""
        self.reassigned = set()
        if not identity and self.speakers:
            return self._spatial_update(frame)

        boxes, keypoints, scores = self.boxes_only(frame)
        embeddings = self.embed(frame, boxes, keypoints) if boxes else []
        qualities = [view_quality(s, k, b) for s, k, b in zip(scores, keypoints, boxes)]

        taken = self._match(embeddings, boxes, qualities)
        for index, speaker in enumerate(self.speakers):
            if index not in taken:
                speaker.mark_missed()

        claimed = set(taken.values())
        free = [r for r in range(len(embeddings)) if r not in claimed]
        self._admit(embeddings, boxes, qualities, free)
        return self.state()

    def _match(self, embeddings, boxes, qualities):
        """Assign detections to known speakers, present or absent."""
        if not self.speakers or not embeddings:
            return {}
        cost = np.array(
            [[speaker.distance(emb) for speaker in self.speakers] for emb in embeddings]
        )
        total = cost.copy()
        for r, box in enumerate(boxes):
            centre = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
            for c, speaker in enumerate(self.speakers):
                if speaker.bbox is None:
                    # Absent: no position to be near, so identity decides alone.
                    # The offline tracker's 1.0 placeholder would penalise
                    # exactly the case this has to get right.
                    continue
                other = np.array([
                    (speaker.bbox[0] + speaker.bbox[2]) / 2,
                    (speaker.bbox[1] + speaker.bbox[3]) / 2,
                ])
                total[r, c] += SPATIAL_WEIGHT * np.linalg.norm(centre - other) / SPATIAL_SCALE

        rows, cols = linear_sum_assignment(total)
        taken = {}
        for r, c in zip(rows, cols):
            if total[r, c] > MAX_MATCH_COST:
                continue
            self.speakers[c].observe(embeddings[r], boxes[r], qualities[r])
            taken[c] = r
        return taken

    def _admit(self, embeddings, boxes, qualities, free):
        """Decide what to do with detections nobody claimed."""
        still_pending = []
        for r in free:
            emb, box, quality = embeddings[r], boxes[r], qualities[r]

            # Someone who left and came back: match the archive on identity
            # alone and give them their row again.
            if self.archive:
                gaps = [s.distance(emb) for s in self.archive]
                best = int(np.argmin(gaps))
                if gaps[best] <= MAX_MATCH_COST:
                    speaker = self.archive.pop(best)
                    if self._seat(speaker):
                        speaker.observe(emb, box, quality)
                        continue
                    self.archive.append(speaker)

            if any(iou(box, s.occupies()) > NEW_SPEAKER_MAX_IOU for s in self.speakers):
                # Where a speaker already is. A bad view of them, not a stranger.
                continue

            known = min(
                [s.distance(emb) for s in self.speakers + self.archive], default=1.0
            )
            if known <= NEW_SPEAKER_COST:
                # Undecided: too far to continue a speaker, too close to call a
                # stranger. Claim nothing this frame.
                continue

            match = None
            for candidate in self.pending:
                if float(cosine_distance([emb], [candidate.emb])[0][0]) <= MAX_MATCH_COST:
                    match = candidate
                    break
            if match is None:
                still_pending.append(Pending(l2(np.asarray(emb, np.float32)), box, quality))
                continue
            match.frames += 1
            match.idle = 0
            match.bbox = box
            if quality > match.quality:
                match.emb, match.quality = l2(np.asarray(emb, np.float32)), quality
            if match.frames >= PENDING_FRAMES:
                speaker = Speaker(-1)
                speaker.remember(match.emb, match.quality)
                if self._seat(speaker):
                    speaker.observe(emb, box, quality)
                    continue
            still_pending.append(match)

        for candidate in self.pending:
            if candidate not in still_pending:
                candidate.idle += 1
                if candidate.idle <= 2:          # survive a dropped detection
                    still_pending.append(candidate)
        self.pending = still_pending

    def _seat(self, speaker) -> bool:
        """Give a speaker a model row, evicting the longest absent if needed."""
        used = {s.speaker_id for s in self.speakers}
        spare = [i for i in range(self.max_speakers) if i not in used]
        if spare:
            speaker.speaker_id = spare[0]
            self.speakers.append(speaker)
            self.reassigned.add(spare[0])
            return True
        absent = [s for s in self.speakers if not s.present]
        if not absent:
            return False                          # every row is a face on screen
        evicted = max(absent, key=lambda s: s.absent_for)
        self.speakers.remove(evicted)
        slot = evicted.speaker_id
        evicted.bbox, evicted.present = None, False
        self.archive.append(evicted)
        speaker.speaker_id = slot
        self.speakers.append(speaker)
        self.reassigned.add(slot)
        return True

    def _spatial_update(self, frame):
        """Keep boxes fresh between identity passes, on position alone.

        Only reached on the CPU fallback, where a full ArcFace pass is 165 ms.
        """
        boxes, _, _ = self.boxes_only(frame)
        if not boxes:
            for speaker in self.speakers:
                speaker.mark_missed()
            return self.state()
        cost = np.ones((len(boxes), len(self.speakers)))
        for r, box in enumerate(boxes):
            centre = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
            for c, speaker in enumerate(self.speakers):
                if speaker.bbox is None:
                    continue
                other = np.array([
                    (speaker.bbox[0] + speaker.bbox[2]) / 2,
                    (speaker.bbox[1] + speaker.bbox[3]) / 2,
                ])
                cost[r, c] = np.linalg.norm(centre - other) / SPATIAL_SCALE
        rows, cols = linear_sum_assignment(cost)
        matched = set()
        for r, c in zip(rows, cols):
            if cost[r, c] > 0.5:                  # moved too far to be the same face
                continue
            self.speakers[c].bbox = boxes[r]
            self.speakers[c].missed = 0
            self.speakers[c].present = True
            matched.add(c)
        for index, speaker in enumerate(self.speakers):
            if index not in matched:
                speaker.mark_missed()
        return self.state()

    @property
    def tracks(self):
        """Alias kept for callers that ask what is currently held."""
        return self.speakers

    def state(self):
        return [
            {
                "speaker": s.speaker_id,
                "bbox": s.bbox,
                "missed": s.missed,
                "seen": s.seen,
                "present": s.present,
                "angles": len(s.exemplars),
            }
            for s in sorted(self.speakers, key=lambda s: s.speaker_id)
        ]


def crop_face(frame, bbox, size=112):
    """The square grayscale crop the ASD module expects.

    Identical to `preprocess/asd/video.crop_face` - square about the box centre,
    grey where it runs off the frame - because the crop geometry is part of what
    the module was trained on.
    """
    blank = np.full((size, size), 128, dtype=np.uint8)
    if bbox is None:
        return blank
    x1, y1, x2, y2 = (int(v) for v in bbox)
    if x1 >= x2 or y1 >= y2:
        return blank
    height, width = frame.shape[:2]
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    half = max(x2 - x1, y2 - y1) // 2
    xs, ys = cx - half, cy - half
    xe, ye = xs + 2 * half, ys + 2 * half
    xc, yc = max(0, xs), max(0, ys)
    xd, yd = min(width, xe), min(height, ye)
    if xc >= xd or yc >= yd:
        return blank
    crop = cv2.cvtColor(frame[yc:yd, xc:xd], cv2.COLOR_BGR2GRAY)
    pads = (yc - ys, ye - yd, xc - xs, xe - xd)
    if any(p > 0 for p in pads):
        crop = cv2.copyMakeBorder(crop, *pads, cv2.BORDER_CONSTANT, value=128)
    return cv2.resize(crop, (size, size))
