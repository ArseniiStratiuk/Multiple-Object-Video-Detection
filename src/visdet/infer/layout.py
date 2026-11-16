"""Split-screen layout detection: is a video segment one camera or a grid of panels?

Several test clips show two or four videos side by side. A detector run on the whole frame sees
squeezed objects, and a tracker sees several cameras at once, so the final pipeline detects and
tracks each panel on its own crop (visdet.infer.crops, visdet.infer.tracking).

1. Segments. PySceneDetect's AdaptiveDetector splits a clip at scene cuts (a layout can change
   at a cut); 12 evenly spaced frames of each segment are analysed in gray at 640 px width.
2. Seam candidates at 1/2, 1/3 and 2/3 of the width (vertical seam) and height (horizontal).
   Per frame and candidate, four features: the drop of column correlation across the candidate
   versus nearby column pairs (gaps of 1, 2 and 4 px; columns of one image are correlated, of
   two unrelated panels are not), and the log ratio of the brightness/contrast step at the
   candidate versus nearby positions. Mean, min and max over the frames: 12 features.
3. A logistic regression gives each candidate a seam probability. It is trained on synthetic
   split screens built from labelled clips (panels from distinct videos, random layouts up to
   3x3, output sizes, squeeze or crop, blur and JPEG) and checked on held-out synthetic seeds.
   An n-split along an axis needs all its n - 1 seams; 3 beats 2.

`visdet layout` trains the classifier and writes the test table (clip, first, last, rows,
cols) to tables_dir()/test_panels.csv.
"""

import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

WIDTH = 640  # analysis width (px)
TOL = 2  # a seam may sit within +-TOL px of k/n (rounding, borders)
FRACTIONS = {2: [1 / 2], 3: [1 / 3, 2 / 3]}
CANDIDATES = [(axis, n, k) for axis in ("v", "h") for n in FRACTIONS for k in range(1, n)]
GAPS = (1, 2, 4)  # column pair distances for the correlation drop
NEIGH = 40  # reference column pairs within +-NEIGH px
STRIP = 10  # strip width on each side of a candidate for the step feature
BANDS = 6  # row bands: a seam breaks every band, a natural line only some
REFS = (-30, -20, 20, 30)  # reference positions (px offsets) for the step ratio
SAMPLE = 12  # frames analysed per segment


def gray(image: np.ndarray) -> np.ndarray:
    """BGR frame -> uint8 gray at the analysis width."""
    h, w = image.shape[:2]
    g = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.resize(g, (WIDTH, round(h * WIDTH / w)), interpolation=cv2.INTER_AREA)


def _corr_drop(g: np.ndarray, pos: int, gap: int) -> float:
    """Median correlation of nearby column pairs minus the lowest correlation across pos."""
    w = g.shape[1]
    if pos - NEIGH < 0 or pos + NEIGH + gap >= w:
        return 0.0
    z = (g - g.mean(0)) / (g.std(0) + 1e-3)

    def corr(c: int) -> float:
        return float((z[:, c] * z[:, c + gap]).mean())

    across = min(corr(pos - gap // 2 + o) for o in range(-TOL, TOL + 1))
    nearby = range(pos - NEIGH, pos + NEIGH, 3)
    ref = np.median([corr(c) for c in nearby if abs(c + gap / 2 - pos) > gap + TOL])
    return ref - across


def _step(g: np.ndarray, pos: int) -> float:
    """Median over row bands of the change in strip mean and contrast across pos."""
    edges = np.linspace(0, g.shape[0], BANDS + 1).astype(int)
    left = g[:, pos - TOL - STRIP : pos - TOL]
    right = g[:, pos + 1 + TOL : pos + 1 + TOL + STRIP]
    d = []
    for a, b in zip(edges, edges[1:], strict=False):
        lo, ro = left[a:b], right[a:b]
        d.append(
            abs(lo.mean() - ro.mean()) / (0.5 * (lo.std() + ro.std()) + 2.0)
            + abs(np.log((lo.std() + 2.0) / (ro.std() + 2.0)))
        )
    return float(np.median(d))


def _step_ratio(g: np.ndarray, pos: int) -> float:
    """Log of the step at pos over the median step at nearby positions (natural edges ~0)."""
    w = g.shape[1]
    refs = [pos + o for o in REFS if STRIP + TOL <= pos + o < w - STRIP - TOL - 1]
    if not refs or pos - TOL - STRIP < 0 or pos + 1 + TOL + STRIP > w:
        return 0.0
    return float(np.log((_step(g, pos) + 0.02) / (np.median([_step(g, r) for r in refs]) + 0.02)))


def features(grays: list[np.ndarray]) -> np.ndarray:
    """(len(CANDIDATES), 12) features of one segment from its gray frames."""
    out = []
    for axis, n, k in CANDIDATES:
        per_frame = []
        for g in grays:
            g = np.asarray(g, np.float32)
            g = g if axis == "v" else g.T
            pos = round(FRACTIONS[n][k - 1] * g.shape[1]) - 1
            per_frame.append([_corr_drop(g, pos, gap) for gap in GAPS] + [_step_ratio(g, pos)])
        v = np.array(per_frame)
        out.append(np.concatenate([v.mean(0), v.min(0), v.max(0)]))
    return np.array(out)


def labels(layout: tuple[int, int]) -> np.ndarray:
    """Seam present per candidate for a (rows, cols) layout."""
    return np.array([(layout[1] if a == "v" else layout[0]) == n for a, n, _ in CANDIDATES])


def decide(prob: np.ndarray) -> tuple[int, int]:
    """(rows, cols) from per-candidate seam probabilities."""
    best = {}
    for axis in ("v", "h"):
        best[axis] = 1
        for n in (2, 3):
            if all(prob[CANDIDATES.index((axis, n, k))] > 0.5 for k in range(1, n)):
                best[axis] = n
    return best["h"], best["v"]


def segments(files: list[Path]) -> list[tuple[int, int]]:
    """[start, end) positions of the PySceneDetect scenes of one clip (files in frame order)."""
    from scenedetect import AdaptiveDetector, detect

    pattern = str(files[0].parent / f"%0{len(files[0].stem)}d{files[0].suffix}")
    scenes = detect(pattern, AdaptiveDetector())
    return [(a.get_frames(), b.get_frames()) for a, b in scenes] or [(0, len(files))]


def segment_grays(files: list[Path], start: int, end: int) -> list[np.ndarray]:
    idx = np.linspace(start, end - 1, min(SAMPLE, end - start)).round().astype(int)
    return [gray(cv2.imread(str(files[i]))) for i in idx]


# Synthetic split screens for training.
SIZES = [(1344, 756), (1920, 1080), (960, 540), (1360, 764), (680, 382), (1904, 1072)]
LAYOUTS = [(1, 1)] * 4 + [(1, 2), (2, 1), (2, 2), (2, 2), (1, 3), (3, 1), (3, 3), (2, 3)]


def compose(sources: list[np.ndarray], rows: int, cols: int, size, fit: str) -> np.ndarray:
    """Tile rows*cols frames into one (w, h) frame; fit = 'resize' (squeeze) or 'crop'."""
    w, h = size
    canvas = np.zeros((h, w, 3), np.uint8)
    xs = np.linspace(0, w, cols + 1).round().astype(int)
    ys = np.linspace(0, h, rows + 1).round().astype(int)
    for i, s in enumerate(sources):
        r, c = divmod(i, cols)
        pw, ph = xs[c + 1] - xs[c], ys[r + 1] - ys[r]
        if fit == "crop":
            sh, sw = s.shape[:2]
            scale = max(pw / sw, ph / sh)
            s = cv2.resize(s, (int(np.ceil(sw * scale)), int(np.ceil(sh * scale))))
            oy, ox = (s.shape[0] - ph) // 2, (s.shape[1] - pw) // 2
            s = s[oy : oy + ph, ox : ox + pw]
        else:
            s = cv2.resize(s, (pw, ph), interpolation=cv2.INTER_AREA)
        canvas[ys[r] : ys[r + 1], xs[c] : xs[c + 1]] = s
    return canvas


def degrade(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if rng.random() < 0.5:
        image = cv2.GaussianBlur(image, (0, 0), rng.uniform(0.3, 1.5))
    quality = int(rng.integers(30, 95))
    return cv2.imdecode(cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])[1], 1)


def synthetic(frames: pd.DataFrame, root: Path, n_clips: int, seed: int):
    """(features, true layout per frame) for each segment of n_clips synthetic clips with 1-3
    segments each; the clips are written as JPEG sequences and segmented like test clips."""
    rng = np.random.default_rng(seed)
    labelled = frames[frames["split"] != "test"]
    clips = {c: g.sort_values("frame")["file"].tolist() for c, g in labelled.groupby("clip")}
    names = list(clips)
    video = labelled.drop_duplicates("clip").set_index("clip")["video"]
    out = []
    for _ in range(n_clips):
        size, fit = SIZES[rng.integers(len(SIZES))], ["resize", "crop"][rng.integers(2)]
        n_segments = int(rng.choice([1, 2, 3], p=[0.4, 0.4, 0.2]))
        folder = Path(tempfile.mkdtemp(prefix="layout_"))
        truth, k = [], 1
        for _ in range(n_segments):
            layout = LAYOUTS[rng.integers(len(LAYOUTS))]
            length = int(rng.integers(25, 70))
            picked = []  # panels from distinct source videos
            for i in rng.permutation(len(names)):
                if video[names[i]] not in {video[p] for p in picked}:
                    picked.append(names[i])
                if len(picked) == layout[0] * layout[1]:
                    break
            src = [clips[p] for p in picked]
            starts = [int(rng.integers(0, max(1, len(s) - length))) for s in src]
            for t in range(length):
                files = [s[min(a + t, len(s) - 1)] for s, a in zip(src, starts, strict=True)]
                frame = compose([cv2.imread(str(root / f)) for f in files], *layout, size, fit)
                cv2.imwrite(str(folder / f"{k:07d}.jpg"), degrade(frame, rng))
                truth.append(layout)
                k += 1
        files = sorted(folder.glob("*.jpg"))
        for a, b in segments(files):
            out.append((features(segment_grays(files, a, b)), truth[a:b]))
        shutil.rmtree(folder)
    return out


def train(frames: pd.DataFrame, root: Path, seeds=(0, 1, 2, 3), n_clips: int = 30):
    """Logistic regression on synthetic segments; label = majority layout of the segment."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    segs = [s for seed in seeds for s in synthetic(frames, root, n_clips, seed)]
    x = np.concatenate([f for f, _ in segs])
    y = np.concatenate([labels(max(set(t), key=t.count)) for _, t in segs])
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)).fit(x, y)


def frame_errors(clf, segs) -> tuple[int, int]:
    """Wrongly labelled frames and total frames over (features, truth) segments."""
    wrong = sum(sum(t != decide(clf.predict_proba(f)[:, 1]) for t in truth) for f, truth in segs)
    return wrong, sum(len(t) for _, t in segs)


def detect_clips(clf, frames: pd.DataFrame, root: Path) -> pd.DataFrame:
    """Layout table (clip, first, last, rows, cols) for every segment of the given clips."""
    rows = []
    for clip, g in frames.groupby("clip"):
        g = g.sort_values("frame")
        files, numbers = [root / f for f in g["file"]], g["frame"].tolist()
        for a, b in segments(files):
            n_rows, n_cols = decide(clf.predict_proba(features(segment_grays(files, a, b)))[:, 1])
            rows.append((clip, numbers[a], numbers[b - 1], n_rows, n_cols))
    return pd.DataFrame(rows, columns=["clip", "first", "last", "rows", "cols"])
