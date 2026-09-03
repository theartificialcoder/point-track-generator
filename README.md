# Offline Point-Tracker Evaluation

`dtf-eval` is an isolated offline trajectory generator and qualification
harness. It does not contain `blob-sim` runtime, simulation, mask correction,
or profile learning.

The project has four responsibilities:

1. create one immutable query set;
2. run tracker providers without modifying their predictions;
3. score correspondence quality, runtime, and memory;
4. render neutral qualitative viewers.

Videos, checkpoints, generated trajectories, and reports are excluded from Git.

## Layout

| Path | Responsibility |
|---|---|
| `src/dtf_eval/` | provider-neutral data, cohort, scoring and report code |
| `scripts/run_*.py` | isolated provider adapters |
| `tests/` | deterministic harness tests; no model downloads |
| `vendor/` | pinned CoTracker3 and LocoTrack submodules |
| `models/`, `reports/`, `data/` | ignored local inputs and generated outputs |

## What The Scores Mean

The reviewed traffic masks are not exact point-trajectory ground truth. They
support a conditional test: after a point starts inside one annotated object,
does it remain inside that same object while the object is annotated?

Report these measures together:

- **same-object recall:** fraction of eligible points retained on their source object;
- **identity leakage:** points landing on another annotated object;
- **background leakage:** points leaving all annotated objects;
- **object-frame coverage:** annotated object-frames retaining at least one point;
- **runtime and peak GPU memory:** measured for the tracker only.

Results must also be split by temporal horizon and object scale. These scores
do not prove exact pixel accuracy, object discovery, or occlusion recovery.

## Setup

```bash
git clone --recurse-submodules <repository> dtf-eval
cd dtf-eval
conda env create -f environment.yml
conda run -n dtf-trackers python -c "import torch; assert torch.cuda.is_available()"
```

Put the reviewed COCO-RLE archive under `data/` and checkpoints under `models/`.
Both directories are ignored. Download official weights:

```bash
wget -P models https://huggingface.co/datasets/hamacojr/LocoTrack-pytorch-weights/resolve/main/locotrack_base.ckpt
wget -P models https://huggingface.co/facebook/cotracker3/resolve/main/scaled_online.pth
```

## Common Sparse Benchmark

Create the query set once. Every provider must consume this same file:

```bash
conda run -n dtf-trackers env PYTHONPATH=src python scripts/write_queries.py \
  --archive data/day-normal.zip --start 0 --length 250 \
  --width 1424 --height 802 --stride 8 --output reports/queries-s8.npz
```

Continuous queries maintain approximately stride-8 coverage as an annotated
object grows. A smaller object receives one interior point immediately. This
uses annotations only to construct a controlled tracker test; points are never
corrected after birth.

LocoTrack is the primary high-resolution candidate. CoTracker3 remains a
published fixed-resolution baseline:

```bash
conda run -n dtf-trackers env \
  PYTHONPATH=src:vendor/locotrack/locotrack_pytorch \
  python scripts/run_locotrack.py \
  --archive data/day-normal.zip --checkpoint models/locotrack_base.ckpt \
  --queries reports/queries-s8.npz --start 0 --length 250 \
  --query-chunk-size 64 --output reports/locotrack-s8.npz

conda run -n dtf-trackers env PYTHONPATH=src python scripts/run_cotracker.py \
  --mode online \
  --archive data/day-normal.zip --checkpoint models/scaled_online.pth \
  --queries reports/queries-s8.npz --start 0 --length 250 \
  --width 1424 --height 802 --query-batch-size 256 \
  --output reports/cotracker-s8.npz
```

Native-resolution LocoTrack retains the full video feature tensor and requires
a high-memory GPU. `--query-chunk-size` limits query work, not that video tensor.

Large CoTracker runs are resumable. Each completed query batch is persisted in
`<output>.partial/`; rerunning the same command resumes from the next batch.

Score the outputs together:

```bash
conda run -n dtf-trackers env PYTHONPATH=src python scripts/score_sparse.py \
  --archive data/day-normal.zip --start 0 --length 250 \
  --width 1424 --height 802 \
  --tracks reports/locotrack-s8.npz reports/cotracker-s8.npz \
  --output reports/trackers/report.json
```

## Visual Inspection

For an annotation-aware comparison:

```bash
conda run -n dtf-trackers env PYTHONPATH=src python scripts/write_sparse_viewer.py \
  --archive data/day-normal.zip --start 0 --length 250 \
  --width 1424 --height 802 \
  --tracks reports/locotrack-s8.npz reports/cotracker-s8.npz \
  --output reports/trackers/viewer.html
```

Build a true constant-rate video from the archive timestamps, then write the
neutral viewer. This keeps playback speed and trajectory overlays synchronized
when the reviewed source frames are irregularly sampled:

```bash
PYTHONPATH=src python scripts/write_cfr_video.py \
  --archive <archive.zip> --start 0 --length 250 --fps 30 \
  --output reports/qualitative/background.mp4

PYTHONPATH=src python scripts/write_neutral_viewer.py \
  --video reports/qualitative/background.mp4 \
  --video-file background.mp4 --archive <archive.zip> --start 0 \
  --tracks /tmp/cotracker.npz \
  --output reports/qualitative/cotracker.html
```

The neutral viewer shows active points by default. Inactive positions and one
selected point's trail are optional. Playback stops at the recorded trajectory
horizon instead of holding the last point positions over untracked video.

`run_farneback.py` is retained only as a diagnostic motion control. Its region
membership score is not used to select the learned point tracker.

## Reproducibility

Provider source is pinned as Git submodules. Record the repository commit,
checkpoint checksum, GPU, provider canvas, query cohort and command with every
reported result.

The active protocol and current evidence are summarized in
[`QUALIFICATION.md`](QUALIFICATION.md).
