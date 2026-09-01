# Point-Trajectory Generator And Evaluation

This is a standalone offline project for generating and qualifying point
trajectories. It does not import or modify the `blob-sim` runtime. Checkpoints,
videos, generated archives and reports are intentionally excluded from Git.

It qualifies correspondence trackers without changing the simulator runtime:

- **DTF-Net:** one reference frame mapped directly across the clip;
- **CoTracker3 online:** joint query-point tracking with uninterrupted temporal state;
- **MFTIQ+RAFT:** causal multi-flow tracking with learned matching quality;
- **Farneback chain:** the current two-frame control baseline accumulated through time.

The reviewed traffic annotations are used only after inference. They measure
whether tracked reference pixels remain within the same annotated object. This
is a useful traffic-scene proxy, but it is not exact per-pixel trajectory truth.
Forward/backward cycle error supplies a second annotation-free consistency test.

## Prerecorded Trajectory Input

The simulator experiment uses three deliberately separate passes:

1. `blob-sim` exports its current calibrated motion support. Point tracks,
   learned masks and body dynamics have no authority in this pass.
2. This project admits native stride-8 points inside that support and records
   point measurements in native video coordinates.
3. A fresh `blob-sim` run consumes the neutral trajectory archive. The simulator
   cannot alter the prerecorded tracker result.

Export 20 seconds of admission support:

```bash
blob-sim bench export-trajectory-support \
  --video videos/BD_2_rain.mov --config config/default.yaml \
  --duration 20 --output /tmp/bd-support.npz
```

The former rolling recorder is retained only to reproduce its failed continuity
audit. It resets CoTracker every eight frames and must not supply the simulator:

```bash
cd ../dtf-eval
PYTHONPATH=src python -m dtf_eval.cli record \
  --video ../blob-sim/videos/BD_2_rain.mov \
  --support /tmp/bd-support.npz \
  --checkpoint models/cotracker3/scaled_online.pth \
  --output /tmp/bd-trajectories.npz --device cuda
```

Its archive contains no object identity, mask correction or simulator state,
but the recorder is not qualified. The replacement must preserve temporal model
state and batch only query points for GPU memory.

## Setup

Create an independent environment and install only the required provider:

```bash
git clone --recurse-submodules <trajectory-repository> dtf-eval
cd dtf-eval
conda create -n dtf-eval python=3.11 -y
conda run -n dtf-eval pip install -e .[dev]
conda run -n dtf-eval pip install -e vendor/co-tracker
```

Place provider checkpoints under `models/`; the directory is ignored by Git.
Each report must record the provider, checkpoint, input resolution, query
spacing, temporal window, runtime and peak GPU memory.

## Qualification

The legacy command below reproduces the rejected reset-every-eight-frames audit:

```bash
dtf-eval qualify-rolling \
  --archive /data/day-normal_000000-000499_v1.zip \
  --checkpoint models/cotracker3/scaled_online.pth \
  --start 0 --length 250 --output reports/rolling-selection
```

Do not run frames 250-499 until support-only admission and the continuous
provider configuration are locked on frames 0-249.

The selection benchmark reports same-object retention, identity/background
leakage, scale, horizon, runtime and memory. Exact point error and occlusion
recovery cannot be claimed because the dataset has neither physical point
trajectories nor valid positive occlusion labels.

The current locked-selection status is recorded in
[`QUALIFICATION.md`](QUALIFICATION.md). Do not run the held-out half while the
selection configuration remains unqualified.

## Run

```bash
conda run -n dtf-eval dtf-eval run \
  --archive ../blob-sim/datasets/primary/day-normal_000000-000499_v1.zip \
  --checkpoint models/dtfnet.pt \
  --start 200 --length 12 --device cuda \
  --output reports/day-normal-200
```

Open `reports/day-normal-200/viewer.html`. The single viewer provides:

- current/warped-reference overlay and residual views;
- a stable reference-coordinate field for dense correspondence inspection;
- final-layer DTF centroid membership and assignment confidence;
- DTF/Farneback selection on the same frames;
- click-to-isolate one complete trajectory;
- optional annotation boundaries;
- region retention, identity leakage, cycle error, and runtime.

Measure component latency separately from the qualitative run:

```bash
conda run -n dtf-eval dtf-eval runtime \
  --archive ../blob-sim/datasets/primary/day-normal_000000-000499_v1.zip \
  --checkpoint models/dtfnet.pt \
  --start 200 --length 12 --device cuda \
  --full-resolution-farneback \
  --output reports/runtime.json
```

This performs warmup runs before measuring median and p95 model compute,
end-to-end tracker latency, rolling-window update rate and peak GPU memory.
Video decoding, annotations, the simulator and GUI are excluded. DTF's future
frame wait is reported separately because batch throughput is not real-time
update latency.

The default 384x216 inference size preserves the traffic video aspect ratio and
keeps the first qualification bounded. Resolution and window length must be
reported with every result because both affect accuracy and compute cost.

## Provenance

`vendor/dtf_core` is copied unmodified from the official DTF-Net repository for
reproducible inference at commit
`f73f87e2f39f74c381af750e22bfc3391c996af9`. Its original license is retained
in `vendor/LICENSE`. The official checkpoint SHA-256 is
`c04d3ce5bca9de1b9d8ede8a264841c35f8f50002b35b2d63fab525af111e0fc`.

Current local qualification findings are recorded in
`reports/qualification.md`. Generated reports remain untracked.

## Continuous Tracker Comparison

The common sparse benchmark seeds the same balanced set of points inside 46
annotated objects. Later annotations are used only for scoring: correct object,
another object, background, or tracker-declared invisible. This evaluates
conditional correspondence, not detection or exact physical point error.

Run each model in its compatible environment, then score their neutral `.npz`
outputs together:

```bash
PYTHONPATH=src python scripts/write_queries.py --archive <archive.zip> \
  --output /tmp/queries.npz

PYTHONPATH=src python scripts/run_cotracker.py --archive <archive.zip> \
  --checkpoint models/cotracker3/scaled_online.pth --queries /tmp/queries.npz \
  --output /tmp/cotracker.npz

PYTHONPATH=src:vendor/tapnet python scripts/run_bootstapir.py --archive <archive.zip> \
  --checkpoint models/causal_bootstapir_checkpoint.pt --queries /tmp/queries.npz \
  --output /tmp/bootstapir.npz

PYTHONPATH=src python scripts/run_mftiq.py --archive <archive.zip> \
  --mftiq-root vendor/MFTIQ --queries /tmp/queries.npz --output /tmp/mftiq.npz

PYTHONPATH=src python scripts/run_farneback.py --archive <archive.zip> \
  --queries /tmp/queries.npz --output /tmp/farneback.npz

PYTHONPATH=src python scripts/score_sparse.py --archive <archive.zip> \
  --tracks /tmp/farneback.npz /tmp/cotracker.npz /tmp/mftiq.npz \
  --output reports/continuous-trackers/report.json
```

On frames 200-325 at 384x216 with 575 points, CoTracker3 reduced background
leakage versus Farneback (10.7% vs 15.3%) but had essentially equal total
same-object recall (78.1% vs 78.3%). It ran at 3.45 input fps and used 2.41 GB
GPU memory; Farneback ran at 101.5 fps on CPU. MFTIQ+RAFT reached 44.9% recall,
24.6% identity leakage and 0.71 fps. These numbers qualify this traffic clip
and configuration only; MFTIQ's stronger RoMA backend was not tested.

## Online BootsTAPIR Comparison

Online BootsTAPIR and CoTracker3 were also run on identical 256x256 RGB inputs.
Their predictions were mapped back to the 384x216 annotation coordinates before
scoring. Both consumed the same 575 immutable queries from 46 objects; neither
received annotations after frame 200.

| Method | Recall | Precision | Identity leak | Background leak | FPS | Peak GPU |
|---|---:|---:|---:|---:|---:|---:|
| CoTracker3 online | 76.9% | 80.5% | 8.1% | 11.4% | 3.38 | 2.35 GiB |
| Online BootsTAPIR | 61.5% | 65.8% | 18.8% | 15.4% | 2.10 | 1.80 GiB |

At frame 325, only 116 queries remained eligible because their annotated source
objects were still present. CoTracker retained 52.6% with 7.9% identity leakage;
BootsTAPIR retained 34.5% with 40.6% identity leakage. Almost all scored source
objects were small at the reference frame, so this is specifically a crowded,
distant-traffic result rather than a general large-object comparison.

The report is `reports/continuous-trackers/bootstap-vs-cotracker-256-126.json`.
Open `reports/continuous-trackers/bootstap-vs-cotracker-viewer.html` for visual
inspection. This test measures conditional correspondence only. Continuous
point registration, stopped-object retention and true occlusion recovery remain
separate qualification tasks.

The official TAPNet source is pinned at commit
`c2cbab81cc06092b5f05bfe2da7bfec54e2079c9`. The Online BootsTAPIR checkpoint
SHA-256 is `87c1e752cf5ce56e3e2f7da460aeb4d40fc826d04ef2939bade86a5c7495377f`.
