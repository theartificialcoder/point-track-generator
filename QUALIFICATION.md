# Point-Tracker Qualification

## Current Verdict

No provider is yet qualified as simulator input. CoTracker3 is the current
conditional-tracking baseline, but point admission from unlabelled motion
support remains unqualified. The held-out evaluation half remains untouched.

## Fixed Protocol

| Item | Value |
|---|---|
| Data | reviewed day-normal frames 0-249 |
| Queries | exact native-coordinate stride-8 lattice |
| Tracker input | 512x288 RGB; output restored to 1424x802 |
| Output coordinates | native 1424x802 |
| Query birth | first labelled-object/lattice intersection; scoring only |
| Temporal state | uninterrupted across the selected clip |
| Memory control | independent query batches; no temporal reset |
| Confidence decision | 0.60 |

Annotations place the selection cohort and score later region membership. They
never correct a trajectory. This tests conditional tracking, not unlabelled
point admission, exact physical point error, or true occlusion recovery.

## CoTracker Selection Results

| Measure | Result |
|---|---:|
| Queries / labelled objects | 3,582 / 173 |
| Same-object recall / precision | 78.7% / 87.4% |
| Object-balanced recall / precision | 68.9% / 71.7% |
| Small / medium / large recall | 59.7% / 93.9% / 79.1% |
| Identity / background leakage | 2.6% / 10.0% |
| Recall at frame 60 / 249 | 88.5% / 69.9% |
| Runtime / peak GPU memory | 563.2 s / 2.36 GiB |

Query batching was checked on the 64-frame cohort. A 128-point batch changed
overall recall from 93.48% to 93.31%; it preserves temporal memory and is an
acceptable GPU-memory control. Feeding native-size frames improved small-object
recall slightly but reduced overall recall, because CoTracker still normalizes
its internal resolution.

## Rejected Configurations

| Configuration | Finding |
|---|---|
| CoTracker reset every 8 frames | Same 450 points fell from 94.7% to 84.4% recall; background leakage rose from 4.4% to 13.8%. |
| Motion-onset query admission | Boundary-biased points reached 83.7% recall and 14.6% background leakage over 64 frames. |
| TAPNext++ 512 | Only marginally exceeded CoTracker over 64 frames, but was about 103 times slower. |
| LocoTrack Base | Native 64-frame input exceeded the available 4 GiB GPU memory. |

## Required Before Held-Out Evaluation

1. Qualify native stride-8 point admission from motion support without labels.
2. Preserve uninterrupted tracker memory; do not restore rolling resets.
3. Lock the admission and provider configuration on frames 0-249.
4. Run frames 250-499 once after the configuration is fixed.
