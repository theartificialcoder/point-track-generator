# Rolling Point-Tracker Qualification

## Current Verdict

CoTracker3 online with the current rolling admission policy is **not qualified**
as simulator input. The held-out evaluation half remains untouched.

## Selection Protocol

| Item | Value |
|---|---|
| Data | reviewed day-normal frames 0-249 |
| Admission | annotation-mask oracle; qualification only |
| Tracker input | 512x288 RGB |
| Output coordinates | native 1424x802 |
| Query spacing | 8 native pixels |
| Rolling window / advance | 16 / 8 frames |
| Active capacity | 1,024 tracks |
| Confidence decision | 0.60 |

Annotations admit test points and score region membership. They never correct a
trajectory. The benchmark cannot measure exact physical point error or true
occlusion recovery because those labels are unavailable.

## Selection Results

| Measure | Result |
|---|---:|
| Annotated objects admitted | 129 / 176 |
| Object-frame coverage | 73.9% |
| Admission delay, median / p90 | 3 / 20 frames |
| Same-object recall / precision | 64.1% / 68.1% |
| Small / medium / large recall | 53.2% / 62.3% / 76.7% |
| Identity / background leakage | 5.9% / 26.0% |
| Capacity-limited windows | 5 |
| Runtime / peak GPU memory | 99.7 s / 2.28 GiB |

Recall falls from 95.2% at frame 8 to 71.6% at frame 60 and 66.4% at frame
249. Raising the confidence decision from 0.30 to 0.90 changes precision only
from 68.0% to 68.5%; provider confidence therefore does not reliably reject
drift in this sequence.

A selection-only sampler that placed one interior point in every occupied tile
was also tested. It increased point count and compute but reduced object
admission and retention in crowded traffic, so it was removed.

## Required Before Held-Out Evaluation

1. Qualify a better continuous admission policy without annotation or object ID.
2. Compare at least one alternative tracker through the same rolling interface.
3. Lock resolution, spacing, advance, capacity and confidence on frames 0-249.
4. Run frames 250-499 once after the configuration is fixed.
