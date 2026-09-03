# Point-Tracker Qualification

## Scope

This project qualifies point correspondence only. Query discovery, motion-mask
filtering, simulation, object management, and profile learning are outside its
active scope.

The current question is narrow: given a point on an annotated object, how
accurately and consistently does each tracker follow it through the clip?

## Fixed Protocol

| Item | Requirement |
|---|---|
| Development data | reviewed day-normal frames 0-249 |
| Query coordinates | native coordinates; approximately stride-8 density with one-point minimum |
| Provider input | documented per run; predictions restored to native coordinates |
| Temporal state | uninterrupted across the selected clip |
| Provider comparison | identical immutable query file |
| Scoring | annotations used after inference only |
| Report | recall, leakage, coverage, horizon, scale, runtime, memory |

Annotations place the evaluation cohort and score later region membership.
They never correct a trajectory. Therefore this is conditional correspondence,
not exact physical point error or unlabelled object discovery.

## Current Evidence

The current development run covers frames 0-249: 13.9 seconds, 175 annotated
objects and continuous point births. CoTracker3 retained 69.4% of eligible
points on the same annotated object, with 48.5% recall for small objects and
93.7% object-frame coverage. It required 1,653.7 seconds for 8,606 queries.

This is not yet a provider-selection result. CoTracker internally resizes the
`1424x802` frames to `512x384`, the source timestamps are irregular, and some
queries lie near imperfect mask boundaries. The result qualifies the current
adapter as inadequate for small traffic detail; it does not establish
CoTracker's general accuracy. Farneback remains a diagnostic motion control and
is deliberately excluded from learned-tracker ranking.

## Rejected Provider Configurations

| Configuration | Reason |
|---|---|
| Reset CoTracker every 8 frames | destroyed temporal state and increased drift |
| LocoTrack Base at native resolution | exceeded available GPU memory |
| CoTracker3 offline on the 64-frame traffic slice | lower recall and coverage than online, with 2.4x runtime and 8.87 GiB peak allocation |
| Precomputed flow-based replenishment | tested query admission rather than tracker quality and introduced stale points |

The final item is deliberately removed from the active implementation. Point
admission and replenishment will be designed only after a tracker is qualified.

## Next Qualification

1. Run native-resolution LocoTrack on the larger GPU using the locked stride-8 cohort.
2. Repeat CoTracker with corrected constant-time, aspect-preserving preprocessing.
3. Compare retention across track age, object scale, mask interior and boundary.
4. Lock the provider on frames 0-249, then evaluate frames 250-499 once.

The primary result must remain unfiltered. Stratification explains failures; it
must not remove difficult points from the score.
