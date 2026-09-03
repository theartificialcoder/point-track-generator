"""Score common sparse-track files against the reviewed traffic annotations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dtf_eval.dataset import CocoTrafficArchive
from dtf_eval.sparse import (
    SparseTracks,
    score_object_frame_coverage,
    score_region_retention,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--tracks", required=True, nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--start", type=int, default=200)
    parser.add_argument("--length", type=int, default=126)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--height", type=int, default=216)
    parser.add_argument("--visibility-level", type=float, default=0.5)
    args = parser.parse_args()

    frames = CocoTrafficArchive(args.archive).clip(args.start, args.length)
    algorithms = {}
    for path in args.tracks:
        tracks = SparseTracks.load(path)
        algorithms[path.stem] = {
            "query_count": len(tracks.queries.points),
            "source_object_count": len(set(tracks.queries.track_ids.tolist())),
            "runtime_seconds": tracks.runtime_seconds,
            "effective_input_fps": len(frames) / tracks.runtime_seconds,
            "peak_gpu_memory_bytes": tracks.peak_gpu_memory_bytes,
            "region_retention": score_region_retention(
                tracks,
                frames,
                (args.width, args.height),
                visibility_level=args.visibility_level,
            ),
            "object_frame_coverage": score_object_frame_coverage(
                tracks,
                frames,
                (args.width, args.height),
                visibility_level=args.visibility_level,
            ),
        }
    report = {
        "methodology": {
            "task": "Track points seeded inside annotated objects without later annotation correction.",
            "ground_truth_limit": "Masks support region membership, not exact point correspondence.",
            "annotations_used_for": "Controlled query admission and scoring only.",
            "track_age": "Reported relative to each query's birth frame, not the video's absolute frame.",
            "visibility_level": args.visibility_level,
        },
        "clip": {"start": args.start, "length": args.length, "size": [args.width, args.height]},
        "algorithms": algorithms,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
