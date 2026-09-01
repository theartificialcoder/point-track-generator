"""Create the immutable query cohort shared by every tracker environment."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from dtf_eval.dataset import CocoTrafficArchive
from dtf_eval.sparse import (
    continuous_strided_queries,
    reference_queries,
    strided_reference_queries,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--start", type=int, default=200)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--height", type=int, default=216)
    parser.add_argument("--points-per-object", type=int, default=16)
    parser.add_argument("--stride", type=int)
    parser.add_argument("--length", type=int)
    args = parser.parse_args()

    archive = CocoTrafficArchive(args.archive)
    frame = archive.frame(args.start)
    if args.stride is None:
        queries = reference_queries(
            frame,
            (args.width, args.height),
            max_points_per_object=args.points_per_object,
        )
    else:
        queries = (
            continuous_strided_queries(
                archive.clip(args.start, args.length),
                (args.width, args.height),
                stride=args.stride,
            )
            if args.length is not None
            else strided_reference_queries(
                frame,
                (args.width, args.height),
                stride=args.stride,
            )
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    queries.save(args.output)
    print(
        {
            "queries": len(queries.points),
            "objects": len(set(queries.track_ids.tolist())),
            "birth_frames": int(np.unique(queries.frame_indices).size),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
