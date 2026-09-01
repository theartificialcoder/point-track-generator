"""Build a self-contained qualitative viewer for sparse tracker results."""

from __future__ import annotations

import argparse
from pathlib import Path

from dtf_eval.dataset import CocoTrafficArchive
from dtf_eval.sparse import SparseTracks
from dtf_eval.sparse_viewer import write_sparse_viewer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--tracks", required=True, nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--start", type=int, default=200)
    parser.add_argument("--length", type=int, default=126)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--height", type=int, default=216)
    args = parser.parse_args()

    frames = CocoTrafficArchive(args.archive).clip(args.start, args.length)
    results = {path.stem: SparseTracks.load(path) for path in args.tracks}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_sparse_viewer(args.output, frames, results, (args.width, args.height))
    print(f"Viewer: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
