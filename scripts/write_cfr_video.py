"""Render archive frames on a true constant-rate media timeline."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from dtf_eval.dataset import CocoTrafficArchive
from dtf_eval.media import constant_rate_source_indices


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--length", type=int, required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()

    archive = CocoTrafficArchive(args.archive)
    _, timestamps = archive.timeline(args.start, args.length)
    source_indices = constant_rate_source_indices(timestamps, args.fps)
    first = archive.frame(args.start).image
    height, width = first.shape[:2]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s:v",
        f"{width}x{height}",
        "-r",
        str(args.fps),
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(args.output),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert process.stdin is not None
    cached_index = -1
    cached_image = first
    try:
        for source_index in source_indices:
            index = int(source_index)
            if index != cached_index:
                cached_image = archive.frame(args.start + index).image
                cached_index = index
            process.stdin.write(cached_image.tobytes())
        process.stdin.close()
        return_code = process.wait()
    except BaseException:
        process.kill()
        process.wait()
        raise
    if return_code:
        raise RuntimeError(f"ffmpeg exited with status {return_code}")
    print(
        {
            "source_frames": args.length,
            "output_frames": len(source_indices),
            "fps": args.fps,
            "duration_seconds": len(source_indices) / args.fps,
            "output": str(args.output),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
