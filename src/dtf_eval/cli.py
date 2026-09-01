"""Command line entry point for the isolated tracker qualification."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .dataset import CocoTrafficArchive
from .metrics import cycle_consistency, region_consistency
from .qualification import qualify_rolling_cotracker
from .recording import record_cotracker
from .runtime import benchmark_tracker
from .trackers import DtfNetTracker, FarnebackChainTracker
from .viewer import write_viewer


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="compare DTF-Net with chained Farneback")
    run.add_argument("--archive", required=True, type=Path)
    run.add_argument("--checkpoint", required=True, type=Path)
    run.add_argument("--start", type=int, default=200)
    run.add_argument("--length", type=int, default=12)
    run.add_argument("--reference", type=int, default=0)
    run.add_argument("--width", type=int, default=384)
    run.add_argument("--height", type=int, default=216)
    run.add_argument("--device", default="cuda")
    run.add_argument("--output", type=Path, default=Path("reports/day-normal"))
    run.add_argument("--save-fields", action="store_true")
    runtime = subparsers.add_parser("runtime", help="measure tracker latency and memory")
    runtime.add_argument("--archive", required=True, type=Path)
    runtime.add_argument("--checkpoint", required=True, type=Path)
    runtime.add_argument("--start", type=int, default=200)
    runtime.add_argument("--length", type=int, default=12)
    runtime.add_argument("--reference", type=int, default=0)
    runtime.add_argument("--width", type=int, default=384)
    runtime.add_argument("--height", type=int, default=216)
    runtime.add_argument("--device", default="cuda")
    runtime.add_argument("--warmup", type=int, default=2)
    runtime.add_argument("--repeats", type=int, default=8)
    runtime.add_argument("--full-resolution-farneback", action="store_true")
    runtime.add_argument("--output", type=Path, default=Path("reports/runtime.json"))
    record = subparsers.add_parser(
        "record",
        help="audit the rejected rolling-reset recorder on blob-sim support",
    )
    record.add_argument("--video", required=True, type=Path)
    record.add_argument("--support", required=True, type=Path)
    record.add_argument("--checkpoint", required=True, type=Path)
    record.add_argument("--output", required=True, type=Path)
    record.add_argument(
        "--tracker-root",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "vendor" / "co-tracker",
    )
    record.add_argument("--device", default="cuda")
    record.add_argument("--tracker-width", type=int, default=512)
    record.add_argument("--tracker-height", type=int, default=288)
    record.add_argument("--query-stride", type=int, default=8)
    record.add_argument("--coverage-radius", type=float, default=8.0)
    record.add_argument("--advance-frames", type=int, default=8)
    record.add_argument("--max-active-tracks", type=int, default=1024)
    record.add_argument("--visibility-threshold", type=float, default=0.6)
    qualify = subparsers.add_parser(
        "qualify-rolling",
        help="reproduce the rejected rolling-reset qualification",
    )
    qualify.add_argument("--archive", required=True, type=Path)
    qualify.add_argument("--checkpoint", required=True, type=Path)
    qualify.add_argument("--output", required=True, type=Path)
    qualify.add_argument(
        "--tracker-root",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "vendor" / "co-tracker",
    )
    qualify.add_argument("--start", type=int, default=0)
    qualify.add_argument("--length", type=int, default=150)
    qualify.add_argument("--device", default="cuda")
    qualify.add_argument("--tracker-width", type=int, default=512)
    qualify.add_argument("--tracker-height", type=int, default=288)
    qualify.add_argument("--query-stride", type=int, default=8)
    qualify.add_argument("--coverage-radius", type=float, default=8.0)
    qualify.add_argument("--advance-frames", type=int, default=8)
    qualify.add_argument("--max-active-tracks", type=int, default=1024)
    qualify.add_argument("--visibility-threshold", type=float, default=0.6)
    return parser


def _run(args: argparse.Namespace) -> int:
    if args.length < 2:
        raise ValueError("length must be at least two frames")
    if not 0 <= args.reference < args.length:
        raise ValueError("reference lies outside the clip")
    archive = CocoTrafficArchive(args.archive)
    frames = archive.clip(args.start, args.length)
    images = tuple(frame.image for frame in frames)
    size = (args.width, args.height)
    project_root = Path(__file__).resolve().parents[2]
    vendor_root = project_root / "vendor"

    trackers = (
        DtfNetTracker(args.checkpoint, vendor_root, args.device, capture_groups=True),
        FarnebackChainTracker(),
    )
    fields = {}
    group_fields = {}
    report: dict[str, object] = {
        "methodology": {
            "exact_ground_truth": False,
            "region_consistency": "Predicted reference pixels should remain inside the same annotated track.",
            "cycle_consistency": "Forward then backward tracking should return to the reference pixel.",
            "caution": "Traffic masks contain imperfect distant boundaries and IDs; region scores are proxies, not exact trajectory accuracy.",
        },
        "clip": {
            "start": frames[0].frame_number,
            "end": frames[-1].frame_number,
            "length": len(frames),
            "inference_size": [args.width, args.height],
            "reference": args.reference,
        },
        "algorithms": {},
    }
    for tracker in trackers:
        forward = tracker.track(images, args.reference, size)
        if isinstance(tracker, DtfNetTracker) and tracker.last_groups is not None:
            group_fields[tracker.name] = tracker.last_groups
        reverse_index = args.length - 1 if args.reference == 0 else 0
        reverse = tracker.track(images, reverse_index, size)
        fields[tracker.name] = forward
        report["algorithms"][tracker.name] = {
            "runtime_seconds": forward.runtime_seconds,
            "region_consistency": region_consistency(forward, frames),
            "cycle": cycle_consistency(forward, reverse),
        }
        if args.save_fields:
            args.output.mkdir(parents=True, exist_ok=True)
            safe_name = tracker.name.lower().replace(" ", "-")
            np.savez_compressed(
                args.output / f"{safe_name}.npz",
                coordinates=forward.coordinates,
                visibility=forward.visibility,
                reference_grid=forward.reference_grid,
                reference_index=forward.reference_index,
            )

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    write_viewer(args.output / "viewer.html", frames, fields, report, group_fields=group_fields)
    print(json.dumps(report, indent=2))
    print(f"Viewer: {(args.output / 'viewer.html').resolve()}")
    return 0


def _runtime(args: argparse.Namespace) -> int:
    if args.length < 2:
        raise ValueError("length must be at least two frames")
    if not 0 <= args.reference < args.length:
        raise ValueError("reference lies outside the clip")
    archive = CocoTrafficArchive(args.archive)
    frames = archive.clip(args.start, args.length)
    images = tuple(frame.image for frame in frames)
    size = (args.width, args.height)
    project_root = Path(__file__).resolve().parents[2]

    load_started = time.perf_counter()
    dtf = DtfNetTracker(args.checkpoint, project_root / "vendor", args.device)
    model_load_seconds = time.perf_counter() - load_started
    dtf_stats = benchmark_tracker(
        dtf,
        images,
        args.reference,
        size,
        warmup=args.warmup,
        repeats=args.repeats,
        measure_cuda_memory=args.device.startswith("cuda"),
    )
    farneback = FarnebackChainTracker()
    farneback_stats = benchmark_tracker(
        farneback,
        images,
        args.reference,
        size,
        warmup=args.warmup,
        repeats=args.repeats,
    )

    reference_time = frames[args.reference].timestamp_seconds
    final_time = frames[-1].timestamp_seconds
    lookahead_seconds = None
    if reference_time is not None and final_time is not None:
        lookahead_seconds = max(0.0, final_time - reference_time)
    dtf_latency = None
    if lookahead_seconds is not None:
        dtf_latency = lookahead_seconds + dtf_stats["end_to_end"]["median_seconds"]
    transitions = len(frames) - 1
    farneback_step = farneback_stats["end_to_end"]["median_seconds"] / transitions

    algorithms: dict[str, object] = {
        "DTF-Net": {
            **dtf_stats,
            "model_load_seconds": model_load_seconds,
            "future_frame_wait_seconds": lookahead_seconds,
            "reference_result_latency_seconds": dtf_latency,
            "execution": "non-causal complete-window inference",
        },
        "Farneback chain": {
            **farneback_stats,
            "causal_step_median_seconds": farneback_step,
            "causal_steps_per_second": 1.0 / farneback_step,
            "execution": "causal adjacent-frame updates",
        },
    }
    if args.full_resolution_farneback:
        original_height, original_width = images[0].shape[:2]
        full_stats = benchmark_tracker(
            farneback,
            images,
            args.reference,
            (original_width, original_height),
            warmup=args.warmup,
            repeats=args.repeats,
        )
        full_step = full_stats["end_to_end"]["median_seconds"] / transitions
        algorithms["Farneback chain (full resolution)"] = {
            **full_stats,
            "causal_step_median_seconds": full_step,
            "causal_steps_per_second": 1.0 / full_step,
            "execution": "causal adjacent-frame updates",
        }

    report = {
        "scope": "tracker component only; excludes video decode, annotations, simulator and GUI",
        "clip": {
            "start": frames[0].frame_number,
            "end": frames[-1].frame_number,
            "length": len(frames),
            "inference_size": [args.width, args.height],
            "reference": args.reference,
            "captured_span_seconds": lookahead_seconds,
        },
        "algorithms": algorithms,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Runtime report: {args.output.resolve()}")
    return 0


def main() -> int:
    args = _parser().parse_args()
    if args.command == "run":
        return _run(args)
    if args.command == "runtime":
        return _runtime(args)
    if args.command == "record":
        record_cotracker(
            args.video,
            args.support,
            args.output,
            checkpoint=args.checkpoint,
            tracker_root=args.tracker_root,
            device=args.device,
            tracker_size=(args.tracker_width, args.tracker_height),
            stride=args.query_stride,
            coverage_radius=args.coverage_radius,
            advance_frames=args.advance_frames,
            max_active_tracks=args.max_active_tracks,
            visibility_threshold=args.visibility_threshold,
        )
        return 0
    if args.command == "qualify-rolling":
        qualify_rolling_cotracker(
            args.archive,
            args.output,
            checkpoint=args.checkpoint,
            tracker_root=args.tracker_root,
            start=args.start,
            length=args.length,
            device=args.device,
            tracker_size=(args.tracker_width, args.tracker_height),
            stride=args.query_stride,
            coverage_radius=args.coverage_radius,
            advance_frames=args.advance_frames,
            max_active_tracks=args.max_active_tracks,
            visibility_threshold=args.visibility_threshold,
        )
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
