"""Measure saved JPEG p95 and emit provisional per-worker evidence budgets."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inferops.domain.evidence_budget import evidence_budget


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jpeg-directory", type=Path, required=True)
    parser.add_argument("--cameras", type=int, required=True)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--sample-fps", type=float, default=10)
    args = parser.parse_args()
    sizes = sorted(path.stat().st_size for path in args.jpeg_directory.rglob("*.jpg"))
    if not sizes:
        parser.error("no measured JPEGs; supply representative saved overlays")
    import math
    budget = evidence_budget(cameras=args.cameras, jpeg_p95_bytes=sizes[math.ceil(len(sizes) * .95) - 1],
                             width=args.width, height=args.height, sample_fps=args.sample_fps)
    print(json.dumps({"measured_jpegs": len(sizes), **budget}, indent=2))


if __name__ == "__main__":
    main()
