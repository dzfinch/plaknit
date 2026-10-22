"""Command-line interface for comparing categorical land-cover rasters."""

from __future__ import annotations

import argparse
from typing import List, Optional, Sequence

from ..processing.landcover_compare import compare_landcover


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="plaknit landcover",
        description="Compare two or more categorical land-cover rasters.",
    )
    parser.add_argument(
        "--rasters", nargs="+", required=True, help="Raster paths, in comparison order."
    )
    parser.add_argument("--out-dir", required=True, help="Output directory.")
    parser.add_argument("--prefix", default="landcover", help="Output file prefix.")
    parser.add_argument("--width", type=float, default=100.0, help="Square width (m).")
    parser.add_argument("--samples", type=int, default=100, help="Squares to select.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--search-limit",
        type=int,
        default=None,
        help="Max candidate centres to examine (default: 100 x samples).",
    )
    parser.add_argument(
        "--classes", type=int, nargs="+", default=None, help="Allowed class codes."
    )
    parser.add_argument(
        "--no-reproject",
        action="store_true",
        help="Error on CRS mismatch instead of reprojecting to the finest CRS.",
    )
    parser.add_argument(
        "--n-jobs", type=int, default=1, help="Worker threads (-1 = all CPUs)."
    )
    parser.add_argument(
        "--block-rows", type=int, default=512, help="Rows per agreement block."
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    result = compare_landcover(
        args.rasters,
        args.out_dir,
        width_m=args.width,
        n_samples=args.samples,
        seed=args.seed,
        search_limit=args.search_limit,
        reproject_to_finest=not args.no_reproject,
        classes=args.classes,
        prefix=args.prefix,
        n_jobs=args.n_jobs,
        block_rows=args.block_rows,
    )
    print(result.selection.message)
    outputs: List = [
        result.squares_gpkg,
        result.squares_csv,
        result.agreement_tif,
        result.legend_csv,
    ]
    for p in outputs:
        print(f"Wrote {p}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
