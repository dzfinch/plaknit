"""CLI for spatial and temporal land-cover comparisons."""

from __future__ import annotations

import argparse
from typing import Dict, Optional, Sequence, Tuple

from ..processing.landcover_analysis import InputSpec, run_analysis
from ..processing.reflectance_analysis import ReflectanceInput, run_reflectance


def _key_value(value: str, option: str) -> Tuple[str, str]:
    if "=" not in value:
        raise ValueError(
            f"{option} must be LABEL=VALUE (or LABEL:BAND=VALUE)."
        )
    key, item = value.split("=", 1)
    if not key or not item:
        raise ValueError(f"{option} must have non-empty key and value.")
    return key, item


def _label_band(value: str, option: str) -> Tuple[str, str, str]:
    key, item = _key_value(value, option)
    if ":" not in key:
        raise ValueError(f"{option} must be LABEL:BAND=VALUE.")
    label, band = key.split(":", 1)
    if not label or not band:
        raise ValueError(f"{option} needs both LABEL and BAND.")
    return label, band, item


def _add_common_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        metavar="LABEL=CLASSIFICATION.tif",
        help=(
            "Classification raster and unique input label; repeat in comparison order."
        ),
    )
    parser.add_argument("--out-dir", required=True, help="Output directory.")
    parser.add_argument(
        "--prefix", default="compare", help="Output filename prefix."
    )
    parser.add_argument(
        "--width", type=float, default=100.0, help="Square width (m)."
    )
    parser.add_argument(
        "--samples", type=int, default=100, help="Squares to select."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--search-limit",
        type=int,
        default=None,
        help="Maximum candidate centres examined (default: 100 x samples).",
    )
    parser.add_argument(
        "--classes",
        type=int,
        nargs="+",
        default=None,
        help="Allowed class codes.",
    )
    parser.add_argument(
        "--no-reproject",
        action="store_true",
        help="Reject classification CRS mismatches instead of reprojecting.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Agreement workers (-1 = all CPUs).",
    )
    parser.add_argument(
        "--block-rows",
        type=int,
        default=512,
        help="Agreement rows per processing block.",
    )
    parser.add_argument(
        "--patch-min-area-m2",
        type=float,
        default=0.0,
        help="Minimum disagreement patch area to export (default: retain all).",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="plaknit compare",
        description=(
            "Compare categorical land-cover classifications spatially or temporally."
        ),
    )
    commands = parser.add_subparsers(dest="mode", required=True)
    spatial = commands.add_parser(
        "spatial",
        help="Compare classifications from different sources.",
        description=(
            "Compare classifications of the same area from different sources; "
            "dates are not required."
        ),
    )
    _add_common_options(spatial)

    temporal = commands.add_parser(
        "temporal",
        help="Compare classifications of different dates.",
        description="Compare classifications of the same area across dates.",
    )
    _add_common_options(temporal)
    temporal.add_argument(
        "--date",
        action="append",
        required=True,
        metavar="LABEL=YYYY-MM-DD",
        help="Acquisition date for an input label; repeat once for each --input.",
    )
    temporal.add_argument(
        "--group-by",
        choices=("date", "year"),
        default="date",
        help="Temporal output grouping (default: acquisition date).",
    )
    reflectance = commands.add_parser(
        "reflectance",
        help="Compare continuous rasters (bands, NDVI) by class.",
        description=(
            "Scale continuous rasters to 0-1 and compare their class-wise "
            "distributions across inputs with Mann-Whitney U tests."
        ),
    )
    reflectance.add_argument(
        "--input",
        action="append",
        required=True,
        metavar="LABEL=CLASSIFICATION.tif",
        help="Classification raster and unique input label; repeat per input.",
    )
    reflectance.add_argument(
        "--raster",
        action="append",
        required=True,
        metavar="LABEL:NAME=PATH",
        help="Continuous raster on its classification grid; repeat as needed.",
    )
    reflectance.add_argument("--out-dir", required=True, help="Output directory.")
    reflectance.add_argument(
        "--prefix", default="reflectance", help="Output filename prefix."
    )
    reflectance.add_argument("--seed", type=int, default=42, help="Random seed.")
    reflectance.add_argument(
        "--classes", type=int, nargs="+", default=None, help="Allowed class codes."
    )
    reflectance.add_argument(
        "--no-reproject",
        action="store_true",
        help="Reject classification CRS mismatches instead of reprojecting.",
    )
    return parser


def _reflectance_inputs(args) -> Sequence[ReflectanceInput]:
    paths = []
    for raw in args.input:
        label, path = _key_value(raw, "--input")
        if ":" in label:
            raise ValueError("Input labels cannot contain ':'.")
        paths.append((label, path))
    rasters: Dict[str, Dict[str, str]] = {label: {} for label, _ in paths}
    for raw in args.raster:
        label, name, path = _label_band(raw, "--raster")
        if label not in rasters:
            raise ValueError(f"--raster references unknown input label {label!r}.")
        if name in rasters[label]:
            raise ValueError(f"--raster: duplicate {label!r}/{name!r}.")
        rasters[label][name] = path
    return [ReflectanceInput(label, path, rasters[label]) for label, path in paths]


def _unique_pairs(values, option: str) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for raw in values:
        key, value = _key_value(raw, option)
        if key in result:
            raise ValueError(f"{option}: duplicate label {key!r}.")
        result[key] = value
    return result


def _input_specs(args) -> Sequence[InputSpec]:
    inputs = []
    labels = set()
    for raw in args.input:
        label, path = _key_value(raw, "--input")
        if ":" in label or "=" in label:
            raise ValueError("Input labels cannot contain ':' or '='.")
        if label in labels:
            raise ValueError(f"--input: duplicate label {label!r}.")
        labels.add(label)
        inputs.append((label, path))
    dates = (
        _unique_pairs(args.date, "--date")
        if args.mode == "temporal"
        else {}
    )
    if args.mode == "temporal" and set(dates) != labels:
        raise ValueError("--date labels must match --input labels exactly.")
    return [
        InputSpec(
            label=label,
            classification=path,
            date=dates.get(label),
        )
        for label, path in inputs
    ]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.mode == "reflectance":
        try:
            outcome = run_reflectance(
                _reflectance_inputs(args),
                args.out_dir,
                prefix=args.prefix,
                seed=args.seed,
                classes=args.classes,
                reproject_to_finest=not args.no_reproject,
            )
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
        for path in outcome.outputs:
            print(f"Wrote {path}")
        return 0
    try:
        result = run_analysis(
            mode=args.mode,
            inputs=_input_specs(args),
            out_dir=args.out_dir,
            prefix=args.prefix,
            width_m=args.width,
            n_samples=args.samples,
            seed=args.seed,
            search_limit=args.search_limit,
            reproject_to_finest=not args.no_reproject,
            classes=args.classes,
            n_jobs=args.n_jobs,
            block_rows=args.block_rows,
            group_by=getattr(args, "group_by", "date"),
            patch_min_area_m2=args.patch_min_area_m2,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(result.selection.message)
    for path in result.outputs:
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
