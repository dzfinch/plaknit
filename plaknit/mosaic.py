"""Command-line interface for the Planet mosaic workflow."""

from __future__ import annotations

import argparse
from typing import Optional, Sequence

from .workflows.mosaic import MosaicJob, MosaicWorkflow, configure_logging


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for the mosaic workflow."""
    parser = argparse.ArgumentParser(
        prog="plaknit mosaic",
        description="Mask Planet strips with UDM rasters and mosaic them with OTB.",
    )
    parser.add_argument(
        "--inputs",
        "-il",
        nargs="+",
        required=True,
        help="Input strip GeoTIFFs or directories containing them.",
    )
    parser.add_argument(
        "--udms",
        "-udm",
        nargs="*",
        help="UDM GeoTIFFs (required unless --skip-masking).",
    )
    parser.add_argument(
        "--output",
        "-out",
        required=True,
        help="Destination GeoTIFF for the final mosaic.",
    )
    parser.add_argument(
        "--workdir",
        "-w",
        default="",
        help="Directory for intermediate masked strips (defaults to a temp directory).",
    )
    parser.add_argument(
        "--tmpdir",
        "-t",
        default="",
        help="Temporary directory for OTB scratch files (defaults to a temp directory).",
    )
    parser.add_argument(
        "--ram",
        "-r",
        type=int,
        default=131072,
        help="Maximum RAM available to OTB in MB (default: 131072 = 128 GB).",
    )
    parser.add_argument(
        "--jobs",
        "-j",
        type=int,
        default=4,
        help="Parallel jobs for the masking step (default: 4).",
    )
    parser.add_argument(
        "--skip-masking",
        action="store_true",
        help="Skip masking and use --inputs directly for mosaicking.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity: -v (info), -vv (debug).",
    )
    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = build_parser()
    return parser.parse_args(argv)


def _blank_to_none(value: str) -> Optional[str]:
    return value or None


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point used by both python -m plaknit.mosaic and the plaknit CLI."""
    args = parse_args(argv)
    logger = configure_logging(args.verbose)

    job = MosaicJob(
        inputs=args.inputs,
        udms=args.udms,
        output=args.output,
        workdir=_blank_to_none(args.workdir),
        tmpdir=_blank_to_none(args.tmpdir),
        ram=args.ram,
        jobs=args.jobs,
        skip_masking=args.skip_masking,
    )

    workflow = MosaicWorkflow(job, logger=logger)
    workflow.run()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
