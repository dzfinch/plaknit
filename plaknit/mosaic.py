"""Planet mosaic workflow orchestration and CLI."""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Union

try:
    import otbApplication  # type: ignore
except ModuleNotFoundError as exc:  # pragma: no cover - environment specific
    otbApplication = None  # type: ignore
    _OTB_IMPORT_ERROR = exc
else:  # pragma: no cover - import success depends on environment
    _OTB_IMPORT_ERROR = None

PathLike = Union[str, Path]
_LOGGING_CONFIGURED = False


@dataclass(frozen=True)
class MosaicJob:
    """Configuration required to run the Planet mosaic workflow."""

    inputs: Sequence[PathLike]
    output: PathLike
    udms: Optional[Sequence[PathLike]] = None
    workdir: Optional[PathLike] = None
    tmpdir: Optional[PathLike] = None
    ram: int = 131072
    jobs: int = 4
    skip_masking: bool = False


def configure_logging(verbosity: int) -> logging.Logger:
    """Configure and return a module-level logger."""
    global _LOGGING_CONFIGURED
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG

    root = logging.getLogger()
    if not _LOGGING_CONFIGURED or not root.handlers:
        logging.basicConfig(level=level, format="%(message)s")
        _LOGGING_CONFIGURED = True
    else:
        root.setLevel(level)

    logger = logging.getLogger("plaknit.mosaic")
    logger.setLevel(level)
    return logger


class MosaicWorkflow:
    """Coordinate masking and OTB mosaicking."""

    def __init__(self, job: MosaicJob, logger: Optional[logging.Logger] = None):
        self.job = job
        self.log = logger or logging.getLogger("plaknit.mosaic")
        self._tmpdir_created: Optional[Path] = None
        self._workdir_created: Optional[Path] = None

    def run(self) -> Path:
        """Execute the workflow and return the output path."""
        job = self.job
        if not job.inputs:
            raise ValueError("At least one input strip must be provided.")
        if job.ram <= 0:
            raise ValueError("RAM must be a positive integer.")

        inputs = self._expand(job.inputs, label="inputs")
        if job.skip_masking:
            masked_paths = inputs
        else:
            if not job.udms:
                raise ValueError(
                    "UDM rasters are required unless --skip-masking is provided."
                )
            udms = self._expand(job.udms, label="UDMs")
            if len(inputs) != len(udms):
                raise ValueError(
                    f"Input/UDM mismatch: expected {len(inputs)} UDMs but received {len(udms)}."
                )
            masked_paths = self._mask_inputs(inputs, udms)

        tmpdir = self._prepare_tmpdir()
        self._prepare_output_directory()
        self._configure_environment()

        try:
            self._run_mosaic(masked_paths, tmpdir)
        finally:
            self._cleanup_tmpdir()

        output_path = Path(job.output).expanduser()
        self.log.info("Mosaic complete: %s", output_path)
        return output_path

    def _expand(self, entries: Sequence[PathLike], label: str) -> List[str]:
        resolved: List[str] = []
        for entry in entries:
            path = Path(entry).expanduser()
            if path.is_dir():
                rasters = sorted(str(p) for p in path.glob("*.tif"))
                if not rasters:
                    raise ValueError(f"No .tif rasters found in directory '{path}'.")
                resolved.extend(rasters)
            elif path.exists():
                resolved.append(str(path))
            else:
                raise FileNotFoundError(f"{label} path '{entry}' does not exist.")

        if not resolved:
            raise ValueError(f"No rasters detected for {label}.")

        return resolved

    def _prepare_tmpdir(self) -> Path:
        if self.job.tmpdir:
            tmpdir = Path(self.job.tmpdir).expanduser()
            tmpdir.mkdir(parents=True, exist_ok=True)
            return tmpdir

        created = Path(tempfile.mkdtemp(prefix="otb_tmp_"))
        self._tmpdir_created = created
        return created

    def _ensure_workdir(self) -> Path:
        if self.job.workdir:
            workdir = Path(self.job.workdir).expanduser()
            workdir.mkdir(parents=True, exist_ok=True)
            return workdir

        if self._workdir_created is None:
            self._workdir_created = Path(tempfile.mkdtemp(prefix="mask_work_"))

        return self._workdir_created

    def _prepare_output_directory(self) -> None:
        output_path = Path(self.job.output).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)

    def _configure_environment(self) -> None:
        jobs = max(1, self.job.jobs)
        os.environ.setdefault("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", str(jobs))
        os.environ.setdefault("OTB_MAX_RAM_HINT", str(self.job.ram))

    def _mask_inputs(self, strips: Sequence[str], udms: Sequence[str]) -> List[str]:
        workdir = self._ensure_workdir()
        jobs = max(1, self.job.jobs)
        self.log.info("Masking %s strips using %s parallel jobs.", len(strips), jobs)

        masked_paths: List[str] = []
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            futures = []
            for strip_path, udm_path in zip(strips, udms):
                strip = Path(strip_path)
                udm = Path(udm_path)
                masked = workdir / f"{strip.stem}_masked.tif"
                futures.append(pool.submit(self._mask_single_strip, strip, udm, masked))

            for future in as_completed(futures):
                masked_paths.append(str(future.result()))

        masked_paths.sort()
        return masked_paths

    def _mask_single_strip(
        self, strip_path: Path, udm_path: Path, masked_path: Path
    ) -> Path:
        cmd = [
            "gdal_calc.py",
            "-A",
            str(strip_path),
            "-B",
            str(udm_path),
            "--allBands",
            "A",
            "--calc",
            "A*(B==1)",
            "--NoDataValue",
            "0",
            "--overwrite",
            "--creation-option",
            "TILED=YES",
            "--creation-option",
            "BLOCKXSIZE=512",
            "--creation-option",
            "BLOCKYSIZE=512",
            "--creation-option",
            "BIGTIFF=YES",
            "--creation-option",
            "COMPRESS=NONE",
            "--outfile",
            str(masked_path),
        ]
        self.log.debug("Masking strip with command: %s", " ".join(cmd))
        subprocess.run(cmd, check=True)
        return masked_path

    def _run_mosaic(self, rasters: Sequence[str], tmpdir: Path) -> None:
        if otbApplication is None:  # pragma: no cover - environment specific
            raise RuntimeError(
                "otbApplication bindings are not available. Install the OTB Python "
                "packages before running the mosaic workflow."
            ) from _OTB_IMPORT_ERROR

        self.log.info(
            "Running OTB Mosaic on %s strips with RAM=%s MB.",
            len(rasters),
            self.job.ram,
        )

        app = otbApplication.Registry.CreateApplication("Mosaic")
        params = {
            "comp.feather": "slim",
            "comp.feather.slim.exponent": 1,
            "comp.feather.slim.length": 0,
            "distancemap.sr": 10,
            "harmo.method": "none",
            "harmo.cost": "rmse",
            "il": list(rasters),
            "interpolator": "bco",
            "interpolator.bco.radius": 2,
            "nodata": 0,
            "output.spacingx": 3,
            "output.spacingy": 3,
            "tmpdir": str(tmpdir),
            "ram": self.job.ram,
            "out": str(Path(self.job.output).expanduser()),
        }

        for key, value in params.items():
            if value is None:
                continue
            if key == "il":
                app.SetParameterStringList(key, [str(v) for v in value])
            elif isinstance(value, list):
                app.SetParameterStringList(key, [str(v) for v in value])
            elif isinstance(value, str):
                app.SetParameterString(key, value)
            elif isinstance(value, int):
                app.SetParameterInt(key, value)
            elif isinstance(value, float):
                app.SetParameterFloat(key, value)

        app.ExecuteAndWriteOutput()

    def _cleanup_tmpdir(self) -> None:
        if self._tmpdir_created and self._tmpdir_created.exists():
            try:
                shutil.rmtree(self._tmpdir_created)
            except OSError:
                pass


def run_mosaic(job: MosaicJob, logger: Optional[logging.Logger] = None) -> Path:
    """Convenience helper for running the workflow in a single call."""
    workflow = MosaicWorkflow(job, logger=logger)
    return workflow.run()


# ------------------------------- CLI Helpers ------------------------------- #


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
