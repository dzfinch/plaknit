"""Console script entry point for plaknit."""

from __future__ import annotations

from typing import Optional, Sequence

from . import mosaic as mosaic_cli


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Delegate to the mosaic workflow CLI."""
    return mosaic_cli.main(argv)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
