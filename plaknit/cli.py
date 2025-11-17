"""Console script entry point for plaknit."""

from __future__ import annotations

from typing import List, Optional, Sequence

from . import mosaic as mosaic_cli
from . import planner as planner_cli


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Dispatch to the requested subcommand."""
    args = list(argv) if argv is not None else None
    if args is None:
        import sys

        args = sys.argv[1:]

    args = list(args)
    if args:
        command = args[0]
        subargv: List[str] = args[1:]
        if command == "plan":
            return planner_cli.main(subargv)
        if command == "mosaic":
            return mosaic_cli.main(subargv)

    return mosaic_cli.main(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
