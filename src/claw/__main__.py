"""Entry point for The Claw — python -m claw."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="claw",
        description="The Claw — Local-first voice AI agent",
    )
    parser.add_argument(
        "--mode",
        choices=["voice", "cli", "both"],
        default="both",
        help="Run mode: voice (audio + admin), cli (text + admin), both (default)",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stderr,
    )

    from claw.main import Claw

    claw = Claw()
    asyncio.run(claw.run(mode=args.mode))


if __name__ == "__main__":
    main()
