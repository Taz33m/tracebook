#!/usr/bin/env python3
"""Install tracebook from the local checkout using package extras.

This helper intentionally delegates dependency selection to setup.py so it
cannot drift from the package metadata. Prefer the README commands for normal
development; this script exists for contributors who want one small installer.
"""

import argparse
import subprocess
import sys
from typing import List

EXTRAS = ("dev", "dashboard", "analysis", "all")


def build_install_target(extras: List[str]) -> str:
    """Build a pip editable target for the requested extras."""
    if not extras:
        return "."
    return f".[{','.join(extras)}]"


def main() -> int:
    """Install the package with explicitly requested extras."""
    parser = argparse.ArgumentParser(description="Install tracebook from this checkout.")
    parser.add_argument(
        "--extra",
        choices=EXTRAS,
        action="append",
        default=[],
        help="Optional extra to install. Repeat for multiple extras.",
    )
    parser.add_argument(
        "--upgrade-pip",
        action="store_true",
        help="Upgrade pip before installing.",
    )
    args = parser.parse_args()

    commands = []
    if args.upgrade_pip:
        commands.append([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    commands.append(
        [sys.executable, "-m", "pip", "install", "-e", build_install_target(args.extra)]
    )

    for command in commands:
        print("+ " + " ".join(command))
        subprocess.check_call(command)

    print("tracebook installation complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
