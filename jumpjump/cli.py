from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .automation import (
    print_window_list,
    run_auto,
    run_calibration,
    run_dry_run,
    run_single_step,
)
from .config import DEFAULT_CONFIG_PATH, DEFAULT_DEBUG_DIR, load_config
from .dependencies import set_dpi_awareness
from .types import DependencyError, JumpAutoError


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Windows WeChat Jump desktop automation helper."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Capture and detect only.")
    mode.add_argument("--calibrate", action="store_true", help="Record manual calibration samples.")
    mode.add_argument("--auto", action="store_true", help="Run continuous automatic jumps.")
    mode.add_argument("--single-step", action="store_true", help="Run exactly one automatic jump.")
    mode.add_argument("--list-windows", action="store_true", help="Print visible window candidates.")
    parser.add_argument("--window-title", default=None, help="Substring of the target window title.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"Config JSON path. Default: {DEFAULT_CONFIG_PATH}",
    )
    parser.add_argument(
        "--debug-dir",
        type=Path,
        default=DEFAULT_DEBUG_DIR,
        help=f"Debug image directory. Default: {DEFAULT_DEBUG_DIR}",
    )
    parser.add_argument("--interval", type=float, default=0.85, help="Seconds to wait after each jump.")
    parser.add_argument("--save-masks", action="store_true", help="Save intermediate mask images.")
    parser.add_argument(
        "--calibration-samples",
        "--samples",
        type=int,
        default=1,
        help="Number of manual successful jumps to collect in --calibrate mode.",
    )
    parser.add_argument(
        "--reset-calibration",
        action="store_true",
        help="Clear previous calibration samples before collecting new ones.",
    )
    parser.add_argument(
        "--no-auto-tune",
        action="store_true",
        help="Disable automatic model updates from successful auto-mode jumps.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    args.config = args.config.resolve()
    args.debug_dir = args.debug_dir.resolve()
    try:
        set_dpi_awareness()
        if args.list_windows:
            print_window_list()
            return 0

        config = load_config(args.config)
        if args.dry_run:
            run_dry_run(args, config)
        elif args.calibrate:
            run_calibration(args, config, args.config)
        elif args.single_step:
            run_single_step(args, config)
        elif args.auto:
            run_auto(args, config)
        else:
            print("No mode selected. Start with --dry-run, --calibrate, --single-step, or --auto.")
            print("Use --list-windows to inspect window titles.")
            return 2
        return 0
    except DependencyError as exc:
        print(f"Dependency error: {exc}", file=sys.stderr)
        return 3
    except KeyboardInterrupt:
        print("Interrupted.")
        return 130
    except JumpAutoError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1