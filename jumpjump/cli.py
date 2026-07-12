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
from .config import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_DEBUG_DIR,
    load_config,
    neural_press_model_config,
    press_model_config,
    save_config,
)
from .dependencies import set_dpi_awareness
from .neural_press_model import evaluate_press_model, train_press_model
from .training_data import import_legacy_samples, load_samples, resolve_runtime_path
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
    mode.add_argument(
        "--train-press-model",
        action="store_true",
        help="Train and validate the neural press residual model.",
    )
    mode.add_argument(
        "--evaluate-press-model",
        action="store_true",
        help="Evaluate the active neural press model without changing configuration.",
    )
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
    parser.add_argument(
        "--no-neural-press",
        action="store_true",
        help="Force legacy press prediction for this run.",
    )
    return parser.parse_args(argv)


def _neural_paths(config: dict, config_path: Path) -> tuple[Path, Path, Path]:
    settings = neural_press_model_config(config)
    return (
        resolve_runtime_path(config_path, str(settings["dataset_path"])),
        resolve_runtime_path(config_path, str(settings["model_path"])),
        resolve_runtime_path(config_path, str(settings["metadata_path"])),
    )


def run_neural_training(config: dict, config_path: Path) -> None:
    dataset_path, model_path, metadata_path = _neural_paths(config, config_path)
    imported = import_legacy_samples(dataset_path, press_model_config(config).get("samples", []))
    metadata = train_press_model(load_samples(dataset_path), config, model_path, metadata_path)
    save_config(config_path, config)
    metrics = metadata["metrics"]
    print(f"Imported legacy samples: {imported}")
    print(
        f"Training samples: {metadata['training_samples']}  "
        f"validation: {metadata['validation_samples']}  device: {metadata['device']}"
    )
    print(
        f"Coverage bins: {len(metadata['coverage_bins'])}  "
        f"covered samples: {metadata['covered_supervised_samples']}  "
        f"excluded OOD: {metadata['excluded_outside_coverage']}  "
        f"split: {metadata['split_strategy']}"
    )
    print(
        f"Validation MAE: legacy={metrics['legacy_mae_ms']:.2f}ms  "
        f"neural={metrics['model_mae_ms']:.2f}ms"
    )
    harmful = metadata["harmful_correction_metrics"]
    if harmful["rate"] is not None:
        print(
            f"Failure-direction harmful correction rate: {harmful['rate']:.1%} "
            f"({harmful['samples']} validation constraints)"
        )
    print(f"Neural model accepted and enabled: {metadata['accepted']}")


def run_neural_evaluation(config: dict, config_path: Path) -> None:
    settings = neural_press_model_config(config)
    if not bool(settings.get("enabled", False)):
        latest = settings.get("training_metrics", {})
        metrics = latest.get("metrics", {}) if isinstance(latest, dict) else {}
        detail = ""
        if metrics:
            detail = (
                f" Latest candidate: legacy MAE={metrics['legacy_mae_ms']:.2f}ms, "
                f"neural MAE={metrics['model_mae_ms']:.2f}ms."
            )
        raise JumpAutoError(
            "No accepted neural press model is active; collect more reliable covered samples "
            f"and run --train-press-model again.{detail}"
        )
    dataset_path, model_path, metadata_path = _neural_paths(config, config_path)
    metrics = evaluate_press_model(load_samples(dataset_path), model_path, metadata_path)
    print(f"Evaluation samples: {metrics['samples']}")
    print(
        f"Covered-sample MAE: legacy={metrics['legacy_mae_ms']:.2f}ms  "
        f"neural={metrics['model_mae_ms']:.2f}ms"
    )
    validation = metrics.get("validation_metrics", {})
    if validation:
        print(
            f"Held-out validation MAE: legacy={validation['legacy_mae_ms']:.2f}ms  "
            f"neural={validation['model_mae_ms']:.2f}ms"
        )
    print(f"Runtime coverage bins: {len(metrics.get('coverage_bins', []))}")
    landing = metrics["landing_comparison"]
    if landing["status"] == "insufficient_data":
        print(
            "Landing comparison: insufficient data "
            f"({landing['matched_pairs']}/{landing['minimum_pairs']} matched pairs)."
        )
    else:
        print(
            f"Landing median: legacy={landing['legacy_median_landing_error_px']:.1f}px  "
            f"neural={landing['neural_median_landing_error_px']:.1f}px  "
            f"accepted={landing['accepted']}"
        )


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
        if args.train_press_model:
            run_neural_training(config, args.config)
        elif args.evaluate_press_model:
            run_neural_evaluation(config, args.config)
        elif args.dry_run:
            run_dry_run(args, config)
        elif args.calibrate:
            run_calibration(args, config, args.config)
        elif args.single_step:
            run_single_step(args, config)
        elif args.auto:
            run_auto(args, config)
        else:
            print(
                "No mode selected. Start with --dry-run, --calibrate, --single-step, "
                "--auto, --train-press-model, or --evaluate-press-model."
            )
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
