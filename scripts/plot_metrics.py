"""Generate plots from experiment CSVs.

Single-run usage:
    python scripts/plot_metrics.py --csv artifacts/logs/p2p/run.csv

Multi-run comparison:
    python scripts/plot_metrics.py \
        --compare local=artifacts/logs/local/run.csv \
                  centralized=artifacts/logs/centralized/run.csv \
                  p2p=artifacts/logs/p2p/run.csv
"""
import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from swarmfed_rl.plotting import generate_comparison_plots, generate_plots


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate experiment plots from CSV")
    parser.add_argument("--csv", type=str, default=None, help="single-run CSV path")
    parser.add_argument("--compare", nargs="+", metavar="LABEL=CSV",
                        help="multi-run comparison: label=path pairs")
    parser.add_argument("--out-dir", type=str, default="artifacts/plots", help="output directory")
    args = parser.parse_args()

    if not args.csv and not args.compare:
        parser.error("Provide --csv for single-run or --compare for multi-run comparison")

    outputs: list[str] = []

    if args.csv:
        outputs.extend(generate_plots(args.csv, args.out_dir))

    if args.compare:
        csv_paths: dict[str, str] = {}
        for item in args.compare:
            if "=" not in item:
                parser.error(f"Expected LABEL=CSV format, got: {item}")
            label, path = item.split("=", 1)
            csv_paths[label] = path
        out = args.out_dir if not args.csv else str(Path(args.out_dir) / "comparison")
        outputs.extend(generate_comparison_plots(csv_paths, out))

    for path in outputs:
        print(f"Generated: {path}")


if __name__ == "__main__":
    main()
