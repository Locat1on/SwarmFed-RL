import argparse

from swarmfed_rl.plotting import generate_plots


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate experiment plots from CSV")
    parser.add_argument("--csv", type=str, required=True, help="input metrics CSV")
    parser.add_argument("--out-dir", type=str, default="artifacts\\plots", help="output plot directory")
    args = parser.parse_args()
    outputs = generate_plots(args.csv, args.out_dir)
    for path in outputs:
        print(f"Generated: {path}")


if __name__ == "__main__":
    main()
