import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run compile + unit tests quality checks")
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args()

    run([sys.executable, "-m", "compileall", "src", "scripts", "tests"])
    if not args.skip_tests and Path("tests").exists():
        run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py"])


if __name__ == "__main__":
    main()
