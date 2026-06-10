from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_compare import main as run_compare_main


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--out-dir", default="outputs/eval_rl")
    parser.add_argument(
        "--controller",
        choices=["rl_ppo", "rl_ppo_feedforward", "rl_ppo_direct"],
        default="rl_ppo",
    )
    args = parser.parse_args()

    import sys

    sys.argv = [
        "run_compare.py",
        "--controllers",
        args.controller,
        "--model-path",
        args.model_path,
        "--out-dir",
        args.out_dir,
    ]
    run_compare_main()


if __name__ == "__main__":
    main()
