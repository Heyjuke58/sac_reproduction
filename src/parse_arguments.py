import argparse
from typing import Any


def parse_arguments() -> dict[str, Any]:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-a",
        "--algs",
        "--algorithms",
        nargs="+",
        type=str,
        dest="algs",
        default=["sac", "td3"],
        choices=["sac", "td3"],
        help="Which algorithm(s) to run.",
    )

    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        dest="seed",
        default=33,  # super max
        help="Seed",
    )

    parser.add_argument(
        "-e",
        "--env",
        "--environment",
        dest="env",
        choices=["hopper-v2", "cheetah-v3"],
        default="Hopper-v2",
        help="Which environment to let the algorithms loose on.",
    )

    args = parser.parse_args()
    return vars(args)
