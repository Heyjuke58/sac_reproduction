import argparse
from typing import Any, Dict, List, Tuple


def parse_arguments() -> Tuple[Any, ...]:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e",
        "--examples",
        type=str,
        dest="examples",
        default="0",
        help="",
    )
    parser.add_argument(
        "--seed",
        type=int,
        dest="seed",
        default=33,  # super max
        help="Seed",
    )
    parser.add_argument(
        "--no-viz-attr",
        dest="viz_attr",
        action="store_false",
        help="Whether to visualize the summed attributions for every example.",
    )
    parser.set_defaults(viz_attr=True)

    args = parser.parse_args()

    return args
