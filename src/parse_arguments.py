import argparse
from typing import Any
import src.hyperparameters as hyperparameters


def parse_arguments() -> dict[str, Any]:
    parser = argparse.ArgumentParser()

    # parser.add_argument(
    #     "-a",
    #     "--algs",
    #     "--algorithms",
    #     nargs="+",
    #     type=str,
    #     dest="algs",
    #     default=["sac", "td3"],
    #     choices=["sac", "td3"],
    #     help="Which algorithm(s) to run.",
    # )

    parser.add_argument(
        "-r",
        "--runs",
        type=int,
        dest="runs",
        default=1,
        help="How many runs per algorithm should be performed.",
    )

    parser.add_argument(
        "--sac",
        "--sac-hpars",
        type=str,
        dest="sac_hpars",
        help="Choice of a set of hyperparameters from src/hyperparameters.py, for running SAC. String is converted to upper case.",
    )

    parser.add_argument(
        "--td3",
        "--td3-hpars",
        type=str,
        dest="td3_hpars",
        help="Choice of a set of hyperparameters from src/hyperparameters.py, for running TD3. String is converted to upper case.",
    )

    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        dest="seed",
        help="Random seed",
    )

    args = parser.parse_args()

    # lookup hyperparameters from the file, based on the passed variable name:
    if args.sac_hpars is not None:
        args.sac_hpars = getattr(hyperparameters, args.sac_hpars.upper())
    if args.td3_hpars is not None:
        args.td3_hpars = getattr(hyperparameters, args.td3_hpars.upper())

    # convert to dict
    dict_args = vars(args)

    if args.seed is not None:
        dict_args.update({"seed": args.seed})

    return dict_args
