import argparse
from typing import Any
import src.hyperparameters as hyperparameters


def parse_arguments() -> dict[str, Any]:
    """
    Argument parsing for the main.py script
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-r",
        "--runs",
        type=int,
        dest="runs",
        default=1,
        help="How many runs per algorithm should be performed. Increments the seed to get different results.",
    )

    parser.add_argument(
        "--sac",
        "--sac-hpars",
        type=str,
        dest="sac_hpars",
        help="Choice of a set of hyperparameters from src/hyperparameters.py, for running SAC. String is converted to upper case.",
    )

    parser.add_argument(
        "--sac-v2",
        "--sac-v2-hpars",
        type=str,
        dest="sac_v2_hpars",
        help="Choice of a set of hyperparameters from src/hyperparameters.py, for running SAC V2. String is converted to upper case.",
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
    if args.sac_v2_hpars is not None:
        args.sac_v2_hpars = getattr(hyperparameters, args.sac_v2_hpars.upper())
    if args.td3_hpars is not None:
        args.td3_hpars = getattr(hyperparameters, args.td3_hpars.upper())

    # convert to dict
    dict_args = vars(args)

    if args.seed is not None:
        if args.sac_hpars is not None:
            dict_args["sac_hpars"].update({"seed": args.seed})
        if args.sac_v2_hpars is not None:
            dict_args["sac_v2_hpars"].update({"seed": args.seed})
        if args.td3_hpars is not None:
            dict_args["td3_hpars"].update({"seed": args.seed})

    dict_args.pop("seed")

    return dict_args
