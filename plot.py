import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.interpolate import interp1d
import argparse
import os
from io import StringIO
from typing import Any

TIME_INTERP_INTERVAL = 1
COLORS = {
    "SAC V2 (learned temperature)": "#1F77B4",
    "SAC V2 (fixed temperature)": "#FF7F0E",
    "SAC (fixed temperature)": "#E377C2",
    "TD3": "#2CA02C",
}
X_AXIS = {
    "time": "time",
    "env_steps": "environment steps",
    "grad_steps": "gradient steps",
}
Y_AXIS = {
    "avg_return": "average eval return",
    "log_probs_alpha": "average log probs & temperature (alpha)",
}


def parse_args() -> dict[str, Any]:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-x",
        type=str,
        dest="x",
        choices=["time", "env_steps", "grad_steps"],
        help="What performance should be plotted against.",
        required=True,
    )

    parser.add_argument(
        "-y",
        type=str,
        dest="y",
        choices=["avg_return", "log_probs_alpha"],
        help="Whether to plot the performance (return) or information about the policy entropy.",
        required=True,
    )

    parser.add_argument(
        "-e",
        "--env",
        "--environment",
        type=str,
        dest="env",
        choices=["HalfCheetah", "Hopper"],
        help="Which files should be read.",
        required=True,
    )

    parser.add_argument(
        "-b",
        "--bin-size",
        type=int,
        dest="bin_size",
        help="Size of bins for x axis.",
        required=True,
    )

    args_dict = vars(parser.parse_args())
    return args_dict


def main(x: str, y: str, env: str, bin_size: int):
    """
    Plots the results currently lying in results/plot with the following options:
        - x: time, environment steps or gradient steps
        - y: average return or log probs and alphas (temperature)
    """
    fig, ax = plt.subplots(1, 1)

    folder_name = "results/plot"
    file_names = os.listdir(folder_name)

    for file_name in file_names:
        # skip dummy files and those that are not relevant for the current environment:
        if not file_name.endswith(".csv"):
            continue
        if not env in file_name:
            continue
        with open(os.path.join(folder_name, file_name)) as f:
            file = f.read()
        hpars, results = file.split("\n\n")

        # infer algorithm name from file name and whether alpha is fixed:
        if "TD3_" in file_name:
            alg_name = "TD3"
        elif "SAC_V2_" in file_name:
            if "Fixed alpha: None" in hpars:
                alg_name = "SAC V2 (learned temperature)"
            else:
                alg_name = "SAC V2 (fixed temperature)"
        elif "SAC_" in file_name:
            alg_name = "SAC (fixed temperature)"
        else:
            raise Exception("Cannot infer algorithm name from file name and hyperparameters")

        results_df = pd.read_csv(StringIO(results))

        # infer alpha (inverse to reward scaling)
        if alg_name == "SAC (fixed temperature)":
            for hpar in hpars.split("\n"):
                if "Reward scaling" in hpar:
                    number = int(hpar.split(":")[-1])
                    alpha = 1 / number

                    results_df["alpha"] = [alpha] * results_df.shape[0]

        # if x == "time":
        #     # when plotting against time, interpolate average rewards:
        #     all_xs = np.asarray([])
        #     all_ys = np.asarray([])
        #     for seed in pd.unique(results_df["seed"]):
        #         split = results_df.loc[results_df["seed"] == seed, ["avg_reward", "time"]]
        #         time_interpolation = interp1d(x=split["time"], y=split["avg_reward"], kind="linear")
        #         xs = np.arange(0, np.max(split["time"]), TIME_INTERP_INTERVAL)
        #         ys = np.asarray([time_interpolation(x) for x in xs])
        #         all_xs = np.append(all_xs, xs)
        #         all_ys = np.append(all_ys, ys)
        #     sns.lineplot(x=all_xs, y=all_ys, ax=ax, label=alg_name, color=COLORS[alg_name])
        if x == "time":
            assert bin_size >= 60  # bin into minute bins

        else:
            if y == "avg_return":
                sns.lineplot(
                    data=results_df,
                    x=x,
                    y="avg_reward",
                    ax=ax,
                    label=alg_name,
                    color=COLORS[alg_name],
                )
            elif y == "log_probs_alpha" and alg_name != "TD3":
                sns.lineplot(
                    data=results_df,
                    x=x,
                    y="avg_log_probs",
                    ax=ax,
                    label=alg_name,
                    color=COLORS[alg_name],
                )
                ax2 = plt.twinx()
                ax.set_ylabel("Temperature")
                sns.lineplot(
                    data=results_df,
                    x=x,
                    y="alpha",
                    ax=ax2,
                    label=alg_name,
                    color=COLORS[alg_name],
                )

    plt.ylabel(X_AXIS[x])
    plt.xlabel(Y_AXIS[y])
    plt.grid()
    plt.legend(loc="upper left")
    plt.show()


if __name__ == "__main__":
    args = parse_args()
    main(**args)
