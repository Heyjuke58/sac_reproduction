import argparse
import os
from io import StringIO
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.collections import PolyCollection

from src.utils import avg_binning

plt.rc("axes", titlesize=28)  # fontsize of the axes title
plt.rc("axes", labelsize=24)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=20)  # fontsize of the tick labels
plt.rc("ytick", labelsize=20)  # fontsize of the tick labels
plt.rc("legend", fontsize=20)  # legend fontsize

TIME_INTERP_INTERVAL = 1
COLORS = {
    "SAC V2 (learned temperature)": "#1F77B4",
    "SAC V2 (fixed temperature)": "#E377C2",
    "SAC (fixed temperature)": "#FF7F0E",
    "TD3": "#2CA02C",
}
HATCHES = {
    "SAC V2 (learned temperature)": "///",
    "SAC V2 (fixed temperature)": "\\\\\\",
    "SAC (fixed temperature)": "||",
    "TD3": "--",
}
# rcParams["hatch.linewidth"] = 2
X_AXIS = {
    "time": "Time",
    "env_steps": "Environment Steps",
    "grad_steps": "Gradient Steps",
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
        - e / --env: Environment to plot (only important for filtering of files)
        - b / --bin-size: Bin size (to make plot less noisy)
    """
    if y == "log_probs_alpha":
        fig, axs = plt.subplots(
            2,
            1,
            figsize=(19.20, 10.80),
            squeeze=False,
            sharex=True,
            # gridspec_kw={"hspace": 0},
        )
    else:
        fig, axs = plt.subplots(
            1,
            1,
            figsize=(9.60, 10.80),
            squeeze=False,
        )

    folder_name = "results/plot"
    file_names = os.listdir(folder_name)
    max_x = None
    n_initial_exploration_steps = None

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

        # read number of initial exploration steps from file:
        if alg_name != "TD3":
            for hpar in hpars.split("\n"):
                if "Number of initial exploration steps" in hpar:
                    n_initial_exploration_steps = int(float(hpar.split(":")[-1]))

        # bin x values:
        if x == "time":
            assert bin_size >= 60  # bin into minute bins
        else:
            assert bin_size % 5000 == 0 and bin_size >= 5000  # frequency of evaluations

        max_x = results_df[x].max()
        num_seeds = len(results_df["seed"].unique())

        def ceil_binning(col):
            # round up, but not higher than biggest original (x) value
            return min(max_x, bin_size * (np.ceil(col / bin_size)))

        if x == "time":
            results_df[x] = results_df[x].apply(ceil_binning)
        else:
            results_df[x] = results_df[x].apply(ceil_binning)
            # results_df = avg_binning(results_df, x, bin_size)

        lineplot_kwargs = {
            "data": results_df,
            "x": x,
            "label": f"{alg_name} n={num_seeds}",
            "color": COLORS[alg_name],
            "err_kws": {
                "hatch": HATCHES[alg_name],
            },
            "linewidth": 2.5,
            # "path_effects": [pe.Stroke(linewidth=2.25, foreground="black"), pe.Normal()],
        }

        if y == "avg_return":
            sns.lineplot(y="avg_reward", ax=axs[0, 0], **lineplot_kwargs)
            axs[0, 0].set_ylabel("Average Return")
            axs[0, 0].grid(visible=True)
        elif y == "log_probs_alpha" and alg_name != "TD3":
            sns.lineplot(y="avg_log_probs", ax=axs[0, 0], **lineplot_kwargs)
            axs[0, 0].set_ylabel("Average Log Probs")
            axs[0, 0].grid(visible=True)
            sns.lineplot(y="alpha", ax=axs[1, 0], **lineplot_kwargs)
            axs[1, 0].set_ylabel("Temperature")
            axs[1, 0].grid(visible=True)

    # initial exploration gray fill
    if x == "env_steps" and y == "log_probs_alpha":
        for ax in axs.reshape(-1):
            ax.axvspan(
                0,
                n_initial_exploration_steps,
                alpha=0.5,
                color="gray",
                label="Initial Exploration",
            )

    # set alpha for error fills:
    for ax in axs.reshape(-1):
        for child in ax.findobj(PolyCollection):
            child.set_alpha(0.35)

    # layout stuff
    fig.tight_layout(rect=[0.01, 0, 0.99, 0.97])
    plt.xlabel(X_AXIS[x])
    plt.xlim(0, max_x)
    if y == "avg_return":
        axs[0, 0].legend(loc="lower right")
        plt.title(
            f"{env}-v3 {'(' + f'{bin_size=}' + ')' if bin_size != 5000 else ''}"
        )  # 5000 was eval frequency
    elif y == "log_probs_alpha" and alg_name != "TD3":
        axs[1, 0].legend(loc="upper right")
        axs[0, 0].legend([], [], frameon=False)
        # axs[0, 0].set_xticks([])
        axs[0, 0].set_title(f"{env}-v3 {'(' + f'{bin_size=}' + ')' if bin_size != 5000 else ''}")
    plt.savefig(f"results/plot/{env}_{x}_{y}")


if __name__ == "__main__":
    args = parse_args()
    main(**args)
