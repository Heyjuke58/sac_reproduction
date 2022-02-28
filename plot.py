import matplotlib.pyplot as plt
from numpy import ndarray
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.interpolate import interp1d
import argparse
import os
from io import StringIO
from typing import Any

TIME_INTERP_INTERVAL = 1


def parse_args() -> dict[str, Any]:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-x",
        type=str,
        dest="x",
        choices=["time", "env_steps", "grad_steps"],
        help="What performance should be plotted against.",
    )
    return vars(parser.parse_args())


def main(x: str):
    fig, ax = plt.subplots(1, 1)

    folder_name = "results/plot"
    file_names = os.listdir(folder_name)

    for file_name in file_names:
        with open(os.path.join(folder_name, file_name)) as f:
            file = f.read()
        hpars, results = file.split("\n\n")
        results_df = pd.read_csv(StringIO(results))

        if x == "time":
            # when plotting against time, interpolate average rewards:
            all_xs = np.asarray([])
            all_ys = np.asarray([])
            for seed in pd.unique(results_df["seed"]):
                split = results_df.loc[results_df["seed"] == seed, ["avg_reward", "time"]]
                time_interpolation = interp1d(x=split["time"], y=split["avg_reward"], kind="linear")
                xs = np.arange(0, np.max(split["time"]), TIME_INTERP_INTERVAL)
                ys = np.asarray([time_interpolation(x) for x in xs])
                all_xs = np.append(all_xs, xs)
                all_ys = np.append(all_ys, ys)

            sns.lineplot(x=all_xs, y=all_ys, ax=ax)
        else:
            sns.lineplot(data=results_df, x=x, y="avg_reward", ax=ax)
    plt.grid()
    plt.show()


if __name__ == "__main__":
    args = parse_args()
    main(**args)
