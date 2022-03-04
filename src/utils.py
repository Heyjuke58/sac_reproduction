import torch
import numpy as np
import time
import pandas as pd


def set_seeds(seed, env):
    """
    Seed all relevant seeds (torch, environment and numpy) to ensure deterministic behaviour
    """
    env.seed(seed)
    env.action_space.seed(seed)
    # set another seed for action space for deterministic sampling behaviour
    # https://github.com/openai/gym/issues/681
    env.action_space.np_random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_timestamp() -> str:
    return time.strftime("%Y_%m_%d-%H-%M-%S")


def avg_binning(df, x: str, bin_size: int):
    """
    Bins result logs (in df) to the average of the x values to make plot less noisy
    :param df: The dataframe to be manipulated
    :param x: Str of the mainpulated column
    :param bin_size: Size of the bins of the x column
    """
    df_copy = df.copy()
    grouped = df_copy.groupby(by="seed")
    cols = []
    for group in grouped.indices.values():
        mask = [index in group for index in df_copy.index]
        col = df_copy[mask][x].copy()
        stepsizes = [col[index + 1] - entry for index, entry in col[:-1].iteritems()]
        acc_step = 0.0
        last_i = 0
        for i, stepsize in enumerate(stepsizes):
            acc_step += stepsize
            if acc_step >= bin_size:
                col[last_i + 1 : i + 2] = [sum(col[last_i + 1 : i + 2]) / (i - last_i + 1)] * (
                    i - last_i + 1
                )
                acc_step = 0.0
                last_i = i + 1
            elif i + 1 == len(stepsizes):
                col[last_i + 1 : len(stepsizes) + 1] = [
                    sum(col[last_i + 1 : len(stepsizes) + 1]) / (len(stepsizes) - last_i)
                ] * (len(stepsizes) - last_i)
        cols.append(col)
    concatted_cols = pd.concat(cols, ignore_index=True)
    df_copy[x] = concatted_cols

    return df_copy
