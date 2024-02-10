import numpy as np


def kullbackLeibler_divergence(distribution_1, distribution_2):
    return np.sum(
        np.where(
            distribution_1 != 0,
            distribution_1 * np.log(distribution_1 / distribution_2),
            0,
        )
    )
