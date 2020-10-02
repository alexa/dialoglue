"""Subsamples reduced test sets from csv files

Copyright PolyAI Limited.
"""

import argparse
import csv

import numpy as np
from tqdm import tqdm

# Fix the seed
np.random.seed(0)

_HEADER = ["text", "category"]


def _get_script_flags():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_file",
        help="Path to train.csv",
        required=True
    )
    parser.add_argument(
        "--n_per_class",
        help="Number of examples subsample per class",
        required=True
    )
    return parser.parse_args()


def _write_data_into_file(path, rows):
    with open(path, "w") as data_file:
        writer = csv.writer(data_file, quoting=csv.QUOTE_ALL)
        writer.writerows(rows)


def _main():
    flags = _get_script_flags()
    fname = flags.train_file
    n = int(flags.n_per_class)

    with open(fname, "r") as data_file:
        reader = list(csv.reader(data_file))
        header = reader[0]
        assert header == _HEADER
        data = np.array(reader[1:])

    labels = np.unique(data[:, 1].flatten())

    total_subsample = np.array([], dtype=int)
    for label in tqdm(labels):
        label_indicies = np.argwhere(data[:, 1] == label)
        n_this_label = label_indicies.shape[0]
        assert n_this_label >= n

        label_subsample = np.random.choice(
            label_indicies.flatten(),
            replace=False,
            size=n
        )
        total_subsample = np.hstack((total_subsample, label_subsample))

    new_dataset = np.vstack((np.array(_HEADER), data[total_subsample]))
    _write_data_into_file(
        path=fname.replace("train.csv", "train_{}.csv".format(n)),
        rows=new_dataset
    )


if __name__ == "__main__":
    _main()
