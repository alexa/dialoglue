"""Gets the data from https://github.com/clinc/oos-eval

Dataset paper: https://arxiv.org/pdf/1909.02027.pdf

Copyright PolyAI Limited.
"""
import argparse
import csv
import json
import os
import urllib.request

_DATA_URL = ("https://raw.githubusercontent.com/clinc/oos-eval/"
             "master/data/data_full.json")
_DESIRED_SETS = ["train", "test", "val"]
_HEADER = ["text", "category"]


def _get_script_flags():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        help="Path to dir where to save train, test, validation, "
             "categories.json",
        required=True
    )
    return parser.parse_args()


def _write_data_into_file(path, rows):
    with open(path, "w") as data_file:
        writer = csv.writer(data_file, quoting=csv.QUOTE_ALL)
        writer.writerows(rows)


def _main():
    flags = _get_script_flags()
    data_dir = flags.data_dir

    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    # get the in domain data
    data = urllib.request.urlopen(_DATA_URL)
    data = json.loads("".join([x.decode("utf-8") for x in data.readlines()]))
    categories = set()

    for set_name, dataset in data.items():
        if set_name not in _DESIRED_SETS:
            continue

        _, new_categories = zip(*dataset)
        categories |= set(new_categories)

        _write_data_into_file(
            path=os.path.join(data_dir, set_name + ".csv"),
            rows=[_HEADER] + dataset
        )

    with open(os.path.join(data_dir, "categories.json"), "w") as cat_file:
        json.dump(sorted(list(categories)), cat_file)


if __name__ == "__main__":
    _main()
