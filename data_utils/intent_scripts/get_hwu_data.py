"""Gets k-fold data (first fold) from:

https://github.com/xliuhw/NLU-Evaluation-Data

Dataset paper: https://arxiv.org/abs/1903.05566

Copyright PolyAI Limited.
"""
import argparse
import csv
import io
import json
import os

import requests
from tqdm import tqdm

_HEADER = ["text", "category"]
PATTERNS = {
    "train": "https://raw.githubusercontent.com/xliuhw/NLU-Evaluation-Data"
             "/master/CrossValidation/autoGeneFromRealAnno/autoGene_2018_03_"
             "22-13_01_25_169/CrossValidation/KFold_1/trainset/{f}",
    "test": "https://raw.githubusercontent.com/xliuhw/NLU-Evaluation-Data/"
            "master/CrossValidation/autoGeneFromRealAnno/autoGene_2018_03_"
            "22-13_01_25_169/CrossValidation/KFold_1/testset/csv/{f}"
}

LIST_OF_FILES = (
    'alarm_query.csv\nalarm_remove.csv\nalarm_set.csv\naudio_volum'
    'e_down.csv\naudio_volume_mute.csv\naudio_volume_up.csv\ncalend'
    'ar_query.csv\t\ncalendar_remove.csv\t\ncalendar_set.csv\t\ncoo'
    'king_recipe.csv\t\ndatetime_convert.csv\t\ndatetime_query.csv'
    '\t\nemail_addcontact.csv\t\nemail_query.csv\t\nemail_querycon'
    'tact.csv\t\nemail_sendemail.csv\t\ngeneral_affirm.csv\t\ngener'
    'al_commandstop.csv\t\ngeneral_confirm.csv\t\ngeneral_dontcare.'
    'csv\t\ngeneral_explain.csv\t\ngeneral_joke.csv\t\ngeneral_neg'
    'ate.csv\t\ngeneral_praise.csv\t\ngeneral_quirky.csv\t\ngenera'
    'l_repeat.csv\t\niot_cleaning.csv\t\niot_coffee.csv\t\niot_hue'
    '_lightchange.csv\t\niot_hue_lightdim.csv\t\niot_hue_lightoff.'
    'csv\t\niot_hue_lighton.csv\t\niot_hue_lightup.csv\t\niot_wemo_'
    'off.csv\t\niot_wemo_on.csv\t\nlists_createoradd.csv\t\nlists_'
    'query.csv\t\nlists_remove.csv\t\nmusic_likeness.csv\t\nmusic_q'
    'uery.csv\t\nmusic_settings.csv\t\nnews_query.csv\t\nplay_audio'
    'book.csv\t\nplay_game.csv\t\nplay_music.csv\t\nplay_podcasts.'
    'csv\t\nplay_radio.csv\t\nqa_currency.csv\t\nqa_definition.csv'
    '\t\nqa_factoid.csv\t\nqa_maths.csv\t\nqa_stock.csv\t\nrecomme'
    'ndation_events.csv\t\nrecommendation_locations.csv\t\nrecomme'
    'ndation_movies.csv\t\nsocial_post.csv\t\nsocial_query.csv\t\n'
    'takeaway_order.csv\t\ntakeaway_query.csv\t\ntransport_query.c'
    'sv\t\ntransport_taxi.csv\t\ntransport_ticket.csv\t\ntransport'
    '_traffic.csv\t\nweather_query.csv\t'.split())


def _get_script_flags():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        help="Path to dir where to save train, test, validation, categories"
             ".json",
        required=True
    )
    return parser.parse_args()


def _get_category_rows(fname: str, set_name: str):
    pattern = PATTERNS[set_name]
    url = pattern.format(f=fname)
    request = requests.get(url)

    reader = csv.reader(
        io.StringIO(request.content.decode("utf-8")), delimiter=";"
    )
    first_row = next(reader)
    scenario_i, intent_i = first_row.index("scenario"), first_row.index(
        "intent")
    answer_i = first_row.index("answer_from_anno")

    rows = []
    for row in reader:
        text = row[answer_i]
        category = f"{row[scenario_i]}_{row[intent_i]}"
        rows.append([text, category])
    return rows


def _get_final_rows(set_name: str):
    final_rows = [_HEADER]
    for f in tqdm(LIST_OF_FILES):
        final_rows += _get_category_rows(f, set_name)
    return final_rows


def _write_data_into_file(path, rows):
    with open(path, "w") as data_file:
        writer = csv.writer(data_file, quoting=csv.QUOTE_ALL)
        writer.writerows(rows)


def _main():
    flags = _get_script_flags()
    data_dir = flags.data_dir

    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    print("Getting train data")
    train_rows = _get_final_rows(set_name="train")
    _write_data_into_file(
        path=os.path.join(data_dir, "train.csv"),
        rows=train_rows
    )

    print("Getting test data")
    test_rows = _get_final_rows(set_name="test")
    _write_data_into_file(
        path=os.path.join(data_dir, "test.csv"),
        rows=test_rows
    )

    print("Creating categories.json file")
    _, train_cats = zip(*train_rows[1:])
    _, test_cats = zip(*test_rows[1:])
    categories = sorted(list(
        set(train_cats) | set(test_cats)
    ))
    with open(os.path.join(data_dir, "categories.json"), "w") as f:
        json.dump(categories, f)


if __name__ == "__main__":
    _main()
