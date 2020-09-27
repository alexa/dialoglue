import json
import numpy as np
import os
import sys

# set seed
np.random.seed(0)

dataset = sys.argv[1]
if not dataset.endswith("/"):
    dataset = dataset + "/"

if "restaurant" in dataset:
    train_data = json.load(open(dataset + "train_0.json"))
    val_data = list(np.random.choice(train_data[-4000:], 1000))
    train_data = [e for e in train_data if e not in val_data]
    few_train_data = list(np.random.choice(train_data, int(len(train_data)/10)))

    json.dump(train_data, open(dataset + "train.json", "w+"))
    json.dump(val_data, open(dataset + "val.json", "w+"))

    os.system("rm {}train_*".format(dataset))
    json.dump(few_train_data, open(dataset + "train_10.json", "w+"))
elif "dstc" in dataset:
    subfolders = ["Buses_1", "Events_1", "Homes_1", "RentalCars_1"]
    train_data = []
    val_data = []
    test_data = []
    for sub in subfolders:
        sub_train_data = json.load(open(dataset + sub + "/train_0.json"))
        sub_test_data = json.load(open(dataset + sub + "/test.json"))
        sub_val_data = list(np.random.choice(sub_train_data[-len(sub_train_data)//2:], len(sub_test_data)//3))
        sub_train_data = [e for e in sub_train_data if e not in sub_val_data]

        train_data += sub_train_data
        val_data += sub_val_data
        test_data += sub_test_data

        os.system("rm -rf {}{}".format(dataset, sub))

    few_train_data = list(np.random.choice(train_data, int(len(train_data)/10)))
    json.dump(train_data, open(dataset + "train.json", "w+"))
    json.dump(few_train_data, open(dataset + "train_10.json", "w+"))
    json.dump(val_data, open(dataset + "val.json", "w+"))
    json.dump(test_data, open(dataset + "test.json", "w+"))

slots = set([slot['slot'] for row in train_data for slot in row.get('labels', [])])
vocab = ["O"] + [prefix + slot for slot in slots for prefix in ["B-", "I-"]]
json.dump(vocab, open(dataset + "vocab.txt", "w+"))
