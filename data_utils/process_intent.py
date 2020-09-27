import csv
import numpy as np
import os
import sys

from collections import Counter, defaultdict

# set seed
np.random.seed(0)

dataset = sys.argv[1]
if not dataset.endswith("/"):
    dataset = dataset + "/"

print("Processing", dataset)

header = [row for row in csv.reader(open(dataset + "train.csv"))][:1]

train_rows = [row for row in csv.reader(open(dataset + "train.csv"))][1:]
test_rows = [row for row in csv.reader(open(dataset + "test.csv"))][1:]

train_10 = [row for row in csv.reader(open(dataset + "train_10.csv"))][1:]
train_30 = [row for row in csv.reader(open(dataset + "train_30.csv"))][1:]

if "clinc" not in dataset:
    train_by_intent = defaultdict(list)
    for i,(k,v) in enumerate(train_rows):
        if [k,v] in train_10 or [k,v] in train_30:
            continue

        train_by_intent[v].append(i)

    val_rows = []
    counts = Counter([e[1] for e in test_rows])
    val_test_ratio = 1 if "hwu" in dataset else 2
    for k,c in counts.items():
        val_rows += list(np.random.choice(train_by_intent[k], c//val_test_ratio))

    new_train_rows = header + [e for i,e in enumerate(train_rows) if i not in val_rows]
    val_rows = header + [train_rows[i] for i in val_rows]
    csv.writer(open(dataset + "train.csv", "w")).writerows(new_train_rows)
    csv.writer(open(dataset + "val.csv", "w")).writerows(val_rows)

train_5 = header + train_10[::2]
csv.writer(open(dataset + "train_5.csv", "w")).writerows(train_5)

# Delete unnecessary 30 file
os.system("rm {}train_30.csv".format(dataset))
