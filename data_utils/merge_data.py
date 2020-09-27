import csv
import json

def load_intent(fn):
    reader = csv.reader(open(fn+"train.csv"))
    next(reader)
    return [row[0] for row in reader]

def load_slot(fn):
    data = json.load(open(fn+"train.json"))
    text_data = []
    for example in data:
        text = example['userInput']['text']
        if 'context' in example:
            for req in example['context'].get('requestedSlots', []):
                text = req + " " + text
        text_data.append(text)
    return text_data

def load_top(fn):
    data = open(fn+"train.txt").readlines()
    text_data = []
    for row in data:
        sentence = row.split(" <=>")[0]
        words = [w.split(":")[0] for w in sentence.strip().split()]
        text_data.append(" ".join(words))

    return text_data

def load_multiwoz(fn):
    data = json.load(open(fn+"train_dials.json"))
    text_data = []
    for example in data.values():
        history = []
        for i,utt in enumerate(example['log']):
            history.append(utt['text'])

            # Continue if assistant speaking
            if i % 2 == 1:
                continue

            text = " [SEP] ".join(history)
            text = " ".join(text.split())
            text_data.append(text)

    return text_data

# All the MLM sentences throughout the dataset
train = []

# Load the intent dataset
train += load_intent("dialoglue/banking/")
train += load_intent("dialoglue/hwu/")
train += load_intent("dialoglue/clinc/")

# Load the slot datasets
train += load_slot("dialoglue/restaurant8k/")
train += load_slot("dialoglue/dstc8_sgd/")

# Load TOP
train += load_top("dialoglue/top/")

# Load MultiWOZ
train += load_multiwoz("dialoglue/multiwoz/MULTIWOZ2.1/")

open("dialoglue/mlm_all.txt", "w+").writelines([e+"\n" for e in train])
