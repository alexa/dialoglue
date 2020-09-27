import json
import numpy as np
import os

np.random.seed(0)

train_dials = json.load(open("dialoglue/multiwoz/train_dials.json"))
train_dials_few_keys = list(np.random.choice(list(train_dials.keys()), int(len(train_dials)/10)))
train_dials_few = {k:train_dials[k] for k in train_dials_few_keys}
json.dump(train_dials_few, open("dialoglue/multiwoz/train_dials_10.json", "w+"))

# Create proper structure:
os.mkdir("dialoglue/multiwoz/MULTIWOZ2.1/")
os.system("mv dialoglue/multiwoz/* dialoglue/multiwoz/MULTIWOZ2.1/")
os.mkdir("dialoglue/multiwoz/MULTIWOZ2.1_fewshot/")
os.system("cp dialoglue/multiwoz/MULTIWOZ2.1/* dialoglue/multiwoz/MULTIWOZ2.1_fewshot/")
os.system("mv dialoglue/multiwoz/MULTIWOZ2.1_fewshot/train_dials_10.json dialoglue/multiwoz/MULTIWOZ2.1_fewshot/train_dials.json")
os.system("rm dialoglue/multiwoz/MULTIWOZ2.1/train_dials_10.json")
