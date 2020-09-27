import json
import numpy as np
import re

def evaluate(test_annotation_file, user_annotation_file, phase_codename, **kwargs):
    """
    DialoGLUE evaluation function for eval.ai 

    See documentation: https://evalai.readthedocs.io/en/latest/evaluation_scripts.html
    See test_annotation_file: gt_test.json

    The same annotation file and function will be used regardless of mode (e.g., full data, few shot, few shot + unlabeled)
    """
    gt_outputs = json.load(open(test_annotation_file))    
    gen_outputs = json.load(open(user_annotation_file))    

    # Iterate over the tasks/datasets
    dataset_to_task = {
        "hwu": "intent",
        "clinc": "intent",
        "banking": "intent",
        "restaurant8k": "slot",
        "dstc8": "slot",
        "multiwoz": "dst",
        "top": "top",
    }

    assert len(gen_outputs.keys()) == 7
    assert all(key in gen_outputs for key in dataset_to_task.keys())

    results = {}
    for dataset, gt_outputs in gt_outputs.items():
        # Calculate score differently depending on the dataset/task
        task = dataset_to_task[dataset]

        if phase_codename == "few":
            dataset_results = []
            for i in range(5):  
                # Accuracy and exact match for intent classification & TOP
                if task == "intent" or task == "top":
                    # Calculate accuracy between generated and ground-truth
                    dataset_results.append(sum(p == t for p,t in zip(gen_outputs.get(dataset)[i], gt_outputs))/len(gt_outputs))
                elif task == "slot":
                    dataset_results.append(evaluate_slot(gen_outputs.get(dataset)[i], gt_outputs))
                elif task == "dst":
                    dataset_results.append(evaluate_mwoz(gen_outputs.get(dataset)[i]))

            results[dataset] = np.mean(dataset_results)
        else:
            # Accuracy and exact match for intent classification & TOP
            if task == "intent" or task == "top":
                # Calculate accuracy between generated and ground-truth
                results[dataset] = sum(p == t for p,t in zip(gen_outputs.get(dataset), gt_outputs))/len(gt_outputs)
            elif task == "slot":
                results[dataset] = evaluate_slot(gen_outputs.get(dataset), gt_outputs)
            elif task == "dst":
                results[dataset] = evaluate_mwoz(gen_outputs.get(dataset))

    rename = {
        "hwu": "HWU64 (Acc)",
        "clinc": "CLINC150 (Acc)",
        "banking": "Banking77 (Acc)",
        "restaurant8k": "Restaurant8k (F-1)",
        "dstc8": "DSTC8 (F-1)",
        "top": "TOP (EM)",
        "multiwoz": "MultiWOZ (Joint Goal Acc)",
    }
    results = {
        rename.get(key): value*100 for key,value in results.items()
    }

    # Add average
    results["Average"] = np.mean(list(results.values()))

    labels = ["Average", "Banking77 (Acc)", "CLINC150 (Acc)", "HWU64 (Acc)", "Restaurant8k (F-1)", "DSTC8 (F-1)", "TOP (EM)", "MultiWOZ (Joint Goal Acc)"]
    output = {"result": [{"test_split": results}], "submission_result": [results.get(e) for e in labels]}
    print(output)
    return output

def evaluate_slot(pred_slots, true_slots):
    slot_types = set([slot.split(":")[0] for row in true_slots for slot in row])
    slot_type_f1_scores = []

    for slot_type in slot_types:
        predictions_for_slot = [
            [p for p in prediction if slot_type in p] for prediction in pred_slots
        ]
        labels_for_slot = [
            [l for l in label if slot_type in l] for label in true_slots
        ]

        proposal_made = [len(p) > 0 for p in predictions_for_slot]
        has_label = [len(l) > 0 for l in labels_for_slot]
        prediction_correct = [
            prediction == label for prediction, label in zip(predictions_for_slot, labels_for_slot)
        ]
        true_positives = sum([
            int(proposed and correct)
            for proposed, correct in zip(proposal_made, prediction_correct)
        ])
        num_predicted = sum([int(proposed) for proposed in proposal_made])
        num_to_recall = sum([int(hl) for hl in has_label])

        precision = true_positives / (1e-5 + num_predicted)
        recall = true_positives / (1e-5 + num_to_recall)

        f1_score = 2 * precision * recall / (1e-5 + precision + recall)
        slot_type_f1_scores.append(f1_score)

    return np.mean(slot_type_f1_scores)

def load_dataset_config(dataset_config):
    raw_config = dataset_config
    return raw_config['class_types'], raw_config['slots'], raw_config['label_maps']

def tokenize(text):
    if "\u0120" in text:
        text = re.sub(" ", "", text)
        text = re.sub("\u0120", " ", text)
        text = text.strip()
    return ' '.join([tok for tok in map(str.strip, re.split("(\W+)", text)) if len(tok) > 0])


def is_in_list(tok, value):
    found = False
    tok_list = [item for item in map(str.strip, re.split("(\W+)", tok)) if len(item) > 0]
    value_list = [item for item in map(str.strip, re.split("(\W+)", value)) if len(item) > 0]
    tok_len = len(tok_list)
    value_len = len(value_list)
    for i in range(tok_len + 1 - value_len):
        if tok_list[i:i + value_len] == value_list:
            found = True
            break
    return found


def check_slot_inform(value_label, inform_label, label_maps):
    value = inform_label
    if value_label == inform_label:
        value = value_label
    elif is_in_list(inform_label, value_label):
        value = value_label
    elif is_in_list(value_label, inform_label):
        value = value_label
    elif inform_label in label_maps:
        for inform_label_variant in label_maps[inform_label]:
            if value_label == inform_label_variant:
                value = value_label
                break
            elif is_in_list(inform_label_variant, value_label):
                value = value_label
                break
            elif is_in_list(value_label, inform_label_variant):
                value = value_label
                break
    elif value_label in label_maps:
        for value_label_variant in label_maps[value_label]:
            if value_label_variant == inform_label:
                value = value_label
                break
            elif is_in_list(inform_label, value_label_variant):
                value = value_label
                break
            elif is_in_list(value_label_variant, inform_label):
                value = value_label
                break
    return value


def get_joint_slot_correctness(preds, class_types, label_maps,
                               key_class_label_id='class_label_id',
                               key_class_prediction='class_prediction',
                               key_start_pos='start_pos',
                               key_start_prediction='start_prediction',
                               key_end_pos='end_pos',
                               key_end_prediction='end_prediction',
                               key_refer_id='refer_id',
                               key_refer_prediction='refer_prediction',
                               key_slot_groundtruth='slot_groundtruth',
                               key_slot_prediction='slot_prediction'):
    class_correctness = [[] for cl in range(len(class_types) + 1)]
    confusion_matrix = [[[] for cl_b in range(len(class_types))] for cl_a in range(len(class_types))]
    pos_correctness = []
    refer_correctness = []
    val_correctness = []
    total_correctness = []
    c_tp = {ct: 0 for ct in range(len(class_types))}
    c_tn = {ct: 0 for ct in range(len(class_types))}
    c_fp = {ct: 0 for ct in range(len(class_types))}
    c_fn = {ct: 0 for ct in range(len(class_types))}

    for pred in preds:
        guid = pred['guid']  # List: set_type, dialogue_idx, turn_idx
        turn_gt_class = pred[key_class_label_id]
       	turn_pd_class = pred[key_class_prediction]
       	gt_start_pos = pred[key_start_pos]
       	pd_start_pos = pred[key_start_prediction]
       	gt_end_pos = pred[key_end_pos]
       	pd_end_pos = pred[key_end_prediction]
       	gt_refer = pred[key_refer_id]
       	pd_refer = pred[key_refer_prediction]
       	gt_slot = pred[key_slot_groundtruth]
       	pd_slot = pred[key_slot_prediction]
       
       	gt_slot = tokenize(gt_slot)
       	pd_slot = tokenize(pd_slot)
       
       	# Make sure the true turn labels are contained in the prediction json file!
       	joint_gt_slot = gt_slot
           
       	if guid[-1] == '0': # First turn, reset the slots
       	    joint_pd_slot = 'none'

	# If turn_pd_class or a value to be copied is "none", do not update the dialog state.
        if turn_pd_class == class_types.index('none'):
            pass
        elif turn_pd_class == class_types.index('dontcare'):
            joint_pd_slot = 'dontcare'
        elif turn_pd_class == class_types.index('copy_value'):
            joint_pd_slot = pd_slot
        elif 'true' in class_types and turn_pd_class == class_types.index('true'):
            joint_pd_slot = 'true'
        elif 'false' in class_types and turn_pd_class == class_types.index('false'):
            joint_pd_slot = 'false'
        elif 'refer' in class_types and turn_pd_class == class_types.index('refer'):
            if pd_slot[0:3] == "§§ ":
                if pd_slot[3:] != 'none':
                    joint_pd_slot = check_slot_inform(joint_gt_slot, pd_slot[3:], label_maps)
            elif pd_slot[0:2] == "§§":
                if pd_slot[2:] != 'none':
                    joint_pd_slot = check_slot_inform(joint_gt_slot, pd_slot[2:], label_maps)
            elif pd_slot != 'none':
                joint_pd_slot = pd_slot
        elif 'inform' in class_types and turn_pd_class == class_types.index('inform'):
            if pd_slot[0:3] == "§§ ":
                if pd_slot[3:] != 'none':
                    joint_pd_slot = check_slot_inform(joint_gt_slot, pd_slot[3:], label_maps)
            elif pd_slot[0:2] == "§§":
                if pd_slot[2:] != 'none':
                    joint_pd_slot = check_slot_inform(joint_gt_slot, pd_slot[2:], label_maps)
            else:
                print("ERROR: Unexpected slot value format. Aborting.")
                exit()
        else:
            print("ERROR: Unexpected class_type. Aborting.")
            exit()

        total_correct = True

        # Check the per turn correctness of the class_type prediction
        if turn_gt_class == turn_pd_class:
            class_correctness[turn_gt_class].append(1.0)
            class_correctness[-1].append(1.0)
            c_tp[turn_gt_class] += 1
            # Only where there is a span, we check its per turn correctness
            if turn_gt_class == class_types.index('copy_value'):
                if gt_start_pos == pd_start_pos and gt_end_pos == pd_end_pos:
                    pos_correctness.append(1.0)
                else:
                    pos_correctness.append(0.0)
            # Only where there is a referral, we check its per turn correctness
            if 'refer' in class_types and turn_gt_class == class_types.index('refer'):
                if gt_refer == pd_refer:
                    refer_correctness.append(1.0)
                else:
                    refer_correctness.append(0.0)
        else:
            if turn_gt_class == class_types.index('copy_value'):
                pos_correctness.append(0.0)
            if 'refer' in class_types and turn_gt_class == class_types.index('refer'):
                refer_correctness.append(0.0)
            class_correctness[turn_gt_class].append(0.0)
            class_correctness[-1].append(0.0)
            confusion_matrix[turn_gt_class][turn_pd_class].append(1.0)
            c_fn[turn_gt_class] += 1
            c_fp[turn_pd_class] += 1
        for cc in range(len(class_types)):
            if cc != turn_gt_class and cc != turn_pd_class:
                c_tn[cc] += 1
 
        # Check the joint slot correctness.
        # If the value label is not none, then we need to have a value prediction.
        # Even if the class_type is 'none', there can still be a value label,
        # it might just not be pointable in the current turn. It might however
        # be referrable and thus predicted correctly.
        if joint_gt_slot == joint_pd_slot:
            val_correctness.append(1.0)
        elif joint_gt_slot != 'none' and joint_gt_slot != 'dontcare' and joint_gt_slot != 'true' and joint_gt_slot != 'false' and joint_gt_slot in label_maps:
            no_match = True
            for variant in label_maps[joint_gt_slot]:
                if variant == joint_pd_slot:
                    no_match = False
                    break
            if no_match:
                val_correctness.append(0.0)
                total_correct = False
            else:
                val_correctness.append(1.0)
        else:
            val_correctness.append(0.0)
            total_correct = False
 
        total_correctness.append(1.0 if total_correct else 0.0)
 
    # Account for empty lists (due to no instances of spans or referrals being seen)
    if pos_correctness == []:
        pos_correctness.append(1.0)
    if refer_correctness == []:
        refer_correctness.append(1.0)
 
    for ct in range(len(class_types)):
        if c_tp[ct] + c_fp[ct] > 0:
            precision = c_tp[ct] / (c_tp[ct] + c_fp[ct])
        else:
            precision = 1.0
        if c_tp[ct] + c_fn[ct] > 0:
            recall = c_tp[ct] / (c_tp[ct] + c_fn[ct])
        else:
            recall = 1.0
        if precision + recall > 0:
            f1 = 2 * ((precision * recall) / (precision + recall))
        else:
            f1 = 1.0
        if c_tp[ct] + c_tn[ct] + c_fp[ct] + c_fn[ct] > 0:
            acc = (c_tp[ct] + c_tn[ct]) / (c_tp[ct] + c_tn[ct] + c_fp[ct] + c_fn[ct])
        else:
            acc = 1.0
 
    return np.asarray(total_correctness), np.asarray(val_correctness), np.asarray(class_correctness), np.asarray(pos_correctness), np.asarray(refer_correctness), np.asarray(confusion_matrix), c_tp, c_tn, c_fp, c_fn

def evaluate_mwoz(preds):
    acc_list = []
    key_class_label_id = 'class_label_id_%s'
    key_class_prediction = 'class_prediction_%s'
    key_start_pos = 'start_pos_%s'
    key_start_prediction = 'start_prediction_%s'
    key_end_pos = 'end_pos_%s'
    key_end_prediction = 'end_prediction_%s'
    key_refer_id = 'refer_id_%s'
    key_refer_prediction = 'refer_prediction_%s'
    key_slot_groundtruth = 'slot_groundtruth_%s'
    key_slot_prediction = 'slot_prediction_%s'
    dataset_config = mwoz_config
    class_types, slots, label_maps = load_dataset_config(dataset_config)

    # Prepare label_maps
    label_maps_tmp = {}
    for v in label_maps:
        label_maps_tmp[tokenize(v)] = [tokenize(nv) for nv in label_maps[v]]
    label_maps = label_maps_tmp

    goal_correctness = 1.0
    cls_acc = [[] for cl in range(len(class_types))]
    cls_conf = [[[] for cl_b in range(len(class_types))] for cl_a in range(len(class_types))]
    c_tp = {ct: 0 for ct in range(len(class_types))}
    c_tn = {ct: 0 for ct in range(len(class_types))}
    c_fp = {ct: 0 for ct in range(len(class_types))}
    c_fn = {ct: 0 for ct in range(len(class_types))}
    for slot in slots:
        tot_cor, joint_val_cor, cls_cor, pos_cor, ref_cor, conf_mat, ctp, ctn, cfp, cfn = get_joint_slot_correctness(preds, class_types, label_maps,
                                                         key_class_label_id=(key_class_label_id % slot),
                                                         key_class_prediction=(key_class_prediction % slot),
                                                         key_start_pos=(key_start_pos % slot),
                                                         key_start_prediction=(key_start_prediction % slot),
                                                         key_end_pos=(key_end_pos % slot),
                                                         key_end_prediction=(key_end_prediction % slot),
                                                         key_refer_id=(key_refer_id % slot),
                                                         key_refer_prediction=(key_refer_prediction % slot),
                                                         key_slot_groundtruth=(key_slot_groundtruth % slot),
                                                         key_slot_prediction=(key_slot_prediction % slot)
                                                         )
        goal_correctness *= tot_cor
        for cl_a in range(len(class_types)):
            cls_acc[cl_a] += cls_cor[cl_a]
            for cl_b in range(len(class_types)):
                cls_conf[cl_a][cl_b] += list(conf_mat[cl_a][cl_b])
            c_tp[cl_a] += ctp[cl_a]
            c_tn[cl_a] += ctn[cl_a]
            c_fp[cl_a] += cfp[cl_a]
            c_fn[cl_a] += cfn[cl_a]

    for ct in range(len(class_types)):
        if c_tp[ct] + c_fp[ct] > 0:
            precision = c_tp[ct] / (c_tp[ct] + c_fp[ct])
        else:
            precision = 1.0
        if c_tp[ct] + c_fn[ct] > 0:
            recall = c_tp[ct] / (c_tp[ct] + c_fn[ct])
        else:
            recall = 1.0
        if precision + recall > 0:
            f1 = 2 * ((precision * recall) / (precision + recall))
        else:
            f1 = 1.0
        if c_tp[ct] + c_tn[ct] + c_fp[ct] + c_fn[ct] > 0:
            acc = (c_tp[ct] + c_tn[ct]) / (c_tp[ct] + c_tn[ct] + c_fp[ct] + c_fn[ct])
        else:
            acc = 1.0

    acc = np.mean(goal_correctness)

    return acc

# MWOZ CONFIG
mwoz_config = {
  "class_types": [
    "none",
    "dontcare",
    "copy_value",
    "true",
    "false",
    "refer",
    "inform"
  ],
  "slots": [
    "taxi-leaveAt",
    "taxi-destination",
    "taxi-departure",
    "taxi-arriveBy",
    "restaurant-book_people",
    "restaurant-book_day",
    "restaurant-book_time",
    "restaurant-food",
    "restaurant-pricerange",
    "restaurant-name",
    "restaurant-area",
    "hotel-book_people",
    "hotel-book_day",
    "hotel-book_stay",
    "hotel-name",
    "hotel-area",
    "hotel-parking",
    "hotel-pricerange",
    "hotel-stars",
    "hotel-internet",
    "hotel-type",
    "attraction-type",
    "attraction-name",
    "attraction-area",
    "train-book_people",
    "train-leaveAt",
    "train-destination",
    "train-day",
    "train-arriveBy",
    "train-departure"
  ],
  "label_maps": {
    "guest house": [
      "guest houses"
    ],
    "hotel": [
      "hotels"
    ],
    "centre": [
      "center",
      "downtown"
    ],
    "north": [
      "northern",
      "northside",
      "northend"
    ],
    "east": [
      "eastern",
      "eastside",
      "eastend"
    ],
    "west": [
      "western",
      "westside",
      "westend"
    ],
    "south": [
      "southern",
      "southside",
      "southend"
    ],
    "cheap": [
      "inexpensive",
      "lower price",
      "lower range",
      "cheaply",
      "cheaper",
      "cheapest",
      "very affordable"
    ],
    "moderate": [
      "moderately",
      "reasonable",
      "reasonably",
      "affordable",
      "mid range",
      "mid-range",
      "priced moderately",
      "decently priced",
      "mid price",
      "mid-price",
      "mid priced",
      "mid-priced",
      "middle price",
      "medium price",
      "medium priced",
      "not too expensive",
      "not too cheap"
    ],
    "expensive": [
      "high end",
      "high-end",
      "high class",
      "high-class",
      "high scale",
      "high-scale",
      "high price",
      "high priced",
      "higher price",
      "fancy",
      "upscale",
      "nice",
      "expensively",
      "luxury"
    ],
    "0": [
      "zero"
    ],
    "1": [
      "one",
      "just me",
      "for me",
      "myself",
      "alone",
      "me"
    ],
    "2": [
      "two"
    ],
    "3": [
      "three"
    ],
    "4": [
      "four"
    ],
    "5": [
      "five"
    ],
    "6": [
      "six"
    ],
    "7": [
      "seven"
    ],
    "8": [
      "eight"
    ],
    "9": [
      "nine"
    ],
    "10": [
      "ten"
    ],
    "11": [
      "eleven"
    ],
    "12": [
      "twelve"
    ],
    "architecture": [
      "architectural",
      "architecturally",
      "architect"
    ],
    "boat": [
      "boating",
      "boats",
      "camboats"
    ],
    "boating": [
      "boat",
      "boats",
      "camboats"
    ],
    "camboats": [
      "boating",
      "boat",
      "boats"
    ],
    "cinema": [
      "cinemas",
      "movie",
      "films",
      "film"
    ],
    "college": [
      "colleges"
    ],
    "concert": [
      "concert hall",
      "concert halls",
      "concerthall",
      "concerthalls",
      "concerts"
    ],
    "concerthall": [
      "concert hall",
      "concert halls",
      "concerthalls",
      "concerts",
      "concert"
    ],
    "entertainment": [
      "entertaining"
    ],
    "gallery": [
      "museum"
    ],
    "gastropubs": [
      "gastropub"
    ],
    "multiple sports": [
      "multiple sport",
      "multi sport",
      "multi sports",
      "sports",
      "sporting"
    ],
    "museum": [
      "museums",
      "gallery",
      "galleries"
    ],
    "night club": [
      "night clubs",
      "nightclub",
      "nightclubs",
      "club",
      "clubs"
    ],
    "park": [
      "parks"
    ],
    "pool": [
      "swimming pool",
      "swimming",
      "pools",
      "swimmingpool",
      "water",
      "swim"
    ],
    "sports": [
      "multiple sport",
      "multi sport",
      "multi sports",
      "multiple sports",
      "sporting"
    ],
    "swimming pool": [
      "swimming",
      "pool",
      "pools",
      "swimmingpool",
      "water",
      "swim"
    ],
    "theater": [
      "theatre",
      "theatres",
      "theaters"
    ],
    "theatre": [
      "theater",
      "theatres",
      "theaters"
    ],
    "abbey pool and astroturf pitch": [
      "abbey pool and astroturf",
      "abbey pool"
    ],
    "abbey pool and astroturf": [
      "abbey pool and astroturf pitch",
      "abbey pool"
    ],
    "abbey pool": [
      "abbey pool and astroturf pitch",
      "abbey pool and astroturf"
    ],
    "adc theatre": [
      "adc theater",
      "adc"
    ],
    "adc": [
      "adc theatre",
      "adc theater"
    ],
    "addenbrookes hospital": [
      "addenbrooke's hospital"
    ],
    "cafe jello gallery": [
      "cafe jello"
    ],
    "cambridge and county folk museum": [
      "cambridge and country folk museum",
      "county folk museum"
    ],
    "cambridge and country folk museum": [
      "cambridge and county folk museum",
      "county folk museum"
    ],
    "county folk museum": [
      "cambridge and county folk museum",
      "cambridge and country folk museum"
    ],
    "cambridge arts theatre": [
      "cambridge arts theater"
    ],
    "cambridge book and print gallery": [
      "book and print gallery"
    ],
    "cambridge contemporary art": [
      "cambridge contemporary art museum",
      "contemporary art museum"
    ],
    "cambridge contemporary art museum": [
      "cambridge contemporary art",
      "contemporary art museum"
    ],
    "cambridge corn exchange": [
      "the cambridge corn exchange"
    ],
    "the cambridge corn exchange": [
      "cambridge corn exchange"
    ],
    "cambridge museum of technology": [
      "museum of technology"
    ],
    "cambridge punter": [
      "the cambridge punter",
      "cambridge punters"
    ],
    "cambridge punters": [
      "the cambridge punter",
      "cambridge punter"
    ],
    "the cambridge punter": [
      "cambridge punter",
      "cambridge punters"
    ],
    "cambridge university botanic gardens": [
      "cambridge university botanical gardens",
      "cambridge university botanical garden",
      "cambridge university botanic garden",
      "cambridge botanic gardens",
      "cambridge botanical gardens",
      "cambridge botanic garden",
      "cambridge botanical garden",
      "botanic gardens",
      "botanical gardens",
      "botanic garden",
      "botanical garden"
    ],
    "cambridge botanic gardens": [
      "cambridge university botanic gardens",
      "cambridge university botanical gardens",
      "cambridge university botanical garden",
      "cambridge university botanic garden",
      "cambridge botanical gardens",
      "cambridge botanic garden",
      "cambridge botanical garden",
      "botanic gardens",
      "botanical gardens",
      "botanic garden",
      "botanical garden"
    ],
    "botanic gardens": [
      "cambridge university botanic gardens",
      "cambridge university botanical gardens",
      "cambridge university botanical garden",
      "cambridge university botanic garden",
      "cambridge botanic gardens",
      "cambridge botanical gardens",
      "cambridge botanic garden",
      "cambridge botanical garden",
      "botanical gardens",
      "botanic garden",
      "botanical garden"
    ],
    "cherry hinton village centre": [
      "cherry hinton village center"
    ],
    "cherry hinton village center": [
      "cherry hinton village centre"
    ],
    "cherry hinton hall and grounds": [
      "cherry hinton hall"
    ],
    "cherry hinton hall": [
      "cherry hinton hall and grounds"
    ],
    "cherry hinton water play": [
      "cherry hinton water play park"
    ],
    "cherry hinton water play park": [
      "cherry hinton water play"
    ],
    "christ college": [
      "christ's college",
      "christs college"
    ],
    "christs college": [
      "christ college",
      "christ's college"
    ],
    "churchills college": [
      "churchill's college",
      "churchill college"
    ],
    "cineworld cinema": [
      "cineworld"
    ],
    "clair hall": [
      "clare hall"
    ],
    "clare hall": [
      "clair hall"
    ],
    "the fez club": [
      "fez club"
    ],
    "great saint marys church": [
      "great saint mary's church",
      "great saint mary's",
      "great saint marys"
    ],
    "jesus green outdoor pool": [
      "jesus green"
    ],
    "jesus green": [
      "jesus green outdoor pool"
    ],
    "kettles yard": [
      "kettle's yard"
    ],
    "kings college": [
      "king's college"
    ],
    "kings hedges learner pool": [
      "king's hedges learner pool",
      "king hedges learner pool"
    ],
    "king hedges learner pool": [
      "king's hedges learner pool",
      "kings hedges learner pool"
    ],
    "little saint marys church": [
      "little saint mary's church",
      "little saint mary's",
      "little saint marys"
    ],
    "mumford theatre": [
      "mumford theater"
    ],
    "museum of archaelogy": [
      "museum of archaeology"
    ],
    "museum of archaelogy and anthropology": [
      "museum of archaeology and anthropology"
    ],
    "peoples portraits exhibition": [
      "people's portraits exhibition at girton college",
      "peoples portraits exhibition at girton college",
      "people's portraits exhibition"
    ],
    "peoples portraits exhibition at girton college": [
      "people's portraits exhibition at girton college",
      "people's portraits exhibition",
      "peoples portraits exhibition"
    ],
    "queens college": [
      "queens' college",
      "queen's college"
    ],
    "riverboat georgina": [
      "riverboat"
    ],
    "saint barnabas": [
      "saint barbabas press gallery"
    ],
    "saint barnabas press gallery": [
      "saint barbabas"
    ],
    "saint catharines college": [
      "saint catharine's college",
      "saint catharine's"
    ],
    "saint johns college": [
      "saint john's college",
      "st john's college",
      "st johns college"
    ],
    "scott polar": [
      "scott polar museum"
    ],
    "scott polar museum": [
      "scott polar"
    ],
    "scudamores punting co": [
      "scudamore's punting co",
      "scudamores punting",
      "scudamore's punting",
      "scudamores",
      "scudamore's",
      "scudamore"
    ],
    "scudamore": [
      "scudamore's punting co",
      "scudamores punting co",
      "scudamores punting",
      "scudamore's punting",
      "scudamores",
      "scudamore's"
    ],
    "sheeps green and lammas land park fen causeway": [
      "sheep's green and lammas land park fen causeway",
      "sheep's green and lammas land park",
      "sheeps green and lammas land park",
      "lammas land park",
      "sheep's green",
      "sheeps green"
    ],
    "sheeps green and lammas land park": [
      "sheep's green and lammas land park fen causeway",
      "sheeps green and lammas land park fen causeway",
      "sheep's green and lammas land park",
      "lammas land park",
      "sheep's green",
      "sheeps green"
    ],
    "lammas land park": [
      "sheep's green and lammas land park fen causeway",
      "sheeps green and lammas land park fen causeway",
      "sheep's green and lammas land park",
      "sheeps green and lammas land park",
      "sheep's green",
      "sheeps green"
    ],
    "sheeps green": [
      "sheep's green and lammas land park fen causeway",
      "sheeps green and lammas land park fen causeway",
      "sheep's green and lammas land park",
      "sheeps green and lammas land park",
      "lammas land park",
      "sheep's green"
    ],
    "soul tree nightclub": [
      "soul tree night club",
      "soul tree",
      "soultree"
    ],
    "soultree": [
      "soul tree nightclub",
      "soul tree night club",
      "soul tree"
    ],
    "the man on the moon": [
      "man on the moon"
    ],
    "man on the moon": [
      "the man on the moon"
    ],
    "the junction": [
      "junction theatre",
      "junction theater"
    ],
    "junction theatre": [
      "the junction",
      "junction theater"
    ],
    "old schools": [
      "old school"
    ],
    "vue cinema": [
      "vue"
    ],
    "wandlebury country park": [
      "the wandlebury"
    ],
    "the wandlebury": [
      "wandlebury country park"
    ],
    "whipple museum of the history of science": [
      "whipple museum",
      "history of science museum"
    ],
    "history of science museum": [
      "whipple museum of the history of science",
      "whipple museum"
    ],
    "williams art and antique": [
      "william's art and antique"
    ],
    "alimentum": [
      "restaurant alimentum"
    ],
    "restaurant alimentum": [
      "alimentum"
    ],
    "bedouin": [
      "the bedouin"
    ],
    "the bedouin": [
      "bedouin"
    ],
    "bloomsbury restaurant": [
      "bloomsbury"
    ],
    "cafe uno": [
      "caffe uno",
      "caffee uno"
    ],
    "caffe uno": [
      "cafe uno",
      "caffee uno"
    ],
    "caffee uno": [
      "cafe uno",
      "caffe uno"
    ],
    "cambridge lodge restaurant": [
      "cambridge lodge"
    ],
    "chiquito": [
      "chiquito restaurant bar",
      "chiquito restaurant"
    ],
    "chiquito restaurant bar": [
      "chiquito restaurant",
      "chiquito"
    ],
    "city stop restaurant": [
      "city stop"
    ],
    "cityr": [
      "cityroomz"
    ],
    "citiroomz": [
      "cityroomz"
    ],
    "clowns cafe": [
      "clown's cafe"
    ],
    "cow pizza kitchen and bar": [
      "the cow pizza kitchen and bar",
      "cow pizza"
    ],
    "the cow pizza kitchen and bar": [
      "cow pizza kitchen and bar",
      "cow pizza"
    ],
    "darrys cookhouse and wine shop": [
      "darry's cookhouse and wine shop",
      "darry's cookhouse",
      "darrys cookhouse"
    ],
    "de luca cucina and bar": [
      "de luca cucina and bar riverside brasserie",
      "luca cucina and bar",
      "de luca cucina",
      "luca cucina"
    ],
    "de luca cucina and bar riverside brasserie": [
      "de luca cucina and bar",
      "luca cucina and bar",
      "de luca cucina",
      "luca cucina"
    ],
    "da vinci pizzeria": [
      "da vinci pizza",
      "da vinci"
    ],
    "don pasquale pizzeria": [
      "don pasquale pizza",
      "don pasquale",
      "pasquale pizzeria",
      "pasquale pizza"
    ],
    "efes": [
      "efes restaurant"
    ],
    "efes restaurant": [
      "efes"
    ],
    "fitzbillies restaurant": [
      "fitzbillies"
    ],
    "frankie and bennys": [
      "frankie and benny's"
    ],
    "funky": [
      "funky fun house"
    ],
    "funky fun house": [
      "funky"
    ],
    "gardenia": [
      "the gardenia"
    ],
    "the gardenia": [
      "gardenia"
    ],
    "grafton hotel restaurant": [
      "the grafton hotel",
      "grafton hotel"
    ],
    "the grafton hotel": [
      "grafton hotel restaurant",
      "grafton hotel"
    ],
    "grafton hotel": [
      "grafton hotel restaurant",
      "the grafton hotel"
    ],
    "hotel du vin and bistro": [
      "hotel du vin",
      "du vin"
    ],
    "Kohinoor": [
      "kohinoor",
      "the kohinoor"
    ],
    "kohinoor": [
      "the kohinoor"
    ],
    "the kohinoor": [
      "kohinoor"
    ],
    "lan hong house": [
      "lan hong",
      "ian hong house",
      "ian hong"
    ],
    "ian hong": [
      "lan hong house",
      "lan hong",
      "ian hong house"
    ],
    "lovel": [
      "the lovell lodge",
      "lovell lodge"
    ],
    "lovell lodge": [
      "lovell"
    ],
    "the lovell lodge": [
      "lovell lodge",
      "lovell"
    ],
    "mahal of cambridge": [
      "mahal"
    ],
    "mahal": [
      "mahal of cambridge"
    ],
    "maharajah tandoori restaurant": [
      "maharajah tandoori"
    ],
    "the maharajah tandoor": [
      "maharajah tandoori restaurant",
      "maharajah tandoori"
    ],
    "meze bar": [
      "meze bar restaurant",
      "the meze bar"
    ],
    "meze bar restaurant": [
      "the meze bar",
      "meze bar"
    ],
    "the meze bar": [
      "meze bar restaurant",
      "meze bar"
    ],
    "michaelhouse cafe": [
      "michael house cafe"
    ],
    "midsummer house restaurant": [
      "midsummer house"
    ],
    "missing sock": [
      "the missing sock"
    ],
    "the missing sock": [
      "missing sock"
    ],
    "nandos": [
      "nando's city centre",
      "nando's city center",
      "nandos city centre",
      "nandos city center",
      "nando's"
    ],
    "nandos city centre": [
      "nando's city centre",
      "nando's city center",
      "nandos city center",
      "nando's",
      "nandos"
    ],
    "oak bistro": [
      "the oak bistro"
    ],
    "the oak bistro": [
      "oak bistro"
    ],
    "one seven": [
      "restaurant one seven"
    ],
    "restaurant one seven": [
      "one seven"
    ],
    "river bar steakhouse and grill": [
      "the river bar steakhouse and grill",
      "the river bar steakhouse",
      "river bar steakhouse"
    ],
    "the river bar steakhouse and grill": [
      "river bar steakhouse and grill",
      "the river bar steakhouse",
      "river bar steakhouse"
    ],
    "pipasha restaurant": [
      "pipasha"
    ],
    "pizza hut city centre": [
      "pizza hut city center"
    ],
    "pizza hut fenditton": [
      "pizza hut fen ditton",
      "pizza express fen ditton"
    ],
    "restaurant two two": [
      "two two",
      "restaurant 22"
    ],
    "saffron brasserie": [
      "saffron"
    ],
    "saint johns chop house": [
      "saint john's chop house",
      "st john's chop house",
      "st johns chop house"
    ],
    "sesame restaurant and bar": [
      "sesame restaurant",
      "sesame"
    ],
    "shanghai": [
      "shanghai family restaurant"
    ],
    "shanghai family restaurant": [
      "shanghai"
    ],
    "sitar": [
      "sitar tandoori"
    ],
    "sitar tandoori": [
      "sitar"
    ],
    "slug and lettuce": [
      "the slug and lettuce"
    ],
    "the slug and lettuce": [
      "slug and lettuce"
    ],
    "st johns chop house": [
      "saint john's chop house",
      "st john's chop house",
      "saint johns chop house"
    ],
    "stazione restaurant and coffee bar": [
      "stazione restaurant",
      "stazione"
    ],
    "thanh binh": [
      "thanh",
      "binh"
    ],
    "thanh": [
      "thanh binh",
      "binh"
    ],
    "binh": [
      "thanh binh",
      "thanh"
    ],
    "the hotpot": [
      "the hotspot",
      "hotpot",
      "hotspot"
    ],
    "hotpot": [
      "the hotpot",
      "the hotpot",
      "hotspot"
    ],
    "the lucky star": [
      "lucky star"
    ],
    "lucky star": [
      "the lucky star"
    ],
    "the peking restaurant: ": [
      "peking restaurant"
    ],
    "the varsity restaurant": [
      "varsity restaurant",
      "the varsity",
      "varsity"
    ],
    "two two": [
      "restaurant two two",
      "restaurant 22"
    ],
    "restaurant 22": [
      "restaurant two two",
      "two two"
    ],
    "zizzi cambridge": [
      "zizzi"
    ],
    "american": [
      "americas"
    ],
    "asian oriental": [
      "asian",
      "oriental"
    ],
    "australian": [
      "australasian"
    ],
    "barbeque": [
      "barbecue",
      "bbq"
    ],
    "corsica": [
      "corsican"
    ],
    "indian": [
      "tandoori"
    ],
    "italian": [
      "pizza",
      "pizzeria"
    ],
    "japanese": [
      "sushi"
    ],
    "latin american": [
      "latin-american",
      "latin"
    ],
    "malaysian": [
      "malay"
    ],
    "middle eastern": [
      "middle-eastern"
    ],
    "traditional american": [
      "american"
    ],
    "modern american": [
      "american modern",
      "american"
    ],
    "modern european": [
      "european modern",
      "european"
    ],
    "north american": [
      "north-american",
      "american"
    ],
    "portuguese": [
      "portugese"
    ],
    "portugese": [
      "portuguese"
    ],
    "seafood": [
      "sea food"
    ],
    "singaporean": [
      "singapore"
    ],
    "steakhouse": [
      "steak house",
      "steak"
    ],
    "the americas": [
      "american",
      "americas"
    ],
    "a and b guest house": [
      "a & b guest house",
      "a and b",
      "a & b"
    ],
    "the acorn guest house": [
      "acorn guest house",
      "acorn"
    ],
    "acorn guest house": [
      "the acorn guest house",
      "acorn"
    ],
    "alexander bed and breakfast": [
      "alexander"
    ],
    "allenbell": [
      "the allenbell"
    ],
    "the allenbell": [
      "allenbell"
    ],
    "alpha-milton guest house": [
      "the alpha-milton",
      "alpha-milton"
    ],
    "the alpha-milton": [
      "alpha-milton guest house",
      "alpha-milton"
    ],
    "arbury lodge guest house": [
      "arbury lodge",
      "arbury"
    ],
    "archway house": [
      "archway"
    ],
    "ashley hotel": [
      "the ashley hotel",
      "ashley"
    ],
    "the ashley hotel": [
      "ashley hotel",
      "ashley"
    ],
    "aylesbray lodge guest house": [
      "aylesbray lodge",
      "aylesbray"
    ],
    "aylesbray lodge guest": [
      "aylesbray lodge guest house",
      "aylesbray lodge",
      "aylesbray"
    ],
    "alesbray lodge guest house": [
      "aylesbray lodge guest house",
      "aylesbray lodge",
      "aylesbray"
    ],
    "alyesbray lodge hotel": [
      "aylesbray lodge guest house",
      "aylesbray lodge",
      "aylesbray"
    ],
    "bridge guest house": [
      "bridge house"
    ],
    "cambridge belfry": [
      "the cambridge belfry",
      "belfry hotel",
      "belfry"
    ],
    "the cambridge belfry": [
      "cambridge belfry",
      "belfry hotel",
      "belfry"
    ],
    "belfry hotel": [
      "the cambridge belfry",
      "cambridge belfry",
      "belfry"
    ],
    "carolina bed and breakfast": [
      "carolina"
    ],
    "city centre north": [
      "city centre north bed and breakfast"
    ],
    "north b and b": [
      "city centre north bed and breakfast"
    ],
    "city centre north b and b": [
      "city centre north bed and breakfast"
    ],
    "el shaddia guest house": [
      "el shaddai guest house",
      "el shaddai",
      "el shaddia"
    ],
    "el shaddai guest house": [
      "el shaddia guest house",
      "el shaddai",
      "el shaddia"
    ],
    "express by holiday inn cambridge": [
      "express by holiday inn",
      "holiday inn cambridge",
      "holiday inn"
    ],
    "holiday inn": [
      "express by holiday inn cambridge",
      "express by holiday inn",
      "holiday inn cambridge"
    ],
    "finches bed and breakfast": [
      "finches"
    ],
    "gonville hotel": [
      "gonville"
    ],
    "hamilton lodge": [
      "the hamilton lodge",
      "hamilton"
    ],
    "the hamilton lodge": [
      "hamilton lodge",
      "hamilton"
    ],
    "hobsons house": [
      "hobson's house",
      "hobson's"
    ],
    "huntingdon marriott hotel": [
      "huntington marriott hotel",
      "huntington marriot hotel",
      "huntingdon marriot hotel",
      "huntington marriott",
      "huntingdon marriott",
      "huntington marriot",
      "huntingdon marriot",
      "huntington",
      "huntingdon"
    ],
    "kirkwood": [
      "kirkwood house"
    ],
    "kirkwood house": [
      "kirkwood"
    ],
    "lensfield hotel": [
      "the lensfield hotel",
      "lensfield"
    ],
    "the lensfield hotel": [
      "lensfield hotel",
      "lensfield"
    ],
    "leverton house": [
      "leverton"
    ],
    "marriot hotel": [
      "marriott hotel",
      "marriott"
    ],
    "rosas bed and breakfast": [
      "rosa's bed and breakfast",
      "rosa's",
      "rosas"
    ],
    "university arms hotel": [
      "university arms"
    ],
    "warkworth house": [
      "warkworth hotel",
      "warkworth"
    ],
    "warkworth hotel": [
      "warkworth house",
      "warkworth"
    ],
    "wartworth": [
      "warkworth house",
      "warkworth hotel",
      "warkworth"
    ],
    "worth house": [
      "the worth house"
    ],
    "the worth house": [
      "worth house"
    ],
    "birmingham new street": [
      "birmingham new street train station"
    ],
    "birmingham new street train station": [
      "birmingham new street"
    ],
    "bishops stortford": [
      "bishops stortford train station"
    ],
    "bishops stortford train station": [
      "bishops stortford"
    ],
    "broxbourne": [
      "broxbourne train station"
    ],
    "broxbourne train station": [
      "broxbourne"
    ],
    "cambridge": [
      "cambridge train station"
    ],
    "cambridge train station": [
      "cambridge"
    ],
    "ely": [
      "ely train station"
    ],
    "ely train station": [
      "ely"
    ],
    "kings lynn": [
      "king's lynn",
      "king's lynn train station",
      "kings lynn train station"
    ],
    "kings lynn train station": [
      "kings lynn",
      "king's lynn",
      "king's lynn train station"
    ],
    "leicester": [
      "leicester train station"
    ],
    "leicester train station": [
      "leicester"
    ],
    "london kings cross": [
      "kings cross",
      "king's cross",
      "london king's cross",
      "kings cross train station",
      "king's cross train station",
      "london king's cross train station",
      "london kings cross train station"
    ],
    "london kings cross train station": [
      "kings cross",
      "king's cross",
      "london king's cross",
      "london kings cross",
      "kings cross train station",
      "king's cross train station",
      "london king's cross train station"
    ],
    "london liverpool": [
      "liverpool street",
      "london liverpool street",
      "london liverpool train station",
      "liverpool street train station",
      "london liverpool street train station"
    ],
    "london liverpool street": [
      "london liverpool",
      "liverpool street",
      "london liverpool train station",
      "liverpool street train station",
      "london liverpool street train station"
    ],
    "london liverpool street train station": [
      "london liverpool",
      "liverpool street",
      "london liverpool street",
      "london liverpool train station",
      "liverpool street train station"
    ],
    "norwich": [
      "norwich train station"
    ],
    "norwich train station": [
      "norwich"
    ],
    "peterborough": [
      "peterborough train station"
    ],
    "peterborough train station": [
      "peterborough"
    ],
    "stansted airport": [
      "stansted airport train station"
    ],
    "stansted airport train station": [
      "stansted airport"
    ],
    "stevenage": [
      "stevenage train station"
    ],
    "stevenage train station": [
      "stevenage"
    ]
  }
}
