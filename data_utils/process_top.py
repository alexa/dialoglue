import os 
import re
import sys
import numpy as np

def read_data(data_file):
    """ Reads all conversations ids in the given subset from the subset_list file"""
    """ Arguments:
          data_file: original TSV file released by facebook.
        Returns:
          annotated_data: list of annotated utterances
    """

    annotated_data = list()
    
    f = open(data_file)
    for line in f:
        line = line.strip()
        line_els = line.split("\t")
        annotated_data.append(line_els[2])

    return annotated_data


def process(utterance):
    """ Reads all conversations ids in the given subset from the subset_list file"""
    """ Arguments:
          utterance: utterance in the original annotation format
        Returns:
          new_utterance: processed utterance in out annotation format
    """

    new_utterance = ""
    tags = list()
    tokens = utterance.split(" ")
    start_utterance = True
    first_instance = True
    for token in tokens:
        if token.startswith("["):
            if start_utterance:
                # the first intent tag in the utterance
                start_utterance = False
                intent = token.strip("[").replace("IN:", "IN_")
            else:
                # new tag seen in the input
                tags.append(token.strip("[").replace("IN:", "IN_").replace("SL:", "SL_"))
                first_instance = True
                #print(tags)
                
        elif token.startswith("]"):
            if len(tags) > 0:
                tags.pop(len(tags)-1)

        else:
            current_tag = "O"
            if len(tags) > 0:
                current_tag = "_".join(tags)
                if first_instance:
                    first_instance = False
                    current_tag = "B-" + current_tag
                else:
                    current_tag = "I-" + current_tag

            if new_utterance == "":
                new_utterance = token + ":" + current_tag
            else:
                new_utterance = new_utterance + " " + token + ":" + current_tag
    new_utterance = new_utterance + " <=> " + intent

    return new_utterance


def process_and_write(data, output_file):
    """ Reads all conversations ids in the given subset from the subset_list file"""
    """ Arguments:
          data: list of utterances in the original annotation format
          output_file: file where the reformatted output will be written
    """
    
    f = open(output_file, "w")

    for utterance in data:
        new_utterance = process(utterance) + "\n"
        f.write(new_utterance)
    f.close()
    return


if __name__ == '__main__':
    data_dir = sys.argv[1]
    if not data_dir.endswith("/"):
        data_dir += "/"

    for subset in ["train", "eval", "test"]:
        output_file = data_dir + subset + ".txt"
        data_file = data_dir + subset + ".tsv"
        data = read_data(data_file)
        process_and_write(data, output_file)

        if subset == 'train':
            np.random.seed(0)
            lines = open(output_file).readlines()
            few_shot_lines = np.random.choice(lines, int(len(lines)/10))
            open(data_dir+"train_10.txt", "w+").writelines(few_shot_lines)

    all_intents = set()
    all_slots = set()
    for subset in ["train", "eval", "test"]:
        intents = set([l.split(" <=> ")[-1].strip() for l in open(data_dir + subset + ".txt").readlines()])
        all_intents.update(intents)

        rows = [l.split(" <=> ")[0] for l in open(data_dir + subset + ".txt").readlines()]
        slots = set([w.split(":")[-1] for l in rows for w in l.split()])
        all_slots.update(slots)

    open(data_dir + "vocab.intent", "w+").writelines([intent+"\n" for intent in sorted(list(all_intents))])
    open(data_dir + "vocab.slot", "w+").writelines([slot+"\n" for slot in sorted(list(all_slots))])
