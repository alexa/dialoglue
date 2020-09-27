import csv
import logging
import json
import numpy as np
import os
import pickle

from collections import defaultdict
from tokenizers import BertWordPieceTokenizer
from torch.utils.data import Dataset
from tqdm import tqdm
from typing import Dict

from constants import SPECIAL_TOKENS

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.DEBUG)
LOGGER = logging.getLogger(__name__)


class IntentDataset(Dataset):
    def __init__(self,
                 data_path: str,
                 tokenizer: BertWordPieceTokenizer,
                 max_seq_length: int,
                 vocab_file_name: str):
        # For caching
        data_dirname = os.path.dirname(os.path.abspath(data_path))
        split = os.path.basename(os.path.abspath(data_path))

        # Intent categories
        intent_vocab_path = os.path.join(data_dirname, "categories.json")
        intent_names = json.load(open(intent_vocab_path))
        self.intent_label_to_idx = dict((label, idx) for idx, label in enumerate(intent_names))
        self.intent_idx_to_label = {idx: label for label, idx in self.intent_label_to_idx.items()}

        # Process data
        self.tokenizer = tokenizer
        cached_path = os.path.join(data_dirname, "{}_{}_intent_cached".format(split, vocab_file_name))
        if not os.path.exists(cached_path):
            self.examples = []
            reader = csv.reader(open(data_path))
            next(reader, None)
            out = []
            for utt, intent in tqdm(reader):
                encoded = tokenizer.encode(utt)

                self.examples.append({
                    "input_ids": np.array(encoded.ids)[-max_seq_length:],
                    "attention_mask": np.array(encoded.attention_mask)[-max_seq_length:],
                    "token_type_ids": np.array(encoded.type_ids)[-max_seq_length:],
                    "intent_label": self.intent_label_to_idx[intent],
                    "ind": len(self.examples),
                })
            with open(cached_path, "wb") as f:
                pickle.dump(self.examples, f)
        else:
            LOGGER.info("Loading from cached path: {}".format(cached_path))
            with open(cached_path, "rb") as f:
                self.examples = pickle.load(f)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

class SlotDataset(Dataset):
    def __init__(self,
                 data_path: str,
                 tokenizer: BertWordPieceTokenizer,
                 max_seq_length: int,
                 vocab_file_name: str):
        # For caching
        data_dirname = os.path.dirname(os.path.abspath(data_path))
        split = os.path.basename(os.path.abspath(data_path))

        # Slot categories
        slot_vocab_path = os.path.join(os.path.dirname(data_path), "vocab.txt")
        slot_names = json.load(open(slot_vocab_path))
        slot_names.insert(0, "[PAD]")
        self.slot_label_to_idx = dict((label, idx) for idx, label in enumerate(slot_names))
        self.slot_idx_to_label = {idx: label for label, idx in self.slot_label_to_idx.items()}

        # Process data
        self.tokenizer = tokenizer
        cached_path = os.path.join(data_dirname, "{}_{}_slots_cached".format(split, vocab_file_name))
        texts = []
        slotss = []
        if not os.path.exists(cached_path):
            self.examples = []
            data = json.load(open(data_path))
            for example in tqdm(data):
                text, slots = self.parse_example(example) 
                texts.append(text)
                slotss.append(slots)
                encoded = tokenizer.encode(text)
                encoded_slot_labels = self.encode_token_labels([text], [slots],
                                                               len(encoded.ids),
                                                               tokenizer,
                                                               self.slot_label_to_idx,
                                                               max_seq_length)
                self.examples.append({
                    "input_ids": np.array(encoded.ids)[-max_seq_length:],
                    "attention_mask": np.array(encoded.attention_mask)[-max_seq_length:],
                    "token_type_ids": np.array(encoded.type_ids)[-max_seq_length:],
                    "slot_labels": encoded_slot_labels[-max_seq_length:],
                    "ind": len(self.examples),
                })
            with open(cached_path, "wb") as f:
                pickle.dump(self.examples, f)
        else:
            LOGGER.info("Loading from cached path: {}".format(cached_path))
            with open(cached_path, "rb") as f:
                self.examples = pickle.load(f)

    def encode_token_labels(self,
                            text_sequences,
                            slot_names,
                            encoded_length,
                            tokenizer,
                            slot_map,
                            max_length) -> np.array:
    
        def _get_word_tokens_len(word: str, tokenizer: BertWordPieceTokenizer) -> int:
            return sum(map(lambda token: 1 if token not in SPECIAL_TOKENS else 0,
                           tokenizer.encode(word).tokens))
    
        encoded = np.zeros(shape=(len(text_sequences), encoded_length), dtype=np.int32)
        for i, (text_sequence, word_labels) in enumerate(
                zip(text_sequences, slot_names)):
            encoded_labels = []
            for word, word_label in zip(text_sequence.split(), word_labels.split()):
                encoded_labels.append(slot_map[word_label])
                expand_label = word_label.replace("B-", "I-")
                if not expand_label in slot_map:
                    expand_label = word_label
                word_tokens_len = _get_word_tokens_len(word, tokenizer)
                encoded_labels.extend([slot_map[expand_label]] * (word_tokens_len - 1))
            encoded[i, 1:len(encoded_labels) + 1] = encoded_labels
        return encoded.squeeze()

    def parse_example(self, example):
        text = example['userInput']['text']

        # Create slots dictionary
        word_to_slot = {}
        for label in example.get('labels', []):
            slot = label['slot']
            start = label['valueSpan'].get('startIndex', 0)
            end = label['valueSpan'].get('endIndex', -1)

            for word in text[start:end].split():
                word_to_slot[word] = slot
          
        # Add context if it's there
        if 'context' in example:
            for req in example['context'].get('requestedSlots', []):
                text = req + " " + text

        # Create slots list
        slots = []
        cur = None
        for word in text.split():
            if word in word_to_slot:
                slot = word_to_slot[word]
                if cur is not None and slot == cur:
                    slots.append("I-" + slot) 
                else:
                    slots.append("B-" + slot) 
                    cur = slot
            else:
                slots.append("O")
                cur = None

        return text, " ".join(slots) 

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

class TOPDataset(Dataset):
    def __init__(self,
                 data_path: str,
                 tokenizer: BertWordPieceTokenizer,
                 max_seq_length: int,
                 vocab_file_name: str):
        # For caching
        data_dirname = os.path.dirname(os.path.abspath(data_path))
        split = os.path.basename(os.path.abspath(data_path))

        # Slot categories
        slot_vocab_path = os.path.join(os.path.dirname(data_path), "vocab.slot")
        slot_names = [e.strip() for e in open(slot_vocab_path).readlines()]
        slot_names.insert(0, "[PAD]")
        self.slot_label_to_idx = dict((label, idx) for idx, label in enumerate(slot_names))
        self.slot_idx_to_label = {idx: label for label, idx in self.slot_label_to_idx.items()}

        # Intent categories
        intent_vocab_path = os.path.join(data_dirname, "vocab.intent")
        intent_names = [e.strip() for e in open(intent_vocab_path).readlines()]
        self.intent_label_to_idx = dict((label, idx) for idx, label in enumerate(intent_names))
        self.intent_idx_to_label = {idx: label for label, idx in self.intent_label_to_idx.items()}

        # Process data
        self.tokenizer = tokenizer
        cached_path = os.path.join(data_dirname, "{}_{}_top_cached".format(split, vocab_file_name))
        if not os.path.exists(cached_path):
            self.examples = []
            data = [e.strip() for e in open(data_path).readlines() ]
            for example in tqdm(data):
                example, intent = example.split(" <=> ")
                text = " ".join([e.split(":")[0] for e in example.split()])
                slots = " ".join([e.split(":")[1] for e in example.split()])
                encoded = tokenizer.encode(text)
                encoded_slot_labels = self.encode_token_labels([text], [slots],
                                                               len(encoded.ids),
                                                               tokenizer,
                                                               self.slot_label_to_idx,
                                                               max_seq_length)
                self.examples.append({
                    "input_ids": np.array(encoded.ids)[-max_seq_length:],
                    "attention_mask": np.array(encoded.attention_mask)[-max_seq_length:],
                    "token_type_ids": np.array(encoded.type_ids)[-max_seq_length:],
                    "slot_labels": encoded_slot_labels[-max_seq_length:],
                    "intent_label": self.intent_label_to_idx[intent],
                    "ind": len(self.examples),
                })
            with open(cached_path, "wb") as f:
                pickle.dump(self.examples, f)
        else:
            LOGGER.info("Loading from cached path: {}".format(cached_path))
            with open(cached_path, "rb") as f:
                self.examples = pickle.load(f)

    def encode_token_labels(self,
                            text_sequences,
                            slot_names,
                            encoded_length,
                            tokenizer,
                            slot_map,
                            max_length) -> np.array:
    
        def _get_word_tokens_len(word: str, tokenizer: BertWordPieceTokenizer) -> int:
            return sum(map(lambda token: 1 if token not in SPECIAL_TOKENS else 0,
                           tokenizer.encode(word).tokens))
    
        encoded = np.zeros(shape=(len(text_sequences), encoded_length), dtype=np.int32)
        for i, (text_sequence, word_labels) in enumerate(
                zip(text_sequences, slot_names)):
            encoded_labels = []
            for word, word_label in zip(text_sequence.split(), word_labels.split()):
                encoded_labels.append(slot_map[word_label])
                expand_label = word_label.replace("B-", "I-")
                if not expand_label in slot_map:
                    expand_label = word_label
                word_tokens_len = _get_word_tokens_len(word, tokenizer)
                encoded_labels.extend([slot_map[expand_label]] * (word_tokens_len - 1))
            encoded[i, 1:len(encoded_labels) + 1] = encoded_labels
        return encoded.squeeze()

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]
