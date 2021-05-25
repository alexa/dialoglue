import torch
import torch.nn.functional as F

from collections import defaultdict
from torch import nn
from torch.nn import CrossEntropyLoss, NLLLoss 
from torch.nn import Dropout
from transformers import BertConfig, BertModel, BertForMaskedLM
from typing import Any

class BertPretrain(torch.nn.Module):
    def __init__(self,
                 model_name_or_path: str):
        super(BertPretrain, self).__init__()
        self.bert_model = BertForMaskedLM.from_pretrained(model_name_or_path)

    def forward(self, 
                input_ids: torch.tensor,
                mlm_labels: torch.tensor):
        outputs = self.bert_model(input_ids, masked_lm_labels=mlm_labels)
        return outputs[0]

class IntentBertModel(torch.nn.Module):
    def __init__(self,
                 model_name_or_path: str,
                 dropout: float,
                 num_intent_labels: int,
                 use_observers: bool = False):
        super(IntentBertModel, self).__init__()
        self.bert_model = BertModel.from_pretrained(model_name_or_path)

        self.dropout = Dropout(dropout)
        self.num_intent_labels = num_intent_labels
        self.intent_classifier = nn.Linear(self.bert_model.config.hidden_size, num_intent_labels)
        self.use_observers = use_observers
        self.num_observers = 20

    def encode(self,
               input_ids: torch.tensor,
               attention_mask: torch.tensor,
               token_type_ids: torch.tensor):
        if not self.use_observers:
            pooled_output = self.bert_model(input_ids=input_ids,
                                            attention_mask=attention_mask,
                                            token_type_ids=token_type_ids)[1]
        else:
            hidden_states = self.bert_model(input_ids=input_ids,
                                            attention_mask=attention_mask,
                                            token_type_ids=token_type_ids)[0]

            pooled_output =  hidden_states[:, -self.num_observers:].mean(dim=1)

        return pooled_output


    def forward(self,
                input_ids: torch.tensor,
                attention_mask: torch.tensor,
                token_type_ids: torch.tensor,
                intent_label: torch.tensor = None):
        if not self.use_observers:
            pooled_output = self.bert_model(input_ids=input_ids,
                                            attention_mask=attention_mask,
                                            token_type_ids=token_type_ids)[1]
        else:
            hidden_states = self.bert_model(input_ids=input_ids,
                                            attention_mask=attention_mask,
                                            token_type_ids=token_type_ids)[0]

            pooled_output =  hidden_states[:, -self.num_observers:].mean(dim=1)

        intent_logits = self.intent_classifier(self.dropout(pooled_output))

        # Compute losses if labels provided
        if intent_label is not None:
            loss_fct = CrossEntropyLoss()
            intent_loss = loss_fct(intent_logits.view(-1, self.num_intent_labels), intent_label.type(torch.long))
        else:
            intent_loss = torch.tensor(0)

        return intent_logits, intent_loss

class ExampleIntentBertModel(torch.nn.Module):
    def __init__(self,
                 model_name_or_path: str,
                 dropout: float,
                 num_intent_labels: int,
                 use_observers: bool = False):
        super(ExampleIntentBertModel, self).__init__()
        self.bert_model = BertModel(BertConfig.from_pretrained(model_name_or_path, output_attentions=True))

        self.dropout = Dropout(dropout)
        self.num_intent_labels = num_intent_labels
        self.use_observers = use_observers
        self.num_observers = 20

    def encode(self,
               input_ids: torch.tensor,
               attention_mask: torch.tensor,
               token_type_ids: torch.tensor):
        if not self.use_observers:
            pooled_output = self.bert_model(input_ids=input_ids,
                                            attention_mask=attention_mask,
                                            token_type_ids=token_type_ids)[1]
        else:
            hidden_states = self.bert_model(input_ids=input_ids,
                                            attention_mask=attention_mask,
                                            token_type_ids=token_type_ids)[0]

            pooled_output =  hidden_states[:, -self.num_observers:].mean(dim=1)

        return pooled_output


    def forward(self,
                input_ids: torch.tensor,
                attention_mask: torch.tensor,
                token_type_ids: torch.tensor,
                intent_label: torch.tensor,
                example_input: torch.tensor,
                example_mask: torch.tensor,
                example_token_types: torch.tensor,
                example_intents: torch.tensor):
        example_pooled_output = self.encode(input_ids=example_input,
                                            attention_mask=example_mask,
                                            token_type_ids=example_token_types)

        pooled_output = self.encode(input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    token_type_ids=token_type_ids)

        pooled_output = self.dropout(pooled_output)
        probs = torch.softmax(pooled_output.mm(example_pooled_output.t()), dim =-1)

        intent_probs = 1e-6 + torch.zeros(probs.size(0), self.num_intent_labels).cuda().scatter_add(-1, example_intents.unsqueeze(0).repeat(probs.size(0), 1), probs)

        # Compute losses if labels provided
        if intent_label is not None:
            loss_fct = NLLLoss()
            intent_lp = torch.log(intent_probs)
            intent_loss = loss_fct(intent_lp.view(-1, self.num_intent_labels), intent_label.type(torch.long))
        else:
            intent_loss = torch.tensor(0)

        return intent_probs, intent_loss


class SlotBertModel(torch.nn.Module):
    def __init__(self,
                 model_name_or_path: str,
                 dropout: float,
                 num_slot_labels: int):
        super(SlotBertModel, self).__init__()
        self.bert_model = BertModel.from_pretrained(model_name_or_path)
        self.dropout = Dropout(dropout)
        self.num_slot_labels = num_slot_labels
        self.slot_classifier = nn.Linear(self.bert_model.config.hidden_size, num_slot_labels)

    def encode(self,
               input_ids: torch.tensor,
               attention_mask: torch.tensor,
               token_type_ids: torch.tensor):
        hidden_states, _ = self.bert_model(input_ids=input_ids,
                                           attention_mask=attention_mask,
                                           token_type_ids=token_type_ids)
        return hidden_states

    def forward(self,
                input_ids: torch.tensor,
                attention_mask: torch.tensor,
                token_type_ids: torch.tensor,
                slot_labels: torch.tensor = None):
        hidden_states = self.encode(input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    token_type_ids=token_type_ids)

        slot_logits = self.slot_classifier(self.dropout(hidden_states))

        # Compute losses if labels provided
        if slot_labels is not None:
            loss_fct = CrossEntropyLoss()

            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = slot_logits.view(-1, self.num_slot_labels)[active_loss]
                active_labels = slot_labels.view(-1)[active_loss]
                slot_loss = loss_fct(active_logits, active_labels.type(torch.long))
            else:
                slot_loss = loss_fct(slot_logits.view(-1, self.num_slot_labels), slot_labels.view(-1).type(torch.long))
        else:
            slot_loss = torch.tensor(0).cuda() if torch.cuda.is_available() else torch.tensor(0)

        return slot_logits, slot_loss

class JointSlotIntentBertModel(torch.nn.Module):
    def __init__(self,
                 model_name_or_path: str,
                 dropout: float,
                 num_intent_labels: int,
                 num_slot_labels: int):
        super(JointSlotIntentBertModel, self).__init__()
        self.bert_model = BertModel.from_pretrained(model_name_or_path)

        self.dropout = Dropout(dropout)
        self.num_intent_labels = num_intent_labels
        self.num_slot_labels = num_slot_labels
        self.intent_classifier = nn.Linear(self.bert_model.config.hidden_size, num_intent_labels)
        self.slot_classifier = nn.Linear(self.bert_model.config.hidden_size, num_slot_labels)

    def forward(self,
                input_ids: torch.tensor,
                attention_mask: torch.tensor,
                token_type_ids: torch.tensor,
                intent_label: torch.tensor = None,
                slot_labels: torch.tensor = None):
        hidden_states, pooled_output = self.bert_model(input_ids=input_ids,
                                                       attention_mask=attention_mask,
                                                       token_type_ids=token_type_ids)


        intent_logits = self.intent_classifier(self.dropout(pooled_output))
        slot_logits = self.slot_classifier(self.dropout(hidden_states))

        # Compute losses if labels provided
        if slot_labels is not None:
            loss_fct = CrossEntropyLoss()

            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = slot_logits.view(-1, self.num_slot_labels)[active_loss]
                active_labels = slot_labels.view(-1)[active_loss]
                slot_loss = loss_fct(active_logits, active_labels.type(torch.long))
            else:
                slot_loss = loss_fct(slot_logits.view(-1, self.num_slot_labels), slot_labels.view(-1).type(torch.long))
        else:
            slot_loss = torch.tensor(0).cuda() if torch.cuda.is_available() else torch.tensor(0)

        # Compute losses if labels provided
        if intent_label is not None:
            loss_fct = CrossEntropyLoss()
            intent_loss = loss_fct(intent_logits.view(-1, self.num_intent_labels), intent_label.type(torch.long))
        else:
            intent_loss = torch.tensor(0)

        return intent_logits, slot_logits, intent_loss + slot_loss
