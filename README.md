# DialoGLUE

DialoGLUE is a conversational AI benchmark designed to encourage dialogue research in representation-based transfer, domain adaptation, and sample-efficient task learning.
For a more detailed write-up of the benchmark check out [our paper](/missing-link).

This repository contains all code related to the benchmark, including scripts for download 
relevant datasets, preprocessing them in a consistent format for benchmark submissions, evaluating any
submission outputs, and running baseline models from the original benchmark description.

## Datasets

As in all science, this benchmark stands on the shoulders of giants, leveraging the following previously-published 
data resources. Thank you to the authors of those works for their great contributions:


| Dataset      	| Size 	| Description                                                                         	| License                    	|
|--------------	|------	|-------------------------------------------------------------------------------------	|----------------------------	|
| [Banking77](https://arxiv.org/abs/2003.04807)      	| 13K  	| online banking queries                                                              	| CC-BY-4.0                  	|
| [HWU64](https://arxiv.org/abs/1903.05566)          	| 11K  	| popular personal assistant queries                                                  	| CC-BY-SA 3.0               	|
| [CLINC150](https://www.aclweb.org/anthology/D19-1131/)        	| 20K  	| popular personal assistant queries                                                  	| CC-BY-SA 3.0               	|
| [Restaurant8k](https://arxiv.org/abs/2005.08866) 	| 8.2K 	| restaurant booking domain queries                                                   	| CC-BY-4.0                  	|
| [DSTC8 SGD](https://arxiv.org/abs/1909.05855)    	| 20K  	| multi-domain, task-oriented conversations   between a human and a virtual assistant 	| CC-BY-SA 4.0 International 	|
| [TOP](https://arxiv.org/abs/1810.07942)          	| 45K  	| compositional queries for hierachical   semantic representations                    	| CC-BY-SA                   	|
| [MultiWOZ 2.1](https://arxiv.org/abs/1907.01669) 	| 12K  	| multi-domain dialogues with multiple turns                                              	| MIT                        	|


## Data Download

To download/process the various datasets that are part of the DialoGLUE benchmark, run `bash download_data.sh` from `data_utils`.

Upon completion, your DialoGLUE folder should contain the following:

```
dialoglue/
├── banking
│   ├── categories.json
│   ├── test.csv
│   ├── train_10.csv
│   ├── train_5.csv
│   ├── train.csv
│   └── val.csv
├── clinc
│   ├── categories.json
│   ├── test.csv
│   ├── train_10.csv
│   ├── train_5.csv
│   ├── train.csv
│   └── val.csv
├── dstc8_sgd
│   ├── stats.csv
│   ├── test.json
│   ├── train_10.json
│   ├── train.json
│   ├── val.json
│   └── vocab.txt
├── hwu
│   ├── categories.json
│   ├── test.csv
│   ├── train_10.csv
│   ├── train_5.csv
│   ├── train.csv
│   └── val.csv
├── mlm_all.txt
├── multiwoz
│   ├── MULTIWOZ2.1
│   │   ├── dialogue_acts.json
│   │   ├── README.txt
│   │   ├── test_dials.json
│   │   ├── train_dials.json
│   │   └── val_dials.json
│   └── MULTIWOZ2.1_fewshot
│       ├── dialogue_acts.json
│       ├── README.txt
│       ├── test_dials.json
│       ├── train_dials.json
│       └── val_dials.json
├── restaurant8k
│   ├── test.json
│   ├── train_10.json
│   ├── train.json
│   ├── val.json
│   └── vocab.txt
└── top
    ├── eval.txt
    ├── test.txt
    ├── train_10.txt
    ├── train.txt
    ├── vocab.intent
    └── vocab.slot
```

The files with a `_10` suffix (e.g., `banking/train_10.csv`) are used for the few-shot experiments, wherein models are trained with only 10% of the datasets.

## EvalAI Leaderboard

The DialoGLUE benchmark is [hosted on EvalAI](https://evalai.cloudcv.org/web/challenges/challenge-page/708/) and we invite submissions to our leaderboard. The submission should be a JSON file with keys corresponding to each of the seven DialoGLUE datasets:

```
{"banking": [*banking outputs*], "hwu": [*hwu outputs*], ..., "multiwoz": [*multiwoz outputs*]}
```

Given a set of seven model checkpoints, you can edit and run `dump_outputs.py` to generate a valid submission file. For the intent classification tasks (HWU64, Banking77, CLINC150), the outputs are a list of intent classes. For the slot filling tasks (Restaurant8k, DSTC8), the outputs are a list of spans. For the TOP dataset, the outputs are a list of (intent, slots) pairs wherein each slot is the path from the root to the leaf node. For MultiWOZ, the output follows the [TripPy format](https://gitlab.cs.uni-duesseldorf.de/general/dsml/trippy-public/-/blob/master/run_dst.py#L295) and corresponds to the `pred_res.test.final.json` output file. We strongly recommend using the `dump_outputs.py` script to generate outputs. 

For the few-shot experimental setting, we only train models on a subset (roughly 10%) of the training data. The specific data splits are produced by running `download_data.sh`. To mitigate the impact of random initialization, we ask that you train **5 models** for each of the few-shot tasks and submit the output of all 5 models.  The scores on the leaderboard will be the average of these five runs.

The few-shot submission file format is as follows:

```
{"banking": [*banking outputs from model 1*, ... *banking outputs from model 5*], ...}
```

You may run `dump_outputs_fewshot.py` to generate a valid submission file given the model paths corresponding to all of the runs.

## Training

Almost all of the models can be trained/evaluated using the `run.py` script. MultiWOZ is the exception, and relies on the modified open-sourced TripPy implementation.

The commands for training/evaluating models are as follows. If you want to *only* run inference/evaluation, simply change `--num_epochs` to 0.

**HWU64**

```
python run.py \
        --train_data_path data_utils/dialoglue/hwu/train.csv \
        --val_data_path data_utils/dialoglue/hwu/val.csv \
        --test_data_path data_utils/dialoglue/hwu/test.csv \
        --token_vocab_path bert-base-uncased-vocab.txt \
        --train_batch_size 64 --dropout 0.1 --num_epochs 100 --learning_rate 6e-5 \
        --model_name_or_path convbert-dg --task intent --do_lowercase --max_seq_length 50 --mlm_pre --mlm_during --dump_outputs \
```

**Banking77**

```
python run.py \
        --train_data_path data_utils/dialoglue/banking/train.csv \
        --val_data_path data_utils/dialoglue/banking/val.csv \
        --test_data_path data_utils/dialoglue/banking/test.csv \
        --token_vocab_path bert-base-uncased-vocab.txt \
        --train_batch_size 32 --grad_accum 2 --dropout 0.1 --num_epochs 100 --learning_rate 6e-5 \
        --model_name_or_path convbert-dg --task intent --do_lowercase --max_seq_length 100 --mlm_pre --mlm_during --dump_outputs \
```

**CLINC150**

```
python run.py \
        --train_data_path data_utils/dialoglue/clinc/train.csv \
        --val_data_path data_utils/dialoglue/clinc/val.csv \
        --test_data_path data_utils/dialoglue/clinc/test.csv \
        --token_vocab_path bert-base-uncased-vocab.txt \
        --train_batch_size 64 --dropout 0.1 --num_epochs 100 --learning_rate 6e-5 \
        --model_name_or_path convbert-dg --task intent --do_lowercase --max_seq_length 50 --mlm_pre --mlm_during --dump_outputs \
```

**Restaurant8k**

```
python run.py \
        --train_data_path data_utils/dialoglue/restaurant8k/train.json \
        --val_data_path data_utils/dialoglue/restaurant8k/val.json \
        --test_data_path data_utils/dialoglue/restaurant8k/test.json \
        --token_vocab_path bert-base-uncased-vocab.txt \
        --train_batch_size 64 --dropout 0.1 --num_epochs 100 --learning_rate 6e-5 \
        --model_name_or_path convbert-dg --task slot --do_lowercase --max_seq_length 50 --mlm_pre --mlm_during --dump_outputs \
```

**DSTC8**

```
python run.py \
        --train_data_path data_utils/dialoglue/dstc8/train.json \
        --val_data_path data_utils/dialoglue/dstc8/val.json \
        --test_data_path data_utils/dialoglue/dstc8/test.json \
        --token_vocab_path bert-base-uncased-vocab.txt \
        --train_batch_size 64 --dropout 0.1 --num_epochs 100 --learning_rate 6e-5 \
        --model_name_or_path convbert-dg --task slot --do_lowercase --max_seq_length 50 --mlm_pre --mlm_during --dump_outputs \
```

**TOP**

```
python run.py \
        --train_data_path data_utils/dialoglue/top/train.txt \
        --val_data_path data_utils/dialoglue/top/eval.txt \
        --test_data_path data_utils/dialoglue/top/test.txt \
        --token_vocab_path bert-base-uncased-vocab.txt \
        --train_batch_size 64 --dropout 0.1 --num_epochs 100 --learning_rate 6e-5 \
        --model_name_or_path convbert-dg --task top --do_lowercase --max_seq_length 50 --mlm_pre --mlm_during --dump_outputs \
```

**MultiWOZ**

The MultiWOZ code builds on the open-sourced [TripPy implementation](https://gitlab.cs.uni-duesseldorf.de/general/dsml/trippy-public/-/tree/master). To train/evaluate the model using our modifications (i.e., MLM pre-training), you can use `trippy/DO.example.advanced`.

## Checkpoints

We release checkpoints for (1) ConvBERT, (2) BERT-DG and (3) ConvBERT-DG. Given these pre-trained models and the code in this repo, all of our results can be reproduced.

## License

This project is licensed under the Apache-2.0 License.

## Citation

If using these scripts or the DialoGLUE benchmark, please cite the following in any relevant work:

